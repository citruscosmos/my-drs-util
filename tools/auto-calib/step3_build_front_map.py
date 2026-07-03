#!/usr/bin/env python3
"""Step 3: build the Front-LiDAR map in the (local-origin) map frame.

Each undistorted scan (Step 2 output, points in the lidar frame at t0) is placed
into the map frame via the RTK trajectory:

    map_T_front(t0) = map_T_drs(t0) @ drs_T_front
    p_map           = map_T_front(t0) @ p_lidar

Points are voxelized (map_voxel_size) and accumulated across all scans. A moving
object occupies a voxel in only a few scans, so we apply a **visibility-
normalized voxel voting filter**: for each voxel we keep it only if it was
occupied in at least `vote_threshold_ratio` of the scans that *could have seen
it* (center inside the sensor's range + FOV cone). Normalizing by visibility —
rather than by the total scan count — stops static structure that the vehicle
only briefly drove past from being deleted on longer runs.

The surviving voxel centroids ARE the downsampled map (one point per voxel), so
no extra voxel-downsample pass is needed.

`build_voxel_map()` is written to be reusable: Step 4 calls it with each other
LiDAR's current extrinsic to build that LiDAR's map for ICP.

Usage:
    python step3_build_front_map.py --out-dir OUT [--rtk-poses OUT/rtk_poses.npy]
        [--lidar front] [--config config.yaml] [--max-scans N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import common
import viz

# int64 voxel-key packing: 21 bits per axis, biased so coords in [-2^20, 2^20)
# (±~100 km at 0.1 m voxels) map to non-negative fields. 3*21 = 63 bits.
_KEY_BITS = 21
_KEY_MASK = (1 << _KEY_BITS) - 1
_KEY_OFFSET = 1 << (_KEY_BITS - 1)


def pack_voxel_keys(ijk: np.ndarray) -> np.ndarray:
    """Pack integer voxel coords (N,3) into a single int64 key per row."""
    a = (ijk[:, 0] + _KEY_OFFSET) & _KEY_MASK
    b = (ijk[:, 1] + _KEY_OFFSET) & _KEY_MASK
    c = (ijk[:, 2] + _KEY_OFFSET) & _KEY_MASK
    return (a << (2 * _KEY_BITS)) | (b << _KEY_BITS) | c


def load_scan_manifest(out_dir: str, lidar: str) -> dict:
    path = os.path.join(out_dir, "undistorted", lidar, "manifest.json")
    with open(path) as f:
        return json.load(f)


def write_pcd(path: str, xyz: np.ndarray, intensity: np.ndarray | None = None) -> None:
    """Write a binary (little-endian float32) PCD readable by Open3D / PCL."""
    n = len(xyz)
    has_i = intensity is not None
    fields = "x y z intensity" if has_i else "x y z"
    size = "4 4 4 4" if has_i else "4 4 4"
    typ = "F F F F" if has_i else "F F F"
    count = "1 1 1 1" if has_i else "1 1 1"
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        f"FIELDS {fields}\n"
        f"SIZE {size}\n"
        f"TYPE {typ}\n"
        f"COUNT {count}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    if has_i:
        arr = np.empty((n, 4), dtype=np.float32)
        arr[:, :3] = xyz
        arr[:, 3] = intensity
    else:
        arr = np.ascontiguousarray(xyz, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())


class _VoxelAccumulator:
    """Growable int64-keyed voxel store: per-voxel sum(xyz), sum(i), pts, scans."""

    def __init__(self, cap: int = 1 << 18):
        self.d: dict[int, int] = {}
        self.key = np.empty(cap, dtype=np.int64)
        self.sum = np.empty((cap, 3), dtype=np.float64)
        self.si = np.empty(cap, dtype=np.float64)
        self.cnt = np.empty(cap, dtype=np.int64)
        self.occ = np.empty(cap, dtype=np.int64)  # distinct scans occupying voxel
        self.n = 0

    def _grow(self, need: int) -> None:
        cap = len(self.key)
        while cap < need:
            cap *= 2
        self.key = np.resize(self.key, cap)
        self.sum = np.resize(self.sum, (cap, 3))
        self.si = np.resize(self.si, cap)
        self.cnt = np.resize(self.cnt, cap)
        self.occ = np.resize(self.occ, cap)

    def add_scan(self, keys: np.ndarray, world: np.ndarray, intensity: np.ndarray) -> None:
        """Merge one scan's per-voxel aggregates (occ += 1 per distinct voxel)."""
        uk, inv = np.unique(keys, return_inverse=True)
        su = np.zeros((len(uk), 3), dtype=np.float64)
        np.add.at(su, inv, world)
        si = np.zeros(len(uk), dtype=np.float64)
        np.add.at(si, inv, intensity)
        cu = np.bincount(inv, minlength=len(uk)).astype(np.int64)

        idx = np.fromiter((self.d.get(int(k), -1) for k in uk.tolist()),
                          dtype=np.int64, count=len(uk))
        ex = idx >= 0
        if np.any(ex):
            ei = idx[ex]
            self.sum[ei] += su[ex]
            self.si[ei] += si[ex]
            self.cnt[ei] += cu[ex]
            self.occ[ei] += 1
        new = ~ex
        m = int(np.count_nonzero(new))
        if m:
            if self.n + m > len(self.key):
                self._grow(self.n + m)
            rows = np.arange(self.n, self.n + m)
            nk = uk[new]
            self.key[rows] = nk
            self.sum[rows] = su[new]
            self.si[rows] = si[new]
            self.cnt[rows] = cu[new]
            self.occ[rows] = 1
            for k, r in zip(nk.tolist(), rows.tolist()):
                self.d[int(k)] = int(r)
            self.n += m

    def centroids(self):
        n = self.n
        centers = self.sum[:n] / self.cnt[:n][:, None]
        inten = self.si[:n] / self.cnt[:n]
        return centers, inten, self.occ[:n].copy()


def _scan_pose(traj, t0_ns, drs_T_lidar):
    """Return (R_map_lidar (3,3), origin_map (3,)) for a scan at t0."""
    pos0, quat0 = traj.interpolate(np.array([t0_ns], dtype=np.float64))
    R_drs = common.quat_to_matrix(quat0[0])
    map_T_drs = np.eye(4)
    map_T_drs[:3, :3] = R_drs
    map_T_drs[:3, 3] = pos0[0]
    map_T_lidar = map_T_drs @ drs_T_lidar
    return map_T_lidar[:3, :3], map_T_lidar[:3, 3]


def build_voxel_map(out_dir, traj, lidar, drs_T_lidar, cfg, max_scans=None, verbose=True):
    """Accumulate one LiDAR's undistorted scans into a visibility-voted voxel map.

    Returns a dict with the surviving voxel centroids and the diagnostic
    occupancy/visibility arrays for every raw voxel.
    """
    voxel = float(cfg["map_voxel_size"])
    stride = int(cfg["map_scan_stride"])
    manifest = load_scan_manifest(out_dir, lidar)
    scan_dir = os.path.join(out_dir, "undistorted", lidar)

    acc = _VoxelAccumulator()
    origins = []
    rots = []
    used = 0
    for i, s in enumerate(manifest["scans"]):
        if i % stride != 0:
            continue
        if max_scans is not None and used >= max_scans:
            break
        pts = np.load(os.path.join(scan_dir, s["file"]))
        if len(pts) == 0:
            continue
        xyz = pts[:, :3].astype(np.float64)
        inten = pts[:, 3].astype(np.float64) if pts.shape[1] > 3 else np.zeros(len(pts))
        R_ml, o_m = _scan_pose(traj, s["t0_ns"], drs_T_lidar)
        world = xyz @ R_ml.T + o_m
        ijk = np.floor(world / voxel).astype(np.int64)
        keys = pack_voxel_keys(ijk)
        acc.add_scan(keys, world, inten)
        origins.append(o_m)
        rots.append(R_ml)
        used += 1
        if verbose and used % 100 == 0:
            print(f"  [{lidar}] accumulated {used} scans, {acc.n} voxels")

    centers, inten, occ = acc.centroids()
    origins = np.asarray(origins, dtype=np.float64)
    rots = np.asarray(rots, dtype=np.float64)

    visible = _visibility_counts(centers, origins, rots, cfg)
    # A voxel is always visible to at least the scans that occupied it.
    denom = np.maximum(visible, occ)
    ratio = np.where(denom > 0, occ / denom, 0.0)
    keep = ratio >= float(cfg["vote_threshold_ratio"])

    return {
        "centers": centers,
        "intensity": inten,
        "occ": occ,
        "visible": visible,
        "ratio": ratio,
        "keep": keep,
        "n_scans": used,
        "voxel_size": voxel,
    }


def _visibility_counts(centers, origins, rots, cfg):
    """For each voxel, count scans whose range+FOV cone contains its center."""
    n = len(centers)
    visible = np.zeros(n, dtype=np.int64)
    if n == 0 or len(origins) == 0:
        return visible
    rmax2 = float(cfg["vote_max_range"]) ** 2
    rmin2 = float(cfg["vote_min_range"]) ** 2
    haz = np.deg2rad(float(cfg["vote_hfov_deg"]) / 2.0)
    hel = np.deg2rad(float(cfg["vote_vfov_deg"]) / 2.0)
    for o_m, R_ml in zip(origins, rots):
        d = centers - o_m
        dist2 = np.einsum("ij,ij->i", d, d)
        m = (dist2 <= rmax2) & (dist2 >= rmin2)
        if not np.any(m):
            continue
        idx = np.nonzero(m)[0]
        ds = d[idx] @ R_ml  # world -> sensor frame (R_ml.T @ d == d @ R_ml)
        az = np.abs(np.arctan2(ds[:, 1], ds[:, 0]))
        el = np.abs(np.arctan2(ds[:, 2], np.hypot(ds[:, 0], ds[:, 1])))
        in_fov = (az <= haz) & (el <= hel)
        visible[idx[in_fov]] += 1
    return visible


def plot_map(result, out_dir, lidar, cfg):
    """Top-down kept-map (colored by height), vote-ratio histogram, and a
    kept-vs-removed overlay to visually judge whether the vote filter is
    deleting real static structure vs. actual transient clutter."""
    keep = result["keep"]
    centers = result["centers"]
    ratio = result["ratio"]
    threshold = float(cfg["vote_threshold_ratio"])
    ps = float(cfg["viz_point_size"])
    pa = float(cfg["viz_point_alpha"])
    paths = []
    kept_xy = centers[keep][:, :2]
    kept_z = centers[keep][:, 2]
    paths.append(viz.topdown_scatter(
        os.path.join(out_dir, f"step3_{lidar}_map_topdown.png"),
        [(kept_xy, {"c": kept_z, "cmap": "turbo", "s": ps, "alpha": pa})],
        cfg, title=f"Step 3 [{lidar}]: kept map, colored by height (z)",
        colorbar_label="z [m]"))
    paths.append(viz.hist_plot(
        os.path.join(out_dir, f"step3_{lidar}_vote_hist.png"), ratio, cfg, bins=40,
        title=f"Step 3 [{lidar}]: voxel vote-ratio distribution", xlabel="occ/visible ratio",
        vlines=[(threshold, dict(color="red", linestyle="--", linewidth=1,
                                 label="vote_threshold_ratio"))]))
    removed_xy = centers[~keep][:, :2]
    paths.append(viz.topdown_scatter(
        os.path.join(out_dir, f"step3_{lidar}_removed_overlay.png"),
        [(kept_xy, {"c": "lightgray", "s": ps, "alpha": pa, "label": "kept"}),
         (removed_xy, {"c": "red", "s": ps, "alpha": pa, "label": "removed (vote filter)"})],
        cfg, title=f"Step 3 [{lidar}]: kept vs. removed voxels"))
    return paths


def _run(args, cfg):
    rtk_poses = args.rtk_poses or os.path.join(args.out_dir, "rtk_poses.npy")
    poses = np.load(rtk_poses)
    traj = common.TrajectoryInterpolator(poses[:, 0], poses[:, 1:4], poses[:, 4:8])

    manifest = load_scan_manifest(args.out_dir, args.lidar)
    drs_T_lidar = np.asarray(manifest["drs_T_lidar"], dtype=np.float64)
    override = common.load_tf_override(args.tf_override)
    if f"lidar_{args.lidar}" in override:
        drs_T_lidar = override[f"lidar_{args.lidar}"]
        print(f"  [override] using tf_override for lidar_{args.lidar}: "
              f"t={np.round(drs_T_lidar[:3, 3], 4).tolist()}")

    result = build_voxel_map(args.out_dir, traj, args.lidar, drs_T_lidar, cfg,
                             max_scans=args.max_scans)

    keep = result["keep"]
    centers = result["centers"][keep]
    inten = result["intensity"][keep]
    n_raw = len(result["keep"])
    n_keep = int(np.count_nonzero(keep))

    # Global origin (map-local frame is Step 1's origin ~ traj origin).
    map_origin = traj.local_origin.tolist()
    rtk_meta_path = os.path.join(args.out_dir, "rtk_meta.json")
    global_origin = None
    if os.path.isfile(rtk_meta_path):
        with open(rtk_meta_path) as f:
            global_origin = json.load(f).get("local_origin")

    out_pcd = os.path.join(args.out_dir, f"{args.lidar}_lidar_map.pcd")
    write_pcd(out_pcd, centers, inten)
    np.save(os.path.join(args.out_dir, f"{args.lidar}_lidar_map.npy"),
            np.column_stack([centers, inten]).astype(np.float32))

    meta = {
        "lidar": args.lidar,
        "n_scans": result["n_scans"],
        "voxel_size": result["voxel_size"],
        "vote_threshold_ratio": float(cfg["vote_threshold_ratio"]),
        "n_voxels_raw": n_raw,
        "n_voxels_kept": n_keep,
        "kept_ratio": (n_keep / n_raw) if n_raw else 0.0,
        "traj_local_origin": map_origin,
        "global_local_origin": global_origin,
        "drs_T_lidar": manifest["drs_T_lidar"],
    }
    if n_keep:
        lo = centers.min(axis=0)
        hi = centers.max(axis=0)
        meta["bbox_min"] = lo.tolist()
        meta["bbox_max"] = hi.tolist()
        meta["footprint_xy_m"] = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]
    with open(os.path.join(args.out_dir, f"{args.lidar}_lidar_map_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    removed = n_raw - n_keep
    print(f"Step 3 [{args.lidar}]: {result['n_scans']} scans -> "
          f"{n_raw} voxels, kept {n_keep} ({meta['kept_ratio']*100:.1f}%), "
          f"removed {removed} by vote filter")
    if n_keep:
        fp = meta["footprint_xy_m"]
        print(f"  map footprint {fp[0]:.1f} x {fp[1]:.1f} m -> {out_pcd}")
    if n_raw:
        r = result["ratio"]
        print(f"  vote ratio percentiles: p10={np.percentile(r,10):.2f} "
              f"p50={np.percentile(r,50):.2f} p90={np.percentile(r,90):.2f}")

    if n_keep and cfg["viz_enabled"] and not args.no_viz:
        for p in plot_map(result, args.out_dir, args.lidar, cfg):
            print(f"  wrote {p}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 3: build Front LiDAR map")
    ap.add_argument("--out-dir", required=True, help="dir with undistorted/ and rtk_poses.npy")
    ap.add_argument("--rtk-poses", default=None, help="default: <out-dir>/rtk_poses.npy")
    ap.add_argument("--lidar", default="front", help="which LiDAR map to build (default front)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tf-override", default=None,
                    help="multi_tf_static YAML overriding drs->lidar extrinsics "
                         "(use when the bag's /tf_static is wrong, e.g. front)")
    ap.add_argument("--max-scans", type=int, default=None, help="cap scans (debug)")
    ap.add_argument("--no-viz", action="store_true", help="skip PNG plot generation")
    args = ap.parse_args(argv)

    cfg = common.load_config(args.config)
    with viz.tee_log(args.out_dir, f"step3_{args.lidar}_log.txt"):
        return _run(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
