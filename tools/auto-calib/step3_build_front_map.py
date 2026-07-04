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
    """int64-keyed voxel store: per-voxel sum(xyz), sum(i), pts, scans.

    Each scan is deduped to unique voxel keys locally (vectorized, cheap); the
    cross-scan merge is deferred to a single global np.unique in centroids()
    instead of an incremental per-scan Python dict merge. The incremental
    version's `self.d.get(int(k), -1)` generator did one Python-level dict
    lookup per unique voxel key per scan (measured as Step 3's single largest
    cost: ~480k dict.get() calls for just 100 scans). Deferring the merge
    replaces that with two global vectorized calls total, independent of
    scan count.
    """

    def __init__(self):
        self._keys: list[np.ndarray] = []
        self._sums: list[np.ndarray] = []
        self._sqs: list[np.ndarray] = []
        self._sis: list[np.ndarray] = []
        self._cnts: list[np.ndarray] = []

    def add_scan(self, keys: np.ndarray, world: np.ndarray, intensity: np.ndarray) -> None:
        """Dedup one scan's points to unique voxel keys; queue for the global merge."""
        uk, inv = np.unique(keys, return_inverse=True)
        su = np.zeros((len(uk), 3), dtype=np.float64)
        np.add.at(su, inv, world)
        sq = np.zeros((len(uk), 3), dtype=np.float64)
        np.add.at(sq, inv, world ** 2)
        si = np.zeros(len(uk), dtype=np.float64)
        np.add.at(si, inv, intensity)
        cu = np.bincount(inv, minlength=len(uk)).astype(np.int64)
        self._keys.append(uk)
        self._sums.append(su)
        self._sqs.append(sq)
        self._sis.append(si)
        self._cnts.append(cu)

    @property
    def n(self) -> int:
        """Voxel-observations queued so far (pre-merge; a progress proxy, not
        the final distinct-voxel count -- that's only known after centroids())."""
        return sum(len(k) for k in self._keys)

    def centroids(self):
        """One global merge across all scans' deduped voxels. occ = number of
        distinct scans touching each voxel, since each scan contributes each
        of its own voxel keys at most once here. spread = RMS radial spread
        of the raw points around their voxel centroid (sqrt of the trace of
        the per-voxel point covariance, using all constituent points across
        all scans, not just per-scan means) -- a direct, cheap (one extra
        sum-of-squares accumulator) measure of front-map "smear": how far the
        observations that got averaged into a centroid actually disagreed
        with each other. Voxels with cnt<2 have spread=0 (undefined, single
        observation trivially matches its own mean) -- filter those out when
        interpreting the spread distribution."""
        if not self._keys:
            return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=np.int64), np.zeros(0)
        uk, inv = np.unique(np.concatenate(self._keys), return_inverse=True)
        n = len(uk)
        sum_xyz = np.zeros((n, 3), dtype=np.float64)
        np.add.at(sum_xyz, inv, np.concatenate(self._sums))
        sum_xyz2 = np.zeros((n, 3), dtype=np.float64)
        np.add.at(sum_xyz2, inv, np.concatenate(self._sqs))
        si = np.zeros(n, dtype=np.float64)
        np.add.at(si, inv, np.concatenate(self._sis))
        cnt = np.bincount(inv, weights=np.concatenate(self._cnts), minlength=n).astype(np.int64)
        occ = np.bincount(inv, minlength=n).astype(np.int64)
        centers = sum_xyz / cnt[:, None]
        inten = si / cnt
        var_xyz = np.maximum(sum_xyz2 / cnt[:, None] - centers ** 2, 0.0)
        spread = np.sqrt(var_xyz.sum(axis=1))
        return centers, inten, occ, spread


def _scan_pose(traj, t0_ns, drs_T_lidar):
    """Return (R_map_lidar (3,3), origin_map (3,)) for a scan at t0."""
    pos0, quat0 = traj.interpolate(np.array([t0_ns], dtype=np.float64))
    R_drs = common.quat_to_matrix(quat0[0])
    map_T_drs = np.eye(4)
    map_T_drs[:3, :3] = R_drs
    map_T_drs[:3, 3] = pos0[0]
    map_T_lidar = map_T_drs @ drs_T_lidar
    return map_T_lidar[:3, :3], map_T_lidar[:3, 3]


# int64 packing for the two arc-length bin indices of the occlusion range
# image (see vote_occlusion_footprint_m). 26 bits/axis, offset-biased so
# indices in [-2^25, 2^25) map to non-negative fields; generous headroom for
# a 60m-range, 120deg-FOV arc-length extent at sub-decimeter footprints.
_OCC_KEY_BITS = 26
_OCC_KEY_MASK = (1 << _OCC_KEY_BITS) - 1
_OCC_KEY_OFFSET = 1 << (_OCC_KEY_BITS - 1)


def _pack_occ_keys(key_az: np.ndarray, key_el: np.ndarray) -> np.ndarray:
    a = (key_az + _OCC_KEY_OFFSET) & _OCC_KEY_MASK
    b = (key_el + _OCC_KEY_OFFSET) & _OCC_KEY_MASK
    return (a.astype(np.int64) << _OCC_KEY_BITS) | b.astype(np.int64)


def _arc_length_keys(az, el, r, footprint_m):
    """Bin by metric arc-length (angle * that point's own range) instead of
    raw angle, so the bin footprint is ~constant in meters at any range
    (see vote_occlusion_footprint_m's config comment for why)."""
    key_az = np.floor((az * r) / footprint_m).astype(np.int64)
    key_el = np.floor((el * r) / footprint_m).astype(np.int64)
    return _pack_occ_keys(key_az, key_el)


def _range_image(xyz, haz, hel, footprint_m):
    """Per-scan sparse min-range lookup keyed by arc-length bin, in the
    LiDAR's own sensor frame (xyz here is pre-world-transform, i.e. already
    sensor-local). Returns (sorted_keys, min_range_per_key)."""
    r = np.linalg.norm(xyz, axis=1)
    m = r > 1e-6
    x, y, z, r = xyz[m, 0], xyz[m, 1], xyz[m, 2], r[m]
    az = np.arctan2(y, x)
    el = np.arctan2(z, np.hypot(x, y))
    infov = (np.abs(az) <= haz) & (np.abs(el) <= hel)
    if not np.any(infov):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    keys = _arc_length_keys(az[infov], el[infov], r[infov], footprint_m)
    uk, inv = np.unique(keys, return_inverse=True)
    minr = np.full(len(uk), np.inf, dtype=np.float64)
    np.minimum.at(minr, inv, r[infov])
    return uk, minr


def build_voxel_map(out_dir, traj, lidar, drs_T_lidar, cfg, max_scans=None, verbose=True,
                    scan_filter=None):
    """Accumulate one LiDAR's undistorted scans into a visibility-voted voxel map.

    scan_filter: optional callable(scan_index, scan_dict, t0_ns) -> bool, applied
    after the map_scan_stride/max_scans selection, to restrict which scans are
    accumulated (e.g. isolating turning vs. straight segments for smear analysis).

    Returns a dict with the surviving voxel centroids and the diagnostic
    occupancy/visibility/spread arrays for every raw voxel.
    """
    voxel = float(cfg["map_voxel_size"])
    stride = int(cfg["map_scan_stride"])
    manifest = load_scan_manifest(out_dir, lidar)
    scan_dir = os.path.join(out_dir, "undistorted", lidar)

    occlusion_on = bool(cfg.get("vote_occlusion_enabled", True))
    haz = np.deg2rad(float(cfg["vote_hfov_deg"]) / 2.0)
    hel = np.deg2rad(float(cfg["vote_vfov_deg"]) / 2.0)
    footprint_m = float(cfg["vote_occlusion_footprint_m"])

    acc = _VoxelAccumulator()
    origins = []
    rots = []
    range_grids = [] if occlusion_on else None
    used = 0
    for i, s in enumerate(manifest["scans"]):
        if i % stride != 0:
            continue
        if max_scans is not None and used >= max_scans:
            break
        if scan_filter is not None and not scan_filter(i, s, s["t0_ns"]):
            continue
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
        if occlusion_on:
            range_grids.append(_range_image(xyz, haz, hel, footprint_m))
        origins.append(o_m)
        rots.append(R_ml)
        used += 1
        if verbose and used % 100 == 0:
            print(f"  [{lidar}] accumulated {used} scans, {acc.n} voxels")

    centers, inten, occ, spread = acc.centroids()
    origins = np.asarray(origins, dtype=np.float64)
    rots = np.asarray(rots, dtype=np.float64)

    visible = _visibility_counts(centers, origins, rots, cfg, range_grids, footprint_m)
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
        "spread": spread,
        "n_scans": used,
        "voxel_size": voxel,
    }


def _gpu_device(cfg):
    """Return an Open3D CUDA device if GPU accel is enabled + available, else None."""
    if not cfg.get("gpu_enabled", True):
        return None
    try:
        import open3d as o3d
        if o3d.core.cuda.is_available():
            return o3d.core.Device("CUDA:0")
    except Exception:
        pass
    return None


def _range_masks_cpu(centers, origins, rmax2, rmin2):
    masks = []
    for o_m in origins:
        d = centers - o_m
        dist2 = np.einsum("ij,ij->i", d, d)
        masks.append((dist2 <= rmax2) & (dist2 >= rmin2))
    return masks


def _range_masks_gpu(centers, origins, rmax2, rmin2, device, cfg):
    """Same range mask as _range_masks_cpu, batched across chunks of scans on
    GPU (Open3D tensor broadcasting). Only the boolean mask is transferred
    back to host, not the (chunk x n_voxels x 3) delta tensor, which is what
    makes this ~13x faster than the CPU loop in practice: the range filter is
    O(n_voxels x n_scans) and dominates Step 3 at full map scale, while
    everything downstream of it (FOV angle check) runs on the much smaller
    in-range subset per scan and stays on CPU unchanged (see _visibility_counts).
    Uses float32 on GPU; at map extents of ~100s of meters this is far more
    precise than the meter-scale range/FOV thresholds being tested, so the
    only expected discrepancy vs. the float64 CPU path is an occasional voxel
    exactly on a range/FOV boundary flipping sides -- immaterial for a
    visibility-vote heuristic."""
    import open3d as o3d
    chunk = max(1, int(cfg.get("gpu_visibility_chunk_scans", 32)))
    n = len(centers)
    centers_t = o3d.core.Tensor(centers.astype(np.float32), device=device).reshape((1, n, 3))
    masks = []
    for s0 in range(0, len(origins), chunk):
        o_chunk = np.asarray(origins[s0:s0 + chunk], dtype=np.float32)
        o_t = o3d.core.Tensor(o_chunk, device=device).reshape((len(o_chunk), 1, 3))
        d = centers_t - o_t
        dist2 = (d * d).sum(dim=2)
        mask = (dist2 <= rmax2).logical_and(dist2 >= rmin2)
        mask_np = mask.cpu().numpy()
        masks.extend(mask_np[k] for k in range(len(o_chunk)))
    return masks


def _visibility_counts(centers, origins, rots, cfg, range_grids=None, footprint_m=0.0):
    """For each voxel, count scans that could have hit it: within range+FOV of
    the scan origin, AND (if occlusion checking is enabled) not blocked by a
    closer actual return in the same arc-length bin of that scan's own range
    image. Without occlusion checking this reduces to the plain range+FOV
    cone test (kept as a fallback / A-B toggle via vote_occlusion_enabled)."""
    n = len(centers)
    visible = np.zeros(n, dtype=np.int64)
    if n == 0 or len(origins) == 0:
        return visible
    rmax2 = float(cfg["vote_max_range"]) ** 2
    rmin2 = float(cfg["vote_min_range"]) ** 2
    haz = np.deg2rad(float(cfg["vote_hfov_deg"]) / 2.0)
    hel = np.deg2rad(float(cfg["vote_vfov_deg"]) / 2.0)
    occlusion_on = bool(cfg.get("vote_occlusion_enabled", True)) and range_grids is not None
    tol = float(cfg.get("vote_occlusion_tol", 0.5))

    device = _gpu_device(cfg)
    range_masks = (_range_masks_gpu(centers, origins, rmax2, rmin2, device, cfg) if device is not None
                  else _range_masks_cpu(centers, origins, rmax2, rmin2))

    for k, R_ml in enumerate(rots):
        m = range_masks[k]
        if not np.any(m):
            continue
        idx = np.nonzero(m)[0]
        d = centers[idx] - origins[k]
        ds = d @ R_ml  # world -> sensor frame (R_ml.T @ d == d @ R_ml)
        rng = np.linalg.norm(ds, axis=1)
        az = np.arctan2(ds[:, 1], ds[:, 0])
        el = np.arctan2(ds[:, 2], np.hypot(ds[:, 0], ds[:, 1]))
        in_fov = (np.abs(az) <= haz) & (np.abs(el) <= hel)
        if not np.any(in_fov):
            continue
        vis_idx = idx[in_fov]
        if occlusion_on:
            uk, minr = range_grids[k]
            cand_keys = _arc_length_keys(az[in_fov], el[in_fov], rng[in_fov], footprint_m)
            if len(uk) == 0:
                not_occluded = np.zeros(len(cand_keys), dtype=bool)
            else:
                pos = np.clip(np.searchsorted(uk, cand_keys), 0, len(uk) - 1)
                found = uk[pos] == cand_keys
                measured = np.where(found, minr[pos], np.inf)
                # A bin with no return at all (inf) is treated as no evidence
                # of visibility, not as "nothing closer so it must be
                # visible" -- see the vote_occlusion_enabled config comment.
                not_occluded = found & (measured >= (rng[in_fov] - tol))
            vis_idx = vis_idx[not_occluded]
        visible[vis_idx] += 1
    return visible


def plot_map(result, out_dir, lidar, cfg):
    """Top-down kept-map (colored by height), vote-ratio histogram, a
    kept-vs-removed overlay to visually judge whether the vote filter is
    deleting real static structure vs. actual transient clutter, and a
    voxel-spread ("smear") histogram + topdown map."""
    keep = result["keep"]
    centers = result["centers"]
    ratio = result["ratio"]
    occ = result["occ"]
    spread = result["spread"]
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
    # Spread ("smear") is only meaningful for voxels with >=2 contributing
    # points; single-observation voxels trivially have spread=0.
    multi = keep & (occ >= 2)
    if np.any(multi):
        paths.append(viz.hist_plot(
            os.path.join(out_dir, f"step3_{lidar}_spread_hist.png"), spread[multi], cfg, bins=40,
            title=f"Step 3 [{lidar}]: kept-voxel spread (smear) distribution [m]",
            xlabel="RMS radial spread [m]"))
        paths.append(viz.topdown_scatter(
            os.path.join(out_dir, f"step3_{lidar}_spread_topdown.png"),
            [(centers[multi][:, :2], {"c": spread[multi], "cmap": "turbo", "s": ps, "alpha": pa,
                                      "vmax": np.percentile(spread[multi], 95)})],
            cfg, title=f"Step 3 [{lidar}]: kept-voxel spread (smear), colored [m]",
            colorbar_label="RMS radial spread [m]"))
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
    multi = keep & (result["occ"] >= 2) if n_keep else np.zeros(0, dtype=bool)
    if np.any(multi):
        sp = result["spread"][multi]
        sp_cm = sp * 100.0
        meta["spread_cm_p10_p50_p90"] = [float(np.percentile(sp_cm, q)) for q in (10, 50, 90)]
        print(f"  kept-voxel spread (smear) [cm], occ>=2 (n={multi.sum()}): "
              f"p10={np.percentile(sp_cm,10):.2f} p50={np.percentile(sp_cm,50):.2f} "
              f"p90={np.percentile(sp_cm,90):.2f}")

    with open(os.path.join(args.out_dir, f"{args.lidar}_lidar_map_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

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
