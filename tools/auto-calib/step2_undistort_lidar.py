#!/usr/bin/env python3
"""Step 2: LiDAR scan de-skew (motion undistortion) using the RTK trajectory.

Each Seyond scan spans ~92 ms; the vehicle moves during the sweep, so points
acquired later in the scan are measured from a different pose. Using the
per-point relative time (t_us/time_stamp, auto-detected unit) we interpolate the
drs_base_link pose at each point's absolute time and transform every point back
into the lidar frame at the scan-start pose (t0 = header.stamp):

    p_world(t)   = map_T_drs(t) @ drs_T_lidar @ p_lidar
    p_lidar'(t0) = drs_T_lidar^-1 @ map_T_drs(t0)^-1 @ p_world(t)

Output stays in the lidar frame so Step 3 can place scans via map_T_lidar(t0).

Usage:
    python step2_undistort_lidar.py <bag> --rtk-poses OUT/rtk_poses.npy \
        --out-dir OUT [--lidars front,left,...] [--stride N] [--max-scans N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import common
import viz


def build_interpolator(rtk_poses_path):
    poses = np.load(rtk_poses_path)
    return common.TrajectoryInterpolator(poses[:, 0], poses[:, 1:4], poses[:, 4:8])


def undistort_scan(msg, traj, drs_T_lidar):
    """Return (out (M,4) float32 [x,y,z,intensity] in lidar frame at t0, t0_ns,
    max_disp, mean_disp, xyz_raw (M,3) float64 as originally measured,
    disp (M,) float64 per-point displacement magnitude)."""
    arr = common.pointcloud2_to_array(msg)
    names = arr.dtype.names
    xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
    intensity = arr["intensity"].astype(np.float32) if "intensity" in names else np.zeros(len(arr), np.float32)

    tfield = common.find_time_field(names)
    t0 = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    # Drop invalid (NaN/inf) points before transforming.
    valid = np.all(np.isfinite(xyz), axis=1)
    xyz = xyz[valid]
    intensity = intensity[valid]

    if tfield is None or len(xyz) == 0:
        out = np.empty((len(xyz), 4), dtype=np.float32)
        out[:, :3] = xyz
        out[:, 3] = intensity
        return out, int(t0), 0.0, 0.0, xyz, np.zeros(len(xyz))

    rel = arr[tfield][valid].astype(np.float64)
    unit_ns = common.infer_time_unit_ns(arr[tfield])
    t_point = t0 + rel * unit_ns

    R_dl = drs_T_lidar[:3, :3]
    t_dl = drs_T_lidar[:3, 3]
    p_drs = xyz @ R_dl.T + t_dl  # (M,3) in drs frame

    # Pose at scan-start and at each point time (local-origin frame; origin cancels).
    pos0, quat0 = traj.interpolate(np.array([t0], dtype=np.float64))
    R0 = common.quat_to_matrix(quat0[0])
    pos0 = pos0[0]
    pos_p, quat_p = traj.interpolate(t_point)
    R_p = common.quat_to_matrix(quat_p)  # (M,3,3)

    world = np.einsum("mij,mj->mi", R_p, p_drs) + pos_p
    p_drs0 = (world - pos0) @ R0  # R0.T @ v == v @ R0
    p_lidar0 = (p_drs0 - t_dl) @ R_dl  # R_dl.T @ v == v @ R_dl

    out = np.empty((len(xyz), 4), dtype=np.float32)
    out[:, :3] = p_lidar0.astype(np.float32)
    out[:, 3] = intensity
    disp = np.linalg.norm(p_lidar0 - xyz, axis=1)
    return out, int(t0), float(disp.max()), float(disp.mean()), xyz, disp


def plot_lidar_diag(ldir, name, idx_list, disp_max_list, disp_mean_list, samples, cfg):
    """samples: dict with 'first' and 'worst' -> (idx, xyz_deskewed (M,3), disp (M,)).

    Colors the deskewed scan by per-point displacement magnitude rather than
    overlaying raw-vs-deskewed points: at whole-scan scale a few-cm shift is
    invisible against a many-meter range, so an overlay renders as a single
    color hiding the other; a magnitude heatmap stays informative regardless
    of scan extent."""
    paths = []
    if idx_list:
        dmax_cm = np.asarray(disp_max_list) * 100
        dmean_cm = np.asarray(disp_mean_list) * 100
        paths.append(viz.line_plot(
            os.path.join(ldir, f"step2_{name}_disp_timeseries.png"),
            [(idx_list, dmax_cm, {"color": "tomato", "linewidth": 0.7, "label": "max"}),
             (idx_list, dmean_cm, {"color": "steelblue", "linewidth": 0.7, "label": "mean"})],
            cfg, title=f"Step 2 [{name}]: per-scan deskew displacement",
            xlabel="scan idx", ylabel="displacement [cm]"))
        paths.append(viz.hist_plot(
            os.path.join(ldir, f"step2_{name}_disp_hist.png"), dmax_cm, cfg, bins=40,
            title=f"Step 2 [{name}]: max per-scan displacement distribution",
            xlabel="max displacement [cm]"))
    for tag, sample in samples.items():
        if sample is None:
            continue
        sidx, desk, disp = sample
        paths.append(viz.topdown_scatter(
            os.path.join(ldir, f"step2_{name}_sample_{tag}_scan{sidx:06d}.png"),
            [(desk[:, :2], {"c": disp * 100, "cmap": "inferno",
                            "s": float(cfg["viz_point_size"]), "alpha": float(cfg["viz_point_alpha"])})],
            cfg, title=f"Step 2 [{name}]: scan {sidx} per-point deskew displacement [cm] ({tag})",
            colorbar_label="displacement [cm]"))
    return paths


def process_lidar(files, name, traj, drs_T_lidar, out_dir, cfg, stride, max_scans):
    from sensor_msgs.msg import PointCloud2
    from rclpy.serialization import deserialize_message

    topic = common.LIDAR_TOPICS[name]
    ldir = os.path.join(out_dir, "undistorted", name)
    os.makedirs(ldir, exist_ok=True)
    manifest = {"lidar": name, "topic": topic, "drs_T_lidar": drs_T_lidar.tolist(),
                "undistort_enabled": bool(cfg["undistort_enabled"]), "scans": []}
    idx = -1
    written = 0
    max_disp = 0.0
    idx_list, disp_max_list, disp_mean_list = [], [], []
    first_sample, worst_sample = None, None
    for tp, _tns, data in common.iter_messages(files, [topic]):
        if tp != topic:
            continue
        idx += 1
        if idx % stride != 0:
            continue
        if max_scans is not None and written >= max_scans:
            break
        msg = deserialize_message(data, PointCloud2)
        disp_entry = {}
        if cfg["undistort_enabled"]:
            out, t0, mx, mean, xyz_raw, disp = undistort_scan(msg, traj, drs_T_lidar)
            max_disp = max(max_disp, mx)
            idx_list.append(idx)
            disp_max_list.append(mx)
            disp_mean_list.append(mean)
            disp_entry = {"disp_max_m": round(mx, 5), "disp_mean_m": round(mean, 5)}
            if first_sample is None:
                first_sample = (idx, out[:, :3].astype(np.float64), disp)
            if worst_sample is None or mx > worst_sample[0]:
                worst_sample = (mx, idx, out[:, :3].astype(np.float64), disp)
        else:
            arr = common.pointcloud2_to_array(msg)
            names = arr.dtype.names
            inten = arr["intensity"] if "intensity" in names else np.zeros(len(arr))
            out = np.stack([arr["x"], arr["y"], arr["z"], inten], axis=1).astype(np.float32)
            t0 = int(msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec)
        fname = f"scan_{idx:06d}_{t0}.npy"
        np.save(os.path.join(ldir, fname), out)
        manifest["scans"].append({"idx": idx, "t0_ns": t0, "file": fname, "n": int(len(out)), **disp_entry})
        written += 1
    with open(os.path.join(ldir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    diag_paths = []
    if cfg["viz_enabled"] and idx_list:
        samples = {"first": first_sample,
                   "worst": worst_sample[1:] if worst_sample else None}
        if samples["worst"] and first_sample and samples["worst"][0] == first_sample[0]:
            samples["worst"] = None  # avoid duplicate plot when the worst scan is the first one
        diag_paths = plot_lidar_diag(ldir, name, idx_list, disp_max_list, disp_mean_list, samples, cfg)

    disp_stats = None
    if disp_max_list:
        dm = np.asarray(disp_max_list) * 100
        disp_stats = {"p50_cm": round(float(np.percentile(dm, 50)), 2),
                      "p95_cm": round(float(np.percentile(dm, 95)), 2),
                      "max_cm": round(float(dm.max()), 2)}
    return written, max_disp, disp_stats, diag_paths


def _run(args, cfg):
    stride = args.stride if args.stride is not None else int(cfg["lidar_scan_stride"])
    files = common.resolve_bag_files(args.bag)
    traj = build_interpolator(args.rtk_poses)
    tf = common.load_tf_static(files)

    lidars = [x.strip() for x in args.lidars.split(",") if x.strip()]
    for name in lidars:
        key = ("drs_base_link", f"lidar_{name}")
        if key not in tf:
            print(f"  [skip] {name}: {key} not in tf_static")
            continue
        n, max_disp, disp_stats, diag_paths = process_lidar(
            files, name, traj, tf[key], args.out_dir, cfg, stride, args.max_scans)
        print(f"Step 2 [{name}]: wrote {n} scans, max point de-skew {max_disp*100:.1f} cm")
        if disp_stats:
            print(f"  disp percentiles [cm]: p50={disp_stats['p50_cm']} "
                  f"p95={disp_stats['p95_cm']} max={disp_stats['max_cm']}")
        if not cfg["viz_enabled"] or args.no_viz:
            continue
        for p in diag_paths:
            print(f"  wrote {p}")
    print(f"  output under {args.out_dir}/undistorted/")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 2: LiDAR scan de-skew")
    ap.add_argument("bag", help="single .mcap or split rosbag2 directory")
    ap.add_argument("--rtk-poses", required=True, help="rtk_poses.npy from Step 1")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--lidars", default=",".join(common.LIDAR_NAMES),
                    help="comma-separated subset of front,left,rear,right")
    ap.add_argument("--stride", type=int, default=None, help="scan stride (default: cfg lidar_scan_stride)")
    ap.add_argument("--max-scans", type=int, default=None, help="cap scans per lidar (debug)")
    ap.add_argument("--no-viz", action="store_true", help="skip PNG plot generation")
    args = ap.parse_args(argv)

    cfg = common.load_config(args.config)
    with viz.tee_log(args.out_dir, "step2_log.txt"):
        return _run(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
