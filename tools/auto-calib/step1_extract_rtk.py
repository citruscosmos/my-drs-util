#!/usr/bin/env python3
"""Step 1: extract the RTK/INS trajectory as drs_base_link poses.

Reads /sensing/ins/oxts/odometry (map -> oxts_link), applies the
drs_base_link -> oxts_link correction (180 deg about X, so map_T_drs =
map_T_oxts @ inv(drs_T_oxts)), shifts positions to a local origin (float64,
safe for ECEF-scale input), and writes:

  rtk_poses.npy  — (N,8) float64 = [timestamp_ns, x,y,z, qx,qy,qz,qw]
  rtk_meta.json  — local_origin, frame_kind, and provenance

Usage:
    python step1_extract_rtk.py <bag> [--config config.yaml] --out-dir OUT
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import common
import viz


def _yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def extract(files, cfg):
    odom = common.load_odometry(files)
    ts = odom["ts_ns"]
    pos = odom["pos"]
    quat = common.quat_normalize(odom["quat"])
    if odom["frame_id"] is None or len(ts) < 2:
        raise RuntimeError("no usable odometry in bag")

    order = np.argsort(ts, kind="stable")
    ts, pos, quat = ts[order], pos[order], quat[order]

    frame_kind = common.detect_frame_kind(pos)
    local_origin = pos[0].copy()
    pos_local = pos - local_origin

    # map_T_drs = map_T_oxts @ inv(drs_T_oxts). drs_T_oxts has zero translation,
    # so drs position == oxts position; only orientation changes:
    #   q_drs = q_oxts ⊗ conj(q_drs_oxts)
    drs_T_oxts = common.make_transform((0, 0, 0), common.DRS_T_OXTS_QUAT)
    q_drs_oxts = common.matrix_to_quat(drs_T_oxts[:3, :3])
    conj = q_drs_oxts * np.array([-1.0, -1.0, -1.0, 1.0])
    quat_drs = common.quat_normalize(common.quat_multiply(quat, conj))

    poses = np.empty((len(ts), 8), dtype=np.float64)
    poses[:, 0] = ts
    poses[:, 1:4] = pos_local
    poses[:, 4:8] = quat_drs

    dt = np.diff(ts) / 1e9
    meta = {
        "n": int(len(ts)),
        "frame_id": odom["frame_id"],
        "child_frame_id": odom["child_frame_id"],
        "frame_kind": frame_kind,
        "local_origin": local_origin.tolist(),
        "drs_correction": "map_T_drs = map_T_oxts @ inv(drs_T_oxts) [180deg X]",
        "t_start_ns": int(ts[0]),
        "t_end_ns": int(ts[-1]),
        "dt_s": {"median": round(float(np.median(dt)), 4), "p95": round(float(np.percentile(dt, 95)), 4),
                 "max": round(float(dt.max()), 4)},
    }
    return poses, meta


def plot_extraction(poses: np.ndarray, out_dir: str, cfg: dict) -> list[str]:
    t = (poses[:, 0] - poses[0, 0]) / 1e9
    xy = poses[:, 1:3]
    z = poses[:, 3]
    yaw_deg = np.degrees(_yaw_from_quat(poses[:, 4:8]))
    paths = []
    paths.append(viz.topdown_scatter(
        os.path.join(out_dir, "step1_trajectory.png"),
        [(xy, {"c": t, "cmap": "viridis", "s": 3, "label": "drs_base_link (color=time)"})],
        cfg, title="Step 1: extracted trajectory (local-origin map frame)",
        extra=lambda ax: (ax.scatter(*xy[0], c="lime", s=80, marker="^", zorder=5, label="start"),
                          ax.scatter(*xy[-1], c="red", s=80, marker="v", zorder=5, label="end"))))
    paths.append(viz.line_plot(
        os.path.join(out_dir, "step1_z_yaw.png"),
        [(t, z, {"color": "steelblue", "linewidth": 0.7})],
        cfg, title="Step 1: z (height) vs time", xlabel="t [s]", ylabel="z [m]"))
    paths.append(viz.line_plot(
        os.path.join(out_dir, "step1_yaw.png"),
        [(t, yaw_deg, {"color": "darkorange", "linewidth": 0.7})],
        cfg, title="Step 1: yaw vs time", xlabel="t [s]", ylabel="yaw [deg]"))
    return paths


def _run(args, cfg):
    files = common.resolve_bag_files(args.bag)
    poses, meta = extract(files, cfg)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "rtk_poses.npy"), poses)
    with open(os.path.join(args.out_dir, "rtk_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    o = poses[:, 1:4]
    print(f"Step 1: extracted {meta['n']} drs_base_link poses ({meta['frame_kind']} map)")
    print(f"  local_origin: {[round(v, 3) for v in meta['local_origin']]}")
    print(f"  footprint: {np.ptp(o[:,0]):.1f} x {np.ptp(o[:,1]):.1f} m, "
          f"z-span {np.ptp(o[:,2]):.2f} m")
    print(f"  dt [s]: median {meta['dt_s']['median']}, p95 {meta['dt_s']['p95']}, "
          f"max {meta['dt_s']['max']}")
    print(f"  wrote {args.out_dir}/rtk_poses.npy, rtk_meta.json")

    if cfg["viz_enabled"] and not args.no_viz:
        for p in plot_extraction(poses, args.out_dir, cfg):
            print(f"  wrote {p}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 1: RTK trajectory extraction")
    ap.add_argument("bag", help="single .mcap or split rosbag2 directory")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", required=True, help="output directory")
    ap.add_argument("--no-viz", action="store_true", help="skip PNG plot generation")
    args = ap.parse_args(argv)

    cfg = common.load_config(args.config)
    with viz.tee_log(args.out_dir, "step1_log.txt"):
        return _run(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
