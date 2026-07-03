#!/usr/bin/env python3
"""Step 0: data validation / quality gate for the auto-calibration pipeline.

Runs first. If any FATAL check fails the process exits non-zero so
``run_pipeline.py`` stops before doing any expensive work. Accuracy-related
signals (RTK covariance, GNSS status) are WARN-only per design: physical
plausibility (speed/continuity) is the primary gate.

Usage:
    python step0_validate.py <bag.mcap | split-bag-dir> [--config config.yaml]
                             [--report validation_report.json] [--require-cameras]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import common
import viz

N_CAMERAS = 12
CAMERA_IMAGE_TOPICS = [f"/sensing/camera/camera{i}/image_raw/compressed" for i in range(N_CAMERAS)]

# tf_static transforms that must be present for calibration to have initial values.
REQUIRED_TF = [
    ("drs_base_link", "lidar_front"),
    ("drs_base_link", "lidar_left"),
    ("drs_base_link", "lidar_rear"),
    ("drs_base_link", "lidar_right"),
    ("drs_base_link", "oxts_link"),
    ("base_link", "drs_base_link"),
]

PASS, WARN, FAIL = "pass", "warn", "fail"


class Report:
    def __init__(self):
        self.checks: list[dict] = []
        self.stats: dict = {}
        self.debug: dict = {}  # raw arrays for plotting only, not serialized to JSON

    def add(self, name, level, detail, **values):
        self.checks.append({"name": name, "level": level, "detail": detail, "values": values})

    @property
    def overall(self):
        return "FAIL" if any(c["level"] == FAIL for c in self.checks) else "PASS"


def _yaw_from_quat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _inspect_lidar_fields(files):
    """Deserialize the first message of each lidar topic to inspect fields."""
    from sensor_msgs.msg import PointCloud2
    from rclpy.serialization import deserialize_message

    result = {}
    for name, topic in common.LIDAR_TOPICS.items():
        result[name] = None
        for tp, _tns, data in common.iter_messages(files, [topic]):
            if tp != topic:
                continue
            m = deserialize_message(data, PointCloud2)
            arr = common.pointcloud2_to_array(m)
            names = [f.name for f in m.fields]
            tfield = common.find_time_field(names)
            unit = None
            if tfield is not None:
                unit = "us" if common.infer_time_unit_ns(arr[tfield]) == 1000.0 else "ns"
            result[name] = {
                "frame_id": m.header.frame_id,
                "fields": names,
                "time_field": tfield,
                "time_unit": unit,
            }
            break
    return result


def validate(path: str, cfg: dict) -> Report:
    files = common.resolve_bag_files(path)
    rep = Report()
    rep.stats["files"] = [f.split("/")[-1] for f in files]

    summary = common.bag_summary(files)
    tc = summary["topic_count"]
    dur = summary["duration_s"]
    rep.stats["duration_s"] = round(dur, 2)

    # --- FATAL: required topics ---
    missing = [t for t in list(common.LIDAR_TOPICS.values()) + [common.ODOM_TOPIC, common.TF_STATIC_TOPIC]
               if tc.get(t, 0) == 0]
    if missing:
        rep.add("required_topics", FAIL, f"missing/empty topics: {missing}", missing=missing)
    else:
        rep.add("required_topics", PASS, "all required lidar/odometry/tf_static topics present")

    # --- cameras (warn, or fatal if require_cameras) ---
    n_cam = sum(1 for t in CAMERA_IMAGE_TOPICS if tc.get(t, 0) > 0)
    if n_cam < N_CAMERAS:
        lvl = FAIL if cfg.get("require_cameras") else WARN
        rep.add("cameras", lvl, f"only {n_cam}/{N_CAMERAS} camera image topics present", n_cameras=n_cam)
    else:
        rep.add("cameras", PASS, f"{n_cam}/{N_CAMERAS} camera image topics present", n_cameras=n_cam)

    # --- FATAL: tf_static completeness ---
    tf = common.load_tf_static(files)
    tf_missing = [f"{p}->{c}" for (p, c) in REQUIRED_TF if (p, c) not in tf]
    n_lidar_cam = sum(1 for (p, c) in tf if c.endswith("/camera_link") and p.startswith("lidar_"))
    rep.stats["tf_static_transforms"] = len(tf)
    rep.stats["tf_lidar_to_camera"] = n_lidar_cam
    if tf_missing:
        rep.add("tf_static", FAIL, f"missing required transforms: {tf_missing}", missing=tf_missing)
    else:
        rep.add("tf_static", PASS,
                f"{len(tf)} transforms; all required present; {n_lidar_cam} lidar->camera_link")
    if not tf_missing and n_lidar_cam < N_CAMERAS:
        rep.add("tf_static_cameras", WARN,
                f"only {n_lidar_cam}/{N_CAMERAS} lidar->camera_link transforms", count=n_lidar_cam)

    # --- LiDAR per-point time field ---
    fields = _inspect_lidar_fields(files)
    no_time = [n for n, info in fields.items() if info is None or info["time_field"] is None]
    rep.stats["lidar_fields"] = fields
    if no_time:
        rep.add("lidar_time_field", FAIL,
                f"lidars without a usable per-point time field: {no_time}", lidars=no_time)
    else:
        units = {n: (info["time_field"], info["time_unit"]) for n, info in fields.items()}
        rep.add("lidar_time_field", PASS, f"per-point time fields present: {units}")

    # --- Odometry / RTK ---
    odom = common.load_odometry(files)
    ts, pos, quat, cov = odom["ts_ns"], odom["pos"], odom["quat"], odom["cov_xyz"]
    if odom["frame_id"] is None or len(ts) < 2:
        rep.add("rtk_present", FAIL, "no usable odometry messages")
        return rep

    frame_kind = common.detect_frame_kind(pos)
    rep.stats["odom_frame"] = f"{odom['frame_id']}->{odom['child_frame_id']}"
    rep.stats["map_frame_kind"] = frame_kind

    if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(quat))):
        rep.add("rtk_finite", FAIL, "odometry contains NaN/Inf in position or orientation")
    else:
        rep.add("rtk_finite", PASS, "odometry position/orientation finite")

    dt = np.diff(ts) / 1e9
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = step / np.clip(dt, 1e-6, None)
    reject = float(np.mean(speed > cfg["rtk_max_speed_mps"]))
    o = pos - pos[0]
    footprint = (float(o[:, 0].max() - o[:, 0].min()), float(o[:, 1].max() - o[:, 1].min()))
    rep.debug["t_rel_s"] = (ts - ts[0]) / 1e9
    rep.debug["xy"] = o[:, :2]
    rep.debug["speed_t_rel_s"] = (ts[1:] - ts[0]) / 1e9
    rep.debug["speed"] = speed
    rep.debug["reject_mask"] = speed > cfg["rtk_max_speed_mps"]
    rep.stats["speed_mps"] = {"median": round(float(np.median(speed)), 2),
                              "p95": round(float(np.percentile(speed, 95)), 2),
                              "max": round(float(speed.max()), 2)}
    rep.stats["footprint_m"] = [round(footprint[0], 1), round(footprint[1], 1)]
    rep.stats["reject_ratio"] = round(reject, 4)
    if reject > cfg["rtk_max_reject_ratio"]:
        rep.add("rtk_plausibility", FAIL,
                f"{reject:.1%} of steps exceed {cfg['rtk_max_speed_mps']} m/s "
                f"(> {cfg['rtk_max_reject_ratio']:.0%}) — INS solution unusable",
                reject_ratio=reject)
    else:
        rep.add("rtk_plausibility", PASS,
                f"speed within bounds (reject {reject:.2%}, max {speed.max():.1f} m/s)")

    # --- Odometry temporal coverage vs lidar scans ---
    max_gap = float(np.max(dt))
    rep.stats["odom_max_gap_s"] = round(max_gap, 4)
    if max_gap > cfg["rtk_max_gap_sec"]:
        rep.add("odom_continuity", FAIL,
                f"odometry gap {max_gap:.3f}s > {cfg['rtk_max_gap_sec']}s — interpolation unsafe",
                max_gap_s=max_gap)
    else:
        rep.add("odom_continuity", PASS, f"max odometry gap {max_gap:.3f}s")

    # Coverage vs the whole recording (bag span ⊇ lidar scan span). Fatal only
    # when odometry is missing for a large fraction of the run; small edge
    # shortfalls are handled by clamping in interpolation.
    odom_min, odom_max = float(ts.min()), float(ts.max())
    bag_start, bag_end = float(summary["t_start_ns"]), float(summary["t_end_ns"])
    tol = cfg["rtk_max_gap_sec"] * 1e9
    lead = (odom_min - bag_start) / 1e9   # seconds of run before odometry starts
    trail = (bag_end - odom_max) / 1e9    # seconds of run after odometry ends
    rep.stats["odom_coverage_s"] = {"lead_uncovered": round(lead, 3), "trail_uncovered": round(trail, 3)}
    if (odom_min - bag_start) > tol or (bag_end - odom_max) > tol:
        rep.add("odom_coverage", FAIL,
                f"odometry misses {lead:.2f}s at start / {trail:.2f}s at end of the recording "
                f"(> {cfg['rtk_max_gap_sec']}s) — scans there cannot be interpolated",
                lead_s=lead, trail_s=trail)
    else:
        rep.add("odom_coverage", PASS, "odometry spans the recording")

    # --- WARN: covariance ---
    cov_med = float(np.median(cov))
    rep.stats["cov_m2"] = {"min": float(cov.min()), "median": cov_med, "max": float(cov.max())}
    if cov_med > cfg["rtk_cov_warn_m2"]:
        rep.add("rtk_covariance", WARN,
                f"median position variance {cov_med:.3g} m² (std≈{np.sqrt(cov_med):.2f}m) "
                f"> {cfg['rtk_cov_warn_m2']} m² — coarser than cm-level RTK; caps achievable accuracy",
                cov_median_m2=cov_med)
    else:
        rep.add("rtk_covariance", PASS, f"median position variance {cov_med:.3g} m²")

    # --- WARN: GNSS status (unreliable) ---
    try:
        from sensor_msgs.msg import NavSatFix
        statuses = [m.status.status for m in common.read_deserialized(files, common.NAVSATFIX_TOPIC, NavSatFix)]
        if statuses:
            uniq = sorted(set(statuses))
            rep.stats["navsatfix_status"] = uniq
            if uniq == [0]:
                rep.add("gnss_status", WARN,
                        "NavSatFix.status always 0 — unreliable on this vehicle, informational only")
            else:
                rep.add("gnss_status", PASS, f"NavSatFix.status values {uniq}")
    except Exception:
        pass

    # --- WARN: yaw excitation ---
    yaw = np.unwrap(_yaw_from_quat(quat))
    yaw_cov_deg = float(np.degrees(np.abs(np.diff(yaw)).sum()))
    rep.stats["yaw_coverage_deg"] = round(yaw_cov_deg, 0)
    if yaw_cov_deg < cfg["min_yaw_coverage_deg"]:
        rep.add("yaw_excitation", WARN,
                f"cumulative |yaw| {yaw_cov_deg:.0f}° < {cfg['min_yaw_coverage_deg']:.0f}° "
                "— low rotational observability for lidar-to-lidar",
                yaw_coverage_deg=yaw_cov_deg)
    else:
        rep.add("yaw_excitation", PASS, f"cumulative |yaw| {yaw_cov_deg:.0f}°")

    return rep


def plot_report(rep: Report, out_dir: str, cfg: dict) -> list[str]:
    """Trajectory (top-down, colored by time, reject-speed points marked) and
    speed-vs-time (with the reject threshold) plots. No-op if odometry debug
    arrays weren't captured (e.g. rtk_present failed before reaching them)."""
    if "xy" not in rep.debug:
        return []
    paths = []
    xy = rep.debug["xy"]
    t = rep.debug["t_rel_s"]
    reject_xy = xy[1:][rep.debug["reject_mask"]]
    paths.append(viz.topdown_scatter(
        os.path.join(out_dir, "step0_trajectory.png"),
        [(xy, {"c": t, "cmap": "viridis", "s": 3, "label": "trajectory (color=time)"}),
         (reject_xy, {"c": "red", "s": 20, "marker": "x", "label": "speed-reject"})],
        cfg, title="Step 0: RTK trajectory (drs_base_link, local origin)"))
    paths.append(viz.line_plot(
        os.path.join(out_dir, "step0_speed.png"),
        [(rep.debug["speed_t_rel_s"], rep.debug["speed"], {"color": "steelblue", "linewidth": 0.7})],
        cfg, title="Step 0: speed vs time", xlabel="t [s]", ylabel="speed [m/s]",
        hlines=[(cfg["rtk_max_speed_mps"], dict(color="red", linestyle="--", linewidth=1,
                                                label="rtk_max_speed_mps"))]))
    return paths


def print_report(rep: Report):
    icon = {PASS: "✓", WARN: "⚠", FAIL: "✗"}
    print("=== Step 0: data validation ===")
    for k, v in rep.stats.items():
        if k not in ("lidar_fields",):
            print(f"  {k}: {v}")
    print("  checks:")
    for c in rep.checks:
        print(f"    [{icon[c['level']]}] {c['name']}: {c['detail']}")
    n_fail = sum(1 for c in rep.checks if c["level"] == FAIL)
    n_warn = sum(1 for c in rep.checks if c["level"] == WARN)
    print(f"=== {rep.overall}  ({n_fail} fatal, {n_warn} warnings) ===")


def _run(args, cfg) -> Report:
    rep = validate(args.bag, cfg)
    print_report(rep)

    report_path = args.report
    if report_path is None and args.out_dir:
        report_path = os.path.join(args.out_dir, "validation_report.json")
    if report_path:
        with open(report_path, "w") as f:
            json.dump({"overall": rep.overall, "checks": rep.checks, "stats": rep.stats},
                      f, indent=2, ensure_ascii=False)
        print(f"report written: {report_path}")

    if args.out_dir and cfg["viz_enabled"] and not args.no_viz:
        paths = plot_report(rep, args.out_dir, cfg)
        for p in paths:
            print(f"  wrote {p}")
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description="Auto-calib data validation / quality gate")
    ap.add_argument("bag", help="single .mcap or split rosbag2 directory")
    ap.add_argument("--config", default=None, help="config.yaml with threshold overrides")
    ap.add_argument("--report", default=None, help="path to write validation_report.json")
    ap.add_argument("--out-dir", default=None,
                    help="dir for validation_report.json (if --report unset), step0_log.txt, "
                         "and plots; omit to only print to stdout")
    ap.add_argument("--no-viz", action="store_true", help="skip PNG plot generation")
    ap.add_argument("--require-cameras", action="store_true",
                    help="treat missing camera topics as fatal")
    args = ap.parse_args(argv)

    cfg = common.load_config(args.config)
    if args.require_cameras:
        cfg["require_cameras"] = True

    if args.out_dir:
        with viz.tee_log(args.out_dir, "step0_log.txt"):
            rep = _run(args, cfg)
    else:
        rep = _run(args, cfg)

    return 0 if rep.overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
