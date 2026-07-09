#!/usr/bin/env python3
"""Run Steps 0-5 of the auto-calibration pipeline end-to-end against one bag.

Step 0 is a hard gate: on FAIL the pipeline stops before any expensive work
(matches the individual step's own contract). Steps 1-5 each write their own
log/plots under --out-dir (step0_log.txt, step1_log.txt, ..., undistorted/,
front_lidar_map.*, lidar_to_lidar_result.json, cam_to_lidar_result.json,
cam_to_lidar_v3_result.json, step5_verify_camera*/, ...); this script
additionally tees its own stdout to OUT/pipeline_log.txt and prints a
per-step wall-clock timing summary at the end.

Step 5 follows the fixed camera routing decided in docs/architecture.md
(narrow-FOV front-facing cameras -> v2 scan-to-image DT minimization;
everything else -> v3 whole-trajectory camera cloud + LiDAR-map
registration, RTK-fix bags only), then verifies every routed camera with
step5_verify_projection.py (before/after overlays at low-speed scenes).

Usage:
    tools/auto-calib/run.sh run_pipeline.py <bag> --out-dir OUT \
        [--config config.yaml] [--tf-override multi_tf_static.yaml] \
        [--lidars front,left,rear,right] [--stride N] [--max-scans N] \
        [--v2-cameras 0,4] [--v3-cameras 1,2,3,5,6,7,8,9,10,11] \
        [--skip-step5] [--skip-verify] \
        [--no-viz] [--require-cameras] [--force]
"""
from __future__ import annotations

import argparse
import sys
import time

import common
import step0_validate
import step1_extract_rtk
import step2_undistort_lidar
import step3_build_front_map
import step4_lidar_to_lidar
import step5_cam_to_lidar
import step5_v3_cam_cloud
import step5_v3_register
import step5_verify_projection
import viz


def _parse_cam_list(s):
    return [int(c) for c in s.split(",") if c.strip()]


def _timed(name, fn):
    print(f"\n===== {name} =====", flush=True)
    t0 = time.time()
    rc = fn()
    dt = time.time() - t0
    print(f"===== {name} done in {dt:.1f}s (rc={rc}) =====", flush=True)
    return rc, dt


def _run(args):
    out = args.out_dir
    common_opts = []
    if args.config:
        common_opts += ["--config", args.config]
    viz_opts = ["--no-viz"] if args.no_viz else []

    timings = []

    def step0():
        argv = [args.bag, "--out-dir", out] + common_opts
        if args.require_cameras:
            argv.append("--require-cameras")
        return step0_validate.main(argv)

    rc, dt = _timed("Step 0: validate", step0)
    timings.append(("step0_validate", dt, rc))
    if rc != 0 and not args.force:
        print("\nStep 0 FAILed the quality gate; stopping (use --force to continue anyway).")
        _print_timings(timings)
        return 1

    def step1():
        argv = [args.bag, "--out-dir", out] + common_opts + viz_opts
        return step1_extract_rtk.main(argv)

    rc, dt = _timed("Step 1: extract RTK", step1)
    timings.append(("step1_extract_rtk", dt, rc))
    if rc != 0 and not args.force:
        _print_timings(timings)
        return rc

    def step2():
        argv = [args.bag, "--rtk-poses", f"{out}/rtk_poses.npy", "--out-dir", out,
                 "--lidars", args.lidars] + common_opts + viz_opts
        if args.stride is not None:
            argv += ["--stride", str(args.stride)]
        if args.max_scans is not None:
            argv += ["--max-scans", str(args.max_scans)]
        return step2_undistort_lidar.main(argv)

    rc, dt = _timed("Step 2: undistort LiDAR scans", step2)
    timings.append(("step2_undistort_lidar", dt, rc))
    if rc != 0 and not args.force:
        _print_timings(timings)
        return rc

    def step3():
        argv = ["--out-dir", out, "--lidar", "front"] + common_opts + viz_opts
        if args.tf_override:
            argv += ["--tf-override", args.tf_override]
        if args.max_scans is not None:
            argv += ["--max-scans", str(args.max_scans)]
        return step3_build_front_map.main(argv)

    rc, dt = _timed("Step 3: build front LiDAR map", step3)
    timings.append(("step3_build_front_map", dt, rc))
    if rc != 0 and not args.force:
        _print_timings(timings)
        return rc

    def step4():
        argv = ["--out-dir", out] + common_opts + viz_opts
        if args.tf_override:
            argv += ["--tf-override", args.tf_override]
        if args.max_scans is not None:
            argv += ["--max-scans", str(args.max_scans)]
        return step4_lidar_to_lidar.main(argv)

    rc, dt = _timed("Step 4: LiDAR-to-LiDAR extrinsic", step4)
    timings.append(("step4_lidar_to_lidar", dt, rc))
    if rc != 0 and not args.force:
        _print_timings(timings)
        return rc

    if args.skip_step5:
        _print_timings(timings)
        return rc

    v2_cams = _parse_cam_list(args.v2_cameras)
    v3_cams = _parse_cam_list(args.v3_cameras)

    def step5v2():
        if not v2_cams:
            return 0
        argv = [args.bag, "--out-dir", out, "--cameras", args.v2_cameras] + common_opts
        if args.tf_override:
            argv += ["--tf-override", args.tf_override]
        return step5_cam_to_lidar.main(argv)

    rc, dt = _timed("Step 5a: camera-to-LiDAR (v2)", step5v2)
    timings.append(("step5a_cam_to_lidar_v2", dt, rc))
    if rc != 0 and not args.force:
        _print_timings(timings)
        return rc

    def step5v3_camcloud():
        if not v3_cams:
            return 0
        argv = [args.bag, "--out-dir", out, "--cameras", args.v3_cameras] + common_opts
        return step5_v3_cam_cloud.main(argv)

    rc, dt = _timed("Step 5b: camera cloud tracking (v3)", step5v3_camcloud)
    timings.append(("step5b_v3_cam_cloud", dt, rc))
    if rc != 0 and not args.force:
        _print_timings(timings)
        return rc

    def step5v3_register():
        if not v3_cams:
            return 0
        argv = [args.bag, "--out-dir", out, "--cameras", args.v3_cameras] + common_opts
        if args.tf_override:
            argv += ["--tf-override", args.tf_override]
        return step5_v3_register.main(argv)

    rc, dt = _timed("Step 5c: camera cloud registration (v3)", step5v3_register)
    timings.append(("step5c_v3_register", dt, rc))
    if rc != 0 and not args.force:
        _print_timings(timings)
        return rc

    if args.skip_verify:
        _print_timings(timings)
        return rc

    def step5_verify():
        rc_final = 0
        for cam in sorted(set(v2_cams) | set(v3_cams)):
            argv = [args.bag, "--out-dir", out, "--camera", str(cam)] + common_opts
            if args.tf_override:
                argv += ["--tf-override", args.tf_override]
            rc_cam = step5_verify_projection.main(argv)
            rc_final = rc_final or rc_cam
        return rc_final

    rc, dt = _timed("Step 5d: projection verification", step5_verify)
    timings.append(("step5d_verify_projection", dt, rc))

    _print_timings(timings)
    return rc


def _print_timings(timings):
    total = sum(dt for _, dt, _ in timings)
    print("\n=== pipeline timing summary ===")
    for name, dt, rc in timings:
        status = "ok" if rc == 0 else f"rc={rc}"
        print(f"  {name:28s} {dt:8.1f}s  [{status}]")
    print(f"  {'total':28s} {total:8.1f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run Step 0-5 of the auto-calibration pipeline")
    ap.add_argument("bag", help="single .mcap or split rosbag2 directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tf-override", default=None,
                    help="multi_tf_static YAML overriding drs->lidar extrinsics "
                         "(passed to Steps 3, 4 and 5; MUST be the same file throughout)")
    ap.add_argument("--lidars", default=",".join(common.LIDAR_NAMES),
                    help="comma-separated subset of front,left,rear,right for Step 2")
    ap.add_argument("--stride", type=int, default=None, help="Step 2 scan stride")
    ap.add_argument("--max-scans", type=int, default=None, help="cap scans per lidar (debug)")
    ap.add_argument("--v2-cameras", default="0,4",
                    help="Step 5a: comma-separated cameras refined via scan-to-image DT "
                         "minimization (docs/architecture.md fixed routing: narrow-FOV "
                         "front-facing cameras). Empty string to skip.")
    ap.add_argument("--v3-cameras", default="1,2,3,5,6,7,8,9,10,11",
                    help="Step 5b/5c: comma-separated cameras refined via whole-trajectory "
                         "camera cloud + LiDAR-map registration (RTK-fix bags only). "
                         "Empty string to skip.")
    ap.add_argument("--skip-step5", action="store_true", help="stop after Step 4")
    ap.add_argument("--skip-verify", action="store_true",
                    help="run Step 5a-c but skip Step 5d (projection verification overlays)")
    ap.add_argument("--no-viz", action="store_true", help="skip PNG plot generation")
    ap.add_argument("--require-cameras", action="store_true",
                    help="Step 0: treat missing camera topics as fatal")
    ap.add_argument("--force", action="store_true",
                    help="continue past a failing step instead of stopping")
    args = ap.parse_args(argv)

    with viz.tee_log(args.out_dir, "pipeline_log.txt"):
        return _run(args)


if __name__ == "__main__":
    sys.exit(main())
