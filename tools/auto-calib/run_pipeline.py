#!/usr/bin/env python3
"""Run Steps 0-4 of the auto-calibration pipeline end-to-end against one bag.

Step 0 is a hard gate: on FAIL the pipeline stops before any expensive work
(matches the individual step's own contract). Steps 1-4 each write their own
log/plots under --out-dir (step0_log.txt, step1_log.txt, ..., undistorted/,
front_lidar_map.*, lidar_to_lidar_result.json, ...); this script additionally
tees its own stdout to OUT/pipeline_log.txt and prints a per-step wall-clock
timing summary at the end.

Usage:
    tools/auto-calib/run.sh run_pipeline.py <bag> --out-dir OUT \
        [--config config.yaml] [--tf-override multi_tf_static.yaml] \
        [--lidars front,left,rear,right] [--stride N] [--max-scans N] \
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
import viz


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
    ap = argparse.ArgumentParser(description="Run Step 0-4 of the auto-calibration pipeline")
    ap.add_argument("bag", help="single .mcap or split rosbag2 directory")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tf-override", default=None,
                    help="multi_tf_static YAML overriding drs->lidar extrinsics "
                         "(passed to Step 3 and Step 4; MUST be the same file for both)")
    ap.add_argument("--lidars", default=",".join(common.LIDAR_NAMES),
                    help="comma-separated subset of front,left,rear,right for Step 2")
    ap.add_argument("--stride", type=int, default=None, help="Step 2 scan stride")
    ap.add_argument("--max-scans", type=int, default=None, help="cap scans per lidar (debug)")
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
