#!/usr/bin/env python3
"""Gradually reflects NFS-side directory trees into Alluxio's metadata view,
one small unit at a time, using the v2 loadMetadata async engine.

Background: a single blocking `alluxio fs loadMetadata -R -F <big root>`
over datasets/DTSETS1..8 (~12k recording-level directories, ~30M+ files)
failed after 39 minutes with UnavailableException, and left the shared
Alluxio Master with degraded heartbeats (13s-130s RPCs) for ~40 minutes
afterward. The v2 engine's `-a` (async) mode plus a short `-p` poll interval
lets each recording-folder-sized unit (~2500-2700 files) complete in ~1-3s
wall time with a clean SUCCEEDED/FAILED result, so many small submissions
(with a throttle delay and a heartbeat-based circuit breaker) are far safer
than one giant recursive call.

Usage:
  python3 sync_metadata.py --nfs-root /mnt/DataPool/SharedData/datasets/DTSETS1 \
      --alluxio-root /datasets/DTSETS1 \
      [--worker-container petadata-alluxio-worker-warm-1] \
      [--delay-sec 2.0] [--progress-file <path>] [--dry-run] [--max-units N]

Re-running with the same --progress-file skips units already marked done.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HEARTBEAT_WARN_RE = re.compile(r"WARN.*returned.* in (\d+) ms")


def discover_units(nfs_root, alluxio_root):
    """Yields (alluxio_path, recursive: bool) sync units.

    One level below each top-level category (e.g. pov/<recording>,
    robotaxi/<recording>, synthetic/predict) is treated as one unit; a
    category with no subdirectories, or a loose top-level file, is synced
    directly. This avoids hardcoding category names.
    """
    units = []
    for name in sorted(os.listdir(nfs_root)):
        nfs_child = os.path.join(nfs_root, name)
        alluxio_child = f"{alluxio_root}/{name}"
        if os.path.isfile(nfs_child):
            units.append((alluxio_child, False))
            continue
        try:
            sub_entries = sorted(os.listdir(nfs_child))
        except OSError as e:
            print(f"[warn] cannot list {nfs_child}: {e}", file=sys.stderr)
            continue
        sub_dirs = [e for e in sub_entries if os.path.isdir(os.path.join(nfs_child, e))]
        sub_files = [e for e in sub_entries if os.path.isfile(os.path.join(nfs_child, e))]
        if sub_dirs:
            for d in sub_dirs:
                units.append((f"{alluxio_child}/{d}", True))
            for f in sub_files:
                units.append((f"{alluxio_child}/{f}", False))
        else:
            units.append((alluxio_child, True))
    return units


def load_progress(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"done": {}, "failed": {}}


def save_progress(path, progress):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def recent_heartbeat_warns(container, since_sec, threshold_ms):
    """Returns the max heartbeat RPC latency (ms) seen in worker logs over
    the last `since_sec` seconds, or 0 if no WARN lines / all under threshold."""
    try:
        out = subprocess.run(
            ["docker", "logs", "--since", f"{since_sec}s", container],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception as e:
        print(f"[warn] could not read worker logs for circuit breaker: {e}", file=sys.stderr)
        return 0
    worst = 0
    for line in out.splitlines():
        m = HEARTBEAT_WARN_RE.search(line)
        if m:
            worst = max(worst, int(m.group(1)))
    return worst


def sync_unit(container, alluxio_path, recursive, poll_ms, timeout_sec):
    cmd = ["docker", "exec", container, "alluxio", "fs", "loadMetadata",
           "-v2", "-a", "-p", str(poll_ms)]
    if recursive:
        cmd.append("-R")
    cmd.append(alluxio_path)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        return False, f"client timed out after {timeout_sec}s (job may still be running server-side)"
    out = proc.stdout + proc.stderr
    if "State: SUCCEEDED" in out:
        # multiple poll rounds are printed; the object count only reflects
        # reality in the final (SUCCEEDED) block, so take the last match
        matches = re.findall(r"# of objects loaded from UFS=(\d+)", out)
        n = matches[-1] if matches else "?"
        return True, f"{n} objects"
    if "State: FAILED" in out:
        return False, out.strip().splitlines()[-1] if out.strip() else "FAILED (no detail)"
    return False, out.strip().splitlines()[-1] if out.strip() else f"exit={proc.returncode}, no SUCCEEDED/FAILED seen"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nfs-root", required=True, help="e.g. /mnt/DataPool/SharedData/datasets/DTSETS1")
    ap.add_argument("--alluxio-root", required=True, help="e.g. /datasets/DTSETS1")
    ap.add_argument("--worker-container", default="petadata-alluxio-worker-warm-1")
    ap.add_argument("--delay-sec", type=float, default=2.0, help="sleep between submissions")
    ap.add_argument("--poll-ms", type=int, default=500, help="loadMetadata -p value")
    ap.add_argument("--unit-timeout-sec", type=int, default=60)
    ap.add_argument("--progress-file", default=None)
    ap.add_argument("--max-units", type=int, default=None, help="stop after N units (for testing)")
    ap.add_argument("--circuit-breaker-ms", type=int, default=20000,
                     help="pause if any worker heartbeat WARN exceeds this many ms")
    ap.add_argument("--circuit-breaker-window-sec", type=int, default=30)
    ap.add_argument("--circuit-breaker-pause-sec", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    units = discover_units(args.nfs_root, args.alluxio_root)
    print(f"discovered {len(units)} sync unit(s) under {args.nfs_root}")

    progress = load_progress(args.progress_file)
    done = progress["done"]
    failed = progress["failed"]

    if args.dry_run:
        remaining = [u for u in units if u[0] not in done]
        print(f"[dry-run] {len(remaining)}/{len(units)} unit(s) remaining, e.g.:")
        for u in remaining[:10]:
            print(f"  {'[R] ' if u[1] else '    '}{u[0]}")
        return

    processed_this_run = 0
    for alluxio_path, recursive in units:
        if alluxio_path in done:
            continue
        if args.max_units is not None and processed_this_run >= args.max_units:
            print(f"reached --max-units {args.max_units}, stopping")
            break

        worst_ms = recent_heartbeat_warns(args.worker_container, args.circuit_breaker_window_sec,
                                           args.circuit_breaker_ms)
        if worst_ms > args.circuit_breaker_ms:
            print(f"[circuit-breaker] worker heartbeat WARN {worst_ms}ms > {args.circuit_breaker_ms}ms "
                  f"in last {args.circuit_breaker_window_sec}s - pausing {args.circuit_breaker_pause_sec}s")
            save_progress(args.progress_file, progress)
            time.sleep(args.circuit_breaker_pause_sec)
            continue

        ok, detail = sync_unit(args.worker_container, alluxio_path, recursive,
                                args.poll_ms, args.unit_timeout_sec)
        processed_this_run += 1
        tag = "OK " if ok else "FAIL"
        print(f"[{len(done)+len(failed)+1}/{len(units)}] {tag} {alluxio_path} - {detail}")
        if ok:
            done[alluxio_path] = detail
            failed.pop(alluxio_path, None)
        else:
            failed[alluxio_path] = detail
        save_progress(args.progress_file, progress)
        time.sleep(args.delay_sec)

    print(f"\n--- Summary ---\ndone: {len(done)}/{len(units)}  failed: {len(failed)}")
    if failed:
        print("failed units:")
        for p, detail in failed.items():
            print(f"  {p} - {detail}")


if __name__ == "__main__":
    main()
