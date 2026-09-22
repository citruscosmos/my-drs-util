#!/usr/bin/env python3
"""USB SSD(portA/portB)から ecu0/ecu1 フォルダを検出し、Alluxio S3 API 経由で
jt10/<group>/{ecu0,ecu1}/ へアップロードする。アップロード後は件数突き合わせで
永続化を検証し、Alluxioワーカーコンテナ内でroot権限のpermission修正と
t4dataset/ 空フォルダ作成まで行う。

前提(2026-09-16 実地検証済み):
- jt10 の Alluxio S3 API での bucket 名は "jt10"。`aws s3 ls`(バケット一覧)には
  表示されないが、`s3://jt10/...` を直接アドレスすれば読み書きとも正常動作する。
- S3 PUT で書き込まれたオブジェクトは root:root 所有・rwxrwx--- (otherは無権限)に
  なるため、`docker exec <alluxio-worker-container>` でroot権限のまま
  chmod -R o+rwX / mkdir する必要がある(NFS越しにshimakiユーザーが直接操作すると
  Operation not permitted)。otherに書き込み権限まで付与するのは、既存の全jt10
  グループが drwxrwxrwx であることに合わせるため — Stage 2(gps-route)以降が
  非特権コンテナから書き込めるようにする必要がある。
- アップロード直後(特に数GB超のファイル)は、S3側が永続化を報告した直後でも
  Alluxioの非同期persist完了とレースしてchmodが失われることがあるため、
  chmod自体もリトライ+再検証する(`fix_permissions`)。
- アップロード直後は永続化に多少ラグがあるため、件数一致をリトライで確認する。

どちらのUSBポートにどちらのECUのSSDが挿さるかは固定されていないため、
マウント後にフォルダ名(ecu0/ecu1)で判別する。1件のrecordingアップロードが
失敗しても他のrecordingの処理は継続する(batch_plot_gps_route.py と同じ方針)。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ECU_NAMES = ("ecu0", "ecu1")


def scan_port(port_path: Path) -> dict[str, Path]:
    """Return {"ecu0": path, "ecu1": path} for whichever are found under port_path.

    Prefers top-level folders; falls back to a depth-2 search (with a warning)
    if nothing is found at the top level. Never guesses beyond that depth.
    """
    found: dict[str, Path] = {}
    if not port_path.is_dir():
        return found
    try:
        children = [c for c in port_path.iterdir() if c.is_dir()]
    except OSError:
        return found

    top = {c.name.lower(): c for c in children}
    for name in ECU_NAMES:
        if name in top:
            found[name] = top[name]
    if found:
        return found

    for child in children:
        try:
            grandchildren = {g.name.lower(): g for g in child.iterdir() if g.is_dir()}
        except OSError:
            continue
        for name in ECU_NAMES:
            if name in grandchildren and name not in found:
                print(f"  [warn] {name} found nested under {child} (not top-level of {port_path})",
                      file=sys.stderr)
                found[name] = grandchildren[name]
    return found


def find_ecu_sources(port_a: Path, port_b: Path) -> dict[str, tuple[str, Path]]:
    """Return {"ecu0": (port_name, path), "ecu1": (port_name, path)}.

    Raises SystemExit with a clear message on ambiguity (same ECU found on
    both ports) - this is never guessed past.
    """
    sources: dict[str, tuple[str, Path]] = {}
    conflicts = []
    for port_name, port_path in (("portA", port_a), ("portB", port_b)):
        found = scan_port(port_path)
        for ecu, path in found.items():
            if ecu in sources:
                conflicts.append(f"{ecu}: found under both {sources[ecu][1]} ({sources[ecu][0]}) "
                                  f"and {path} ({port_name})")
            else:
                sources[ecu] = (port_name, path)
    if conflicts:
        for c in conflicts:
            print(f"  [error] {c}", file=sys.stderr)
        sys.exit(1)
    return sources


def s3_env(access_key: str, secret_key: str) -> dict[str, str]:
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = access_key
    env["AWS_SECRET_ACCESS_KEY"] = secret_key
    return env


def aws_s3_cmd(endpoint_url: str, *rest: str) -> list[str]:
    return ["aws", "--endpoint-url", endpoint_url, "s3", *rest, "--region", "us-east-1"]


def count_local_files(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file())


def count_remote_files(s3_uri: str, endpoint_url: str, env: dict[str, str]) -> int:
    proc = subprocess.run(["timeout", "60s", *aws_s3_cmd(endpoint_url, "ls", "--recursive", s3_uri)],
                           capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return 0
    return sum(1 for line in proc.stdout.splitlines() if line.strip())


def abort_incomplete_multipart_uploads(bucket: str, prefix: str, endpoint_url: str, env: dict[str, str]) -> int:
    """Abort any incomplete multipart uploads under `prefix` and return how many were aborted.

    A `cp` killed mid-transfer (by `timeout`, or any other SIGTERM/SIGKILL) leaves its
    in-flight multipart upload(s) uncompleted. This Alluxio S3 gateway deployment lists
    an incomplete upload's already-received parts as if they were ordinary objects under
    a `<filename>_<uploadid>/` pseudo-directory - `count_remote_files`'s plain `ls
    --recursive` count picks these up, so a large recording that timed out with e.g. 24
    real files done and 1 stuck mid-upload can show a remote object count *higher* than
    the local file count (dozens of leftover 8MiB part entries for that one stuck file),
    making `main()`'s "remote_count >= local_count" resume check wrongly conclude the
    recording is fully persisted and skip re-uploading it (confirmed directly: a 100GB
    recording with 9 of 25 files actually missing was reported "SAFE TO WIPE" before this
    fix). Aborting here keeps every future count accurate; called both right after a
    failed upload and once per --group at startup, in case a previous run was killed
    outright rather than hitting its own `timeout` wrapper.
    """
    proc = subprocess.run(["timeout", "60s", "aws", "--endpoint-url", endpoint_url, "s3api",
                            "list-multipart-uploads", "--bucket", bucket, "--prefix", prefix,
                            "--region", "us-east-1", "--output", "json"],
                           capture_output=True, text=True, env=env)
    if proc.returncode != 0 or not proc.stdout.strip():
        return 0
    try:
        uploads = json.loads(proc.stdout).get("Uploads") or []
    except json.JSONDecodeError:
        return 0
    for u in uploads:
        subprocess.run(["aws", "--endpoint-url", endpoint_url, "s3api", "abort-multipart-upload",
                         "--bucket", bucket, "--key", u["Key"], "--upload-id", u["UploadId"],
                         "--region", "us-east-1"], capture_output=True, text=True, env=env)
    return len(uploads)


def count_remote_files_retry(s3_uri: str, endpoint_url: str, env: dict[str, str], expected: int,
                              attempts: int = 5, delay_s: float = 3.0) -> int:
    """Poll for the uploaded object count to reach `expected`, retrying briefly.

    Handles the short persistence lag observed after an S3 PUT to this
    Alluxio deployment (~15-30s before a direct NFS view catches up; the S3
    listing itself is usually faster but this adds margin).
    """
    n = count_remote_files(s3_uri, endpoint_url, env)
    for _ in range(attempts - 1):
        if n >= expected:
            break
        time.sleep(delay_s)
        n = count_remote_files(s3_uri, endpoint_url, env)
    return n


def upload_recording(local_dir: Path, s3_uri: str, endpoint_url: str, env: dict[str, str],
                      timeout_s: int) -> bool:
    # --no-progress: the default progress bar rewrites one line via \r, which
    # (when captured non-interactively) turns into a single, sometimes
    # multi-hundred-KB "line" - noisy and pointless outside a live terminal.
    cmd = ["timeout", f"{timeout_s}s",
           *aws_s3_cmd(endpoint_url, "cp", "--recursive", "--no-progress",
                       f"{local_dir}/", f"{s3_uri}/")]
    return subprocess.run(cmd, env=env).returncode == 0


def verify_permissions_fixed(container: str, path: str) -> bool:
    """True if nothing under `path` still lacks other-read/write access."""
    proc = subprocess.run(["docker", "exec", container, "find", path,
                            "!", "-perm", "-o+rw"], capture_output=True, text=True)
    return proc.returncode == 0 and not proc.stdout.strip()


def fix_permissions(container: str, group_path: str, attempts: int = 5, delay_s: float = 3.0) -> bool:
    """chmod o+rwX recursively, retrying briefly and re-verifying each time.

    Grants "other" read+write (not just read) to match the permissive mode
    every pre-existing jt10 group directory already has (confirmed: every
    group's ecu0/ecu1/gps-route/t4dataset observed this session is
    drwxrwxrwx) - downstream tools (gps-route, t4dataset conversion) run as
    an unprivileged container user and need to write into gps-route/ and
    t4dataset/ without any further root/docker-exec trick.

    A just-uploaded large file can still be mid-finalization by Alluxio's
    async persist job for a few seconds after the S3 API reports it present -
    a chmod that races that finalization can be silently clobbered (observed
    directly: a 2.3GB test upload's chmod was lost this way; a 37-byte test
    file was not affected, so this only bites large files). chmod is
    idempotent, so retrying is safe.
    """
    for _ in range(attempts):
        subprocess.run(["docker", "exec", container, "chmod", "-R", "o+rwX", group_path])
        if verify_permissions_fixed(container, group_path):
            return True
        time.sleep(delay_s)
    return False


def docker_exec(container: str, *cmd: str) -> bool:
    return subprocess.run(["docker", "exec", container, *cmd]).returncode == 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", required=True,
                     help="jt10 group name to upload into, e.g. 20260916_jt10_example")
    ap.add_argument("--port-a", default="/mnt/ssd/portA", type=Path)
    ap.add_argument("--port-b", default="/mnt/ssd/portB", type=Path)
    ap.add_argument("--bucket", default="jt10")
    ap.add_argument("--s3-endpoint-url", default="http://127.0.0.1:39999/api/v1/s3")
    ap.add_argument("--s3-access-key", default="root",
                     help="Must be a real Unix username inside the Alluxio worker container")
    ap.add_argument("--s3-secret-key", default="unused-not-verified", help="Not verified by Alluxio")
    ap.add_argument("--alluxio-worker-container", default="petadata-alluxio-worker-warm-1")
    ap.add_argument("--alluxio-ufs-root", default="/opt/alluxio/underFSStorage",
                     help="Path inside the worker container where the S3 root UFS is mounted")
    ap.add_argument("--transfer-timeout-sec", type=int, default=1800,
                     help="Hard ceiling per recording upload (default 1800s / 30min)")
    ap.add_argument("--allow-partial", action="store_true",
                     help="Proceed even if only one of ecu0/ecu1 was found (default: refuse)")
    ap.add_argument("--dry-run", action="store_true", help="Only scan and report; upload nothing")
    args = ap.parse_args(argv)

    print(f"Scanning {args.port_a} and {args.port_b} for ecu0/ecu1 ...")
    sources = find_ecu_sources(args.port_a, args.port_b)
    if not sources:
        print("  [error] no ecu0/ecu1 folder found on either port", file=sys.stderr)
        return 1
    for ecu, (port_name, path) in sorted(sources.items()):
        print(f"  found {ecu} on {port_name}: {path}")

    missing = [e for e in ECU_NAMES if e not in sources]
    if missing:
        print(f"  [warn] missing: {', '.join(missing)}", file=sys.stderr)
        if not args.allow_partial:
            print("  [error] refusing to proceed with a partial set (pass --allow-partial to override)",
                  file=sys.stderr)
            return 1

    env = s3_env(args.s3_access_key, args.s3_secret_key)

    if not args.dry_run:
        n_aborted = abort_incomplete_multipart_uploads(args.bucket, f"{args.group}/", args.s3_endpoint_url, env)
        if n_aborted:
            print(f"  [cleanup] aborted {n_aborted} incomplete multipart upload(s) left over from a prior run")

    report: dict[str, list[tuple[str, bool, int, int]]] = {}

    for ecu, (port_name, ecu_dir) in sorted(sources.items()):
        recordings = sorted(p for p in ecu_dir.iterdir() if p.is_dir())
        print(f"\n{ecu} ({port_name}): {len(recordings)} recording(s)")
        results: list[tuple[str, bool, int, int]] = []
        for rec in recordings:
            local_count = count_local_files(rec)
            if local_count == 0:
                print(f"  [skip] {rec.name}: empty, nothing to upload")
                continue
            s3_uri = f"s3://{args.bucket}/{args.group}/{ecu}/{rec.name}"

            if args.dry_run:
                print(f"  [dry-run] would upload {rec} -> {s3_uri} ({local_count} files)")
                continue

            remote_count = count_remote_files(s3_uri, args.s3_endpoint_url, env)
            if remote_count >= local_count:
                print(f"  [skip] {rec.name}: already uploaded+persisted ({remote_count} objects)")
                results.append((rec.name, True, remote_count, local_count))
                continue

            print(f"  uploading {rec.name} ({local_count} files) ...")
            ok = upload_recording(rec, s3_uri, args.s3_endpoint_url, env, args.transfer_timeout_sec)
            if not ok:
                print(f"  [fail] upload failed: {rec.name}", file=sys.stderr)
                abort_incomplete_multipart_uploads(args.bucket, f"{args.group}/{ecu}/{rec.name}/",
                                                    args.s3_endpoint_url, env)
                results.append((rec.name, False, 0, local_count))
                continue

            verified = count_remote_files_retry(s3_uri, args.s3_endpoint_url, env, local_count)
            persisted = verified >= local_count
            status = "persisted" if persisted else "UNVERIFIED"
            print(f"  {rec.name}: uploaded, {status} ({verified}/{local_count} objects)")
            results.append((rec.name, persisted, verified, local_count))

        report[ecu] = results

    if args.dry_run:
        print("\n[dry-run] no changes made")
        return 0

    if any(report.values()):
        group_ufs_path = f"{args.alluxio_ufs_root}/{args.bucket}/{args.group}"
        print(f"\nFixing up {args.group} on {args.alluxio_worker_container} ...")
        docker_exec(args.alluxio_worker_container, "mkdir", "-p",
                    f"{group_ufs_path}/t4dataset", f"{group_ufs_path}/gps-route")
        if not fix_permissions(args.alluxio_worker_container, group_ufs_path):
            print("  [warn] permission fix did not fully verify after retries - "
                  "some files may still be unreadable by other users", file=sys.stderr)

    print("\n--- Summary ---")
    all_ok = True
    safe_ports = []
    for ecu, (port_name, _path) in sorted(sources.items()):
        results = report.get(ecu, [])
        n_ok = sum(1 for r in results if r[1])
        n_total = len(results)
        print(f"{ecu} ({port_name}): {n_ok}/{n_total} recording(s) persisted")
        if n_total == 0 or n_ok < n_total:
            all_ok = False
        else:
            safe_ports.append(f"{port_name} ({ecu})")
        for name, persisted, verified, local_count in results:
            if not persisted:
                print(f"    [unverified] {name}: {verified}/{local_count} objects")

    if all_ok and safe_ports:
        print(f"\nSAFE TO WIPE: {', '.join(safe_ports)}")
        return 0
    print("\nDO NOT WIPE: one or more recordings are not fully verified - see [unverified] lines above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
