#!/usr/bin/env python3
"""t4dataset フォルダ配下の全クリップを WebDataset shards (+ parquet index) に一括変換する。

<input_dir> 配下を再帰的に走査し、annotation/ と data/ を両方持つディレクトリを
「クリップ」として検出する (例: HRdqo3pf_..._0)。各クリップは
convert_clip.py を subprocess として呼び出して変換され、出力は

    <output_dir>/<クリップの親(シーン)ディレクトリの input_dir からの相対パス>/

に書き込まれる。<output_dir> に NAS 上のマウントパスを直接指定すれば、
変換 = 転送 が同時に完了する (tar化されているためファイル数の多い
nuScenes形式を直接rsyncするより転送効率が良い想定)。

既に変換済み (index-<clip_id>.manifest.json が存在し、フレーム数が
元データの sample.json と一致) なクリップはデフォルトでスキップされる
(--overwrite で強制再変換)。中断しても再実行するだけで再開できる。

実行の最後に <output_dir> 直下へ sensor-shardlist.json / anno-shardlist.json
(wids の ShardListDataset がそのまま読める形式: {"wids_version": 1, "shardlist":
[{"url": ..., "nsamples": ...}, ...]}) を書き出す。output_dir 配下の
index-*.manifest.json を毎回スキャンして作るので、過去の実行分も含めて
その時点の全クリップが反映される。

注意: nsamples は各tar内の実サンプル数 (= フレーム数 + 1) で、
"+1" は各tarの先頭にあるクリップ単位メタデータ __meta__.json 分。
wids は basename (拡張子より前) が同じファイル群を1サンプルとして
数えるため、__meta__.json も1サンプルとしてカウントされる。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent
CONVERT_SCRIPT = SCRIPT_DIR / "convert_clip.py"


def find_clip_dirs(input_dir: Path) -> list[Path]:
    clips = []
    for dirpath, dirnames, _ in os.walk(input_dir, followlinks=True):
        if "annotation" in dirnames and "data" in dirnames:
            clips.append(Path(dirpath))
    return sorted(clips)


def expected_n_frames(clip_dir: Path) -> int | None:
    try:
        with open(clip_dir / "annotation" / "sample.json") as f:
            return len(json.load(f))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def already_converted(clip_dir: Path, out_dir: Path) -> bool:
    manifest_path = out_dir / f"index-{clip_dir.name}.manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False
    n_frames = expected_n_frames(clip_dir)
    if n_frames is None:
        return False
    if manifest.get("n_frames") != n_frames:
        return False
    for key in ("sensor_shard", "anno_shard"):
        if not (out_dir / manifest[key]).is_file():
            return False
    return (out_dir / f"index-{clip_dir.name}.parquet").is_file()


def build_shardlists(output_dir: Path) -> tuple[Path, Path]:
    sensor_shards = []
    anno_shards = []
    for manifest_path in sorted(output_dir.rglob("index-*.manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            continue
        clip_out_dir = manifest_path.parent
        nsamples = manifest["n_frames"] + 1  # +1 for the __meta__.json entry in each tar
        sensor_rel = (clip_out_dir / manifest["sensor_shard"]).relative_to(output_dir)
        anno_rel = (clip_out_dir / manifest["anno_shard"]).relative_to(output_dir)
        sensor_shards.append({"url": str(sensor_rel), "nsamples": nsamples})
        anno_shards.append({"url": str(anno_rel), "nsamples": nsamples})

    sensor_path = output_dir / "sensor-shardlist.json"
    anno_path = output_dir / "anno-shardlist.json"
    sensor_path.write_text(json.dumps({"wids_version": 1, "shardlist": sensor_shards}, indent=2, ensure_ascii=False))
    anno_path.write_text(json.dumps({"wids_version": 1, "shardlist": anno_shards}, indent=2, ensure_ascii=False))
    return sensor_path, anno_path


def convert_one(clip_dir: Path, out_dir: Path) -> tuple[Path, bool, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(CONVERT_SCRIPT), str(clip_dir), "--out-dir", str(out_dir)],
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0
    log = result.stdout if ok else (result.stdout + result.stderr)
    return clip_dir, ok, log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", type=Path, help="t4dataset ルートフォルダ (例: .../ssd2/t4dataset)")
    ap.add_argument("output_dir", type=Path, help="WebDataset 出力先ルート (NASのマウントパスを直接指定可)")
    ap.add_argument("-j", "--jobs", type=int, default=4, help="同時に変換するクリップ数 (デフォルト: 4)")
    ap.add_argument("--overwrite", action="store_true", help="変換済みクリップも強制的に再変換する")
    ap.add_argument("--dry-run", action="store_true", help="変換対象を一覧表示するだけで実行しない")
    args = ap.parse_args(argv)

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        ap.error(f"not a directory: {input_dir}")
    if args.jobs < 1:
        ap.error("--jobs must be >= 1")

    clip_dirs = find_clip_dirs(input_dir)
    if not clip_dirs:
        print(f"[batch] no clip folders (annotation/ + data/) found under {input_dir}", file=sys.stderr)
        return 1

    targets = []
    skipped = 0
    for clip_dir in clip_dirs:
        scene_rel = clip_dir.parent.relative_to(input_dir)
        out_dir = output_dir / scene_rel
        if not args.overwrite and already_converted(clip_dir, out_dir):
            skipped += 1
            continue
        targets.append((clip_dir, out_dir))

    print(f"[batch] {len(clip_dirs)} clip(s) found under {input_dir}", flush=True)
    print(f"[batch] {skipped} already converted (skipped), {len(targets)} to convert, jobs={args.jobs}", flush=True)
    print(f"[batch] output root: {output_dir}", flush=True)

    if args.dry_run:
        for clip_dir, out_dir in targets:
            print(f"  {clip_dir}  ->  {out_dir}")
        return 0

    ok = 0
    failed: list[tuple[Path, str]] = []
    if targets:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {pool.submit(convert_one, clip_dir, out_dir): clip_dir for clip_dir, out_dir in targets}
            with tqdm(total=len(futures), unit="clip") as pbar:
                for future in as_completed(futures):
                    clip_dir, success, log = future.result()
                    if success:
                        ok += 1
                    else:
                        failed.append((clip_dir, log))
                        tqdm.write(f"[batch] FAILED: {clip_dir}\n{log}")
                    pbar.update(1)

        print(f"\n[batch] done: {ok} ok, {len(failed)} failed, {skipped} skipped, {len(clip_dirs)} total", flush=True)
        if failed:
            print("[batch] failed clips:", file=sys.stderr)
            for clip_dir, _ in failed:
                print(f"  {clip_dir}", file=sys.stderr)

    sensor_path, anno_path = build_shardlists(output_dir)
    print(f"[batch] wrote {sensor_path}", flush=True)
    print(f"[batch] wrote {anno_path}", flush=True)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
