#!/usr/bin/env python3
"""親フォルダ直下の各サブフォルダに対して plot_gps_route.py を順に実行する。

各サブフォルダは、単独の .mcap ファイルの代わりに渡される split rosbag2
ディレクトリ (*.mcap + metadata.yaml) として扱われる。1つのサブフォルダの
処理が失敗しても中断せず、次のサブフォルダに進む。

plot_gps_route.py 本体への追加オプションは "--" の後ろにそのまま渡せる。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TARGET_SCRIPT = SCRIPT_DIR / "plot_gps_route.py"


def find_subfolders(parent: Path) -> list[Path]:
    return sorted(p for p in parent.iterdir() if p.is_dir())


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if "--" in argv:
        sep = argv.index("--")
        own_argv, extra_args = argv[:sep], argv[sep + 1:]
    else:
        own_argv, extra_args = argv, []

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parent_dir",
                     help="この直下の各サブフォルダを1つのbagとして処理する")
    ap.add_argument("--out-dir", default=None,
                     help="出力先ルート。省略時は各サブフォルダ自身に書き込む "
                          "(plot_gps_route.py のデフォルト動作)。指定時は "
                          "<out-dir>/<サブフォルダ名>/ にミラーして書き込む")
    args = ap.parse_args(own_argv)

    parent = Path(args.parent_dir)
    if not parent.is_dir():
        print(f"[batch] ERROR: not a directory: {parent}", file=sys.stderr)
        return 1

    subfolders = find_subfolders(parent)
    if not subfolders:
        print(f"[batch] no subfolders found under {parent}", file=sys.stderr)
        return 1

    print(f"[batch] {len(subfolders)} subfolder(s) under {parent}", flush=True)

    ok = 0
    failed = 0
    for i, sub in enumerate(subfolders, 1):
        if args.out_dir:
            out_dir = Path(args.out_dir) / sub.name
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            out_dir = sub

        print(f"\n{'=' * 60}", flush=True)
        print(f"[batch] ({i}/{len(subfolders)}) {sub.name}", flush=True)
        print(f"{'=' * 60}", flush=True)

        cmd = [sys.executable, str(TARGET_SCRIPT), str(sub), "--out-dir", str(out_dir)] + extra_args
        print(f"  $ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd)

        if result.returncode == 0:
            ok += 1
        else:
            print(f"[batch] failed: {sub.name} (exit {result.returncode})", file=sys.stderr)
            failed += 1

    print(f"\n[batch] done: {ok} ok, {failed} failed, out of {len(subfolders)} subfolder(s)",
          flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
