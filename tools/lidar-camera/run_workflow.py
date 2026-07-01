#!/usr/bin/env python3
"""lidar-camera ワークフローを一括実行するスクリプト。

ステップ:
  1. カメラ画像抽出       (extract_cameras.py)    -- MCAP 内の全カメラを自動検出
  2. LiDAR 点群抽出       (extract_lidar_pcd.py)
  3. LiDAR→カメラ投影     (project_lidar_to_cam.py) -- camera2/3/6/7 のみ

出力先: <MCAP と同じフォルダ>/<MCAP ファイル名(拡張子なし)>_proj/
"""
import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# project_lidar_to_cam.py の CAM_CONFIGS から対応カメラとペア LiDAR を動的に取得
sys.path.insert(0, str(SCRIPT_DIR))
from project_lidar_to_cam import CAM_CONFIGS, _load_tf_yaml  # noqa: E402


def run_step(cmd, description):
    print(f"\n{'=' * 60}", flush=True)
    print(f"[workflow] {description}", flush=True)
    print(f"  $ {' '.join(str(c) for c in cmd)}", flush=True)
    print(f"{'=' * 60}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[workflow] ERROR: {description} が失敗しました (exit {result.returncode})",
              file=sys.stderr)
        sys.exit(result.returncode)


def main():
    ap = argparse.ArgumentParser(
        description="lidar-camera ワークフローを一括実行する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("mcap", help="入力 MCAP ファイルパス")
    ap.add_argument("--start", type=float, default=None, help="開始時刻 (UNIX 秒)")
    ap.add_argument("--end",   type=float, default=None, help="終了時刻 (UNIX 秒)")
    ap.add_argument("--distortion-model", default=None,
                    choices=["rational_polynomial", "equidistant"],
                    help="カメラ歪みモデル強制指定 (省略時は各カメラの camera_info.json から自動検出)")
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="投影点の不透明度 (0~1)")
    ap.add_argument("--tf-yaml", default=None,
                    help="multi_tf_static.yaml から外部パラメータを読み込む (省略時はハードコード値を使用)")
    args = ap.parse_args()

    cam_configs = _load_tf_yaml(args.tf_yaml) if args.tf_yaml else CAM_CONFIGS
    cam_lidar_pairs = {
        cam: Path(cfg["lidar"]).name
        for cam, cfg in cam_configs.items()
    }
    if args.tf_yaml:
        print(f"[workflow] tf_yaml : {args.tf_yaml} ({len(cam_configs)} cameras)", flush=True)

    mcap_path = Path(args.mcap).resolve()
    if not mcap_path.exists():
        print(f"[workflow] ERROR: {mcap_path} が見つかりません", file=sys.stderr)
        sys.exit(1)

    proj_dir = mcap_path.parent / f"{mcap_path.stem}_proj"
    proj_dir.mkdir(exist_ok=True)
    print(f"[workflow] MCAP    : {mcap_path}", flush=True)
    print(f"[workflow] 出力先  : {proj_dir}", flush=True)
    if args.start is not None:
        print(f"[workflow] start   : {args.start}", flush=True)
    if args.end is not None:
        print(f"[workflow] end     : {args.end}", flush=True)

    ts_opts = []
    if args.start is not None:
        ts_opts += ["--start", str(args.start)]
    if args.end is not None:
        ts_opts += ["--end", str(args.end)]

    # ---- Step 1: カメラ画像抽出 ----
    run_step(
        [sys.executable, SCRIPT_DIR / "extract_cameras.py",
         str(mcap_path), str(proj_dir)] + ts_opts,
        "Step 1/3: カメラ画像抽出",
    )

    # ---- Step 2: LiDAR 点群抽出 ----
    run_step(
        [sys.executable, SCRIPT_DIR / "extract_lidar_pcd.py",
         str(mcap_path), str(proj_dir)] + ts_opts,
        "Step 2/3: LiDAR 点群抽出",
    )

    # ---- Step 3: LiDAR→カメラ投影 ----
    projected = 0
    for cam, lidar_name in cam_lidar_pairs.items():
        cam_dir   = proj_dir / cam
        lidar_dir = proj_dir / lidar_name

        if not cam_dir.exists() or not any(cam_dir.glob("*.jpg")):
            print(f"[workflow] skip: {cam} (カメラ画像なし)", flush=True)
            continue
        if not lidar_dir.exists() or not any(lidar_dir.glob("*.pcd")):
            print(f"[workflow] skip: {cam} ({lidar_name} PCD なし)", flush=True)
            continue

        dm_opts = ["--distortion-model", args.distortion_model] if args.distortion_model else []
        tf_opts = ["--tf-yaml", args.tf_yaml] if args.tf_yaml else []
        run_step(
            [sys.executable, SCRIPT_DIR / "project_lidar_to_cam.py",
             cam, str(proj_dir / f"proj_{cam}"),
             "--cam-dir",   str(cam_dir),
             "--lidar-dir", str(lidar_dir),
             "--alpha",     str(args.alpha)] + dm_opts + tf_opts + ts_opts,
            f"Step 3/3: LiDAR→カメラ投影 ({cam})",
        )
        projected += 1

    if projected == 0:
        print("[workflow] 警告: 投影対象カメラのデータが見つかりませんでした",
              flush=True)

    print(f"\n[workflow] 完了 -> {proj_dir}", flush=True)


if __name__ == "__main__":
    main()
