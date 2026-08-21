#!/usr/bin/env python3
"""可視化(点群投影+アノテーション投影) + LLM/VLMによるアノテーション品質チェックのCLI。

例:
  # 可視化のみ(LLMなし)のスモークテスト
  python3 run_qc.py --adapter t4dataset-webdataset --input /path/to/webdataset --out-dir ./qc-out \\
      --clips <clip_id> --every-n 30 --channels CAM_FRONT_WIDE --max-frames 5 --skip-llm

  # Ollamaでのフル実行、歩行者だけを対象にカテゴリ別QA観点を適用
  python3 run_qc.py --adapter t4dataset-webdataset --input /path/to/webdataset --out-dir ./qc-out \\
      --categories pedestrian --llm-base-url http://localhost:11434/v1 --llm-model gemma3:27b
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

from adapter_base import FrameRecord
from llm_client import ChatClient
from qc_schema import build_user_prompt, describe_frame_annotations, load_prompt_guidance
from report import ReportWriter, query_with_retry
from t4dataset_adapter import T4DatasetWebdatasetAdapter
from tlr_crops_adapter import TLRCropsAdapter
from viz2d import draw_bbox2d, draw_box3d_wireframe, draw_lidar_points, draw_mask_overlay

SCRIPT_DIR = Path(__file__).parent

ADAPTERS = {
    "t4dataset-webdataset": T4DatasetWebdatasetAdapter,
    "tlr-crops": TLRCropsAdapter,
}


def render_frame_channel(frame: FrameRecord, channel: str, args, target_categories: list[str]) -> Path:
    cam = frame.cameras[channel]
    img = cam.image_loader()

    if not args.no_lidar and frame.lidar is not None:
        img = draw_lidar_points(img, frame.lidar, cam.ego_pose, cam.calib, args.ann3d_frame, alpha=args.alpha)

    if not args.no_3d_boxes:
        for box in frame.boxes3d:
            if args.only_target_categories and target_categories and box.category_name not in target_categories:
                continue
            img = draw_box3d_wireframe(img, box, cam.ego_pose, cam.calib, args.ann3d_frame)

    if not args.no_masks:
        for mask in cam.masks2d:
            if args.only_target_categories and target_categories and mask.category_name not in target_categories:
                continue
            img = draw_mask_overlay(img, mask, alpha=args.alpha)

    if not args.no_2d_boxes:
        for box in cam.boxes2d:
            if args.only_target_categories and target_categories and box.category_name not in target_categories:
                continue
            img = draw_bbox2d(img, box)

    # 小さい画像(TLRクロップ等)はここで、全レイヤー描画が終わった後にアップスケールする。
    # box/mask座標はネイティブ解像度基準のため、描画前に拡大すると座標とサイズが
    # 食い違う(実データで確認済みのバグ)。拡大は必ず描画完了後に行うこと。
    h, w = img.shape[:2]
    short_side = min(h, w)
    if short_side < args.min_display_size:
        scale = args.min_display_size / short_side
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)

    out_dir = Path(args.out_dir) / "viz" / frame.clip_id
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{frame.key}.{channel.lower()}.png"
    cv2.imwrite(str(png_path), img)
    return png_path


def process_one(frame: FrameRecord, channel: str, args, guidance: dict, requested_categories: list[str] | None, llm_base_url: str):
    present = frame.categories_present() | {b.category_name for b in frame.cameras[channel].boxes2d} | {m.category_name for m in frame.cameras[channel].masks2d}
    if requested_categories:
        target_categories = sorted(present & set(requested_categories))
    else:
        target_categories = sorted(present)

    png_path = render_frame_channel(frame, channel, args, target_categories)

    if args.skip_llm:
        return {
            "clip_id": frame.clip_id, "frame_index": frame.frame_index, "key": frame.key, "channel": channel,
            "viz_png_path": str(png_path), "llm_model": None, "frame_ok": None, "issues": [],
            "overall_notes": "", "parse_ok": None,
        }

    client = ChatClient(llm_base_url, args.llm_model, args.llm_api_key_env)
    annotation_lines = describe_frame_annotations(frame, channel, target_categories)
    user_prompt = build_user_prompt(frame.clip_id, frame.key, channel, target_categories, guidance, annotation_lines)
    parsed, parse_ok, _raw = query_with_retry(client, user_prompt, [str(png_path)], max_tokens=args.llm_max_tokens)
    return {
        "clip_id": frame.clip_id, "frame_index": frame.frame_index, "key": frame.key, "channel": channel,
        "viz_png_path": str(png_path), "llm_model": args.llm_model,
        "frame_ok": parsed.get("frame_ok"), "issues": parsed.get("issues", []),
        "overall_notes": parsed.get("overall_notes", ""), "parse_ok": parse_ok,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, choices=list(ADAPTERS))
    ap.add_argument("--input", required=True, help="データセットのルート(例: t4dataset-webdatasetの出力先)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clips", default=None, help="カンマ区切りのclip_id。省略時は全クリップ")
    ap.add_argument("--every-n", type=int, default=1)
    ap.add_argument("--channels", default=None, help="カンマ区切りのチャンネル名。省略時はカメラ全ch")
    ap.add_argument("--categories", default=None, help="カンマ区切りの対象カテゴリ。省略時は全カテゴリ")
    ap.add_argument("--only-target-categories", action="store_true", help="対象カテゴリ以外は描画しない")
    ap.add_argument("--qc-prompts-file", default=str(SCRIPT_DIR / "qc_prompts.yaml"))
    ap.add_argument("--no-lidar", action="store_true")
    ap.add_argument("--no-3d-boxes", action="store_true")
    ap.add_argument("--no-2d-boxes", action="store_true")
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--ann3d-frame", choices=["world", "ego"], default="world")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--min-display-size", type=int, default=320, help="合成画像の短辺がこれ未満なら描画完了後にアップスケールする(小さいTLRクロップ等向け)")
    ap.add_argument("--llm-base-url", default="http://localhost:11434/v1", help="カンマ区切りで複数指定するとタスクをラウンドロビンで振り分ける(例: GPUごとに立てた複数インスタンス)")
    ap.add_argument("--llm-api-key-env", default=None)
    ap.add_argument("--llm-model", default="qwen3.8:27b")
    ap.add_argument("--llm-max-tokens", type=int, default=2048, help="対象カテゴリ数が多いフレームほど出力が長くなるため、必要なら増やす")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None, help="デバッグ用: 処理する(frame,channel)組の総数上限")
    ap.add_argument("--jobs", type=int, default=1, help="LLM呼び出しの並列数")
    args = ap.parse_args(argv)

    adapter = ADAPTERS[args.adapter]()
    requested_categories = args.categories.split(",") if args.categories else None
    guidance = load_prompt_guidance(args.qc_prompts_file)
    llm_endpoints = [u.strip() for u in args.llm_base_url.split(",") if u.strip()]

    clip_ids = args.clips.split(",") if args.clips else adapter.discover_clips(args.input)
    if not clip_ids:
        print(f"[qc] no clips found under {args.input}", file=sys.stderr)
        return 1
    print(f"[qc] {len(clip_ids)} clip(s), jobs={args.jobs}, categories={requested_categories or 'all'}, llm_endpoints={len(llm_endpoints)}", flush=True)

    writer = ReportWriter(Path(args.out_dir))
    tasks = []
    n_queued = 0
    for clip_id in clip_ids:
        for frame in adapter.iter_frames(args.input, clip_id, every_n=args.every_n, channels=(args.channels.split(",") if args.channels else None), categories=requested_categories):
            for channel in frame.cameras:
                if args.max_frames is not None and n_queued >= args.max_frames:
                    break
                tasks.append((frame, channel))
                n_queued += 1
            if args.max_frames is not None and n_queued >= args.max_frames:
                break
        if args.max_frames is not None and n_queued >= args.max_frames:
            break

    print(f"[qc] {len(tasks)} (frame, channel) task(s) queued", flush=True)

    if args.jobs <= 1:
        endpoint = llm_endpoints[0]
        for frame, channel in tqdm(tasks, unit="frame"):
            row = process_one(frame, channel, args, guidance, requested_categories, endpoint)
            writer.add(**row)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(process_one, frame, channel, args, guidance, requested_categories, llm_endpoints[i % len(llm_endpoints)])
                for i, (frame, channel) in enumerate(tasks)
            ]
            for future in tqdm(futures, unit="frame"):
                row = future.result()
                writer.add(**row)

    summary = writer.finalize()
    print(f"[qc] done: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
