#!/usr/bin/env python3
"""TLRクロップのデータ拡張・合成画像そのものの自然さ/破綻チェック。

run_qc.py --adapter tlr-crops はアノテーション(信号の状態ラベル等)が実物と一致しているかを
チェックするのに対し、こちらはアノテーションを見ず、「拡張・合成処理そのものが視覚的に
破綻していないか・不自然でないか」だけを別のプロンプト・別の出力スキーマ(aug_qc_schema.py,
aug_prompts.yaml)でチェックする。元画像と拡張後画像のペアをLLMに渡し、適用された拡張の
種類・パラメータも明示する。

例:
  # 可視化のみ(LLMなし)のスモークテスト
  python3 run_aug_qc.py --input /path/to/baseline2/val_dataset --out-dir ./aug-qc-out \\
      --clips <clip_id> --max-frames 5 --skip-llm

  # Ollamaでのフル実行、GPUごとに立てた複数インスタンスに振り分け
  python3 run_aug_qc.py --input /path/to/baseline2/val_dataset --out-dir ./aug-qc-out \\
      --jobs 3 --llm-base-url http://127.0.0.1:11435/v1,http://127.0.0.1:11436/v1,http://127.0.0.1:11437/v1
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

from aug_pairs import discover_pairs
from aug_qc_schema import SYSTEM_PROMPT, build_aug_user_prompt, load_prompt_guidance
from llm_client import ChatClient
from report import ReportWriter, query_with_retry
from tlr_crops_adapter import TLRCropsAdapter, _CHANNEL_RE, _split_clip_and_rest

SCRIPT_DIR = Path(__file__).parent


def _channel_of(stem: str) -> str:
    split = _split_clip_and_rest(stem)
    m = _CHANNEL_RE.match(split[1]) if split else None
    return m.group(1) if m else "UNKNOWN"


def _upscale_for_display(path: Path, min_size: int) -> "cv2.Mat":
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    h, w = img.shape[:2]
    short_side = min(h, w)
    if short_side < min_size:
        scale = min_size / short_side
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)
    return img


def process_one(pair: dict, args, guidance: dict, llm_base_url: str) -> dict:
    channel = _channel_of(pair["base_stem"])

    out_dir = Path(args.out_dir) / "viz" / pair["clip_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    aug_png = out_dir / f"{pair['aug_stem']}.aug.png"
    base_png = out_dir / f"{pair['aug_stem']}.base.png"
    cv2.imwrite(str(aug_png), _upscale_for_display(pair["aug_path"], args.min_display_size))
    cv2.imwrite(str(base_png), _upscale_for_display(pair["base_path"], args.min_display_size))

    if args.skip_llm:
        return {
            "clip_id": pair["clip_id"], "frame_index": 0, "key": pair["aug_stem"], "channel": channel,
            "viz_png_path": str(aug_png), "llm_model": None, "frame_ok": None, "issues": [],
            "overall_notes": "", "parse_ok": None,
        }

    client = ChatClient(llm_base_url, args.llm_model, args.llm_api_key_env)
    user_prompt = build_aug_user_prompt(pair["clip_id"], pair["aug_stem"], channel, pair["tags"], guidance)
    # image_paths[0]=元画像, [1]=拡張後画像 (SYSTEM_PROMPT/USER_PROMPT_HEADERの"[1]"/"[2]"表記と対応)
    parsed, parse_ok, _raw = query_with_retry(
        client, user_prompt, [str(base_png), str(aug_png)],
        max_tokens=args.llm_max_tokens, system_prompt=SYSTEM_PROMPT,
    )
    return {
        "clip_id": pair["clip_id"], "frame_index": 0, "key": pair["aug_stem"], "channel": channel,
        "viz_png_path": str(aug_png), "llm_model": args.llm_model,
        # ReportWriterのフィールド名はrun_qc.pyと共通化のため"frame_ok"のままだが、
        # ここでは意味的にaug_qc_schemaの"aug_ok"(拡張が自然か)を格納している。
        "frame_ok": parsed.get("aug_ok"), "issues": parsed.get("issues", []),
        "overall_notes": parsed.get("overall_notes", ""), "parse_ok": parse_ok,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="tlr-cropsレイアウトのデータセットルート")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--clips", default=None, help="カンマ区切りのclip_id。省略時は全クリップ")
    ap.add_argument("--every-n", type=int, default=1, help="拡張画像をN件ごとにサンプリング")
    ap.add_argument("--aug-prompts-file", default=str(SCRIPT_DIR / "aug_prompts.yaml"))
    ap.add_argument("--min-display-size", type=int, default=320)
    ap.add_argument("--llm-base-url", default="http://localhost:11434/v1", help="カンマ区切りで複数指定するとラウンドロビンで振り分ける")
    ap.add_argument("--llm-api-key-env", default=None)
    ap.add_argument("--llm-model", default="qwen3.8:27b")
    ap.add_argument("--llm-max-tokens", type=int, default=2048)
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None, help="デバッグ用: 処理するペア総数の上限")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args(argv)

    guidance = load_prompt_guidance(args.aug_prompts_file)
    llm_endpoints = [u.strip() for u in args.llm_base_url.split(",") if u.strip()]

    clip_ids = args.clips.split(",") if args.clips else TLRCropsAdapter().discover_clips(args.input)
    if not clip_ids:
        print(f"[aug-qc] no clips found under {args.input}", file=sys.stderr)
        return 1
    print(f"[aug-qc] {len(clip_ids)} clip(s), jobs={args.jobs}, llm_endpoints={len(llm_endpoints)}", flush=True)

    pairs = []
    for clip_id in clip_ids:
        for pair in discover_pairs(args.input, clip_id, every_n=args.every_n):
            pairs.append(pair)
            if args.max_frames is not None and len(pairs) >= args.max_frames:
                break
        if args.max_frames is not None and len(pairs) >= args.max_frames:
            break

    print(f"[aug-qc] {len(pairs)} augmented sample(s) queued", flush=True)

    writer = ReportWriter(Path(args.out_dir))
    if args.jobs <= 1:
        endpoint = llm_endpoints[0]
        for pair in tqdm(pairs, unit="sample"):
            writer.add(**process_one(pair, args, guidance, endpoint))
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [
                pool.submit(process_one, pair, args, guidance, llm_endpoints[i % len(llm_endpoints)])
                for i, pair in enumerate(pairs)
            ]
            for future in tqdm(futures, unit="sample"):
                writer.add(**future.result())

    summary = writer.finalize()
    print(f"[aug-qc] done: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
