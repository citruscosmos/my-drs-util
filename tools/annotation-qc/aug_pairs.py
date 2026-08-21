#!/usr/bin/env python3
"""拡張・合成済みTLRクロップ画像と、その元になった実クロップ(Annotations有り)のペアを発見する。

拡張ファイル名は "<実クロップのstem>_<拡張タグチェイン>.jpg" という形式
(例: "..._996_390_rcn00_whole_green_deg015.jpg")。拡張タグの並びは可変(単独 or 複数組み合わせ)
なので、bbox数値の位置を正規表現で当てにいくのではなく、そのクリップの実アノテーションstem集合
から「最長一致する接頭辞」を探すことで、どの拡張がどの実クロップに属するかを頑健に特定する。

各拡張タグの意味は train_gen.log/val_gen.log の生成設定から確認したもの:
    Aug: random_crop=true (wider×2/narrower×2, expand=0.25/shrink=0.15),
         arrow_aug=true, color_aug=true, rot_step=1,
         whole_rot=true (max=90, step=15, dir=ccw), photo=2, lowres=true, flicker=true
"whole_rot"/"color_aug" -> "_whole_<color>_deg<NNN>"、"photo" -> "_ph00/01"、
"random_crop" -> "_rcn00/01"(narrower)・"_rcw00/01"(wider)、"flicker" -> "_flicker"。
"lowres" -> "_lr00〜03" と推定(生成ツールのソース未確認のため断定はできない)。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

_SIMPLE_TAG_RE = re.compile(r"^(ph|rcn|rcw|lr)(\d{2})$")

_TYPE_NAMES = {
    "ph": "photometric",
    "rcn": "random_crop_narrow",
    "rcw": "random_crop_wide",
    "lr": "lowres",
}


def _describe_simple_tag(kind: str, variant: str) -> str:
    if kind == "ph":
        return f"photometric jitter (brightness/contrast/color, variant {variant})"
    if kind == "rcn":
        return f"crop narrowed ~15% from the original detection box (variant {variant})"
    if kind == "rcw":
        return f"crop widened ~25% from the original detection box (variant {variant})"
    if kind == "lr":
        return f"low-resolution/blur degradation, inferred meaning (variant {variant})"
    return f"{kind}{variant}"


def parse_aug_tags(tag_chain: str) -> list[dict]:
    """タグチェイン文字列(例: "rcn00_whole_green_deg015")を構造化して返す。"""
    tokens = tag_chain.split("_")
    tags = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "flicker":
            tags.append({"type": "flicker", "description": "flicker simulation (LED PWM dimming artifact)"})
            i += 1
        elif tok == "whole" and i + 2 < len(tokens):
            color, deg_tok = tokens[i + 1], tokens[i + 2]
            deg = deg_tok[3:] if deg_tok.startswith("deg") else deg_tok
            tags.append({
                "type": "whole_recolor",
                "description": f"whole-lens synthetic recolor to '{color}' via a {deg}° hue rotation",
            })
            i += 3
        else:
            m = _SIMPLE_TAG_RE.match(tok)
            if m:
                kind, variant = m.groups()
                tags.append({"type": _TYPE_NAMES[kind], "description": _describe_simple_tag(kind, variant)})
                i += 1
            else:
                tags.append({"type": "unknown", "description": f"unrecognized tag token: {tok}"})
                i += 1
    return tags


def discover_pairs(input_path: str, clip_id: str, every_n: int = 1) -> Iterator[dict]:
    root = Path(input_path)
    anno_dir = root / "Annotations"
    image_dir = root / "JPEGImages"

    real_stems = sorted((p.stem for p in anno_dir.glob(f"{clip_id}_CAM_*.json")), key=len, reverse=True)
    real_stem_set = set(real_stems)

    aug_images = sorted(image_dir.glob(f"{clip_id}_CAM_*.jpg"))
    aug_images = [p for p in aug_images if p.stem not in real_stem_set]

    for img_path in aug_images[::every_n]:
        stem = img_path.stem
        base = next((b for b in real_stems if stem.startswith(b + "_")), None)
        if base is None:
            continue  # 対応する実クロップを特定できない(想定外の命名) -> 安全にスキップ
        tag_chain = stem[len(base) + 1:]
        base_path = image_dir / f"{base}.jpg"
        if not base_path.is_file():
            continue
        yield {
            "clip_id": clip_id,
            "aug_stem": stem,
            "base_stem": base,
            "tags": parse_aug_tags(tag_chain),
            "aug_path": img_path,
            "base_path": base_path,
        }
