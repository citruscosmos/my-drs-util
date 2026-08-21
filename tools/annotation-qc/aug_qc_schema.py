#!/usr/bin/env python3
"""データ拡張・合成画像そのものの自然さ/破綻チェック用のスキーマとプロンプト組み立て。

qc_schema.py(アノテーションが実物と一致しているかを見る)とは目的が異なるため、
意図的に別モジュールにしている: ここでは「アノテーションは無視して、合成処理
そのものが視覚的に破綻していないか・不自然でないか」だけを見る。
"""
from __future__ import annotations

from pathlib import Path

import yaml

ISSUE_CATEGORIES = [
    "unnatural_color",         # 色変換結果が現実の信号灯の色として不自然
    "visible_artifact",        # 継ぎ目・ブレンドの縁・バンディングなど合成の痕跡が見える
    "implausible_lighting",    # 周囲の照明・露出と矛盾している
    "structure_loss",          # クロップが狭すぎて灯火本体が欠けている/広すぎて無関係な背景が支配的
    "unrealistic_degradation", # flicker/低解像度劣化が実際のカメラ挙動として不自然(綺麗すぎる、過剰、ノイズの種類が違う等)
    "other",
]
SEVERITIES = ["low", "medium", "high"]

SYSTEM_PROMPT = f"""You are reviewing synthetically augmented training images for a traffic-light-\
recognition classifier's data pipeline. You will be shown two images: [1] the ORIGINAL real \
captured crop, and [2] an AUGMENTED version of it produced by an automated pipeline. You are told \
exactly which transformation(s) were applied and their parameters. Your job is NOT to judge \
whether the traffic-light label/annotation is correct — it is to judge whether the AUGMENTATION \
ITSELF looks realistic and free of visible defects, by comparing image [2] against image [1] and \
against what a real camera would plausibly capture.

Respond with ONLY a single JSON object, no markdown fences, no extra prose, matching exactly this schema:
{{"aug_ok": bool, "issues": [{{"category": one of {ISSUE_CATEGORIES}, "severity": one of {SEVERITIES}, \
"description": string}}], "overall_notes": string}}"""

USER_PROMPT_HEADER = (
    "Review this augmented sample from clip {clip_id}, key {key}, camera {channel}.\n"
    "Applied transformation(s): {aug_summary}\n"
    "Image [1] is the original; image [2] is the augmented result. "
    "List any concrete defects in the augmentation itself."
)


def load_prompt_guidance(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_aug_user_prompt(clip_id: str, key: str, channel: str, aug_tags: list[dict], guidance: dict) -> str:
    aug_summary = "; ".join(t["description"] for t in aug_tags) or "(unknown)"
    parts = [USER_PROMPT_HEADER.format(clip_id=clip_id, key=key, channel=channel, aug_summary=aug_summary)]
    default_text = guidance.get("default", "")
    tag_guidance = guidance.get("tags", {}) or {}
    for t in aug_tags:
        text = tag_guidance.get(t["type"], default_text)
        if text:
            parts.append(f"\n### Focus: {t['type']} ({t['description']})\n{text}")
    return "\n".join(parts)
