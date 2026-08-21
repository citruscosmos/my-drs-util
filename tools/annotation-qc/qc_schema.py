#!/usr/bin/env python3
"""LLM/VLMによる品質チェックの出力スキーマとプロンプト組み立て。

カテゴリ別のQA観点は qc_prompts.yaml に外出ししてあり、コード変更なしに追記・改善できる。
"""
from __future__ import annotations

from pathlib import Path

import yaml

ISSUE_CATEGORIES = [
    "missed_detection",
    "misclassification",
    "box_misalignment",
    "mask_misalignment",
    "duplicate_annotation",
    "wrong_attribute",
    "other",
]
SEVERITIES = ["low", "medium", "high"]

# 汎用VLMはchat completion経由でピクセル/正規化bbox座標を出させても実物体に対して
# グラウンディングされない(実データで確認済み: gemma4:latestが返したbboxは実際には
# 木の葉を指しており、車・歩行者とは無関係だった)。そのため座標ではなく、9分割グリッド
# 位置 + 自由記述のランドマーク(何の近くか)という言葉による位置表現に切り替える。
REGION_GRID = [
    "top-left", "top-center", "top-right",
    "center-left", "center", "center-right",
    "bottom-left", "bottom-center", "bottom-right",
]

SYSTEM_PROMPT = f"""You are an expert QA reviewer for autolabeled autonomous-driving perception data. \
You will be shown one camera image with the current auto-generated annotations already drawn on it: \
colored LiDAR points, 3D bounding-box wireframes, 2D bounding boxes, and semantic segmentation mask \
overlays, each labeled with its category. Your job is to find places where the DRAWN ANNOTATIONS \
disagree with what is actually visible in the image (not to re-annotate the scene from scratch).

For each issue, describe WHERE it is using words only, never pixel or normalized coordinates \
(you cannot reliably estimate exact coordinates, so do not attempt to). Give:
  - "region": one of {REGION_GRID} (divide the frame into a 3x3 grid and name the cell)
  - "landmark": a short free-text phrase naming what's nearby in the scene, e.g. \
"on the roadside near the parked car", "on the sidewalk behind the pole", "at the crosswalk", \
"near the horizon between the two buildings"

Respond with ONLY a single JSON object, no markdown fences, no extra prose, matching exactly this schema:
{{"frame_ok": bool, "issues": [{{"category": one of {ISSUE_CATEGORIES}, \
"severity": one of {SEVERITIES}, "description": string, "region": one of {REGION_GRID}, \
"landmark": string, "instance_name": string or null}}], "overall_notes": string}}"""

USER_PROMPT_HEADER = (
    "Review this frame from clip {clip_id}, frame {key}, camera {channel}. "
    "List any concrete annotation issues you see."
)


def load_prompt_guidance(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_user_prompt(clip_id: str, key: str, channel: str, target_categories: list[str], guidance: dict) -> str:
    parts = [USER_PROMPT_HEADER.format(clip_id=clip_id, key=key, channel=channel)]
    default_text = guidance.get("default", "")
    categories_guidance = guidance.get("categories", {}) or {}
    for cat in target_categories:
        text = categories_guidance.get(cat, default_text)
        if text:
            parts.append(f"\n### Focus: {cat}\n{text}")
    return "\n".join(parts)
