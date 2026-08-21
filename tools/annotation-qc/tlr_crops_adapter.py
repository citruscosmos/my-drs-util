#!/usr/bin/env python3
"""信号機認識(TLR)用クロップデータセットのアダプタ。

想定入力レイアウト (例: /data5/v-shimaki2507/tlr_ws/datasets/baseline3/train_dataset):
    Annotations/<crop_name>.json   1クロップ(信号機グループ)ごとの構造化アノテーション
    JPEGImages/<crop_name>.jpg     そのクロップ自体の画像(数十px四方など非常に小さい)
    labels/<crop_name>.txt         Darknet分類器用ラベル(このアダプタでは使わない)

Annotations/*.json のみを正とする(データ拡張で生成された whole_*/flicker サフィックス
付きの画像は JSON を持たないため、Annotations を基準にすることで自然に除外される)。

各JSONの"lights"配列は、1クロップ内に複数の灯火(例: 矢印+丸型の複合信号)を持ちうる
(実データで確認済み)。category_nameは常に"traffic_light"にし、色状態(red/green/...)や
lit_frac/overexposed_frac/orientation/typeはattribute_namesに乗せる。こうすることで
既存の"traffic_light"カテゴリ向けQA観点(qc_prompts.yaml)がそのまま使え、--categories
traffic_light での絞り込みも他アダプタと同じ感覚で機能する。

クロップは点群も3Dアノテーションも持たない(supports_lidar=False, supports_3d=False)。
クロップは数十pxしかないことが多く、VLMがそのままでは色や形をほぼ見えないため、
image_loaderは短辺が最低 MIN_DISPLAY_SIZE になるようアップスケールしてから返す。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from adapter_base import Box2D, CameraView, DatasetAdapter, EgoPose, FrameRecord, SensorCalib

# "CAM_BACK_NARROW_00041_traffic_light_..." -> channel="CAM_BACK_NARROW". チャンネル名自体に
# アンダースコアが含まれるため、5桁のフレーム番号が始まる位置までを非貪欲にまとめて取る。
_CHANNEL_RE = re.compile(r"^(CAM_[A-Z0-9]+(?:_[A-Z0-9]+)*?)_(\d{5})_")
_ZERO_EGO = EgoPose(translation=[0.0, 0.0, 0.0], rotation=[1.0, 0.0, 0.0, 0.0], timestamp=0)
_ZERO_CALIB = SensorCalib(translation=[0.0, 0.0, 0.0], rotation=[1.0, 0.0, 0.0, 0.0], camera_intrinsic=[], camera_distortion=[])


def _split_clip_and_rest(stem: str) -> Optional[tuple[str, str]]:
    """ファイル名を最初の "_CAM_" の前後で分割する。

    クロップ由来のデータソースが複数あり(t4dataset風の命名, "rosbag2_...", "SemiAnnotation_..." 等)
    clip_id自体の形式は統一されていないが、カメラチャンネル名は常に "CAM_" で始まるため、
    この分割方法はソースの命名規則に依存せず頑健。
    """
    idx = stem.find("_CAM_")
    if idx == -1:
        return None
    return stem[:idx], stem[idx + 1:]


def _load_native(path: Path) -> np.ndarray:
    """ネイティブ解像度でそのまま読む。

    アップスケールはここではしない: box座標(bbox_relative)はネイティブ解像度基準なので、
    ここで拡大すると描画時に座標とサイズが食い違う(実データで踏んだバグ)。拡大は
    run_qc.py側で全レイヤー描画が終わった後の合成画像に対して行う。
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"failed to read image: {path}")
    return img


class TLRCropsAdapter:
    name = "tlr-crops"
    supports_lidar = False
    supports_3d = False

    def discover_clips(self, input_path: str) -> list[str]:
        root = Path(input_path) / "Annotations"
        clip_ids = set()
        for p in root.glob("*.json"):
            split = _split_clip_and_rest(p.stem)
            if split is not None:
                clip_ids.add(split[0])
        return sorted(clip_ids)

    def iter_frames(
        self,
        input_path: str,
        clip_id: str,
        every_n: int = 1,
        channels: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
    ) -> Iterator[FrameRecord]:
        root = Path(input_path)
        anno_dir = root / "Annotations"
        image_dir = root / "JPEGImages"

        # traffic_light はこのアダプタの唯一のカテゴリなので、対象カテゴリに含まれて
        # いなければ何もせず即終了する(--categories car のような指定の場合)。
        if categories is not None and "traffic_light" not in categories:
            return

        prefix = f"{clip_id}_CAM_"
        matches = sorted(anno_dir.glob(f"{prefix}*.json"))

        for i, json_path in enumerate(matches[::every_n]):
            with open(json_path) as f:
                data = json.load(f)

            split = _split_clip_and_rest(json_path.stem)
            channel_match = _CHANNEL_RE.match(split[1]) if split else None
            channel = channel_match.group(1) if channel_match else "UNKNOWN"
            if channels is not None and channel not in channels:
                continue

            image_path = image_dir / f"{json_path.stem}.jpg"

            def _make_loader(p=image_path):
                return lambda: _load_native(p)

            boxes2d = []
            for light in data.get("lights") or []:
                bbox = light["bbox_relative"]
                attrs = [
                    f"state={light.get('label')}",
                    f"orientation={light.get('orientation', 'n_a')}",
                    f"type={light.get('type', 'unknown')}",
                    f"lit_frac={light.get('lit_frac', 0):.2f}",
                    f"overexposed_frac={light.get('overexposed_frac', 0):.2f}",
                ]
                boxes2d.append(Box2D(bbox=list(bbox), category_name="traffic_light", attribute_names=attrs))

            cam = CameraView(channel=channel, image_loader=_make_loader(), ego_pose=_ZERO_EGO, calib=_ZERO_CALIB, boxes2d=boxes2d)

            yield FrameRecord(
                clip_id=clip_id,
                frame_index=i,
                key=json_path.stem,
                sample_token=json_path.stem,
                timestamp=0,
                frame_convention="world",
                cameras={channel: cam},
                lidar=None,
                boxes3d=[],
                metadata={
                    "original_full_path": data.get("original_full_path"),
                    "signal_bbox_global": data.get("signal_bbox_global"),
                    "crop_size": data.get("crop_size"),
                },
            )
