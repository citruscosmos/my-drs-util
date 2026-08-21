#!/usr/bin/env python3
"""データセット形式に依存しない共通のフレーム表現と、アダプタのインターフェース定義。

新しいデータセット形式(例: 信号検出用train/val, 2D画像のみで点群・3Dアノテーションなし)を
追加する場合は、DatasetAdapter を実装した新しいクラスを書き、FrameRecord を返せばよい。
lidar=None / boxes3d=[] の場合、可視化・LLM QC側は該当レイヤーの処理を自動的にスキップする。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal, Optional, Protocol

import numpy as np


@dataclass
class EgoPose:
    translation: list[float]  # [x, y, z]
    rotation: list[float]     # [qw, qx, qy, qz] スカラー先置
    timestamp: int


@dataclass
class SensorCalib:
    translation: list[float]     # sensor -> ego, [x, y, z]
    rotation: list[float]        # sensor -> ego, [qw, qx, qy, qz]
    camera_intrinsic: list        # 3x3 nested list、非カメラ(lidar等)は []
    camera_distortion: list       # fisheyeは4係数、pinholeは5係数。非カメラは []
    width: int = 0
    height: int = 0

    @property
    def is_fisheye(self) -> bool:
        return len(self.camera_distortion) == 4


@dataclass
class Box3D:
    translation: list[float]   # box中心、world(global)フレーム
    size: list[float]          # [w, l, h]
    rotation: list[float]      # [qw, qx, qy, qz]、translationと同じフレーム
    category_name: str
    attribute_names: list[str] = field(default_factory=list)
    instance_name: str = ""
    instance_token: str = ""
    score: float = 1.0


@dataclass
class Box2D:
    bbox: list[float]   # [x1, y1, x2, y2] ピクセル座標(投影済み、追加の変換不要)
    category_name: str
    attribute_names: list[str] = field(default_factory=list)
    instance_name: str = ""


@dataclass
class Mask2D:
    rle: dict   # {"size": [h, w], "counts": "<COCO RLE str>"}
    category_name: str
    attribute_names: list[str] = field(default_factory=list)


@dataclass
class LidarPoints:
    xyz: np.ndarray          # (N, 3) このセンサー自身のローカルフレーム
    intensity: np.ndarray    # (N,)
    ego_pose: EgoPose         # このセンサー自身のego pose
    calib: SensorCalib         # sensor -> ego (LIDAR_CONCATは恒等変換)
    channel: str = "LIDAR_CONCAT"


@dataclass
class CameraView:
    channel: str
    image_loader: Callable[[], np.ndarray]   # 遅延読み出し。呼ぶまでデコードしない
    ego_pose: EgoPose
    calib: SensorCalib
    boxes2d: list[Box2D] = field(default_factory=list)
    masks2d: list[Mask2D] = field(default_factory=list)


@dataclass
class FrameRecord:
    clip_id: str
    frame_index: int
    key: str                 # ゼロ埋めフレームキー、例: "00042"
    sample_token: str
    timestamp: int
    frame_convention: Literal["world", "ego"]   # boxes3d / lidarのworld変換で使うフレーム規約
    cameras: dict[str, CameraView]
    lidar: Optional[LidarPoints]
    boxes3d: list[Box3D]
    metadata: dict = field(default_factory=dict)

    def categories_present(self) -> set[str]:
        cats = {b.category_name for b in self.boxes3d}
        for cam in self.cameras.values():
            cats.update(b.category_name for b in cam.boxes2d)
            cats.update(m.category_name for m in cam.masks2d)
        return cats


class DatasetAdapter(Protocol):
    name: str
    supports_lidar: bool
    supports_3d: bool

    def discover_clips(self, input_path: str) -> list[str]:
        """input_path 配下から処理可能なクリップID一覧を返す。"""
        ...

    def iter_frames(
        self,
        input_path: str,
        clip_id: str,
        every_n: int = 1,
        channels: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
    ) -> Iterator[FrameRecord]:
        """指定クリップのフレームを順に返す。

        categories が指定された場合、そのカテゴリを1つも含まないフレームは
        (可能なら画像デコード等の重い処理をする前に)スキップしてよい。
        """
        ...
