#!/usr/bin/env python3
"""tools/t4dataset-webdataset/ が生成した sensor/anno tar + parquet index を読むアダプタ。

parquetのoffset列を使い、tarをスキャンせず直接seekして読む
(t4dataset-webdatasetの本来の読み出しパターン)。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import pandas as pd

from adapter_base import Box2D, Box3D, CameraView, DatasetAdapter, EgoPose, FrameRecord, LidarPoints, Mask2D, SensorCalib


class T4DatasetWebdatasetAdapter:
    name = "t4dataset-webdataset"
    supports_lidar = True
    supports_3d = True

    def discover_clips(self, input_path: str) -> list[str]:
        root = Path(input_path)
        clip_ids = []
        for manifest_path in sorted(root.rglob("index-*.manifest.json")):
            clip_id = manifest_path.name[len("index-"):-len(".manifest.json")]
            clip_ids.append(clip_id)
        return clip_ids

    def _locate_clip_dir(self, input_path: str, clip_id: str) -> Path:
        root = Path(input_path)
        matches = list(root.rglob(f"index-{clip_id}.manifest.json"))
        if not matches:
            raise FileNotFoundError(f"clip not found under {input_path}: {clip_id}")
        return matches[0].parent

    def iter_frames(
        self,
        input_path: str,
        clip_id: str,
        every_n: int = 1,
        channels: Optional[list[str]] = None,
        categories: Optional[list[str]] = None,
    ) -> Iterator[FrameRecord]:
        clip_dir = self._locate_clip_dir(input_path, clip_id)
        manifest = json.loads((clip_dir / f"index-{clip_id}.manifest.json").read_text())
        df = pd.read_parquet(clip_dir / f"index-{clip_id}.parquet")

        all_channels = manifest["channels"]
        cam_channels = channels or [c for c in all_channels if c != "LIDAR_CONCAT"]
        want_lidar = "LIDAR_CONCAT" in all_channels

        cat_set = set(categories) if categories else None

        sensor_path = clip_dir / manifest["sensor_shard"]
        anno_path = clip_dir / manifest["anno_shard"]

        with open(sensor_path, "rb") as sensor_f, open(anno_path, "rb") as anno_f:
            for row in df.iloc[::every_n].itertuples():
                if cat_set is not None:
                    present = set(row.ann3d_categories) | set(row.ann2d_categories)
                    if not (present & cat_set):
                        continue

                sensor_f.seek(int(row.meta_offset))
                frame_meta = json.loads(sensor_f.read(int(row.meta_size)))
                anno_f.seek(int(row.ann3d_offset))
                ann3d_raw = json.loads(anno_f.read(int(row.ann3d_size)))
                anno_f.seek(int(row.ann2d_offset))
                ann2d_raw = json.loads(anno_f.read(int(row.ann2d_size)))

                cameras: dict[str, CameraView] = {}
                for ch in cam_channels:
                    chmeta = frame_meta["channels"].get(ch)
                    if chmeta is None:
                        continue
                    cs = chmeta["calibrated_sensor"]
                    calib = SensorCalib(
                        translation=cs["translation"], rotation=cs["rotation"],
                        camera_intrinsic=cs["camera_intrinsic"], camera_distortion=cs["camera_distortion"],
                        width=chmeta["width"], height=chmeta["height"],
                    )
                    ego = EgoPose(chmeta["ego_pose"]["translation"], chmeta["ego_pose"]["rotation"], chmeta["ego_pose"]["timestamp"])

                    off = getattr(row, f"{ch.lower()}_offset")
                    size = getattr(row, f"{ch.lower()}_size")
                    if pd.isna(off) or pd.isna(size):
                        # このセンサーだけこのフレームで欠落(実データで確認済み: fisheye 4chが
                        # まとめて欠落するフレームがある)。他チャンネルは正常なので、この
                        # チャンネルだけスキップして処理を継続する。
                        continue
                    off, size = int(off), int(size)

                    def _make_loader(off=off, size=size, sensor_path=sensor_path):
                        def _load():
                            with open(sensor_path, "rb") as f:
                                f.seek(off)
                                buf = f.read(size)
                            return cv2.imdecode(np.frombuffer(buf, dtype=np.uint8), cv2.IMREAD_COLOR)
                        return _load

                    ann2d_ch = ann2d_raw.get(ch, {"boxes": [], "masks": []})
                    boxes2d = [
                        Box2D(b["bbox"], b["category_name"], b.get("attribute_names", []), b.get("instance_name", "") or "")
                        for b in ann2d_ch["boxes"]
                    ]
                    masks2d = [
                        Mask2D(m["mask"], m["category_name"], m.get("attribute_names", []))
                        for m in ann2d_ch["masks"]
                    ]
                    cameras[ch] = CameraView(
                        channel=ch, image_loader=_make_loader(), ego_pose=ego, calib=calib,
                        boxes2d=boxes2d, masks2d=masks2d,
                    )

                lidar = None
                if want_lidar and not pd.isna(row.lidar_concat_offset) and not pd.isna(row.lidar_concat_size):
                    lch = frame_meta["channels"]["LIDAR_CONCAT"]
                    cs = lch["calibrated_sensor"]
                    l_calib = SensorCalib(cs["translation"], cs["rotation"], [], [])
                    l_ego = EgoPose(lch["ego_pose"]["translation"], lch["ego_pose"]["rotation"], lch["ego_pose"]["timestamp"])
                    off, size = int(row.lidar_concat_offset), int(row.lidar_concat_size)
                    sensor_f.seek(off)
                    raw = np.frombuffer(sensor_f.read(size), dtype=np.float32).reshape(-1, 5)
                    lidar = LidarPoints(
                        xyz=raw[:, :3].astype(np.float64), intensity=raw[:, 3], ego_pose=l_ego, calib=l_calib,
                    )

                boxes3d = [
                    Box3D(
                        translation=a["translation"], size=a["size"], rotation=a["rotation"],
                        category_name=a["category_name"], attribute_names=a.get("attribute_names", []),
                        instance_name=a.get("instance_name", "") or "", instance_token=a.get("instance_token", ""),
                        score=a.get("score", 1.0) or 1.0,
                    )
                    for a in ann3d_raw
                ]

                yield FrameRecord(
                    clip_id=clip_id, frame_index=row.frame_index, key=row.key, sample_token=row.sample_token,
                    timestamp=row.timestamp, frame_convention="world",
                    cameras=cameras, lidar=lidar, boxes3d=boxes3d, metadata={},
                )
