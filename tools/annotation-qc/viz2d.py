#!/usr/bin/env python3
"""画像への描画: LiDAR点群投影, 3Dボックスワイヤーフレーム, 2Dbbox, RLEマスクオーバーレイ。

color_by_intensity / draw_points は tools/lidar-camera/project_lidar_to_cam.py から移植
(キャリブレーション規約に依存しない部分のみ)。
"""
from __future__ import annotations

import base64

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from adapter_base import Box2D, Box3D, EgoPose, LidarPoints, Mask2D, SensorCalib
from geometry import BOX3D_EDGES, box3d_to_camera, lidar_points_to_camera

# ported from tools/lidar-camera/project_lidar_to_cam.py (color_by_intensity)
IMIN, IMAX = 0.0, 40.0


def color_by_intensity(inten: np.ndarray) -> np.ndarray:
    vals = np.clip((inten - IMIN) / (IMAX - IMIN), 0, 1)
    cm = cv2.applyColorMap((vals * 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)
    return cm  # BGR


# ported from tools/lidar-camera/project_lidar_to_cam.py (draw_points)
def draw_points(img: np.ndarray, u: np.ndarray, v: np.ndarray, colors: np.ndarray, rad: int = 1, alpha: float = 0.45) -> np.ndarray:
    h, w = img.shape[:2]
    overlay = img.copy()
    mask = np.zeros((h, w), dtype=bool)
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)
    for du in range(-rad, rad + 1):
        for dv in range(-rad, rad + 1):
            uu = u + du
            vv = v + dv
            m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            overlay[vv[m], uu[m]] = colors[m]
            mask[vv[m], uu[m]] = True
    img[mask] = (alpha * overlay[mask].astype(np.float32) + (1.0 - alpha) * img[mask].astype(np.float32)).astype(np.uint8)
    return img


def draw_lidar_points(img: np.ndarray, lidar: LidarPoints, cam_ego_pose: EgoPose, cam_calib: SensorCalib,
                       frame_convention: str = "world", alpha: float = 0.45) -> np.ndarray:
    uv, keep = lidar_points_to_camera(lidar, cam_ego_pose, cam_calib, frame_convention)
    if not np.any(keep):
        return img
    h, w = img.shape[:2]
    u, v = uv[keep, 0], uv[keep, 1]
    inbounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(inbounds):
        return img
    colors = color_by_intensity(lidar.intensity[keep][inbounds])
    return draw_points(img, u[inbounds], v[inbounds], colors, rad=1, alpha=alpha)


def _category_color(category_name: str) -> tuple[int, int, int]:
    h = abs(hash(category_name)) % 180
    hsv = np.uint8([[[h, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_box3d_wireframe(img: np.ndarray, box: Box3D, cam_ego_pose: EgoPose, cam_calib: SensorCalib,
                          frame_convention: str = "world", thickness: int = 2) -> np.ndarray:
    uv, keep = box3d_to_camera(box, cam_ego_pose, cam_calib, frame_convention)
    color = _category_color(box.category_name)
    for i, j in BOX3D_EDGES:
        if not (keep[i] and keep[j]):
            continue
        p1 = tuple(np.round(uv[i]).astype(int))
        p2 = tuple(np.round(uv[j]).astype(int))
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
    if keep[0]:
        label_pt = tuple(np.round(uv[0]).astype(int))
        cv2.putText(img, box.category_name, label_pt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def draw_bbox2d(img: np.ndarray, box: Box2D, thickness: int = 1) -> np.ndarray:
    color = _category_color(box.category_name)
    x1, y1, x2, y2 = [int(round(v)) for v in box.bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    label = box.category_name
    cv2.putText(img, label, (x1, max(0, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return img


def draw_mask_overlay(img: np.ndarray, mask2d: Mask2D, alpha: float = 0.4) -> np.ndarray:
    rle = dict(mask2d.rle)
    counts = rle.get("counts")
    if isinstance(counts, str):
        # このデータセットのRLE countsはpycocotools圧縮RLE文字列をさらにbase64した形式
        # (実データで確認済み: base64デコード後のバイト長が5〜数千程度で、pycocotools.decode
        # がエラーなく妥当なマスク面積を返すことを確認)。素のASCIIエンコードでは
        # "Invalid RLE mask representation" になる。
        rle["counts"] = base64.b64decode(counts)
    m = mask_utils.decode(rle).astype(bool)
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    color = np.array(_category_color(mask2d.category_name), dtype=np.uint8)
    overlay = img.copy()
    overlay[m] = color
    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img
