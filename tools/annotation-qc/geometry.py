#!/usr/bin/env python3
"""nuScenes/t4dataset規約(スカラー先置クォータニオン, sensor->ego直接)での座標変換・投影。

tools/lidar-camera/project_lidar_to_cam.py の quat_to_R/make_T 等は ROS tf_static 規約
(スカラー後置クォータニオン + camera_link->camera_optical_link の固定回転)を前提にしており、
ここでは使えない。calibrated_sensor/ego_pose の実データ(恒等回転が [1,0,0,0])を確認した上で
このモジュールを新規に書いている。
"""
from __future__ import annotations

import numpy as np
import cv2

from adapter_base import Box3D, EgoPose, LidarPoints, SensorCalib

# 立方体8頂点のうちワイヤーフレームを描くための12エッジ (底面4 + 上面4 + 垂直4)
BOX3D_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def quat_wxyz_to_R(q: list[float]) -> np.ndarray:
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n == 0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def world_to_ego(p_world: np.ndarray, ego_pose: EgoPose) -> np.ndarray:
    R = quat_wxyz_to_R(ego_pose.rotation)
    t = np.array(ego_pose.translation)
    return (p_world - t) @ R


def ego_to_world(p_ego: np.ndarray, ego_pose: EgoPose) -> np.ndarray:
    R = quat_wxyz_to_R(ego_pose.rotation)
    t = np.array(ego_pose.translation)
    return p_ego @ R.T + t


def ego_to_sensor(p_ego: np.ndarray, calib: SensorCalib) -> np.ndarray:
    R = quat_wxyz_to_R(calib.rotation)
    t = np.array(calib.translation)
    return (p_ego - t) @ R


def sensor_to_ego(p_sensor: np.ndarray, calib: SensorCalib) -> np.ndarray:
    R = quat_wxyz_to_R(calib.rotation)
    t = np.array(calib.translation)
    return p_sensor @ R.T + t


def project_cam_frame_points(p_cam: np.ndarray, calib: SensorCalib) -> tuple[np.ndarray, np.ndarray]:
    """p_cam: (N,3) このカメラの光学フレームに既にある点。

    戻り値: (uv (N,2) — front=Falseの行は無効値, front (N,) bool マスク)
    """
    n = p_cam.shape[0]
    front = p_cam[:, 2] > 0.1
    uv = np.zeros((n, 2), dtype=np.float64)
    if not np.any(front):
        return uv, front

    p = p_cam[front].astype(np.float64)
    K = np.array(calib.camera_intrinsic, dtype=np.float64)
    D = np.array(calib.camera_distortion, dtype=np.float64)
    rvec = np.zeros(3)
    tvec = np.zeros(3)

    if calib.is_fisheye:
        pts = p.reshape(-1, 1, 3)
        proj, _ = cv2.fisheye.projectPoints(pts, rvec, tvec, K, D.reshape(4, 1))
    else:
        pts = p.reshape(-1, 1, 3)
        proj, _ = cv2.projectPoints(pts, rvec, tvec, K, D)

    uv[front] = proj.reshape(-1, 2)
    return uv, front


def world_points_to_camera(p_world: np.ndarray, cam_ego_pose: EgoPose, cam_calib: SensorCalib):
    p_ego = world_to_ego(p_world, cam_ego_pose)
    p_cam = ego_to_sensor(p_ego, cam_calib)
    return project_cam_frame_points(p_cam, cam_calib)


def lidar_points_to_camera(
    lidar: LidarPoints, cam_ego_pose: EgoPose, cam_calib: SensorCalib, frame_convention: str = "world"
):
    """lidarローカル -> ego(lidar時刻) -> [world ->] ego(カメラ時刻) -> カメラ、のフルチェーン。

    frame_convention="ego" の場合はworld経由を省略する近似(センサー間でego_poseが
    実質共有されている前提)。t4dataset-webdatasetの実データではこの近似でも
    ego_pose がチャンネル間でほぼ同一だったため実用上問題ないが、汎用性のため両方実装する。
    """
    p_ego_lidar_time = sensor_to_ego(lidar.xyz, lidar.calib)
    if frame_convention == "world":
        p_world = ego_to_world(p_ego_lidar_time, lidar.ego_pose)
        return world_points_to_camera(p_world, cam_ego_pose, cam_calib)
    p_cam = ego_to_sensor(p_ego_lidar_time, cam_calib)
    return project_cam_frame_points(p_cam, cam_calib)


def box3d_corners(box: Box3D) -> np.ndarray:
    """nuscenes-devkit標準の8頂点算出。size=[w, l, h]。戻り値はbox.translationと同じフレーム。"""
    w, l, h = box.size
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners_local = np.vstack((x, y, z))  # (3, 8)
    R = quat_wxyz_to_R(box.rotation)
    return (R @ corners_local).T + np.array(box.translation)  # (8, 3)


def box3d_to_camera(box: Box3D, cam_ego_pose: EgoPose, cam_calib: SensorCalib, frame_convention: str = "world"):
    corners = box3d_corners(box)
    if frame_convention == "world":
        uv, keep = world_points_to_camera(corners, cam_ego_pose, cam_calib)
    else:
        p_cam = ego_to_sensor(corners, cam_calib)
        uv, keep = project_cam_frame_points(p_cam, cam_calib)
    return uv, keep
