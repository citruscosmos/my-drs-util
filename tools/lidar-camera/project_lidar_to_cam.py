#!/usr/bin/env python3
"""LiDAR 点群をカメラ画像へ投影する。(camera0–11 対応)

2 パターン出力:
  - undistort/ : 画像を歪み補正 -> ピンホール投影(新カメラ行列 newK)で点を重畳
  - distort/   : 原画像はそのまま -> 点群側を歪みモデルで投影

歪みモデルは --distortion-model で切り替え:
  rational_polynomial (デフォルト): cv2.projectPoints / cv2.initUndistortRectifyMap
  equidistant (魚眼):               cv2.fisheye.projectPoints / cv2.fisheye.initUndistortRectifyMap

equidistant を使う場合は D を 4 係数(k1,k2,k3,k4)に更新すること。

点の色は intensity を 0~40 で正規化した JET カラーマップ。半透過(alpha)で重畳。
内部パラメータは全カメラ共通(ユーザ提供 camera2 の camera_info を使い回し)。
カメラごとに変わるのは外部パラメータ(lidarX -> cameraN/camera_link)のみ。
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

# ---- カメラ内部パラメータ フォールバック値 (camera_info.json がない場合のみ使用) ----
_DEFAULT_K = np.array([[1495.316895, 0, 1424.459106],
                       [0, 1494.778564,  943.463684],
                       [0, 0, 1]], dtype=np.float64)
_DEFAULT_D = np.array([0.929988205433, 0.165922805667, -0.000024369263, -0.000013859207,
                       0.002885974478, 1.331432461739, 0.435830116272, 0.027607271448],
                      dtype=np.float64)
_DEFAULT_W, _DEFAULT_H = 2880, 1860
_DEFAULT_MODEL = "rational_polynomial"

# ---- 外部パラメータ (/tf_static より): lidarX -> cameraN/camera_link ----
# RPY (rad) -> quaternion (x,y,z,w) 変換: ZYX extrinsic 規約
def _rpy_to_q(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2),  np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2),   np.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


_BASE = "/data/ssd2/calib_dump"

CAM_CONFIGS = {
    # lidar_front ペア
    "camera0": {
        "dir":   f"{_BASE}/camera0",
        "lidar": f"{_BASE}/lidar_front",
        "t": np.array([-0.017608,  0.026447, -0.084288]),
        "q": _rpy_to_q(-0.001498,  0.000024,  0.007782),
    },
    "camera1": {
        "dir":   f"{_BASE}/camera1",
        "lidar": f"{_BASE}/lidar_front",
        "t": np.array([ 0.088317, -0.025023, -0.098224]),
        "q": _rpy_to_q(-0.002738, -0.001877,  0.007352),
    },
    "camera8": {
        "dir":   f"{_BASE}/camera8",
        "lidar": f"{_BASE}/lidar_front",
        "t": np.array([ 2.544210,  0.023070, -1.455230]),
        "q": _rpy_to_q(-0.015700,  0.514310, -0.026270),
    },
    # lidar_right ペア
    "camera2": {
        "dir":   f"{_BASE}/camera2",
        "lidar": f"{_BASE}/lidar_right",
        "t": np.array([ 0.029606,  0.108714, -0.064901]),
        "q": _rpy_to_q( 0.012516,  0.434295,  0.480116),
    },
    "camera3": {
        "dir":   f"{_BASE}/camera3",
        "lidar": f"{_BASE}/lidar_right",
        "t": np.array([ 0.022396, -0.129673, -0.061051]),
        "q": _rpy_to_q( 0.018060,  0.463019, -0.504767),
    },
    "camera9": {
        "dir":   f"{_BASE}/camera9",
        "lidar": f"{_BASE}/lidar_right",
        "t": np.array([ 0.259260,  1.942450, -1.059810]),
        "q": _rpy_to_q(-0.021740,  0.456320, -0.014260),
    },
    # lidar_rear ペア
    "camera4": {
        "dir":   f"{_BASE}/camera4",
        "lidar": f"{_BASE}/lidar_rear",
        "t": np.array([ 0.038273,  0.030474, -0.097639]),
        "q": _rpy_to_q( 0.004094, -0.004286, -0.008806),
    },
    "camera5": {
        "dir":   f"{_BASE}/camera5",
        "lidar": f"{_BASE}/lidar_rear",
        "t": np.array([ 0.120238, -0.026997, -0.095939]),
        "q": _rpy_to_q(-0.005269, -0.003393, -0.006281),
    },
    "camera10": {
        "dir":   f"{_BASE}/camera10",
        "lidar": f"{_BASE}/lidar_rear",
        "t": np.array([ 1.180760,  0.005190, -0.923770]),
        "q": _rpy_to_q(-0.002840,  0.268340,  0.003090),
    },
    # lidar_left ペア
    "camera6": {
        "dir":   f"{_BASE}/camera6",
        "lidar": f"{_BASE}/lidar_left",
        "t": np.array([ 0.025694,  0.115418, -0.066878]),
        "q": _rpy_to_q( 0.007910,  0.459430,  0.501650),
    },
    "camera7": {
        "dir":   f"{_BASE}/camera7",
        "lidar": f"{_BASE}/lidar_left",
        "t": np.array([ 0.019590, -0.124525, -0.065413]),
        "q": _rpy_to_q( 0.007517,  0.456098, -0.498016),
    },
    "camera11": {
        "dir":   f"{_BASE}/camera11",
        "lidar": f"{_BASE}/lidar_left",
        "t": np.array([ 0.280710, -1.928760, -1.079930]),
        "q": _rpy_to_q( 0.006410,  0.446770,  0.033720),
    },
}
# camera_link -> camera_optical_link (全カメラ共通)
T2_t = np.array([0.0, 0.0, 0.0])
T2_q = np.array([0.5, -0.5, 0.5, -0.5])  # x,y,z,w

# intensity カラーマップ正規化範囲
IMIN, IMAX = 0.0, 40.0


def quat_to_R(q):
    x, y, z, w = q
    n = np.linalg.norm([x, y, z, w])
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def make_T(t, q):
    M = np.eye(4)
    M[:3, :3] = quat_to_R(q)
    M[:3, 3] = t
    return M


def extrinsic_lr_to_optical(cam_cfg):
    """lidar 座標 -> camera optical 座標 への (R, t) を返す。"""
    M_lr_cl = make_T(cam_cfg["t"], cam_cfg["q"])  # cl -> lr
    M_cl_col = make_T(T2_t, T2_q)                 # col -> cl
    M_lr_col = M_lr_cl @ M_cl_col                 # col -> lr
    M_col_lr = np.linalg.inv(M_lr_col)            # lr -> col
    return M_col_lr[:3, :3], M_col_lr[:3, 3]


def _load_camera_info(cam_dir):
    """cam_dir/camera_info.json から (K, D, w, h, model) を返す。なければ None。"""
    path = Path(cam_dir) / "camera_info.json"
    if not path.exists():
        return None
    with open(path) as f:
        info = json.load(f)
    return (
        np.array(info["K"], dtype=np.float64),
        np.array(info["D"], dtype=np.float64),
        info["width"],
        info["height"],
        info["distortion_model"],
    )


def build_undistort(K, D, img_w, img_h, distortion_model):
    """モデルに応じた (new_K, map1, map2, project_fn) を返す。

    project_fn(pts_Nx3) -> uv_Nx2: カメラ座標系の点をピクセル座標へ投影する関数。
    """
    if distortion_model == "equidistant":
        if D.size < 4:
            raise ValueError("equidistant モデルには D が 4 係数以上必要です")
        D4 = D[:4].reshape(4, 1)
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D4, (img_w, img_h), np.eye(3), balance=1.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D4, np.eye(3), new_K, (img_w, img_h), cv2.CV_16SC2)

        def project_fn(pts):
            uv, _ = cv2.fisheye.projectPoints(
                pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D4)
            return uv.reshape(-1, 2)
    else:  # plumb_bob / rational_polynomial
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, (img_w, img_h), 1, (img_w, img_h))
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, None, new_K, (img_w, img_h), cv2.CV_16SC2)

        def project_fn(pts):
            uv, _ = cv2.projectPoints(
                pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
            return uv.reshape(-1, 2)

    return new_K, map1, map2, project_fn


def read_pcd_xyzi(path):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"DATA binary\n"):
            b = f.read(1)
            if not b:
                break
            hdr += b
        raw = f.read()
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)
    return arr[:, :3].astype(np.float64), arr[:, 3].astype(np.float64)


def color_by_intensity(inten):
    vals = np.clip((inten - IMIN) / (IMAX - IMIN), 0, 1)
    cm = cv2.applyColorMap((vals * 255).astype(np.uint8).reshape(-1, 1),
                           cv2.COLORMAP_JET).reshape(-1, 3)
    return cm  # BGR


def draw_points(img, u, v, colors, rad=2, alpha=0.45):
    """点を alpha 合成で重畳(背景画像が透けて見えるように)。
    overlay+mask 方式なので、点が重なる画素でも透過率は一定になる。
    alpha=1.0 で不透明、小さいほど背景がよく見える。"""
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
    img[mask] = (alpha * overlay[mask].astype(np.float32)
                 + (1.0 - alpha) * img[mask].astype(np.float32)).astype(np.uint8)
    return img


def run(cam, out_root, sample, limit, alpha, start_ns, end_ns,
        cam_dir=None, lidar_dir=None, distortion_model=None):
    cfg = CAM_CONFIGS[cam]
    R_col_lr, t_col_lr = extrinsic_lr_to_optical(cfg)
    cam_dir = cam_dir or cfg["dir"]
    lidar_dir = lidar_dir or cfg["lidar"]

    # camera_info.json があればカメラ固有の intrinsics を使用
    cam_info = _load_camera_info(cam_dir)
    if cam_info is not None:
        _K, _D, _W, _H, info_model = cam_info
        _model = distortion_model or info_model
        print(f"[proj] camera_info loaded: {_W}x{_H} model={info_model}"
              + (f" (overridden -> {_model})" if distortion_model else ""), flush=True)
    else:
        _K, _D, _W, _H = _DEFAULT_K, _DEFAULT_D, _DEFAULT_W, _DEFAULT_H
        _model = distortion_model or _DEFAULT_MODEL
        print(f"[proj] camera_info not found, using default intrinsics "
              f"(model={_model})", flush=True)

    new_K, map1, map2, project_fn = build_undistort(_K, _D, _W, _H, _model)
    nfx, nfy = new_K[0, 0], new_K[1, 1]
    ncx, ncy = new_K[0, 2], new_K[1, 2]

    cam_files = sorted(glob.glob(f"{cam_dir}/*.jpg"))
    cam_ts = np.array([int(os.path.basename(p).split('.')[0]) for p in cam_files])
    lidar_files = sorted(glob.glob(f"{lidar_dir}/*.pcd"))
    lidar_ts = np.array([int(os.path.basename(p).split('.')[0]) for p in lidar_files])

    if start_ns is not None:
        mask = lidar_ts >= start_ns
        lidar_files = [f for f, m in zip(lidar_files, mask) if m]
        lidar_ts = lidar_ts[mask]
    if end_ns is not None:
        mask = lidar_ts <= end_ns
        lidar_files = [f for f, m in zip(lidar_files, mask) if m]
        lidar_ts = lidar_ts[mask]

    idxs = list(range(len(lidar_files)))
    if sample:
        idxs = list(np.linspace(0, len(lidar_files) - 1, sample).astype(int))
    elif limit:
        idxs = idxs[:limit]

    out_un = Path(out_root) / "undistort"
    out_di = Path(out_root) / "distort"
    out_un.mkdir(parents=True, exist_ok=True)
    out_di.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    done = 0
    for li in idxs:
        lts = lidar_ts[li]
        ci = int(np.argmin(np.abs(cam_ts - lts)))
        dt_ms = abs(int(cam_ts[ci]) - int(lts)) / 1e6
        img = cv2.imread(cam_files[ci])
        if img is None:
            print(f"  skip(image read fail) {cam_files[ci]}", flush=True)
            continue
        pts_lr, inten = read_pcd_xyzi(lidar_files[li])
        p_col = (R_col_lr @ pts_lr.T).T + t_col_lr
        front = p_col[:, 2] > 0.1
        p_col = p_col[front]
        inten = inten[front]
        colors = color_by_intensity(inten)

        # --- distort: 原画像 + 歪み込み投影 ---
        img_di = img.copy()
        if p_col.shape[0] > 0:
            uv = project_fn(p_col)
            draw_points(img_di, uv[:, 0], uv[:, 1], colors, alpha=alpha)

        # --- undistort: 歪み補正画像 + ピンホール投影(newK) ---
        img_un = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        if p_col.shape[0] > 0:
            u = nfx * p_col[:, 0] / p_col[:, 2] + ncx
            v = nfy * p_col[:, 1] / p_col[:, 2] + ncy
            draw_points(img_un, u, v, colors, alpha=alpha)

        name = f"{lts}.jpg"
        cv2.imwrite(str(out_di / name), img_di, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out_un / name), img_un, [cv2.IMWRITE_JPEG_QUALITY, 95])
        done += 1
        if done % 200 == 0 or sample:
            print(f"  [{done}/{len(idxs)}] lidar_ts={lts} cam_dt={dt_ms:.1f}ms "
                  f"pts_in_front={p_col.shape[0]}", flush=True)

    print(f"DONE cam={cam} {done} frames / {time.time()-t0:.0f}s -> {out_root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam", choices=sorted(CAM_CONFIGS.keys()))
    ap.add_argument("out_root")
    ap.add_argument("--sample", type=int, default=0, help="時間均等にN枚だけ(検証用)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="点の不透明度(0~1)。小さいほど背景が透ける")
    ap.add_argument("--start", type=float, default=None, help="開始時刻(UNIX秒)")
    ap.add_argument("--end", type=float, default=None, help="終了時刻(UNIX秒)")
    ap.add_argument("--cam-dir", default=None,
                    help="カメラ画像ディレクトリ(省略時はCAM_CONFIGSのパスを使用)")
    ap.add_argument("--lidar-dir", default=None,
                    help="LiDAR PCDディレクトリ(省略時はCAM_CONFIGSのパスを使用)")
    ap.add_argument("--distortion-model", default=None,
                    choices=["rational_polynomial", "equidistant"],
                    help="カメラ歪みモデル(省略時はcamera_info.jsonから自動検出)")
    args = ap.parse_args()
    start_ns = int(args.start * 1e9) if args.start is not None else None
    end_ns = int(args.end * 1e9) if args.end is not None else None

    print(f"cam={args.cam} alpha={args.alpha}", flush=True)

    run(args.cam, args.out_root, args.sample, args.limit, args.alpha, start_ns, end_ns,
        cam_dir=args.cam_dir, lidar_dir=args.lidar_dir,
        distortion_model=args.distortion_model)


if __name__ == "__main__":
    main()
