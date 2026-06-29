#!/usr/bin/env python3
"""lidar_right の点群を camera(2/3) 画像へ投影する。

2 パターン出力:
  - undistort/ : 画像を歪み補正(rational_polynomial, balance=1 相当=alpha=1 黒縁あり/全画角保持)
                 -> ピンホール投影(新カメラ行列 newK)で点を重畳
  - distort/   : 原画像はそのまま -> 点群側を歪みモデルで投影(cv2.projectPoints)

点の色は intensity を 0~40 で正規化した JET カラーマップ。半透過(alpha)で重畳。
内部パラメータは全カメラ共通(ユーザ提供 camera2 の camera_info を使い回し)。
カメラごとに変わるのは外部パラメータ(lidar_right -> cameraN/camera_link)のみ。
"""
import argparse
import glob
import os
import time
from pathlib import Path

import cv2
import numpy as np

# ---- カメラ内部パラメータ (ユーザ提供 camera_info, 全カメラ共通で使い回し) ----
FX, FY = 1495.316895, 1494.778564
CX, CY = 1424.459106, 943.463684
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)
D = np.array([0.929988205433, 0.165922805667, -0.000024369263, -0.000013859207,
              0.002885974478, 1.331432461739, 0.435830116272, 0.027607271448], dtype=np.float64)
IMG_W, IMG_H = 2880, 1860

# ---- 外部パラメータ (/tf_static より): lidarX -> cameraN/camera_link ----
# 各カメラはペアとなる lidar を持つ(camera2,3->lidar_right / camera6,7->lidar_left)
CAM_CONFIGS = {
    "camera2": {
        "dir": "/data/ssd2/calib_dump/camera2",
        "lidar": "/data/ssd2/calib_dump/lidar_right",
        "t": np.array([0.029606, 0.108714, -0.064901]),
        "q": np.array([-0.04528721558963694, 0.2107157777742464,
                       0.23086129675254347, 0.9488155725760757]),  # x,y,z,w
    },
    "camera3": {
        "dir": "/data/ssd2/calib_dump/camera3",
        "lidar": "/data/ssd2/calib_dump/lidar_right",
        "t": np.array([0.022396, -0.129673, -0.061051]),
        "q": np.array([0.06580402306228364, 0.21997435190097678,
                       -0.2450469434952197, 0.941930523201268]),  # x,y,z,w
    },
    "camera6": {
        "dir": "/data/ssd2/calib_dump/camera6",
        "lidar": "/data/ssd2/calib_dump/lidar_left",
        "t": np.array([0.025694, 0.115418, -0.066878]),
        "q": np.array([-0.05278484897985, 0.22152895632862807,
                       0.24080899743044448, 0.943477454941382]),  # x,y,z,w
    },
    "camera7": {
        "dir": "/data/ssd2/calib_dump/camera7",
        "lidar": "/data/ssd2/calib_dump/lidar_left",
        "t": np.array([0.01959, -0.124525, -0.065413]),
        "q": np.array([0.05926300354961544, 0.2182008250684628,
                       -0.24088391677898185, 0.943849159022212]),  # x,y,z,w
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
    """lidar_right 座標 -> camera optical 座標 への (R, t) を返す。"""
    M_lr_cl = make_T(cam_cfg["t"], cam_cfg["q"])  # cl -> lr
    M_cl_col = make_T(T2_t, T2_q)                 # col -> cl
    M_lr_col = M_lr_cl @ M_cl_col                 # col -> lr
    M_col_lr = np.linalg.inv(M_lr_col)            # lr -> col
    return M_col_lr[:3, :3], M_col_lr[:3, 3]


# balance=1 相当: alpha=1 で全画角を保持(黒縁あり)。点投影もこの newK を使う。
NEW_K, _roi = cv2.getOptimalNewCameraMatrix(K, D, (IMG_W, IMG_H), 1, (IMG_W, IMG_H))
NFX, NFY = NEW_K[0, 0], NEW_K[1, 1]
NCX, NCY = NEW_K[0, 2], NEW_K[1, 2]


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


def run(cam, out_root, sample, limit, alpha, start_ns, end_ns, cam_dir=None, lidar_dir=None):
    cfg = CAM_CONFIGS[cam]
    R_col_lr, t_col_lr = extrinsic_lr_to_optical(cfg)
    cam_dir = cam_dir or cfg["dir"]
    lidar_dir = lidar_dir or cfg["lidar"]

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

    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, NEW_K, (IMG_W, IMG_H), cv2.CV_16SC2)

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
            uv, _ = cv2.projectPoints(p_col.reshape(-1, 1, 3),
                                      np.zeros(3), np.zeros(3), K, D)
            uv = uv.reshape(-1, 2)
            draw_points(img_di, uv[:, 0], uv[:, 1], colors, alpha=alpha)

        # --- undistort: 歪み補正画像(balance=1, 黒縁あり) + ピンホール投影(newK) ---
        img_un = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        if p_col.shape[0] > 0:
            u = NFX * p_col[:, 0] / p_col[:, 2] + NCX
            v = NFY * p_col[:, 1] / p_col[:, 2] + NCY
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
    args = ap.parse_args()
    start_ns = int(args.start * 1e9) if args.start is not None else None
    end_ns = int(args.end * 1e9) if args.end is not None else None
    print(f"cam={args.cam} alpha={args.alpha}\nNEW_K(alpha=1)=\n{NEW_K}", flush=True)
    run(args.cam, args.out_root, args.sample, args.limit, args.alpha, start_ns, end_ns,
        cam_dir=args.cam_dir, lidar_dir=args.lidar_dir)


if __name__ == "__main__":
    main()
