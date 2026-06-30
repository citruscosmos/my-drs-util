#!/usr/bin/env python3
"""LiDAR-Camera 外部パラメータ 手動調整GUI

機能:
  - 補正済み画像に LiDAR 点群を深度カラーで重畳表示
  - ペアモード: 投影点をクリック(最近傍の3D点を取得) → 正しい画像位置をクリック →
                2D-3D 対応ペアを登録。4組以上で solvePnP により粗合わせ。
  - スライダ: x/y/z/roll/pitch/yaw の6自由度を手動微調整(動かすと即再投影)
  - YAML 出力 (multi_tf_static.yaml 形式)。tf2 static_transform_publisher 引数も併記。

座標系の前提:
  self.T = T_col_lr: lidar 座標の点を camera optical 座標へ変換する 4x4。
  保存時は project_lidar_to_cam.extrinsic_lr_to_optical の逆算で
  T_lr_cl (camera_link in lidar frame) に戻し multi_tf_static.yaml 形式で出力する。

典型的な使い方:
  python tune_extrinsic.py \
    --image  /data/dump/camera0/undistort/1234567890.jpg \
    --points /data/dump/lidar_front/1234567890.pcd \
    --cam-dir /data/dump/camera0 \
    --tf-yaml /path/to/multi_tf_static.yaml --cam camera0 \
    --out extrinsic_adjust.yaml

依存:
  pip install PySide6 matplotlib numpy opencv-python pyyaml
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from PySide6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# 共通ユーティリティを project_lidar_to_cam から再利用
sys.path.insert(0, str(Path(__file__).parent))
from project_lidar_to_cam import (
    T2_q, T2_t, _load_camera_info, _load_tf_yaml,
    build_undistort, extrinsic_lr_to_optical, quat_to_R, read_pcd_xyzi,
)


# ----------------------------------------------------------------------------
# 幾何ユーティリティ (Euler ベース; project_lidar_to_cam は quaternion ベースのため別定義)
# ----------------------------------------------------------------------------
def euler_to_R(roll, pitch, yaw):
    """ZYX 外因性 RPY (rad) -> 3x3 (= Rz @ Ry @ Rx)。"""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr,  cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler(R):
    """3x3 -> (roll, pitch, yaw) rad、ZYX 外因性。"""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 0.99999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw  = np.arctan2(R[1, 0], R[0, 0])
    else:  # ジンバルロック
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw  = 0.0
    return roll, pitch, yaw


def make_T_euler(x, y, z, roll, pitch, yaw):
    T = np.eye(4)
    T[:3, :3] = euler_to_R(roll, pitch, yaw)
    T[:3, 3]  = [x, y, z]
    return T


def decompose_T(T):
    x, y, z = T[:3, 3]
    roll, pitch, yaw = R_to_euler(T[:3, :3])
    return x, y, z, roll, pitch, yaw


def _T2_mat():
    """project_lidar_to_cam の T2_t/T2_q から M_cl_col (optical→camera_link) を構築。"""
    M = np.eye(4)
    M[:3, :3] = quat_to_R(T2_q)
    M[:3, 3]  = T2_t
    return M


def T_col_lr_to_lr_cl(T_col_lr):
    """GUI の T_col_lr (lidar→optical) を T_lr_cl (camera_link in lidar frame) に変換。

    extrinsic_lr_to_optical の逆算:
      M_lr_col = inv(T_col_lr)
      M_lr_cl  = M_lr_col @ inv(M_cl_col)   ただし M_cl_col = _T2_mat()
    """
    T2 = _T2_mat()
    return np.linalg.inv(T_col_lr) @ np.linalg.inv(T2)


# ----------------------------------------------------------------------------
# 投影
# ----------------------------------------------------------------------------
def project(points_lidar, T_cam_lidar, K):
    """LiDAR点(Nx3) を画像へ投影。前方(z>0.001)のみ。

    Returns: uv (Mx2), depth (M,), idx (M,) — 元配列インデックス。
    """
    Xc   = (T_cam_lidar[:3, :3] @ points_lidar.T + T_cam_lidar[:3, 3:4]).T
    front = Xc[:, 2] > 1e-3
    Xc, idx = Xc[front], np.nonzero(front)[0]
    if Xc.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,), dtype=int)
    uvw = (K @ Xc.T).T
    uv  = uvw[:, :2] / uvw[:, 2:3]
    return uv, Xc[:, 2], idx


# ----------------------------------------------------------------------------
# メインウィンドウ
# ----------------------------------------------------------------------------
class TunerWindow(QtWidgets.QMainWindow):
    TRANS_RANGE = 0.10              # 基準値から ±10 cm
    TRANS_STEP  = 0.0005            # 0.5 mm
    ROT_RANGE   = np.deg2rad(5.0)   # 基準値から ±5 deg
    ROT_STEP    = np.deg2rad(0.02)  # 0.02 deg

    def __init__(self, image, points, K, init_T, out_path,
                 lidar_frame="lidar", cam_frame="camera", intensity=None):
        super().__init__()
        self.setWindowTitle("LiDAR-Camera Extrinsic Tuner")

        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        self.h, self.w = self.image.shape[:2]
        self.points    = points.astype(np.float64)
        self.intensity = intensity.astype(np.float64) if intensity is not None else None
        self.K         = K.astype(np.float64)
        self.T         = init_T.copy()
        self.base_T    = init_T.copy()
        self.out_path    = out_path
        self.lidar_frame = lidar_frame
        self.cam_frame   = cam_frame

        self.pairs        = []
        self.pending_3d   = None
        self.pair_mode    = False
        self.pair_submode = "match"   # "match" | "mismatch"

        self.color_mode = "intensity" if intensity is not None else "depth"
        self.pt_size    = 4
        self.pt_alpha   = 0.7
        self.depth_min  = -6.0
        self.depth_max  = 30.0

        self._panning          = False
        self._view_initialized = False

        self._build_ui()
        self._sync_sliders_from_T()
        self.redraw()

    # ---------- UI 構築 ----------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout  = QtWidgets.QHBoxLayout(central)

        self.fig    = Figure(figsize=(9, 7))
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self.canvas = FigureCanvas(self.fig)
        self.ax     = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event",         self._on_scroll)
        layout.addWidget(self.canvas, stretch=3)

        panel = QtWidgets.QVBoxLayout()
        layout.addLayout(panel, stretch=1)

        self.sliders = {}
        specs = [
            ("x",     self.TRANS_RANGE, self.TRANS_STEP, "m"),
            ("y",     self.TRANS_RANGE, self.TRANS_STEP, "m"),
            ("z",     self.TRANS_RANGE, self.TRANS_STEP, "m"),
            ("roll",  self.ROT_RANGE,   self.ROT_STEP,   "deg"),
            ("pitch", self.ROT_RANGE,   self.ROT_STEP,   "deg"),
            ("yaw",   self.ROT_RANGE,   self.ROT_STEP,   "deg"),
        ]
        for name, rng, step, unit in specs:
            row   = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(name)
            label.setFixedWidth(45)
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            n = int(rng / step)
            s.setMinimum(-n); s.setMaximum(n); s.setValue(0); s.setPageStep(1)
            s.valueChanged.connect(self.on_slider)
            val = QtWidgets.QLabel("+0.000")
            val.setFixedWidth(80)
            self.sliders[name] = dict(slider=s, step=step, unit=unit, label=val)
            row.addWidget(label); row.addWidget(s); row.addWidget(val)
            panel.addLayout(row)

        panel.addSpacing(6)
        self.range_label = QtWidgets.QLabel(
            f"微調整幅: 並進±{self.TRANS_RANGE*100:.0f}cm / "
            f"回転±{np.rad2deg(self.ROT_RANGE):.0f}°  "
            "(スライダ選択中に ←→ で1ステップ)")
        self.range_label.setWordWrap(True)
        panel.addWidget(self.range_label)

        rebase_btn = QtWidgets.QPushButton("基準を現在値にリセット (スライダ→0)")
        rebase_btn.clicked.connect(self.rebase)
        panel.addWidget(rebase_btn)

        panel.addSpacing(10)
        panel.addWidget(QtWidgets.QLabel("── 表示設定 ──"))

        # color モード選択
        color_row = QtWidgets.QHBoxLayout()
        color_row.addWidget(QtWidgets.QLabel("color:"))
        self._radio_inten = QtWidgets.QRadioButton("intensity")
        self._radio_depth = QtWidgets.QRadioButton("depth")
        self._radio_inten.setChecked(self.color_mode == "intensity")
        self._radio_depth.setChecked(self.color_mode == "depth")
        if self.intensity is None:
            self._radio_inten.setEnabled(False)
        self._radio_inten.toggled.connect(self._on_color_mode_change)
        color_row.addWidget(self._radio_inten)
        color_row.addWidget(self._radio_depth)
        color_row.addStretch()
        panel.addLayout(color_row)

        disp_specs = [
            ("size",  1,    30,  4,   lambda v: str(v)),
            ("alpha", 1,   100, 70,   lambda v: f"{v/100:.2f}"),
            ("c min", -100, 300, -6,  lambda v: str(v)),
            ("c max", -100, 300, 30,  lambda v: str(v)),
        ]
        self._disp_sliders = {}
        for name, lo, hi, default, fmt in disp_specs:
            row   = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(name)
            label.setFixedWidth(65)
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setMinimum(lo); s.setMaximum(hi); s.setValue(default)
            s.valueChanged.connect(self._on_disp_change)
            val = QtWidgets.QLabel(fmt(default))
            val.setFixedWidth(40)
            self._disp_sliders[name] = dict(slider=s, fmt=fmt, label=val)
            row.addWidget(label); row.addWidget(s); row.addWidget(val)
            panel.addLayout(row)

        panel.addSpacing(10)

        self.pair_btn = QtWidgets.QPushButton("ペアモード: OFF")
        self.pair_btn.setCheckable(True)
        self.pair_btn.toggled.connect(self.toggle_pair_mode)
        panel.addWidget(self.pair_btn)

        submode_row = QtWidgets.QHBoxLayout()
        self._radio_match    = QtWidgets.QRadioButton("一致点追加")
        self._radio_mismatch = QtWidgets.QRadioButton("相違点追加")
        self._radio_match.setChecked(True)
        self._radio_match.setEnabled(False)
        self._radio_mismatch.setEnabled(False)
        self._radio_match.toggled.connect(self._on_pair_submode_change)
        submode_row.addWidget(self._radio_match)
        submode_row.addWidget(self._radio_mismatch)
        panel.addLayout(submode_row)

        self.pair_label = QtWidgets.QLabel("ペア数: 0")
        panel.addWidget(self.pair_label)

        solve_btn = QtWidgets.QPushButton("solvePnP で粗合わせ (要4組)")
        solve_btn.clicked.connect(self.run_pnp)
        panel.addWidget(solve_btn)

        clear_btn = QtWidgets.QPushButton("ペアをクリア")
        clear_btn.clicked.connect(self.clear_pairs)
        panel.addWidget(clear_btn)

        panel.addStretch(1)

        save_btn = QtWidgets.QPushButton("YAML 保存")
        save_btn.clicked.connect(self.save_yaml)
        panel.addWidget(save_btn)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        panel.addWidget(self.status)

    # ---------- スライダ <-> T 同期 ----------
    def _read_offsets(self):
        return {name: cfg["slider"].value() * cfg["step"]
                for name, cfg in self.sliders.items()}

    def _read_T_from_sliders(self):
        """base_T に微小 RPY オフセットと並進オフセットを適用した T を返す。"""
        o  = self._read_offsets()
        dR = euler_to_R(o["roll"], o["pitch"], o["yaw"])
        T  = np.eye(4)
        T[:3, :3] = dR @ self.base_T[:3, :3]
        T[:3, 3]  = self.base_T[:3, 3] + np.array([o["x"], o["y"], o["z"]])
        return T

    def rebase(self):
        """現在の T を新たな基準にして全スライダを 0 へ戻す。"""
        self.base_T = self.T.copy()
        for cfg in self.sliders.values():
            cfg["slider"].blockSignals(True)
            cfg["slider"].setValue(0)
            cfg["slider"].blockSignals(False)
        self._update_slider_labels()
        self.redraw()

    def _sync_sliders_from_T(self):
        self.base_T = self.T.copy()
        for cfg in self.sliders.values():
            cfg["slider"].blockSignals(True)
            cfg["slider"].setValue(0)
            cfg["slider"].blockSignals(False)
        self._update_slider_labels()

    def _update_slider_labels(self):
        for name, cfg in self.sliders.items():
            raw  = cfg["slider"].value() * cfg["step"]
            disp = np.rad2deg(raw) if cfg["unit"] == "deg" else raw
            cfg["label"].setText(f"{disp:+.3f}{cfg['unit']}")

    # ---------- イベント ----------
    def on_slider(self):
        self.T = self._read_T_from_sliders()
        self._update_slider_labels()
        self.redraw()

    def _on_disp_change(self):
        s = self._disp_sliders
        self.pt_size   = s["size"]["slider"].value()
        self.pt_alpha  = s["alpha"]["slider"].value() / 100.0
        self.depth_min = float(s["c min"]["slider"].value())
        self.depth_max = float(s["c max"]["slider"].value())
        for cfg in s.values():
            cfg["label"].setText(cfg["fmt"](cfg["slider"].value()))
        self.redraw()

    def _on_color_mode_change(self):
        self.color_mode = "intensity" if self._radio_inten.isChecked() else "depth"
        defaults = {"intensity": (-6, 30), "depth": (0, 50)}
        lo, hi = defaults[self.color_mode]
        for key, val in [("c min", lo), ("c max", hi)]:
            sl = self._disp_sliders[key]["slider"]
            sl.blockSignals(True)
            sl.setValue(val)
            sl.blockSignals(False)
        self._on_disp_change()

    def toggle_pair_mode(self, checked):
        self.pair_mode  = checked
        self.pending_3d = None
        self.pair_btn.setText(f"ペアモード: {'ON' if checked else 'OFF'}")
        self._radio_match.setEnabled(checked)
        self._radio_mismatch.setEnabled(checked)
        self._update_pair_status()
        if not checked:
            self.redraw()

    def _update_pair_status(self):
        if not self.pair_mode:
            self.status.setText("")
            return
        if self.pair_submode == "match":
            self.status.setText("投影点をクリック → 現在の投影位置でペア登録")
        else:
            self.status.setText("投影点をクリック → 正しい画像位置をクリック")

    def _on_pair_submode_change(self):
        self.pair_submode = "match" if self._radio_match.isChecked() else "mismatch"
        self.pending_3d   = None
        self._update_pair_status()
        self.redraw()

    def clear_pairs(self):
        self.pairs      = []
        self.pending_3d = None
        self.pair_label.setText("ペア数: 0")
        self.redraw()

    # ---------- ズーム / パン ----------
    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        if self.pair_mode:
            self.on_click(event)
            return
        if event.button == 1:
            self._pan_start_disp = (event.x, event.y)
            self._pan_xlim = self.ax.get_xlim()
            self._pan_ylim = self.ax.get_ylim()
            self._panning  = True

    def _on_motion(self, event):
        if not self._panning or event.x is None:
            return
        bbox = self.ax.get_window_extent()
        if bbox.width == 0 or bbox.height == 0:
            return
        # 元の座標系でのみ計算することでドラッグ中のドリフトを防ぐ
        def disp2data(px, py):
            rx = (px - bbox.x0) / bbox.width
            ry = (py - bbox.y0) / bbox.height
            return (self._pan_xlim[0] + rx * (self._pan_xlim[1] - self._pan_xlim[0]),
                    self._pan_ylim[0] + ry * (self._pan_ylim[1] - self._pan_ylim[0]))
        sx, sy = disp2data(*self._pan_start_disp)
        cx, cy = disp2data(event.x, event.y)
        dx, dy = cx - sx, cy - sy
        self.ax.set_xlim(self._pan_xlim[0] - dx, self._pan_xlim[1] - dx)
        self.ax.set_ylim(self._pan_ylim[0] - dy, self._pan_ylim[1] - dy)
        self.canvas.draw_idle()

    def _on_release(self, event):
        self._panning = False

    def _on_scroll(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        scale = 0.85 ** event.step  # step>0=ズームイン, step<0=ズームアウト
        cx, cy = event.xdata, event.ydata
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.ax.set_xlim(cx + (xlim[0] - cx) * scale, cx + (xlim[1] - cx) * scale)
        self.ax.set_ylim(cy + (ylim[0] - cy) * scale, cy + (ylim[1] - cy) * scale)
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or not self.pair_mode:
            return
        click = np.array([event.xdata, event.ydata])

        if self.pair_submode == "match":
            # 1クリック: 最近傍の投影点を見つけ、現在の投影位置をペアとして登録
            uv, _depth, idx = project(self.points, self.T, self.K)
            if uv.shape[0] == 0:
                return
            j = int(np.argmin(np.linalg.norm(uv - click, axis=1)))
            if np.linalg.norm(uv[j] - click) > 30:
                self.status.setText("近くに投影点がありません")
                return
            self.pairs.append((self.points[idx[j]].copy(), uv[j].copy()))
            self.pair_label.setText(f"ペア数: {len(self.pairs)}")
            self.status.setText(f"一致点を登録しました (計{len(self.pairs)}組)")
            self.redraw()
        else:
            # 相違点: 2クリック
            if self.pending_3d is None:
                # 第1クリック: 最近傍の投影点の3D座標を保持
                uv, _depth, idx = project(self.points, self.T, self.K)
                if uv.shape[0] == 0:
                    return
                j = int(np.argmin(np.linalg.norm(uv - click, axis=1)))
                if np.linalg.norm(uv[j] - click) > 30:
                    self.status.setText("近くに投影点がありません")
                    return
                self.pending_3d = self.points[idx[j]].copy()
                self.status.setText("正しい画像位置をクリックしてください")
                self.redraw(highlight_uv=uv[j])
            else:
                # 第2クリック: 正しい画像位置でペア確定
                self.pairs.append((self.pending_3d.copy(), click.copy()))
                self.pending_3d = None
                self.pair_label.setText(f"ペア数: {len(self.pairs)}")
                self.status.setText(f"相違点を登録しました (計{len(self.pairs)}組)")
                self.redraw()

    def run_pnp(self):
        if len(self.pairs) < 4:
            self.status.setText("ペアが4組未満です")
            return
        obj   = np.array([p[0] for p in self.pairs], dtype=np.float64)
        img2d = np.array([p[1] for p in self.pairs], dtype=np.float64)
        rvec0 = cv2.Rodrigues(self.T[:3, :3])[0]
        tvec0 = self.T[:3, 3].reshape(3, 1)
        # new_K で投影するため distCoeffs はゼロを明示
        ok, rvec, tvec = cv2.solvePnP(
            obj, img2d, self.K, np.zeros((4, 1)),
            rvec0, tvec0,
            useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            self.status.setText("solvePnP 失敗")
            return
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(rvec)[0]
        T[:3, 3]  = tvec.ravel()
        self.T = T
        self._sync_sliders_from_T()
        self.redraw()
        self.status.setText(f"PnP 完了。平均再投影誤差 {self.pair_reproj_error():.2f}px")

    def save_yaml(self):
        # T_col_lr (lidar→optical) を T_lr_cl (camera_link in lidar frame) に変換して保存
        M_lr_cl = T_col_lr_to_lr_cl(self.T)
        x, y, z, roll, pitch, yaw = decompose_T(M_lr_cl)
        child_key = f"{self.cam_frame}/camera_link"
        data = {
            self.lidar_frame: {
                child_key: {
                    "x": float(x), "y": float(y), "z": float(z),
                    "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
                }
            }
        }
        with open(self.out_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        # ROS2 Iron/Humble+ のキーワード引数形式
        cmd = (
            f"ros2 run tf2_ros static_transform_publisher "
            f"--x {x:.6f} --y {y:.6f} --z {z:.6f} "
            f"--roll {roll:.6f} --pitch {pitch:.6f} --yaw {yaw:.6f} "
            f"--frame-id {self.lidar_frame} --child-frame-id {child_key}"
        )
        self.status.setText(f"保存: {self.out_path}\n{cmd}")
        print(cmd)

    # ---------- 誤差計算 ----------
    def pair_reproj_error(self):
        if not self.pairs:
            return None
        obj = np.array([p[0] for p in self.pairs], dtype=np.float64)
        img = np.array([p[1] for p in self.pairs], dtype=np.float64)
        Xc  = (self.T[:3, :3] @ obj.T + self.T[:3, 3:4]).T
        uvw = (self.K @ Xc.T).T
        uv  = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
        return float(np.linalg.norm(uv - img, axis=1).mean())

    # ---------- 描画 ----------
    def redraw(self, highlight_uv=None):
        if self._view_initialized:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
        else:
            xlim = (0, self.w)
            ylim = (self.h, 0)
            self._view_initialized = True

        self.ax.clear()
        self.ax.set_axis_off()
        self.ax.imshow(self.image)

        uv, depth, idx = project(self.points, self.T, self.K)
        if uv.shape[0]:
            inb = ((uv[:, 0] >= 0) & (uv[:, 0] < self.w) &
                   (uv[:, 1] >= 0) & (uv[:, 1] < self.h))
            if self.color_mode == "intensity" and self.intensity is not None:
                color_vals = self.intensity[idx[inb]]
            else:
                color_vals = depth[inb]
            self.ax.scatter(uv[inb, 0], uv[inb, 1],
                            c=color_vals, cmap="jet_r",
                            s=self.pt_size, alpha=self.pt_alpha,
                            vmin=self.depth_min, vmax=self.depth_max)

        for _, target in self.pairs:
            self.ax.plot(target[0], target[1], "g+", markersize=12, mew=2)

        if highlight_uv is not None:
            self.ax.plot(highlight_uv[0], highlight_uv[1],
                         "wo", markersize=10, mfc="none", mew=2)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

        err = self.pair_reproj_error()
        if err is not None and self.pending_3d is None:
            self.status.setText(f"平均再投影誤差 {err:.2f}px ({len(self.pairs)}組)")


# ----------------------------------------------------------------------------
# 入力読み込み
# ----------------------------------------------------------------------------
def _derive_lidar_frame(tf_yaml_path, cam_name):
    """multi_tf_static.yaml から cam_name の親 lidar フレーム名を返す。"""
    with open(tf_yaml_path) as f:
        data = yaml.safe_load(f)
    for parent, children in data.items():
        if isinstance(children, dict) and f"{cam_name}/camera_link" in children:
            return parent
    return "lidar"


def load_inputs(args):
    # 画像
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"画像が読み込めません: {args.image}")

    # 点群 (.pcd は extract_lidar_pcd.py の出力、.npy は Nx3 以上の配列)
    p = Path(args.points)
    if p.suffix == ".pcd":
        xyz, inten = read_pcd_xyzi(str(p))
        pts = xyz
        intensity = inten
    else:
        raw = np.load(args.points)
        if raw.ndim != 2 or raw.shape[1] < 3:
            raise ValueError("--points は Nx3 以上の .npy が必要")
        pts = raw[:, :3].astype(np.float64)
        intensity = raw[:, 3].astype(np.float64) if raw.shape[1] >= 4 else None

    # カメラ行列 new_K: --cam-dir (camera_info.json) 優先、次に --K
    if args.cam_dir:
        cam_info = _load_camera_info(args.cam_dir)
        if cam_info is None:
            raise FileNotFoundError(f"camera_info.json が見つかりません: {args.cam_dir}")
        _K, _D, _W, _H, info_model, _P = cam_info
        _model = args.distortion_model or info_model
        new_K, map1, map2, _ = build_undistort(_K, _D, _W, _H, _model, P=_P, fisheye_balance=None)
        img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        print(f"[tuner] camera_info: {_W}x{_H} model={_model} (undistort applied)", flush=True)
    elif args.K:
        if args.K.endswith(".npy"):
            new_K = np.load(args.K).astype(np.float64)
        else:
            new_K = np.array(
                yaml.safe_load(Path(args.K).read_text()), dtype=np.float64
            ).reshape(3, 3)
    else:
        raise ValueError("--cam-dir または --K を指定してください")

    # 初期外パラ T_col_lr: --tf-yaml > --init-T > 単位行列
    lidar_frame = args.lidar_frame
    cam_frame   = args.cam_frame or args.cam or "camera"

    if args.tf_yaml and args.cam:
        cam_configs = _load_tf_yaml(args.tf_yaml, base_dir=args.base_dir)
        if args.cam not in cam_configs:
            raise ValueError(f"--cam '{args.cam}' が --tf-yaml に見つかりません")
        R_col_lr, t_col_lr = extrinsic_lr_to_optical(cam_configs[args.cam])
        init_T = np.eye(4)
        init_T[:3, :3] = R_col_lr
        init_T[:3, 3]  = t_col_lr
        if lidar_frame is None:
            lidar_frame = _derive_lidar_frame(args.tf_yaml, args.cam)
        print(f"[tuner] init_T from tf_yaml: {args.cam} parent={lidar_frame}", flush=True)
    elif args.init_T:
        init_T = np.load(args.init_T)
        if init_T.shape != (4, 4):
            raise ValueError("--init-T は 4x4 行列が必要")
    else:
        init_T = np.eye(4)

    return img, pts, new_K, init_T, lidar_frame or "lidar", cam_frame, intensity


# ----------------------------------------------------------------------------
# エントリポイント
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="LiDAR-Camera 外部パラメータ 手動調整GUI")
    ap.add_argument("--image",   required=True,
                    help="カメラ画像 (image_raw / distort/ 等)。--cam-dir 使用時は自動的に歪み補正を適用")
    ap.add_argument("--points",  required=True,
                    help="LiDAR 点群 (.npy Nx3 または extract_lidar_pcd.py 出力 .pcd)")
    ap.add_argument("--cam-dir", default=None,
                    help="camera_info.json を含むディレクトリ (extract_cameras.py の出力先)")
    ap.add_argument("--K",       default=None,
                    help="new_K 3x3 (.npy または平行列YAML)。--cam-dir 省略時に参照")
    ap.add_argument("--distortion-model", default=None,
                    choices=["rational_polynomial", "equidistant"],
                    help="歪みモデル上書き (--cam-dir 使用時)")
    ap.add_argument("--tf-yaml", default=None,
                    help="multi_tf_static.yaml。--cam と組み合わせて初期外パラを読み込む")
    ap.add_argument("--cam",     default=None,
                    help="カメラ名 (例: camera0)。--tf-yaml 使用時に必要")
    ap.add_argument("--base-dir", default=None,
                    help="--tf-yaml 使用時のデータベースディレクトリ")
    ap.add_argument("--init-T",  default=None,
                    help="初期 T_col_lr 4x4 .npy。--tf-yaml より低優先")
    ap.add_argument("--lidar-frame", default=None,
                    help="YAML 出力の lidar フレーム名 (省略時は --tf-yaml から自動取得)")
    ap.add_argument("--cam-frame",   default=None,
                    help="YAML 出力のカメラ名 (省略時は --cam の値)")
    ap.add_argument("--out",     default="extrinsic_adjust.yaml",
                    help="出力 YAML パス (デフォルト: extrinsic_adjust.yaml)")
    args = ap.parse_args()

    img, pts, K, init_T, lidar_frame, cam_frame, intensity = load_inputs(args)

    app = QtWidgets.QApplication(sys.argv)
    win = TunerWindow(img, pts, K, init_T, args.out,
                      lidar_frame=lidar_frame, cam_frame=cam_frame,
                      intensity=intensity)
    win.resize(1280, 760)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
