#!/usr/bin/env python3
"""Step 5 v3, part 1: whole-trajectory camera point cloud (RTK-fix prerequisite).

Reconstructs a METRIC 3D point cloud per camera by tracking 2D features across
many frames (KLT) and triangulating each track using the KNOWN camera pose at
each observation time:

    W_T_optical(t) = W_T_drs(t; fix-RTK) @ drs_T_lidar(Step4) @ lidar_T_camera_link(x0) @ camera_link_T_optical(fixed)

This is NOT classical SfM (no egomotion estimation from images) — poses come
entirely from the RTK/INS trajectory, so triangulation is a straightforward
multi-view ray intersection. Scale is metric because the poses are metric,
which is only trustworthy once RTK is at an INTEGER (fix) solution: under a
float solution (~0.5m), a several-meter triangulation baseline inherits the
RTK's relative-position error directly into depth (this is exactly the
concern raised when evaluating this approach — see docs/architecture.md
"Step 5 v3 設計"). Do not run this for calibration-quality output on float-RTK
bags; it is still useful there as a mechanical smoke test (see --self-test in
step5_v3_register.py).

Output per camera: <out>/v3_tracks_camera<N>.npz with flat arrays
(track_id, t_ns, u, v) — one row per observation, grouped by track_id. The
registration step (step5_v3_register.py) re-triangulates from this raw track
data every outer iteration as its extrinsic estimate improves.

Usage:
    python step5_v3_cam_cloud.py <bag> --out-dir OUT --cameras 0,2
        [--tf-override YAML] [--config config.yaml] [--max-frames N]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

import common
import step5_cam_to_lidar as s5


# --------------------------------------------------------------------------- #
# Camera pose in map frame (shared with step5_v3_register.py)
# --------------------------------------------------------------------------- #
def camera_pose_in_map(params, drs_T_lidar, traj, t_ns):
    """W_T_optical (4x4) at a single time, given lidar->camera_link params."""
    R_ol, t_ol = s5.lidar_to_optical(*params)          # optical_T_lidar
    T_lo = np.linalg.inv(s5.make_T(t_ol, R_ol))         # lidar_T_optical
    pos, quat = traj.interpolate(np.array([t_ns], np.float64))
    W_T_drs = common.make_transform(pos[0], quat[0])
    W_T_lidar = W_T_drs @ drs_T_lidar
    return W_T_lidar @ T_lo


def camera_poses_in_map_batch(params, drs_T_lidar, traj, t_ns_arr):
    """Vectorized W_T_optical for many times. Returns (R (N,3,3), t (N,3))."""
    R_ol, t_ol = s5.lidar_to_optical(*params)
    T_lo = np.linalg.inv(s5.make_T(t_ol, R_ol))
    pos, quat = traj.interpolate(np.asarray(t_ns_arr, np.float64))
    R_wd = common.quat_to_matrix(quat)                                    # (N,3,3)
    R_wl = np.einsum("nij,jk->nik", R_wd, drs_T_lidar[:3, :3])
    t_wl = np.einsum("nij,j->ni", R_wd, drs_T_lidar[:3, 3]) + pos
    R_wo = np.einsum("nij,jk->nik", R_wl, T_lo[:3, :3])
    t_wo = np.einsum("nij,j->ni", R_wl, T_lo[:3, 3]) + t_wl
    return R_wo, t_wo


# --------------------------------------------------------------------------- #
# Feature tracking (forward-backward-checked KLT)
# --------------------------------------------------------------------------- #
def iter_frames(files, cam, cfg, max_frames):
    import cv2
    from sensor_msgs.msg import CompressedImage

    topic = f"/sensing/camera/camera{cam}/image_raw/compressed"
    stride = max(1, int(cfg["v3_frame_stride"]))
    idx = -1
    n = 0
    for m in common.read_deserialized(files, topic, CompressedImage):
        idx += 1
        if idx % stride != 0:
            continue
        if max_frames is not None and n >= max_frames:
            break
        img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        t_ns = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        yield t_ns, img
        n += 1


# Rolling-shutter line-scan time [s] by camera (hardware spec: frame_period ==
# num_lines*line_scan_s + fixed_exposure_s for all three sensor types below).
# header.stamp is calibrated to the image-CENTER row's capture time, so a
# feature at row v needs a (v - height/2) * line_scan_s correction relative to
# header.stamp. Fisheye cameras (8-11) are excluded: their header.stamp is not
# calibrated to the center-row convention.
CAM_LINE_SCAN_S = {
    0: 0.02e-3, 1: 0.02e-3, 4: 0.02e-3, 5: 0.02e-3,          # C3: 2160 lines
    2: 0.0125e-3, 3: 0.0125e-3, 6: 0.0125e-3, 7: 0.0125e-3,  # C2: 1860 lines
}


def rolling_shutter_t_ns(cam, t_ns, v, height):
    line_s = CAM_LINE_SCAN_S.get(cam)
    if line_s is None:
        return t_ns
    return t_ns + round((v - height / 2.0) * line_s * 1e9)


def klt_track(files, cam, cfg, max_frames):
    """Forward-backward-checked KLT tracking. Returns a list of tracks, each a
    list of (t_ns, u, v) with t_ns already rolling-shutter-corrected per point
    (see rolling_shutter_t_ns); re-seeds goodFeaturesToTrack when the active
    set thins, and force-closes tracks at v3_track_max_len (bounds
    triangulation baseline/cost, not an accuracy requirement)."""
    import cv2

    max_len = int(cfg["v3_track_max_len"])
    min_len = int(cfg["v3_track_min_len"])
    max_corners = int(cfg["v3_klt_max_corners"])
    reseed_below = int(cfg["v3_klt_reseed_below"])
    quality = float(cfg["v3_klt_quality"])
    min_dist = float(cfg["v3_klt_min_dist"])
    lk_params = dict(winSize=(21, 21), maxLevel=3,
                      criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    def seed(gray, existing):
        mask = np.full(gray.shape, 255, np.uint8)
        if existing is not None:
            for p in existing.reshape(-1, 2):
                cv2.circle(mask, (int(p[0]), int(p[1])), int(min_dist), 0, -1)
        return cv2.goodFeaturesToTrack(gray, max_corners, quality, min_dist, mask=mask)

    completed = []
    active_pts, active_tracks, prev_gray = None, [], None

    for t_ns, gray in iter_frames(files, cam, cfg, max_frames):
        height = gray.shape[0]
        if prev_gray is None:
            pts = seed(gray, None)
            active_pts = pts
            active_tracks = ([[(rolling_shutter_t_ns(cam, t_ns, float(p[0][1]), height),
                                float(p[0][0]), float(p[0][1]))] for p in pts]
                              if pts is not None else [])
            prev_gray = gray
            continue

        if active_pts is not None and len(active_pts):
            fwd, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, active_pts, None, **lk_params)
            back, bstatus, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, fwd, None, **lk_params)
            fb_err = np.linalg.norm((active_pts - back).reshape(-1, 2), axis=1)
            good = (status.reshape(-1) == 1) & (bstatus.reshape(-1) == 1) & (fb_err < 1.5)
        else:
            fwd, good = None, np.zeros(0, bool)

        next_pts, next_tracks = [], []
        for i, ok in enumerate(good):
            if not ok:
                if len(active_tracks[i]) >= min_len:
                    completed.append(active_tracks[i])
                continue
            u, v = float(fwd[i, 0, 0]), float(fwd[i, 0, 1])
            tr = active_tracks[i] + [(rolling_shutter_t_ns(cam, t_ns, v, height), u, v)]
            if len(tr) >= max_len:
                completed.append(tr)
                continue
            next_pts.append([[u, v]])
            next_tracks.append(tr)

        if len(next_pts) < reseed_below:
            existing = np.array(next_pts, np.float32) if next_pts else None
            seeded = seed(gray, existing)
            if seeded is not None:
                for p in seeded:
                    u, v = float(p[0][0]), float(p[0][1])
                    next_pts.append([[u, v]])
                    next_tracks.append([(rolling_shutter_t_ns(cam, t_ns, v, height), u, v)])

        active_pts = np.array(next_pts, np.float32) if next_pts else None
        active_tracks = next_tracks
        prev_gray = gray

    for tr in active_tracks:
        if len(tr) >= min_len:
            completed.append(tr)
    return completed


# --------------------------------------------------------------------------- #
# Multi-view triangulation
# --------------------------------------------------------------------------- #
def triangulate_track(track, params, drs_T_lidar, traj, unproject_fn, cfg):
    """Linear least-squares ray intersection in map frame. Returns
    (P_world, mean_reproj_norm, parallax_deg) or None if filtered out
    (insufficient parallax, behind the camera in any view, or high
    reprojection error — see docs/architecture.md Step 5 v3 for why parallax
    matters: translation is only observable via depth/parallax diversity).

    Only every v3_triangulate_obs_stride-th observation is used for the ray
    intersection itself (widens the baseline between views cheaply — adjacent
    tracked frames are close together in time/position and add little
    independent parallax while adding tracking noise); the final reprojection
    refinement in step5_v3_register.py still uses every observation in the
    track, since that step benefits from more residual terms."""
    stride = max(1, int(cfg["v3_triangulate_obs_stride"]))
    sub = track[::stride] if len(track) // stride >= 2 else [track[0], track[-1]]
    ts = np.array([o[0] for o in sub], np.float64)
    uv = np.array([[o[1], o[2]] for o in sub], np.float64)
    ray_n = unproject_fn(uv)

    R_wo, t_wo = camera_poses_in_map_batch(params, drs_T_lidar, traj, ts)
    dirs = np.einsum("nij,nj->ni", R_wo, np.concatenate(
        [ray_n, np.ones((len(ray_n), 1))], axis=1))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    A = np.eye(3) * 0.0
    b = np.zeros(3)
    for o, d in zip(t_wo, dirs):
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    try:
        P = np.linalg.solve(A + 1e-9 * np.eye(3), b)
    except np.linalg.LinAlgError:
        return None

    cosang = np.clip(dirs @ dirs.T, -1.0, 1.0)
    parallax = float(np.degrees(np.arccos(cosang.min())))
    if parallax < float(cfg["v3_min_parallax_deg"]):
        return None

    errs = []
    for R, t in zip(R_wo, t_wo):
        p_cam = R.T @ (P - t)
        if p_cam[2] <= 0.2:
            return None
        errs.append(np.linalg.norm(p_cam[:2] / p_cam[2] - ray_n[len(errs)]))
    reproj = float(np.mean(errs))
    if reproj > float(cfg["v3_max_reproj_norm"]):
        return None
    return P, reproj, parallax


# --------------------------------------------------------------------------- #
def save_tracks(out_dir, cam, tracks):
    track_id, t_ns, u, v = [], [], [], []
    for k, tr in enumerate(tracks):
        for t, uu, vv in tr:
            track_id.append(k)
            t_ns.append(t)
            u.append(uu)
            v.append(vv)
    path = os.path.join(out_dir, f"v3_tracks_camera{cam}.npz")
    np.savez(path, track_id=np.array(track_id, np.int64), t_ns=np.array(t_ns, np.float64),
              u=np.array(u, np.float32), v=np.array(v, np.float32))
    return path


def load_tracks(out_dir, cam):
    """Return list of tracks (each a list of (t_ns,u,v)), grouped from the flat npz."""
    d = np.load(os.path.join(out_dir, f"v3_tracks_camera{cam}.npz"))
    track_id, t_ns, u, v = d["track_id"], d["t_ns"], d["u"], d["v"]
    order = np.argsort(track_id, kind="stable")
    track_id, t_ns, u, v = track_id[order], t_ns[order], u[order], v[order]
    tracks = []
    start = 0
    for i in range(1, len(track_id) + 1):
        if i == len(track_id) or track_id[i] != track_id[start]:
            tracks.append(list(zip(t_ns[start:i].tolist(), u[start:i].tolist(), v[start:i].tolist())))
            start = i
    return tracks


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 5 v3 part 1: camera feature tracking")
    ap.add_argument("bag")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cameras", default=",".join(str(c) for c in range(12)))
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = common.load_config(args.config)
    files = common.resolve_bag_files(args.bag)
    cams = [int(c) for c in args.cameras.split(",") if c.strip() != ""]

    for cam in cams:
        tracks = klt_track(files, cam, cfg, args.max_frames)
        cap = int(cfg["v3_max_tracks_per_cam"])
        if len(tracks) > cap:
            rng = np.random.default_rng(0)
            tracks = [tracks[i] for i in rng.choice(len(tracks), cap, replace=False)]
        lens = [len(t) for t in tracks]
        path = save_tracks(args.out_dir, cam, tracks)
        avg_len = float(np.mean(lens)) if lens else 0.0
        print(f"camera{cam}: {len(tracks)} tracks, avg len {avg_len:.1f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
