#!/usr/bin/env python3
"""Step 5: Camera-to-LiDAR extrinsic refinement (lidar_* -> cameraN/camera_link).

Cost: Canny edges of the raw camera image -> Distance Transform (DT). The LiDAR
environment map (in map-local frame) is projected into the camera at each frame's
timestamp via the RTK trajectory + the (Step 4) drs->lidar + the optimized
lidar->camera_link extrinsic + the fixed camera_link->optical. The 6-DOF
lidar->camera_link is optimized (L-BFGS-B) to minimize the DT sampled at the
projected points, i.e. to snap projected structure onto image edges.

Projection is done on the RAW (distorted) image (cv2.projectPoints for pinhole,
cv2.fisheye.projectPoints for equidistant fisheye), so Canny/DT run on the raw
image too — consistent, and avoids fisheye undistortion warping.

Normal (plumb_bob/rational) cameras 0-7 and fisheye (equidistant) cameras 8-11
share this code; only the projection model differs. Camera->lidar association
follows the vehicle layout (see CAM_LIDAR).

Usage:
    python step5_cam_to_lidar.py <bag> --out-dir OUT [--cameras 0,1,...]
        [--tf-override YAML] [--config config.yaml] [--max-frames N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import common

# Camera -> associated LiDAR (vehicle layout).
CAM_LIDAR = {0: "front", 1: "front", 8: "front",
             2: "right", 3: "right", 9: "right",
             4: "rear", 5: "rear", 10: "rear",
             6: "left", 7: "left", 11: "left"}

# camera_link -> camera_optical_link fixed rotation q_xyzw=(0.5,-0.5,0.5,-0.5).
_T2_Q = np.array([0.5, -0.5, 0.5, -0.5])


# --------------------------------------------------------------------------- #
# Small transform helpers (ZYX extrinsic RPY, quaternion [x,y,z,w])
# --------------------------------------------------------------------------- #
def euler_to_R(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler(R):
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 0.99999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def make_T(t, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def lidar_to_optical(x, y, z, roll, pitch, yaw):
    """6-DOF lidar->camera_link -> (R, t) mapping lidar points to optical frame."""
    M_lr_cl = make_T([x, y, z], euler_to_R(roll, pitch, yaw))   # cl -> lr
    M_cl_col = make_T([0, 0, 0], common.quat_to_matrix(_T2_Q))  # col -> cl
    M_col_lr = np.linalg.inv(M_lr_cl @ M_cl_col)                # lr -> col
    return M_col_lr[:3, :3], M_col_lr[:3, 3]


def make_project_fn(K, D, model):
    """Return fn(pts_optical Nx3) -> uv (raw-image pixels)."""
    import cv2

    if model == "equidistant":
        D4 = np.asarray(D, np.float64)[:4].reshape(4, 1)

        def project_fn(pts):
            obj = pts.reshape(1, -1, 3).astype(np.float64)
            uv, _ = cv2.fisheye.projectPoints(obj, np.zeros(3), np.zeros(3), K, D4)
            return uv.reshape(-1, 2)
    else:
        Dd = np.asarray(D, np.float64)

        def project_fn(pts):
            uv, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, Dd)
            return uv.reshape(-1, 2)
    return project_fn


def make_unproject_fn(K, D, model):
    """Return fn(uv Nx2 raw pixels) -> normalized ray coords (x,y); ray = (x,y,1)."""
    import cv2

    if model == "equidistant":
        D4 = np.asarray(D, np.float64)[:4].reshape(4, 1)

        def unproject_fn(uv):
            pts = uv.reshape(-1, 1, 2).astype(np.float64)
            und = cv2.fisheye.undistortPoints(pts, K, D4)
            return und.reshape(-1, 2)
    else:
        Dd = np.asarray(D, np.float64)

        def unproject_fn(uv):
            pts = uv.reshape(-1, 1, 2).astype(np.float64)
            und = cv2.undistortPoints(pts, K, Dd)
            return und.reshape(-1, 2)
    return unproject_fn


# --------------------------------------------------------------------------- #
# Camera data
# --------------------------------------------------------------------------- #
def load_camera_info(files, cam):
    from sensor_msgs.msg import CameraInfo

    topic = f"/sensing/camera/camera{cam}/camera_info"
    for m in common.read_deserialized(files, topic, CameraInfo):
        return (np.array(m.k, np.float64).reshape(3, 3), np.array(m.d, np.float64),
                int(m.width), int(m.height), m.distortion_model)
    raise RuntimeError(f"no camera_info for camera{cam}")


def distance_transform(gray, canny_lo, canny_hi, dt_max):
    import cv2

    edges = cv2.Canny(gray, canny_lo, canny_hi)
    dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    return np.minimum(dt, dt_max).astype(np.float32)


def load_camera_frames(files, cam, cfg, max_frames, resize_to=None):
    """Yield (t_ns, DistanceTransform float32 (H,W)) for strided camera frames.

    resize_to, if given, is a (w,h) the raw image is downscaled to before
    Canny/DT -- the DT for every frame is held in RAM simultaneously by the
    caller (optimize_camera), so at native resolution this is the dominant
    memory cost (see cfg["image_scale"])."""
    import cv2
    from sensor_msgs.msg import CompressedImage

    topic = f"/sensing/camera/camera{cam}/image_raw/compressed"
    stride = int(cfg["cam_frame_stride"])
    lo, hi = int(cfg["canny_low"]), int(cfg["canny_high"])
    dt_max = float(cfg["dist_transform_max"])
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
        if resize_to is not None:
            img = cv2.resize(img, resize_to, interpolation=cv2.INTER_AREA)
        t_ns = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        yield t_ns, distance_transform(img, lo, hi, dt_max)
        n += 1


# --------------------------------------------------------------------------- #
# LiDAR edge extraction
# --------------------------------------------------------------------------- #
def extract_edge_points(pts, inten, cfg):
    """Keep LiDAR points on structure edges: high local surface variation
    (corners/poles = normal discontinuities) OR high intensity gradient (lane
    markings / curbs / object boundaries). The dense cloud aliases against the
    cluttered image's ubiquitous edges; edges give a sharp, aliasing-free cost."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(0)
    cap = int(cfg["cam_edge_max_points"])
    if len(pts) > cap:
        sel = rng.choice(len(pts), cap, replace=False)
        pts, inten = pts[sel], inten[sel]
    k = int(cfg["cam_edge_knn"])
    tree = cKDTree(pts)
    _d, idx = tree.query(pts, k=k, workers=-1)          # (N,k)
    nb = pts[idx]                                        # (N,k,3)
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k           # (N,3,3)
    ev = np.linalg.eigvalsh(cov)                        # ascending
    surf_var = ev[:, 0] / (ev.sum(axis=1) + 1e-12)      # planar~0, edge/corner high
    istd = inten[idx].std(axis=1)

    def _norm(a):
        lo, hi = np.percentile(a, [5, 95])
        return np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)

    score = _norm(surf_var) + _norm(istd)
    thr = np.quantile(score, 1.0 - float(cfg["cam_edge_ratio"]))
    return pts[score >= thr]


# --------------------------------------------------------------------------- #
# Geometry / cost
# --------------------------------------------------------------------------- #
def bilinear_sample(dt, u, v):
    h, w = dt.shape
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, w - 1)
    v1 = np.clip(v0 + 1, 0, h - 1)
    u0 = np.clip(u0, 0, w - 1)
    v0 = np.clip(v0, 0, h - 1)
    fu = u - u0
    fv = v - v0
    return (dt[v0, u0] * (1 - fu) * (1 - fv) + dt[v0, u1] * fu * (1 - fv)
            + dt[v1, u0] * (1 - fu) * fv + dt[v1, u1] * fu * fv)


def map_points_in_lidar(p_map, drs_T_lidar, traj, t_ns):
    """Transform map-local points into the lidar frame at time t_ns."""
    pos, quat = traj.interpolate(np.array([t_ns], np.float64))
    map_T_drs = make_T(pos[0], common.quat_to_matrix(quat[0]))
    lidar_T_map = np.linalg.inv(map_T_drs @ drs_T_lidar)
    return p_map @ lidar_T_map[:3, :3].T + lidar_T_map[:3, 3]


def crop_frustum(p_lidar, params0, project_fn, w, h, margin, max_pts, rng):
    """Keep points that project inside the image (with margin) at the init extrinsic."""
    R, t = lidar_to_optical(*params0)
    pc = p_lidar @ R.T + t
    front = pc[:, 2] > 0.2
    pc = pc[front]
    pl = p_lidar[front]
    if len(pc) == 0:
        return pl
    uv = project_fn(pc)
    inb = ((uv[:, 0] > -margin) & (uv[:, 0] < w + margin)
           & (uv[:, 1] > -margin) & (uv[:, 1] < h + margin))
    pl = pl[inb]
    if len(pl) > max_pts:
        pl = pl[rng.choice(len(pl), max_pts, replace=False)]
    return pl


def cost_fn(params, frames, project_fn, w, h, dt_max):
    """Mean DT over a FIXED point set per frame; points that fall behind the
    camera or outside the image are penalized at dt_max (so the optimizer cannot
    lower the cost by simply projecting points out of frame)."""
    R, t = lidar_to_optical(*params)
    total = 0.0
    cnt = 0
    for pl, dt in frames:
        n = len(pl)
        pc = pl @ R.T + t
        front = pc[:, 2] > 0.2
        vals = np.full(n, dt_max, dtype=np.float64)
        if np.any(front):
            uv = project_fn(pc[front])
            u, v = uv[:, 0], uv[:, 1]
            inb = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
            fv = np.full(int(front.sum()), dt_max, dtype=np.float64)
            if np.any(inb):
                fv[inb] = bilinear_sample(dt, u[inb], v[inb])
            vals[front] = fv
        total += float(vals.sum())
        cnt += n
    return total / max(cnt, 1)


def residual_px(params, frames, project_fn, w, h):
    """Median DT (px) at projected points — the reprojection residual."""
    R, t = lidar_to_optical(*params)
    vals = []
    for pl, dt in frames:
        pc = pl @ R.T + t
        front = pc[:, 2] > 0.2
        pc = pc[front]
        if len(pc) == 0:
            continue
        uv = project_fn(pc)
        u, v = uv[:, 0], uv[:, 1]
        inb = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if np.any(inb):
            vals.append(bilinear_sample(dt, u[inb], v[inb]))
    return float(np.median(np.concatenate(vals))) if vals else float("inf")


# --------------------------------------------------------------------------- #
def load_drs_T_lidar(out_dir, lidar, files, override):
    """drs->lidar from override, else Step 4 result, else bag tf_static."""
    key = f"lidar_{lidar}"
    if key in override:
        return override[key]
    res = os.path.join(out_dir, "lidar_to_lidar_result.json")
    if lidar != "front" and os.path.isfile(res):
        with open(res) as f:
            r = json.load(f)
        if lidar in r:
            return np.asarray(r[lidar]["drs_T_lidar"], np.float64)
    tf = common.load_tf_static(files)
    return tf[("drs_base_link", key)]


def init_cam_extrinsic(files, cam, lidar, override):
    """Initial lidar->camera_link (x,y,z,roll,pitch,yaw) from override or tf_static."""
    key = f"camera{cam}/camera_link"
    if key in override:
        T = override[key]
    else:
        tf = common.load_tf_static(files)
        T = tf[(f"lidar_{lidar}", key)]
    r, p, yw = R_to_euler(T[:3, :3])
    return [T[0, 3], T[1, 3], T[2, 3], r, p, yw]


def optimize_camera(files, cam, out_dir, traj, p_map, cfg, override, max_frames):
    from scipy.optimize import minimize

    lidar = CAM_LIDAR[cam]
    K, D, w, h, model = load_camera_info(files, cam)
    scale = float(cfg["image_scale"])
    resize_to = None
    if scale != 1.0:
        w2, h2 = max(1, round(w * scale)), max(1, round(h * scale))
        K = K.copy()
        K[0, 0] *= w2 / w
        K[0, 2] *= w2 / w
        K[1, 1] *= h2 / h
        K[1, 2] *= h2 / h
        resize_to, w, h = (w2, h2), w2, h2
    project_fn = make_project_fn(K, D, model)
    drs_T_lidar = load_drs_T_lidar(out_dir, lidar, files, override)
    x0 = np.array(init_cam_extrinsic(files, cam, lidar, override), np.float64)

    rng = np.random.default_rng(0)
    max_pts = int(cfg["cam_proj_max_points"])
    frames = []
    for t_ns, dt in load_camera_frames(files, cam, cfg, max_frames, resize_to):
        pl = map_points_in_lidar(p_map, drs_T_lidar, traj, t_ns)
        pl = crop_frustum(pl, x0, project_fn, w, h, 50.0, max_pts, rng)
        if len(pl):
            frames.append((pl, dt))
    if not frames:
        return None
    import cv2

    r0 = residual_px(x0, frames, project_fn, w, h)
    dt_max = float(cfg["dist_transform_max"])
    tb = float(cfg["cam_trans_bound_m"])
    rb = float(cfg["cam_rot_bound_rad"])
    bounds = [(x0[i] - tb, x0[i] + tb) for i in range(3)] + \
             [(x0[i] - rb, x0[i] + rb) for i in range(3, 6)]
    opts = {"maxiter": int(cfg["lbfgsb_max_iter"]),
            "ftol": float(cfg["lbfgsb_ftol"]), "eps": float(cfg["cam_lbfgsb_eps"])}

    # Coarse-to-fine: blur the DT heavily first (wide smooth basin), then sharpen.
    sigmas = list(cfg["cam_dt_blur_sigmas"]) or [0.0]
    xf = x0.copy()
    for sigma in sigmas:
        if sigma > 0:
            fr = [(pl, cv2.GaussianBlur(dt, (0, 0), sigma)) for pl, dt in frames]
        else:
            fr = frames

        def obj(p, _fr=fr):
            return cost_fn(p, _fr, project_fn, w, h, dt_max)

        xf = minimize(obj, xf, method="L-BFGS-B", bounds=bounds, options=opts).x
    rf = residual_px(xf, frames, project_fn, w, h)
    return {"cam": cam, "lidar": lidar, "model": model, "n_frames": len(frames),
            "params": xf.tolist(), "resid_px_init": r0, "resid_px_final": rf,
            "accept": rf <= float(cfg["cam_reproj_accept_px"]),
            "delta_trans_cm": float(np.linalg.norm(xf[:3] - x0[:3]) * 100),
            "delta_rot_deg": float(np.degrees(np.linalg.norm(xf[3:] - x0[3:])))}


# Per-worker shared state (loaded once per process via the Pool initializer).
_G: dict = {}


def _init_worker(files, out_dir, rtk_path, cfg, override, max_frames, proj_path):
    _G.update(files=files, out_dir=out_dir, cfg=cfg, override=override, max_frames=max_frames)
    poses = np.load(rtk_path)
    _G["traj"] = common.TrajectoryInterpolator(poses[:, 0], poses[:, 1:4], poses[:, 4:8])
    _G["p_map"] = np.load(proj_path).astype(np.float64)


def _worker(cam):
    try:
        r = optimize_camera(_G["files"], cam, _G["out_dir"], _G["traj"], _G["p_map"],
                            _G["cfg"], _G["override"], _G["max_frames"])
        return r if r is not None else {"cam": cam, "n_frames": 0}
    except Exception as e:  # noqa: BLE001
        return {"cam": cam, "error": repr(e)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 5: Camera-to-LiDAR extrinsic")
    ap.add_argument("bag")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rtk-poses", default=None)
    ap.add_argument("--cameras", default=",".join(str(c) for c in range(12)))
    ap.add_argument("--tf-override", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None, help="override cam_parallel_workers")
    args = ap.parse_args(argv)

    cfg = common.load_config(args.config)
    files = common.resolve_bag_files(args.bag)
    override = common.load_tf_override(args.tf_override)
    rtk = args.rtk_poses or os.path.join(args.out_dir, "rtk_poses.npy")
    cams = [int(c) for c in args.cameras.split(",") if c.strip() != ""]
    workers = args.workers if args.workers is not None else int(cfg["cam_parallel_workers"])
    workers = max(1, min(workers, len(cams)))

    # Projection cloud: LiDAR edge points (default) or the dense map.
    p_full = np.load(os.path.join(args.out_dir, "front_lidar_map.npy"))
    xyz = p_full[:, :3].astype(np.float64)
    if cfg["cam_use_edges"]:
        inten = (p_full[:, 3] if p_full.shape[1] > 3 else np.zeros(len(p_full))).astype(np.float64)
        proj = extract_edge_points(xyz, inten, cfg)
        print(f"Step 5: edge points {len(proj)} of {len(xyz)} map points")
    else:
        proj = xyz
    proj_path = os.path.join(args.out_dir, "_cam_proj_points.npy")
    np.save(proj_path, proj.astype(np.float32))

    init_args = (files, args.out_dir, rtk, cfg, override, args.max_frames, proj_path)
    print(f"Step 5: {len(cams)} cameras, {workers} workers")
    if workers == 1:
        _init_worker(*init_args)
        outs = [_worker(c) for c in cams]
    else:
        import multiprocessing as mp
        with mp.Pool(workers, initializer=_init_worker, initargs=init_args) as pool:
            outs = pool.map(_worker, cams)

    results = {}
    yaml_tf = {}
    for r in outs:
        cam = r["cam"]
        if r.get("error"):
            print(f"  [error] camera{cam}: {r['error']}")
            continue
        if r.get("n_frames", 0) == 0:
            print(f"  [skip] camera{cam}: no usable frames")
            continue
        results[f"camera{cam}"] = r
        p = r["params"]
        yaml_tf.setdefault(f"lidar_{r['lidar']}", {})[f"camera{cam}/camera_link"] = {
            "x": p[0], "y": p[1], "z": p[2], "roll": p[3], "pitch": p[4], "yaw": p[5]}
        flag = "OK" if r["accept"] else "HIGH-RESIDUAL"
        print(f"  camera{cam} [{r['model']}] {r['n_frames']}f resid {r['resid_px_init']:.2f}"
              f"->{r['resid_px_final']:.2f}px ({flag}); moved {r['delta_trans_cm']:.1f}cm/"
              f"{r['delta_rot_deg']:.2f}deg")

    with open(os.path.join(args.out_dir, "cam_to_lidar_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    try:
        import yaml
        with open(os.path.join(args.out_dir, "cam_to_lidar_tf.yaml"), "w") as f:
            yaml.safe_dump(yaml_tf, f, sort_keys=False)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not write yaml: {e}")
    print(f"  wrote cam_to_lidar_result.json / cam_to_lidar_tf.yaml to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
