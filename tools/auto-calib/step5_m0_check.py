#!/usr/bin/env python3
"""Step 5 M0: identifiability sweep check for the v2 scan-to-image cost.

Before any optimizer is trusted (see docs/architecture.md, "Step 5 v2 設計"),
this answers a narrower question per camera: does the proposed cost function
have a single, correctly-located maximum near tf_static in each of the 6
lidar->camera_link DOF? If it doesn't, no optimizer built on top of it can
recover a good extrinsic no matter how it searches — this is exactly how v1's
DT-minimization cost failed silently (residual dropped while the true optimum
drifted 50cm+ away, an aliased/displaced global minimum).

Chain (decoupled from Step 4 / RTK, unlike v1's map-projection chain): for a
handful of camera frames, take the single LiDAR scan nearest in time (Step 2
output, already de-skewed to its own t0), motion-compensate it forward to the
image timestamp via the RTK-relative pose over that short (<=m0_max_time_gap_s)
interval, and project directly — no map, no Step 4 drs_T_lidar dependency
beyond the rigid mount transform itself.

Two edge classes are extracted per scan and swept independently so the
identifiability test also picks the better cue for this LiDAR (see the
Levinson & Thrun 2013 vs. Yuan et al. 2021 discussion in the architecture doc):
  - occlusion  : depth-discontinuity, foreground side (range-image based)
  - planar     : local high-curvature / non-planar points (single-scan KNN)
  - combined   : union of both, weights normalized per-class before merging
Mutual information (all frustum points, LiDAR reflectivity vs. image gray) is
computed too as an independent-failure-mode cross-check (Pandey et al. 2012).

Output: <out>/m0_sweep_result.json (per-axis peak displacement / prominence /
GO-NO-GO verdict) and <out>/m0_sweep_camera<N>.png (sweep curves).

Usage:
    python step5_m0_check.py <bag> --out-dir OUT --cameras 0,4
        [--tf-override YAML] [--config config.yaml] [--no-plot]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import common
import step5_cam_to_lidar as s5

AXES = ["x", "y", "z", "roll", "pitch", "yaw"]


# --------------------------------------------------------------------------- #
# Scan/image pairing
# --------------------------------------------------------------------------- #
def load_manifest(out_dir, lidar):
    path = os.path.join(out_dir, "undistorted", lidar, "manifest.json")
    with open(path) as f:
        return json.load(f)


def nearest_scan(manifest, t_img_ns, max_dt_ns):
    scans = manifest["scans"]
    ts = np.array([s["t0_ns"] for s in scans], dtype=np.float64)
    i = int(np.argmin(np.abs(ts - t_img_ns)))
    if abs(ts[i] - t_img_ns) > max_dt_ns:
        return None
    return scans[i]


def select_pairs(files, cam, lidar, out_dir, cfg, n_pairs):
    """Camera frames (blur-filtered) matched to their nearest LiDAR scan,
    spread evenly across the candidates found (not just the first N)."""
    import cv2
    from sensor_msgs.msg import CompressedImage

    manifest = load_manifest(out_dir, lidar)
    topic = f"/sensing/camera/camera{cam}/image_raw/compressed"
    stride = max(1, int(cfg["cam_frame_stride"]))
    max_dt_ns = int(float(cfg["m0_max_time_gap_s"]) * 1e9)
    blur_min = float(cfg["m0_blur_var_min"])

    candidates = []
    idx = -1
    for m in common.read_deserialized(files, topic, CompressedImage):
        idx += 1
        if idx % stride != 0:
            continue
        img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if float(cv2.Laplacian(img, cv2.CV_64F).var()) < blur_min:
            continue
        t_ns = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
        scan = nearest_scan(manifest, t_ns, max_dt_ns)
        if scan is None:
            continue
        candidates.append((t_ns, img, scan))
    if len(candidates) <= n_pairs:
        return candidates
    sel = np.linspace(0, len(candidates) - 1, n_pairs).astype(int)
    return [candidates[i] for i in sel]


def compensate_scan_points(p_lidar0, drs_T_lidar, traj, t0_ns, t_img_ns):
    """Rigid-transform a de-skewed scan (lidar frame @ t0) to the lidar frame
    @ t_img via the short RTK-relative motion. Only the mount transform
    (drs_T_lidar) and the RTK-relative pose over the short gap are used — no
    map, no accumulated structure."""
    pos0, quat0 = traj.interpolate(np.array([t0_ns], np.float64))
    pos1, quat1 = traj.interpolate(np.array([t_img_ns], np.float64))
    M0 = common.make_transform(pos0[0], quat0[0])
    M1 = common.make_transform(pos1[0], quat1[0])
    Ti = np.linalg.inv(drs_T_lidar) @ np.linalg.inv(M1) @ M0 @ drs_T_lidar
    return p_lidar0 @ Ti[:3, :3].T + Ti[:3, 3]


# --------------------------------------------------------------------------- #
# Range image + occlusion edges
# --------------------------------------------------------------------------- #
def spherical_coords(pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    rng = np.sqrt(x * x + y * y + z * z)
    az_raw = np.arctan2(y, x)
    # Recenter on the circular mean so a rear-facing LiDAR's FOV (which may
    # straddle the +-pi wrap boundary) doesn't get split into two extremes.
    mean_c = np.arctan2(np.mean(np.sin(az_raw)), np.mean(np.cos(az_raw)))
    az = np.angle(np.exp(1j * (az_raw - mean_c)))
    el = np.arcsin(np.clip(z / np.maximum(rng, 1e-6), -1.0, 1.0))
    return az, el, rng


def build_range_image(pts, az_res_rad, el_res_rad, min_range):
    """Bin points into a (elevation, azimuth) grid, keeping the nearest point
    (min range) per bin. Returns None if too few valid points."""
    az, el, rng = spherical_coords(pts)
    valid = rng > min_range
    idx_all = np.nonzero(valid)[0]
    az, el, rng = az[valid], el[valid], rng[valid]
    if len(az) < 4:
        return None
    i_az = np.floor((az - az.min()) / az_res_rad).astype(np.int64)
    i_el = np.floor((el - el.min()) / el_res_rad).astype(np.int64)
    n_az = int(i_az.max()) + 1
    n_el = int(i_el.max()) + 1
    flat = i_el * n_az + i_az
    order = np.argsort(rng)
    flat_sorted = flat[order]
    _, first_pos = np.unique(flat_sorted, return_index=True)
    winners = order[first_pos]
    range_img = np.full(n_el * n_az, np.inf, dtype=np.float64)
    point_img = np.full(n_el * n_az, -1, dtype=np.int64)
    range_img[flat_sorted[first_pos]] = rng[winners]
    point_img[flat_sorted[first_pos]] = idx_all[winners]
    return {"range_img": range_img.reshape(n_el, n_az),
            "point_img": point_img.reshape(n_el, n_az)}


def extract_occlusion_edges(ri, jump_abs_m, jump_rel):
    """Foreground-side points at depth discontinuities (Levinson & Thrun
    2013). Weight = clipped depth-jump magnitude."""
    range_img, point_img = ri["range_img"], ri["point_img"]
    idx_parts, w_parts = [], []

    def scan_pairs(ra, pa, rb, pb):
        valid = np.isfinite(ra) & np.isfinite(rb) & (pa >= 0) & (pb >= 0)
        if not np.any(valid):
            return
        d = rb[valid] - ra[valid]
        near = np.minimum(ra[valid], rb[valid])
        jump = np.abs(d) > np.maximum(jump_abs_m, jump_rel * near)
        if not np.any(jump):
            return
        fg_is_a = d[jump] > 0
        near_idx = np.where(fg_is_a, pa[valid][jump], pb[valid][jump])
        idx_parts.append(near_idx)
        w_parts.append(np.minimum(np.abs(d[jump]), 5.0))

    scan_pairs(range_img[:, :-1], point_img[:, :-1], range_img[:, 1:], point_img[:, 1:])
    scan_pairs(range_img[:-1, :], point_img[:-1, :], range_img[1:, :], point_img[1:, :])
    if not idx_parts:
        return np.zeros(0, np.int64), np.zeros(0, np.float64)
    idx = np.concatenate(idx_parts)
    w = np.concatenate(w_parts)
    order = np.argsort(-w)
    idx_sorted, w_sorted = idx[order], w[order]
    _, first = np.unique(idx_sorted, return_index=True)
    return idx_sorted[first], w_sorted[first]


def extract_planar_edges(pts, cfg, exclude_idx, cap):
    """Local high-curvature points on the single scan (Yuan et al. 2021's
    counter-finding for dense solid-state LiDAR: depth-continuous / planar-
    intersection edges, not occlusion boundaries). Weight = surface variation."""
    from scipy.spatial import cKDTree

    n = len(pts)
    rng = np.random.default_rng(0)
    sel = rng.choice(n, cap, replace=False) if n > cap else np.arange(n)
    if len(exclude_idx):
        mask = np.ones(n, dtype=bool)
        mask[exclude_idx] = False
        sel = sel[mask[sel]]
    k = int(cfg["m0_planar_knn"])
    if len(sel) < k + 1:
        return np.zeros(0, np.int64), np.zeros(0, np.float64)
    sub = pts[sel]
    tree = cKDTree(sub)
    _d, idx = tree.query(sub, k=k, workers=-1)
    nb = sub[idx]
    c = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", c, c) / k
    ev = np.linalg.eigvalsh(cov)
    surf_var = ev[:, 0] / (ev.sum(axis=1) + 1e-12)
    thr = np.quantile(surf_var, 1.0 - float(cfg["m0_planar_ratio"]))
    keep = surf_var >= thr
    return sel[keep], surf_var[keep]


def _norm01(w):
    if len(w) == 0:
        return w
    mx = w.max()
    return w / mx if mx > 1e-9 else w


def crop_to_frustum(pts, aux, x0, project_fn, w, h, margin):
    """pts (N,3) in lidar frame @ t_img, aux (N,) parallel weight/intensity
    array; keep points that project inside the image (with margin) at init."""
    if len(pts) == 0:
        return pts, aux
    R, t = s5.lidar_to_optical(*x0)
    pc = pts @ R.T + t
    front = pc[:, 2] > 0.2
    pc, pts, aux = pc[front], pts[front], aux[front]
    if len(pc) == 0:
        return pts, aux
    uv = project_fn(pc)
    inb = ((uv[:, 0] > -margin) & (uv[:, 0] < w + margin)
           & (uv[:, 1] > -margin) & (uv[:, 1] < h + margin))
    return pts[inb], aux[inb]


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def gradient_magnitude_image(gray, blur_sigma):
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    if blur_sigma > 0:
        mag = cv2.GaussianBlur(mag, (0, 0), blur_sigma)
    mx = float(mag.max())
    return (mag / mx).astype(np.float32) if mx > 1e-6 else mag.astype(np.float32)


def corr_cost(params, frames, project_fn, w, h):
    """frames: list of (pts, weights, gradient-magnitude image). Weighted mean
    of the gradient magnitude at the projected pixels (MAXIMIZE). Fixed
    per-frame weight-sum denominator: out-of-frame points contribute 0 to the
    numerator but the denominator doesn't shrink, so the optimizer can't
    inflate the mean by hiding low-scoring points (the maximization mirror of
    v1's fixed-denominator fix)."""
    R, t = s5.lidar_to_optical(*params)
    num = den = 0.0
    for pl, wts, E in frames:
        if len(pl) == 0:
            continue
        pc = pl @ R.T + t
        front = pc[:, 2] > 0.2
        vals = np.zeros(len(pl), dtype=np.float64)
        if np.any(front):
            uv = project_fn(pc[front])
            u, v = uv[:, 0], uv[:, 1]
            inb = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
            fv = np.zeros(int(front.sum()), dtype=np.float64)
            if np.any(inb):
                fv[inb] = s5.bilinear_sample(E, u[inb], v[inb])
            vals[front] = fv
        num += float(np.sum(vals * wts))
        den += float(np.sum(wts))
    return num / den if den > 0 else 0.0


def mi_cost(params, frames, project_fn, w, h, bins):
    """frames: list of (pts, lidar-intensity, gray image). Mutual information
    between reflectivity and image gray at projected points, pooled across all
    frames (MAXIMIZE). Independent failure mode from the edge-correlation
    costs (Pandey et al. 2012) — used as a cross-check, not the primary cost."""
    R, t = s5.lidar_to_optical(*params)
    xs, ys = [], []
    for pl, inten, gray in frames:
        if len(pl) == 0:
            continue
        pc = pl @ R.T + t
        front = pc[:, 2] > 0.2
        if not np.any(front):
            continue
        uv = project_fn(pc[front])
        u, v = uv[:, 0], uv[:, 1]
        inb = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
        if not np.any(inb):
            continue
        ui = u[inb].astype(np.int64)
        vi = v[inb].astype(np.int64)
        xs.append(inten[front][inb])
        ys.append(gray[vi, ui].astype(np.float64))
    if not xs:
        return 0.0
    x, y = np.concatenate(xs), np.concatenate(ys)
    if len(x) < 20:
        return 0.0
    hist2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist2d / hist2d.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pxy > 0, pxy / (px * py), 1.0)
        term = np.where(pxy > 0, pxy * np.log(ratio), 0.0)
    return float(np.nansum(term))


# --------------------------------------------------------------------------- #
# Sweep + verdict
# --------------------------------------------------------------------------- #
def sweep_axis(cost_fn, x0, axis_i, span, n_steps):
    vals = np.linspace(x0[axis_i] - span, x0[axis_i] + span, n_steps)
    costs = np.empty(n_steps, dtype=np.float64)
    for i, v in enumerate(vals):
        p = x0.copy()
        p[axis_i] = v
        costs[i] = cost_fn(p)
    return vals, costs


def analyze_sweep(vals, costs, center_val, accept_disp):
    """Peak location / displacement from tf_static / prominence (peak vs. the
    2nd-best local max) / unimodal pass. Costs are MAXIMIZE-oriented."""
    peak_i = int(np.argmax(costs))
    peak_val = float(vals[peak_i])
    disp = abs(peak_val - center_val)
    lmax_idx = [i for i in range(1, len(costs) - 1)
                if costs[i] > costs[i - 1] and costs[i] > costs[i + 1]]
    lmax_vals = sorted({costs[i] for i in lmax_idx} | {costs[peak_i]}, reverse=True)
    prom = float(lmax_vals[0] / lmax_vals[1]) if len(lmax_vals) > 1 and lmax_vals[1] > 1e-9 else float("inf")
    return {"peak_value": peak_val, "displacement": float(disp),
            "n_local_maxima": len(lmax_idx), "prominence": prom,
            "unimodal_pass": bool(disp <= accept_disp and prom >= 1.05)}


# --------------------------------------------------------------------------- #
# Per-camera driver
# --------------------------------------------------------------------------- #
def run_m0_for_camera(files, cam, out_dir, traj, cfg, override, plot):
    lidar = s5.CAM_LIDAR[cam]
    K, D, w, h, model = s5.load_camera_info(files, cam)
    project_fn = s5.make_project_fn(K, D, model)
    drs_T_lidar = s5.load_drs_T_lidar(out_dir, lidar, files, override)
    x0 = np.array(s5.init_cam_extrinsic(files, cam, lidar, override), np.float64)

    pairs = select_pairs(files, cam, lidar, out_dir, cfg, int(cfg["m0_num_pairs"]))
    if not pairs:
        return {"cam": cam, "lidar": lidar, "error": "no scan-image pairs found"}

    az_res = np.radians(float(cfg["m0_az_res_deg"]))
    el_res = np.radians(float(cfg["m0_el_res_deg"]))
    jump_abs = float(cfg["m0_occ_jump_abs_m"])
    jump_rel = float(cfg["m0_occ_jump_rel"])
    grad_sigma = float(cfg["m0_grad_blur_sigma"])
    margin = float(cfg["m0_proj_margin"])
    mi_cap = int(cfg["m0_mi_max_points"])
    planar_cap = int(cfg["m0_planar_max_points"])

    occ_frames, planar_frames, comb_frames, mi_frames = [], [], [], []
    rng = np.random.default_rng(0)
    for t_ns, img, scan in pairs:
        p0 = np.load(os.path.join(out_dir, "undistorted", lidar, scan["file"]))
        xyz0 = p0[:, :3].astype(np.float64)
        inten0 = p0[:, 3].astype(np.float64)

        ri = build_range_image(xyz0, az_res, el_res, min_range=1.0)
        if ri is None:
            continue
        occ_idx, occ_w = extract_occlusion_edges(ri, jump_abs, jump_rel)
        planar_idx, planar_w = extract_planar_edges(xyz0, cfg, occ_idx, planar_cap)
        occ_w, planar_w = _norm01(occ_w), _norm01(planar_w)

        xyz_t = compensate_scan_points(xyz0, drs_T_lidar, traj, scan["t0_ns"], t_ns)

        occ_pts, occ_w2 = crop_to_frustum(xyz_t[occ_idx], occ_w, x0, project_fn, w, h, margin)
        pl_pts, pl_w2 = crop_to_frustum(xyz_t[planar_idx], planar_w, x0, project_fn, w, h, margin)

        sel = rng.choice(len(inten0), mi_cap, replace=False) if len(inten0) > mi_cap else np.arange(len(inten0))
        mi_pts, mi_int2 = crop_to_frustum(xyz_t[sel], inten0[sel], x0, project_fn, w, h, margin)

        E = gradient_magnitude_image(img, grad_sigma)
        if len(occ_pts):
            occ_frames.append((occ_pts, occ_w2, E))
        if len(pl_pts):
            planar_frames.append((pl_pts, pl_w2, E))
        if len(occ_pts) or len(pl_pts):
            cpts = np.concatenate([occ_pts, pl_pts]) if len(occ_pts) and len(pl_pts) else (
                occ_pts if len(occ_pts) else pl_pts)
            cw = np.concatenate([occ_w2, pl_w2]) if len(occ_w2) and len(pl_w2) else (
                occ_w2 if len(occ_w2) else pl_w2)
            comb_frames.append((cpts, cw, E))
        if len(mi_pts):
            mi_frames.append((mi_pts, mi_int2, img))

    if not (occ_frames or planar_frames):
        return {"cam": cam, "lidar": lidar, "n_pairs": len(pairs),
                "error": "no edge points survived frustum crop"}

    span_t = float(cfg["m0_sweep_trans_m"])
    span_r = np.radians(float(cfg["m0_sweep_rot_deg"]))
    n_steps = int(cfg["m0_sweep_steps"])
    accept_t = float(cfg["m0_unimodal_disp_trans_m"])
    accept_r = np.radians(float(cfg["m0_unimodal_disp_rot_deg"]))

    result = {"cam": cam, "lidar": lidar, "model": model, "n_pairs": len(pairs),
              "n_occ_pts_mean": float(np.mean([len(f[0]) for f in occ_frames])) if occ_frames else 0.0,
              "n_planar_pts_mean": float(np.mean([len(f[0]) for f in planar_frames])) if planar_frames else 0.0,
              "sweeps": {}}
    plot_data = {}

    def _run_class(name, frames, cost_builder):
        result["sweeps"][name] = {}
        plot_data[name] = {}
        n_pass = 0
        for ai, axis in enumerate(AXES):
            span = span_t if ai < 3 else span_r
            accept = accept_t if ai < 3 else accept_r
            vals, costs = sweep_axis(cost_builder(frames), x0, ai, span, n_steps)
            met = analyze_sweep(vals, costs, x0[ai], accept)
            result["sweeps"][name][axis] = met
            plot_data[name][axis] = (vals.tolist(), costs.tolist())
            n_pass += int(met["unimodal_pass"])
        result["sweeps"][name]["_n_pass"] = n_pass
        return n_pass

    verdicts = []
    for name, frames in (("occlusion", occ_frames), ("planar", planar_frames), ("combined", comb_frames)):
        if not frames:
            continue
        n_pass = _run_class(name, frames,
                             lambda fr: (lambda p: corr_cost(p, fr, project_fn, w, h)))
        verdicts.append((name, n_pass))
    if mi_frames:
        _run_class("mi", mi_frames,
                    lambda fr: (lambda p: mi_cost(p, fr, project_fn, w, h, int(cfg["m0_mi_bins"]))))

    best_cls, best_pass = max(verdicts, key=lambda kv: kv[1]) if verdicts else (None, 0)
    result["best_class"] = best_cls
    result["best_class_n_pass"] = best_pass
    result["go_no_go"] = "GO" if best_pass >= int(cfg["m0_go_axes_min"]) else "NO-GO"

    if plot:
        _save_plot(plot_data, out_dir, cam, x0)
    return result


def _save_plot(plot_data, out_dir, cam, x0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = list(plot_data.keys())
    fig, axs = plt.subplots(len(classes), 6, figsize=(24, 4 * len(classes)), squeeze=False)
    for ci, cls_name in enumerate(classes):
        for ai, axis in enumerate(AXES):
            ax = axs[ci][ai]
            if axis not in plot_data[cls_name]:
                ax.axis("off")
                continue
            vals, costs = plot_data[cls_name][axis]
            ax.plot(vals, costs, "-o", ms=2)
            ax.axvline(x0[ai], color="r", ls="--", lw=1)
            ax.set_title(f"{cls_name}/{axis}", fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"camera{cam} M0 sweep (red dashed = tf_static init)")
    fig.tight_layout()
    path = os.path.join(out_dir, f"m0_sweep_camera{cam}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 5 M0: identifiability sweep check")
    ap.add_argument("bag")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rtk-poses", default=None)
    ap.add_argument("--cameras", default="0,4")
    ap.add_argument("--tf-override", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    cfg = common.load_config(args.config)
    files = common.resolve_bag_files(args.bag)
    override = common.load_tf_override(args.tf_override)
    rtk = args.rtk_poses or os.path.join(args.out_dir, "rtk_poses.npy")
    poses = np.load(rtk)
    traj = common.TrajectoryInterpolator(poses[:, 0], poses[:, 1:4], poses[:, 4:8])
    cams = [int(c) for c in args.cameras.split(",") if c.strip() != ""]

    results = {}
    for cam in cams:
        print(f"Step 5 M0: camera{cam} ...")
        r = run_m0_for_camera(files, cam, args.out_dir, traj, cfg, override, plot=not args.no_plot)
        results[f"camera{cam}"] = r
        if r.get("error"):
            print(f"  [error] {r['error']}")
            continue
        print(f"  {r['n_pairs']} pairs, occ~{r['n_occ_pts_mean']:.0f}pt planar~{r['n_planar_pts_mean']:.0f}pt"
              f" -> best_class={r['best_class']} pass={r['best_class_n_pass']}/6 => {r['go_no_go']}")
        for cls_name, sweeps in r["sweeps"].items():
            line = f"    {cls_name:10s}"
            for axis in AXES:
                if axis not in sweeps:
                    continue
                m = sweeps[axis]
                flag = "OK" if m["unimodal_pass"] else "**"
                line += f" {axis}={m['displacement']:.3f}/{m['prominence']:.2f}{flag}"
            print(line)

    out_path = os.path.join(args.out_dir, "m0_sweep_result.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  wrote m0_sweep_result.json to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
