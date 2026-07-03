#!/usr/bin/env python3
"""Step 4: LiDAR-to-LiDAR extrinsic estimation (drs_base_link -> lidar_{rear,left,right}).

Combined optimization (front held fixed as the reference):

  term B (each-lidar vs front map, point-to-plane): constrains Z + globally
     observable DOF. Residual r = n · (M_i E p - y),  M_i = map_T_drs(t_i).
  term A (inter-lidar near-simultaneous overlap, point-to-plane): constrains
     X/Y. Adjacent pairs front-right / front-left / rear-right / rear-left; the
     two lidars co-observe the SAME static structure at (almost) the same
     instant, so the overlap supplies the normal diversity that map-matching
     lacked. Residual r = n_A · (E_B p_B - q_A), with A's points q_A expressed
     in the common drs@t_B frame (RTK only bridges the ~50 ms phase gap).

Why not the doc's naive "accumulate other_map + single global ICP": a lidar-
frame translation error δt places each scan at map displacement R_i·δt, which
averages to ~0 over a full-yaw figure-8, so a single rigid world transform can't
recover δt. And map-matching ALONE leaves X/Y flat (ground-dominated overlap
constrains only Z) — verified by cost slices. See docs/architecture.md Step 4.

Solve order (the pair graph front-right-rear-left-front has a loop, but front
anchors right/left strongly, so we chain): right & left each from front (term B
+ front-* term A), then rear from the now-fixed right & left (term B + rear-*
term A). Per-axis observability is reported; unobservable axes stay at tf_static.

Open3D: target voxel-downsample + normals. scipy.spatial.cKDTree: correspondence.
numpy: robust (Huber) Levenberg-Marquardt with step rejection.

Usage:
    python step4_lidar_to_lidar.py --out-dir OUT [--rtk-poses OUT/rtk_poses.npy]
        [--config config.yaml] [--max-scans N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import namedtuple

import numpy as np

import common
import step3_build_front_map as step3
import viz

OTHER_LIDARS = ("rear", "left", "right")
# Adjacent overlap pairs (reference A, moving B). front is fixed; rear chains.
STAGE1 = {"right": ("front",), "left": ("front",)}
STAGE2_REFS = ("right", "left")  # rear pairs with right and left

# One point-to-plane constraint block: source points p (M,3) mapped by A (4x4)
# then by the unknown E, matched to a fixed target (pts/normals/kdtree).
Block = namedtuple("Block", "p A tpts tnorm tree corr huber weight")


# --------------------------------------------------------------------------- #
# SE(3) helpers (numpy only)
# --------------------------------------------------------------------------- #
def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def so3_exp(w):
    theta = float(np.linalg.norm(w))
    W = skew(w)
    if theta < 1e-12:
        return np.eye(3) + W
    return (np.eye(3) + np.sin(theta) / theta * W
            + (1.0 - np.cos(theta)) / (theta * theta) * (W @ W))


def se3_exp(xi):
    w = xi[:3]
    v = xi[3:]
    theta = float(np.linalg.norm(w))
    R = so3_exp(w)
    W = skew(w)
    if theta < 1e-12:
        V = np.eye(3) + 0.5 * W
    else:
        V = (np.eye(3) + (1.0 - np.cos(theta)) / (theta * theta) * W
             + (theta - np.sin(theta)) / (theta ** 3) * (W @ W))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ v
    return T


def R_to_euler_zyx(R):
    """3x3 -> (roll, pitch, yaw) rad, ZYX extrinsic (matches tools/lidar-camera)."""
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(R[2, 0]) < 0.99999:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def _pose_at(traj, t0_ns):
    p, q = traj.interpolate(np.array([t0_ns], dtype=np.float64))
    M = np.eye(4)
    M[:3, :3] = common.quat_to_matrix(q[0])
    M[:3, 3] = p[0]
    return M


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_front_map_pts(out_dir):
    npy = os.path.join(out_dir, "front_lidar_map.npy")
    if os.path.isfile(npy):
        return np.load(npy)[:, :3].astype(np.float64)
    import open3d as o3d
    p = o3d.io.read_point_cloud(os.path.join(out_dir, "front_lidar_map.pcd"))
    return np.asarray(p.points, dtype=np.float64)


def build_target(pts, voxel, normal_radius):
    """Voxel-downsample + estimate normals; return (pts, normals, cKDTree)."""
    import open3d as o3d
    from scipy.spatial import cKDTree

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd = pcd.voxel_down_sample(voxel)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    p = np.asarray(pcd.points, dtype=np.float64)
    n = np.asarray(pcd.normals, dtype=np.float64)
    return p, n, cKDTree(p)


def load_lidar_scans(out_dir, lidar, traj, cfg, max_scans=None, subsample=True):
    """Return (scans, drs_T_lidar). scans = list of (p_lidar (Mi,3), M_i, t0_ns)."""
    manifest = step3.load_scan_manifest(out_dir, lidar)
    drs_T_lidar = np.asarray(manifest["drs_T_lidar"], dtype=np.float64)
    scan_dir = os.path.join(out_dir, "undistorted", lidar)
    stride = int(cfg["l2l_scan_stride"])
    per_scan = int(cfg["l2l_points_per_scan"])
    rmax = float(cfg["l2l_max_range"])
    rng = np.random.default_rng(0)

    scans = []
    used = 0
    for i, s in enumerate(manifest["scans"]):
        if i % stride != 0:
            continue
        if max_scans is not None and used >= max_scans:
            break
        pts = np.load(os.path.join(scan_dir, s["file"]))
        if len(pts) == 0:
            continue
        p = pts[:, :3].astype(np.float64)
        rr = np.linalg.norm(p, axis=1)
        p = p[(rr > 1.0) & (rr < rmax)]
        if len(p) == 0:
            continue
        if subsample and len(p) > per_scan:
            p = p[rng.choice(len(p), per_scan, replace=False)]
        scans.append((p, _pose_at(traj, s["t0_ns"]), s["t0_ns"]))
        used += 1
    return scans, drs_T_lidar


def build_inter_geoms(out_dir, A_name, B_name, E_A, traj, cfg, max_scans=None):
    """Precompute inter-lidar overlap geometry for pair (A ref, B moving).

    For each B scan, gather nearby A scans, express A's points in the common
    drs@t_B frame (using E_A + RTK for the phase gap), estimate normals. Returns
    a list of (p_B_lidar, A_common_pts, A_common_normals, cKDTree).
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    manA = step3.load_scan_manifest(out_dir, A_name)
    EA0 = np.asarray(manA["drs_T_lidar"], dtype=np.float64)
    E_A = EA0 if E_A is None else E_A
    sA = manA["scans"]
    tA = np.array([s["t0_ns"] for s in sA], dtype=np.float64)
    manB = step3.load_scan_manifest(out_dir, B_name)
    sB = manB["scans"]
    dt_max = float(cfg["l2l_pair_dt_max"]) * 1e9
    ref_n = int(cfg["l2l_pair_ref_scans"])
    stride = int(cfg["l2l_scan_stride"])
    per_scan = int(cfg["l2l_points_per_scan"])
    rmax = float(cfg["l2l_max_range"])
    rng = np.random.default_rng(1)

    geoms = []
    used = 0
    for j, sb in enumerate(sB):
        if j % stride != 0:
            continue
        if max_scans is not None and used >= max_scans:
            break
        t0b = sb["t0_ns"]
        Mbi = np.linalg.inv(_pose_at(traj, t0b))
        order = np.argsort(np.abs(tA - t0b))[:ref_n]
        acc = []
        for k in order:
            if abs(tA[k] - t0b) > dt_max:
                continue
            pa = np.load(os.path.join(out_dir, "undistorted", A_name, sA[k]["file"]))
            if len(pa) == 0:
                continue
            pa = pa[:, :3].astype(np.float64)
            pad = pa @ E_A[:3, :3].T + E_A[:3, 3]          # A -> drs@t_Ak
            T = Mbi @ _pose_at(traj, tA[k])                 # drs@t_Ak -> drs@t_b
            acc.append(pad @ T[:3, :3].T + T[:3, 3])
        if not acc:
            continue
        A_common = np.concatenate(acc)
        if len(A_common) > 40000:
            A_common = A_common[rng.choice(len(A_common), 40000, replace=False)]
        pb = np.load(os.path.join(out_dir, "undistorted", B_name, sb["file"]))[:, :3].astype(np.float64)
        rr = np.linalg.norm(pb, axis=1)
        pb = pb[(rr > 1.0) & (rr < rmax)]
        if len(pb) == 0:
            continue
        if len(pb) > per_scan:
            pb = pb[rng.choice(len(pb), per_scan, replace=False)]
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(A_common)
        pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=20))
        geoms.append((pb, A_common, np.asarray(pc.normals), cKDTree(A_common)))
        used += 1
    return geoms


# --------------------------------------------------------------------------- #
# Robust point-to-plane normal equations over a list of blocks
# --------------------------------------------------------------------------- #
def _huber_w(r, d):
    ar = np.abs(r)
    return np.where(ar <= d, 1.0, d / np.maximum(ar, 1e-12))


def _accumulate(E, blocks):
    H = np.zeros((6, 6))
    g = np.zeros(6)
    cost = 0.0
    sse = 0.0
    cnt = 0
    for b in blocks:
        AE = b.A @ E
        R = AE[:3, :3]
        x = b.p @ R.T + AE[:3, 3]
        dist, idx = b.tree.query(x, workers=-1)
        m = dist < b.corr
        if not np.any(m):
            continue
        pm = b.p[m]
        n = b.tnorm[idx[m]]
        r = np.einsum("ij,ij->i", n, x[m] - b.tpts[idx[m]])
        w = _huber_w(r, b.huber) * b.weight
        a = n @ R
        J = np.concatenate([np.cross(pm, a), a], axis=1)
        Jw = J * w[:, None]
        H += Jw.T @ J
        g += J.T @ (w * r)
        ar = np.abs(r)
        cost += float(b.weight * np.sum(np.where(ar <= b.huber, 0.5 * r * r,
                                                 b.huber * (ar - 0.5 * b.huber))))
        sse += float(r @ r)
        cnt += int(m.sum())
    return H, g, cost, sse, cnt


def _eval_cost(E, blocks):
    cost = 0.0
    sse = 0.0
    cnt = 0
    for b in blocks:
        AE = b.A @ E
        x = b.p @ AE[:3, :3].T + AE[:3, 3]
        dist, idx = b.tree.query(x, workers=-1)
        m = dist < b.corr
        if not np.any(m):
            continue
        n = b.tnorm[idx[m]]
        r = np.einsum("ij,ij->i", n, x[m] - b.tpts[idx[m]])
        ar = np.abs(r)
        cost += float(b.weight * np.sum(np.where(ar <= b.huber, 0.5 * r * r,
                                                 b.huber * (ar - 0.5 * b.huber))))
        sse += float(r @ r)
        cnt += int(m.sum())
    return cost, (np.sqrt(sse / cnt) if cnt else float("inf")), cnt


def _collect_residuals(E, blocks):
    """Raw signed point-to-plane residuals (r = n.(x-target)) for all inlier
    correspondences across blocks, for before/after histogram diagnostics."""
    out = []
    for b in blocks:
        AE = b.A @ E
        x = b.p @ AE[:3, :3].T + AE[:3, 3]
        dist, idx = b.tree.query(x, workers=-1)
        m = dist < b.corr
        if not np.any(m):
            continue
        n = b.tnorm[idx[m]]
        out.append(np.einsum("ij,ij->i", n, x[m] - b.tpts[idx[m]]))
    return np.concatenate(out) if out else np.zeros(0)


def _project_out_degenerate(delta, H, rel):
    """Zero the update along Hessian eigenvectors below eig_max*rel.

    Keeps unobservable DOF at their prior (tf_static, since E starts there and
    every step is projected). Returns (delta, min_eig, n_removed).
    """
    evals, evecs = np.linalg.eigh(H)
    thr = evals.max() * rel if evals.max() > 0 else 0.0
    weak = evals < thr
    for k in np.nonzero(weak)[0]:
        v = evecs[:, k]
        delta = delta - (v @ delta) * v
    return delta, float(evals.min()), int(weak.sum())


def _combine(E, map_blocks, inter_blocks, w_map, w_int):
    """Per-term-normalized normal equations so term A (few overlap pts) is not
    drowned by term B (many map pts). Each group is divided by its own inlier
    count, then weighted. Returns H, g, cost, rmse, (n_map, n_int)."""
    Hm, gm, cm, sm, nm = _accumulate(E, map_blocks)
    if inter_blocks:
        Hi, gi, ci, si, ni = _accumulate(E, inter_blocks)
    else:
        Hi, gi, ci, si, ni = np.zeros((6, 6)), np.zeros(6), 0.0, 0.0, 0
    dm, di = max(nm, 1), max(ni, 1)
    H = w_map * Hm / dm + w_int * Hi / di
    g = w_map * gm / dm + w_int * gi / di
    cost = w_map * cm / dm + w_int * ci / di
    rmse = np.sqrt((sm + si) / max(nm + ni, 1))
    return H, g, cost, rmse, (nm, ni)


def _combine_cost(E, map_blocks, inter_blocks, w_map, w_int):
    cm, _, nm = _eval_cost(E, map_blocks)
    if inter_blocks:
        ci, _, ni = _eval_cost(E, inter_blocks)
    else:
        ci, ni = 0.0, 0
    dm, di = max(nm, 1), max(ni, 1)
    cost = w_map * cm / dm + w_int * ci / di
    return cost, nm + ni


def optimize(E0, map_targets, other_scans, inter_geoms, cfg, verbose=True, tag=""):
    """Coarse-to-fine robust LM over term B (map) + term A (inter), normalized."""
    E = E0.copy()
    voxels = list(cfg["icp_multiscale_voxels"])
    corr_scale = float(cfg["icp_corr_dist_scale"])
    corr_cap = float(cfg["icp_max_correspondence_dist"])
    huber_scale = float(cfg["icp_huber_scale"])
    deg_rel = float(cfg["icp_degeneracy_rel"])
    max_iter = int(cfg["icp_max_iterations"])
    step_tol = float(cfg["icp_step_tol"])
    w_map = float(cfg["l2l_map_weight"])
    w_int = float(cfg["l2l_interlidar_weight"])

    rmse = float("inf")
    n_map = n_int = 0
    min_eig = 0.0
    n_deg = 0
    step = float("inf")
    Hfinal = np.zeros((6, 6))
    history = []  # (voxel, it, rmse) per accepted-or-not iteration, for convergence plots
    for voxel in voxels:
        sp, sn, tree = map_targets[voxel]
        corr = min(voxel * corr_scale, corr_cap)
        huber = voxel * huber_scale
        map_blocks = [Block(p, M, sp, sn, tree, corr, huber, 1.0) for p, M, _t in other_scans]
        inter_blocks = [Block(pb, np.eye(4), rp, rn, tr, corr, huber, 1.0)
                        for (pb, rp, rn, tr) in inter_geoms]

        cost, _ = _combine_cost(E, map_blocks, inter_blocks, w_map, w_int)
        lam = 1e-3
        for it in range(max_iter):
            H, g, cost_h, rmse, (n_map, n_int) = _combine(
                E, map_blocks, inter_blocks, w_map, w_int)
            if n_map + n_int == 0:
                break
            Hfinal = H
            diagH = np.diag(np.diag(H))
            accepted = False
            for _try in range(8):
                try:
                    delta = np.linalg.solve(H + lam * diagH + 1e-12 * np.eye(6), -g)
                except np.linalg.LinAlgError:
                    delta = -np.linalg.pinv(H) @ g
                delta, min_eig, n_deg = _project_out_degenerate(delta, H, deg_rel)
                delta[:3] = np.clip(delta[:3], -0.2, 0.2)
                delta[3:] = np.clip(delta[3:], -0.5, 0.5)
                E_try = E @ se3_exp(delta)
                cost_try, _ = _combine_cost(E_try, map_blocks, inter_blocks, w_map, w_int)
                if cost_try <= cost:
                    E, cost = E_try, cost_try
                    lam = max(lam * 0.5, 1e-7)
                    accepted = True
                    break
                lam = min(lam * 3.0, 1e4)
            step = float(max(np.linalg.norm(delta[:3]), np.linalg.norm(delta[3:])))
            history.append((voxel, it, rmse))
            if verbose:
                print(f"    {tag} voxel {voxel:.2f} it {it:02d}: rmse {rmse*100:.2f}cm "
                      f"map {n_map} int {n_int} step {step:.2e} deg {n_deg} "
                      f"{'acc' if accepted else 'REJ'}")
            if not accepted or step < step_tol:
                break
    axis_curv = np.diag(Hfinal)[3:6].tolist()  # tx,ty,tz curvature (lidar frame)
    return {
        "E": E, "rmse": rmse, "inliers": n_map + n_int, "converged": step < step_tol,
        "min_eig": min_eig, "n_degenerate": n_deg, "axis_curv_trans": axis_curv,
        "history": history,
    }


# --------------------------------------------------------------------------- #
def _record(results, yaml_tf, name, E, E0, res, n_scans, n_inter):
    d_trans = float(np.linalg.norm(E[:3, 3] - E0[:3, 3]))
    dR = E0[:3, :3].T @ E[:3, :3]
    d_rot = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    xyz = E[:3, 3].tolist()
    rpy = list(R_to_euler_zyx(E[:3, :3]))
    results[name] = {
        "drs_T_lidar": E.tolist(), "xyz": xyz, "rpy": rpy,
        "rmse_m": res["rmse"], "inliers": res["inliers"],
        "n_scans": n_scans, "n_inter_pairs": n_inter,
        "converged": bool(res["converged"]), "min_eig": res["min_eig"],
        "n_degenerate": res["n_degenerate"],
        "axis_curv_trans_xyz": res["axis_curv_trans"],
        "init_vs_final_trans_m": d_trans, "init_vs_final_rot_deg": d_rot,
    }
    yaml_tf[f"lidar_{name}"] = {"x": xyz[0], "y": xyz[1], "z": xyz[2],
                               "roll": rpy[0], "pitch": rpy[1], "yaw": rpy[2]}
    cx, cy, cz = res["axis_curv_trans"]
    print(f"[{name}] rmse {res['rmse']*100:.2f}cm corr {d_trans*100:.1f}cm/{d_rot:.2f}deg "
          f"| trans-curv x={cx:.1e} y={cy:.1e} z={cz:.1e} deg-dirs {res['n_degenerate']}")


def aligned_world_xy(scans, E, cap_scans):
    """Map-frame XY of a (subsampled) scan list, transformed by each scan's own
    pose times the given extrinsic E. Shared by the per-lidar overlay and the
    all-lidar combined overlay so both use identical sampling/transform logic."""
    use_scans = scans
    if len(scans) > cap_scans:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(scans), cap_scans, replace=False)
        use_scans = [scans[i] for i in idx]
    world_pts = [p @ (M @ E)[:3, :3].T + (M @ E)[:3, 3] for p, M, _t in use_scans]
    return np.concatenate(world_pts)[:, :2] if world_pts else np.zeros((0, 2))


def _bbox_crop(xy_list, margin_m=5.0):
    """Bounding box (with margin) of the union of xy_list, or None if empty.

    At the front map's full extent (100s of meters) a cm-scale extrinsic error
    is sub-pixel and invisible no matter the marker size; cropping to where the
    overlaid lidar data actually is makes fine misalignment visible."""
    arrs = [xy for xy in xy_list if len(xy)]
    if not arrs:
        return None
    pts = np.concatenate(arrs)
    lo, hi = pts.min(axis=0) - margin_m, pts.max(axis=0) + margin_m
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def _crop_xy(xy, bbox):
    if bbox is None or len(xy) == 0:
        return xy
    xmin, xmax, ymin, ymax = bbox
    m = (xy[:, 0] >= xmin) & (xy[:, 0] <= xmax) & (xy[:, 1] >= ymin) & (xy[:, 1] <= ymax)
    return xy[m]


def plot_combined_overlay(out_dir, front_pts_xy, lidar_xy, cfg):
    """All-lidar-at-once overlay: front map + every other lidar's aligned
    points in one figure, one color per lidar. The per-lidar overlay plots
    only ever show one lidar against the front map at a time; this is the
    single-glance check that all extrinsics agree on the same structure
    simultaneously (e.g. a wall or curb that right/left/rear all cross).
    Cropped to the aligned points' bounding box (see _bbox_crop)."""
    ps, pa = float(cfg["viz_point_size"]), float(cfg["viz_point_alpha"])
    bbox = _bbox_crop(list(lidar_xy.values()))
    colors = {"right": "tab:blue", "left": "tab:green", "rear": "tab:red"}
    series = [(_crop_xy(front_pts_xy, bbox), {"c": "lightgray", "s": ps, "alpha": pa, "label": "front map"})]
    for name, xy in lidar_xy.items():
        if len(xy):
            series.append((xy, {"c": colors.get(name, "black"), "s": ps, "alpha": pa,
                                "label": f"{name} (aligned)"}))
    extra = (lambda ax, b=bbox: (ax.set_xlim(b[0], b[1]), ax.set_ylim(b[2], b[3]))) if bbox else None
    return viz.topdown_scatter(
        os.path.join(out_dir, "step4_combined_overlay_topdown.png"), series, cfg,
        title="Step 4: all-lidar aligned overlay onto front map", extra=extra)


def plot_lidar_diag(out_dir, name, res, E0, map_targets, scans, geoms, front_pts_xy, cfg):
    """Convergence curve, before/after residual histogram, aligned-overlay onto
    the front map, and per-axis translation curvature — visual/quantitative
    sanity checks beyond the summary line printed by _record()."""
    paths = []
    hist = res["history"]
    if hist:
        x = list(range(len(hist)))
        rmse_cm = [r * 100 for _v, _it, r in hist]
        voxels_seen = []
        vlines = []
        for i, (v, it, _r) in enumerate(hist):
            if v not in voxels_seen:
                voxels_seen.append(v)
                if i > 0:
                    vlines.append((i - 0.5, dict(color="gray", linestyle=":", linewidth=1)))
        paths.append(viz.line_plot(
            os.path.join(out_dir, f"step4_{name}_convergence.png"),
            [(x, rmse_cm, {"color": "steelblue", "linewidth": 1.0, "marker": "o", "markersize": 2})],
            cfg, title=f"Step 4 [{name}]: RMSE convergence (coarse->fine voxel scales {voxels_seen})",
            xlabel="global iteration", ylabel="rmse [cm]", vlines=vlines))

    voxel_f = list(cfg["icp_multiscale_voxels"])[-1]
    corr = min(voxel_f * float(cfg["icp_corr_dist_scale"]), float(cfg["icp_max_correspondence_dist"]))
    huber = voxel_f * float(cfg["icp_huber_scale"])
    sp, sn, tree = map_targets[voxel_f]
    map_blocks = [Block(p, M, sp, sn, tree, corr, huber, 1.0) for p, M, _t in scans]
    inter_blocks = [Block(pb, np.eye(4), rp, rn, tr, corr, huber, 1.0) for (pb, rp, rn, tr) in geoms]
    if map_blocks or inter_blocks:
        r_before = np.concatenate([_collect_residuals(E0, map_blocks), _collect_residuals(E0, inter_blocks)])
        r_after = np.concatenate([_collect_residuals(res["E"], map_blocks), _collect_residuals(res["E"], inter_blocks)])
        paths.append(viz.hist_compare_plot(
            os.path.join(out_dir, f"step4_{name}_residual_hist.png"),
            r_before * 100, r_after * 100, cfg, bins=60,
            title=f"Step 4 [{name}]: point-to-plane residuals (init tf_static vs. final)",
            xlabel="signed residual [cm]", labels=("init", "final")))

    cap_scans = int(cfg.get("viz_max_scans_overlay", 60))
    world_xy = aligned_world_xy(scans, res["E"], cap_scans)
    if len(world_xy):
        ps, pa = float(cfg["viz_point_size"]), float(cfg["viz_point_alpha"])
        bbox = _bbox_crop([world_xy])
        extra = (lambda ax, b=bbox: (ax.set_xlim(b[0], b[1]), ax.set_ylim(b[2], b[3]))) if bbox else None
        paths.append(viz.topdown_scatter(
            os.path.join(out_dir, f"step4_{name}_overlay_topdown.png"),
            [(_crop_xy(front_pts_xy, bbox), {"c": "lightgray", "s": ps, "alpha": pa, "label": "front map"}),
             (world_xy, {"c": "crimson", "s": ps, "alpha": pa, "label": f"{name} (aligned)"})],
            cfg, title=f"Step 4 [{name}]: aligned overlay onto front map", extra=extra))

    cx, cy, cz = res["axis_curv_trans"]
    paths.append(viz.bar_plot(
        os.path.join(out_dir, f"step4_{name}_axis_curv.png"), ["x", "y", "z"], [cx, cy, cz],
        cfg, title=f"Step 4 [{name}]: translation-axis Hessian curvature (low = weakly observable)",
        ylabel="curvature", log_y=True))
    return paths, world_xy


def _run(args, cfg):
    if cfg["icp_method"] != "point_to_plane":
        print(f"  [warn] icp_method={cfg['icp_method']} not implemented; using point_to_plane")
    rtk = args.rtk_poses or os.path.join(args.out_dir, "rtk_poses.npy")
    poses = np.load(rtk)
    traj = common.TrajectoryInterpolator(poses[:, 0], poses[:, 1:4], poses[:, 4:8])
    override = common.load_tf_override(args.tf_override)

    voxels = list(cfg["icp_multiscale_voxels"])
    nrad = float(cfg["l2l_normal_radius_scale"])
    front_pts = load_front_map_pts(args.out_dir)
    print(f"Step 4: front map {len(front_pts)} pts; building {len(voxels)} scale targets")
    map_targets = {v: build_target(front_pts, v, v * nrad) for v in voxels}
    E_front = override.get(
        "lidar_front",
        np.asarray(step3.load_scan_manifest(args.out_dir, "front")["drs_T_lidar"], float))
    if "lidar_front" in override:
        print(f"  [override] fixed front reference t={np.round(E_front[:3, 3], 4).tolist()}")

    results, yaml_tf, E = {}, {}, {}
    lidar_xy = {}  # name -> aligned world XY, accumulated for the combined overlay
    # Stage 1: right, left (each anchored to fixed front via front-* term A)
    for name in ("right", "left"):
        scans, E0 = load_lidar_scans(args.out_dir, name, traj, cfg, args.max_scans)
        E0 = override.get(f"lidar_{name}", E0)
        geoms = build_inter_geoms(args.out_dir, "front", name, E_front, traj, cfg, args.max_scans)
        print(f"[{name}] {len(scans)} map-scans, {len(geoms)} front-{name} overlap pairs")
        res = optimize(E0, map_targets, scans, geoms, cfg, tag=name)
        E[name] = res["E"]
        _record(results, yaml_tf, name, res["E"], E0, res, len(scans), len(geoms))
        if cfg["viz_enabled"] and not args.no_viz:
            paths, lidar_xy[name] = plot_lidar_diag(args.out_dir, name, res, E0, map_targets, scans, geoms,
                                                     front_pts[:, :2], cfg)
            for p in paths:
                print(f"  wrote {p}")

    # Stage 2: rear (term A against the now-fixed right and left)
    scans, E0 = load_lidar_scans(args.out_dir, "rear", traj, cfg, args.max_scans)
    E0 = override.get("lidar_rear", E0)
    geoms = []
    for ref in STAGE2_REFS:
        if ref in E:
            geoms += build_inter_geoms(args.out_dir, ref, "rear", E[ref], traj, cfg, args.max_scans)
    print(f"[rear] {len(scans)} map-scans, {len(geoms)} rear-(right/left) overlap pairs")
    res = optimize(E0, map_targets, scans, geoms, cfg, tag="rear")
    _record(results, yaml_tf, "rear", res["E"], E0, res, len(scans), len(geoms))
    if cfg["viz_enabled"] and not args.no_viz:
        paths, lidar_xy["rear"] = plot_lidar_diag(args.out_dir, "rear", res, E0, map_targets, scans, geoms,
                                                   front_pts[:, :2], cfg)
        for p in paths:
            print(f"  wrote {p}")
        p = plot_combined_overlay(args.out_dir, front_pts[:, :2], lidar_xy, cfg)
        print(f"  wrote {p}")

    with open(os.path.join(args.out_dir, "lidar_to_lidar_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    try:
        import yaml
        with open(os.path.join(args.out_dir, "lidar_to_lidar_tf.yaml"), "w") as f:
            yaml.safe_dump({"drs_base_link": yaml_tf}, f, sort_keys=False)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not write yaml: {e}")
    print(f"  wrote lidar_to_lidar_result.json / lidar_to_lidar_tf.yaml to {args.out_dir}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 4: combined LiDAR-to-LiDAR extrinsic")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rtk-poses", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tf-override", default=None,
                    help="multi_tf_static YAML overriding drs->lidar extrinsics "
                         "(MUST match the --tf-override used for Step 3's front map)")
    ap.add_argument("--max-scans", type=int, default=None)
    ap.add_argument("--no-viz", action="store_true", help="skip PNG plot generation")
    args = ap.parse_args(argv)

    cfg = common.load_config(args.config)
    with viz.tee_log(args.out_dir, "step4_log.txt"):
        return _run(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
