#!/usr/bin/env python3
"""Step 5 v3, part 2: register the accumulated camera cloud to the LiDAR map.

Solves lidar_T_camera_link by minimizing REPROJECTION error (not a post-hoc
rigid ICP on the pooled world-frame cloud): each observation (pixel, time)
contributes a residual tying the SHARED unknown extrinsic to a specific known
base pose W_T_drs(t). This avoids the failure mode already identified in
Step 4 ("a naive accumulate + single rigid-transform correction averages a
translation error to ~0 over a full-yaw loop") — a pooled-cloud rigid ICP
would have the same defect, since the systematic position error from a wrong
extrinsic varies with vehicle heading and does not collapse to one global
rigid offset. Per-observation reprojection residuals don't have this
degeneracy: each ties E to one specific (time, pixel) pair.

The LiDAR map enters by SNAPPING each triangulated 3D point onto the nearest
LiDAR map surface (point-to-plane, using Step 3/4's map target) before the
reprojection refinement — this is what "GICP" contributes in the user's
proposed design: it corrects/anchors the visually-reconstructed structure to
independently-acquired LiDAR ground truth, rather than trusting triangulation
depth alone. Points with no nearby LiDAR support are dropped (can't
cross-validate depth there).

Outer loop (reconstruct -> snap -> refine), v3_outer_iters times: each
iteration re-triangulates with the latest extrinsic estimate, so triangulation
error introduced by an imperfect estimate shrinks as the estimate improves
(benign for the small errors this refines from — tf_static is already good).

The snap distance follows a coarse-to-fine schedule (v3_snap_max_dist_schedule,
holding at the last value once outer iterations exceed the schedule length)
rather than a single fixed threshold. Rotation error turns into a snap-distance
error proportional to point depth (depth * tan(theta)), and triangulated points
can span tens of meters, so a single-digit-degree error already pushes most
points outside a tight fixed threshold before the extrinsic has a chance to
improve — verified empirically on camera1 via --self-test: 63% snap rate at
the true extrinsic collapsed to 39% under a 3deg perturbation, and never
recovered over 3 outer iterations at a fixed 0.5m threshold. Starting loose
and tightening lets far points pull the estimate in the right direction first.

Reuses Step 4's point-to-plane registration primitives (build_target for the
LiDAR map KD-tree+normals) but NOT its analytic SE(3) Jacobian — the
reprojection residual here uses scipy.optimize.least_squares (numerical
Jacobian, robust Huber loss), matching how v1/v2 already handle camera-side
optimization.

Usage:
    python step5_v3_register.py <bag> --out-dir OUT --cameras 0,2
        [--tf-override YAML] [--config config.yaml]
        [--self-test --perturb-trans-m 0.10 --perturb-rot-deg 3.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import common
import step4_lidar_to_lidar as step4
import step5_cam_to_lidar as s5
import step5_v3_cam_cloud as v3c


def snap_to_map(P, map_target, max_dist):
    """Point-to-plane projection of each point onto the nearest LiDAR map
    surface; points farther than max_dist are dropped (keep mask returned)."""
    sp, sn, tree = map_target
    dist, idx = tree.query(P, workers=-1)
    keep = dist < max_dist
    Pn = P.copy()
    q = sp[idx[keep]]
    n = sn[idx[keep]]
    r = np.einsum("ij,ij->i", n, P[keep] - q)
    Pn[keep] = P[keep] - r[:, None] * n
    return Pn, keep


def build_obs_arrays(tracks):
    k_idx, t_ns, uv = [], [], []
    for k, tr in enumerate(tracks):
        for t, u, v in tr:
            k_idx.append(k)
            t_ns.append(t)
            uv.append((u, v))
    return (np.array(k_idx, np.int64), np.array(t_ns, np.float64),
            np.array(uv, np.float64))


def reprojection_residuals(params, k_idx, t_ns, uv, pts, drs_T_lidar, traj, project_fn):
    """Vectorized (predicted - observed) pixel residual per observation.
    Behind-camera points get a fixed large penalty (keeps a bounded, non-degenerate
    residual vector without letting the optimizer hide points, mirroring the
    fixed-denominator lesson from Step 5 v1/v2)."""
    R_wo, t_wo = v3c.camera_poses_in_map_batch(params, drs_T_lidar, traj, t_ns)
    P = pts[k_idx]
    p_cam = np.einsum("nij,nj->ni", R_wo.transpose(0, 2, 1), P - t_wo)
    behind = p_cam[:, 2] <= 0.2
    p_cam_safe = p_cam.copy()
    p_cam_safe[behind, 2] = 0.2
    uv_pred = project_fn(p_cam_safe)
    res = uv_pred - uv
    res[behind] = 50.0
    return res.reshape(-1)


def optimize_camera_v3(files, cam, out_dir, traj, map_target, cfg, override,
                        tracks=None, start_params=None):
    from scipy.optimize import least_squares

    lidar = s5.CAM_LIDAR[cam]
    K, D, w, h, model = s5.load_camera_info(files, cam)
    project_fn = s5.make_project_fn(K, D, model)
    unproject_fn = s5.make_unproject_fn(K, D, model)
    drs_T_lidar = s5.load_drs_T_lidar(out_dir, lidar, files, override)
    x0 = np.array(s5.init_cam_extrinsic(files, cam, lidar, override), np.float64)
    if tracks is None:
        tracks = v3c.load_tracks(out_dir, cam)
    if not tracks:
        return {"cam": cam, "lidar": lidar, "error": "no tracks"}

    tb, rb = float(cfg["v3_trans_bound_m"]), float(cfg["v3_rot_bound_rad"])
    bound = np.array([tb, tb, tb, rb, rb, rb])
    lo, hi = x0 - bound, x0 + bound
    huber_px = float(cfg["v3_huber_px"])
    snap_schedule = [float(v) for v in cfg["v3_snap_max_dist_schedule"]]
    max_nfev = int(cfg["v3_ls_max_nfev"])

    params = x0.copy() if start_params is None else np.asarray(start_params, np.float64)
    history = []
    for outer in range(int(cfg["v3_outer_iters"])):
        snap_max = snap_schedule[min(outer, len(snap_schedule) - 1)]
        pts, tr_kept, reprojs, parallaxes = [], [], [], []
        for tr in tracks:
            res = v3c.triangulate_track(tr, params, drs_T_lidar, traj, unproject_fn, cfg)
            if res is None:
                continue
            P, reproj, parallax = res
            pts.append(P)
            tr_kept.append(tr)
            reprojs.append(reproj)
            parallaxes.append(parallax)
        if len(pts) < 20:
            return {"cam": cam, "lidar": lidar, "n_tracks": len(tracks),
                    "error": f"insufficient triangulated points ({len(pts)})"}
        pts = np.array(pts)
        pts_snapped, keep = snap_to_map(pts, map_target, snap_max)
        pts_final = pts_snapped[keep]
        tracks_final = [tr_kept[i] for i in np.nonzero(keep)[0]]
        if len(tracks_final) < 20:
            return {"cam": cam, "lidar": lidar, "n_tracks": len(tracks), "n_triangulated": len(pts),
                    "error": f"insufficient points near LiDAR map ({len(tracks_final)}/{len(pts)})"}

        k_idx, t_ns_arr, uv_arr = build_obs_arrays(tracks_final)

        def residual_fn(p, _k=k_idx, _t=t_ns_arr, _uv=uv_arr, _pts=pts_final):
            return reprojection_residuals(p, _k, _t, _uv, _pts, drs_T_lidar, traj, project_fn)

        r0 = residual_fn(params)
        result = least_squares(residual_fn, params, method="trf", loss="huber",
                                f_scale=huber_px, bounds=(lo, hi), max_nfev=max_nfev)
        params = result.x
        rf = residual_fn(params)
        history.append({
            "outer_it": outer, "snap_max_dist": snap_max,
            "n_triangulated": len(pts), "n_snapped": len(tracks_final),
            "n_obs": len(uv_arr), "resid_px_rms_init": float(np.sqrt(np.mean(r0 ** 2))),
            "resid_px_rms_final": float(np.sqrt(np.mean(rf ** 2))),
            "mean_parallax_deg": float(np.mean(parallaxes)), "mean_reproj_norm": float(np.mean(reprojs)),
        })
        Jlast = result.jac

    R0 = s5.euler_to_R(*x0[3:])
    Rf = s5.euler_to_R(*params[3:])
    d_trans = float(np.linalg.norm(params[:3] - x0[:3]))
    dR = R0.T @ Rf
    d_rot = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    H = Jlast.T @ Jlast
    return {"cam": cam, "lidar": lidar, "model": model, "params": params.tolist(), "x0": x0.tolist(),
            "history": history, "delta_trans_cm": d_trans * 100, "delta_rot_deg": d_rot,
            "resid_px_rms_final": history[-1]["resid_px_rms_final"], "n_obs_final": history[-1]["n_obs"],
            "axis_curv_trans": np.diag(H)[:3].tolist(), "axis_curv_rot": np.diag(H)[3:].tolist()}


def load_map_target(out_dir, cfg):
    front_pts = step4.load_front_map_pts(out_dir)
    voxel = float(cfg["icp_multiscale_voxels"][-1])
    nrad = voxel * float(cfg["l2l_normal_radius_scale"])
    return step4.build_target(front_pts, voxel, nrad)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 5 v3 part 2: camera cloud -> LiDAR map registration")
    ap.add_argument("bag")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rtk-poses", default=None)
    ap.add_argument("--cameras", default="0,2,4,6")
    ap.add_argument("--tf-override", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--self-test", action="store_true",
                     help="mechanical smoke test: perturb tf_static by a known amount and "
                          "check recovery. Runs on any RTK (incl. float) since it only checks "
                          "that the pipeline finds its way back to the SAME unperturbed result "
                          "-- it does NOT validate absolute calibration accuracy (needs fix RTK).")
    ap.add_argument("--perturb-trans-m", type=float, default=0.10)
    ap.add_argument("--perturb-rot-deg", type=float, default=3.0)
    args = ap.parse_args(argv)

    cfg = common.load_config(args.config)
    files = common.resolve_bag_files(args.bag)
    override = common.load_tf_override(args.tf_override)
    rtk = args.rtk_poses or os.path.join(args.out_dir, "rtk_poses.npy")
    poses = np.load(rtk)
    traj = common.TrajectoryInterpolator(poses[:, 0], poses[:, 1:4], poses[:, 4:8])
    cams = [int(c) for c in args.cameras.split(",") if c.strip() != ""]

    print("Step 5 v3: building LiDAR map registration target ...")
    map_target = load_map_target(args.out_dir, cfg)

    rng = np.random.default_rng(42)
    results = {}
    yaml_tf = {}
    for cam in cams:
        start_params = None
        if args.self_test:
            lidar = s5.CAM_LIDAR[cam]
            x0 = np.array(s5.init_cam_extrinsic(files, cam, lidar, override), np.float64)
            dtr = rng.normal(size=3)
            dtr = dtr / np.linalg.norm(dtr) * args.perturb_trans_m
            drot = rng.normal(size=3)
            drot = drot / np.linalg.norm(drot) * np.radians(args.perturb_rot_deg)
            start_params = x0 + np.concatenate([dtr, drot])

        r = optimize_camera_v3(files, cam, args.out_dir, traj, map_target, cfg, override,
                                start_params=start_params)
        results[f"camera{cam}"] = r
        if r.get("error"):
            print(f"  [error] camera{cam}: {r['error']}")
            continue
        h = r["history"]
        print(f"  camera{cam} [{r['model']}] {len(h)} outer iters, "
              f"final {h[-1]['n_snapped']}/{h[-1]['n_triangulated']} pts snapped, "
              f"resid {h[0]['resid_px_rms_init']:.2f}->{r['resid_px_rms_final']:.2f}px, "
              f"moved {r['delta_trans_cm']:.1f}cm/{r['delta_rot_deg']:.2f}deg"
              + (f" (perturbed {args.perturb_trans_m*100:.1f}cm/{args.perturb_rot_deg:.2f}deg)"
                 if args.self_test else ""))
        p = r["params"]
        yaml_tf.setdefault(f"lidar_{r['lidar']}", {})[f"camera{cam}/camera_link"] = {
            "x": p[0], "y": p[1], "z": p[2], "roll": p[3], "pitch": p[4], "yaw": p[5]}

    with open(os.path.join(args.out_dir, "cam_to_lidar_v3_result.json"), "w") as f:
        json.dump(results, f, indent=2)
    try:
        import yaml
        with open(os.path.join(args.out_dir, "cam_to_lidar_v3_tf.yaml"), "w") as f:
            yaml.safe_dump(yaml_tf, f, sort_keys=False)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not write yaml: {e}")
    print(f"  wrote cam_to_lidar_v3_result.json / cam_to_lidar_v3_tf.yaml to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
