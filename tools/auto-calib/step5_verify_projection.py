#!/usr/bin/env python3
"""Step 5 verification: before/after camera-LiDAR projection overlays.

For a handful of scenes spread across the bag, projects a single RAW LiDAR
scan (the camera's associated lidar's own frame, nearest in time to the
camera frame -- no RTK trajectory, accumulated map, or Step 3/4 registration
involved) onto the RAW (distorted) camera image: once with the
pre-optimization extrinsic (tf_static / --tf-override -- Step 5's x0) and
once with the optimized extrinsic from a Step 5 result JSON (v2 or v3). This
isolates the camera<->lidar extrinsic itself from trajectory-interpolation /
map-registration error, unlike projecting the Step 3 accumulated map.

Mirrors tools/lidar-camera/project_lidar_to_cam.py's approach (single scan,
intensity-colored, alpha-blended, raw-distorted projection), adapted to read
directly from the bag (matching nearest lidar scan <-> camera frame by
timestamp) and to compare Step 5's before/after extrinsic side by side.

Usage:
    python step5_verify_projection.py <bag> --out-dir OUT --camera 1
        [--n-scenes 5] [--result-json PATH] [--tf-override YAML] [--alpha 0.45]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

import common
import step5_cam_to_lidar as s5c


def find_result(out_dir, cam):
    """Locate the Step 5 result JSON containing camera{cam}; error if ambiguous
    (e.g. multiple v3 self-test/perturbation variants) so a stale/perturbed
    result is never picked silently."""
    candidates = []
    v2 = os.path.join(out_dir, "cam_to_lidar_result.json")
    if os.path.isfile(v2):
        candidates.append(v2)
    candidates += sorted(glob.glob(os.path.join(out_dir, "cam_to_lidar_v3_result*.json")))
    hits = []
    for p in candidates:
        with open(p) as f:
            d = json.load(f)
        if f"camera{cam}" in d:
            hits.append((p, d))
    if not hits:
        raise SystemExit(
            f"no Step 5 result JSON under {out_dir} contains camera{cam}; "
            f"run step5_cam_to_lidar.py / step5_v3_register.py first, or pass --result-json")
    if len(hits) > 1:
        names = "\n  ".join(p for p, _ in hits)
        raise SystemExit(
            f"multiple Step 5 result JSONs contain camera{cam}, pass --result-json to pick one:\n  {names}")
    return hits[0]


def label_bar(w, text, height=40):
    import cv2
    bar = np.full((height, w, 3), 30, np.uint8)
    cv2.putText(bar, text, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return bar


def color_by_intensity(inten, imin, imax):
    import cv2
    vals = np.clip((inten - imin) / (imax - imin), 0, 1)
    return cv2.applyColorMap((vals * 255).astype(np.uint8).reshape(-1, 1),
                              cv2.COLORMAP_JET).reshape(-1, 3)


def draw_points(img, u, v, colors, rad, alpha):
    """Alpha-blended point overlay so background stays visible (same scheme as
    tools/lidar-camera/project_lidar_to_cam.py:draw_points)."""
    h, w = img.shape[:2]
    overlay = img.copy()
    mask = np.zeros((h, w), dtype=bool)
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)
    for du in range(-rad, rad + 1):
        for dv in range(-rad, rad + 1):
            uu, vv = u + du, v + dv
            m = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
            overlay[vv[m], uu[m]] = colors[m]
            mask[vv[m], uu[m]] = True
    img[mask] = (alpha * overlay[mask].astype(np.float32)
                 + (1.0 - alpha) * img[mask].astype(np.float32)).astype(np.uint8)
    return img


def render_overlay(img, xyz, inten, params, project_fn, w, h, cfg, imin, imax, radius, alpha):
    """Transform a raw LiDAR scan (native lidar frame) into the optical frame at
    params and alpha-blend intensity-colored points onto the raw image.

    Returns (image, n_drawn, median DT residual [px] at the drawn points --
    the same Canny-edge cost Step 5 v2 optimizes, as an independent alignment
    check)."""
    import cv2

    R, t = s5c.lidar_to_optical(*params)
    pc = xyz @ R.T + t
    front = pc[:, 2] > 0.2
    pc, inten = pc[front], inten[front]
    out = img.copy()
    if len(pc) == 0:
        return out, 0, float("nan")
    uv = project_fn(pc)
    inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, inten = uv[inb], inten[inb]
    if len(uv) == 0:
        return out, 0, float("nan")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dt = s5c.distance_transform(gray, int(cfg["canny_low"]), int(cfg["canny_high"]),
                                 float(cfg["dist_transform_max"]))
    resid = float(np.median(s5c.bilinear_sample(dt, uv[:, 0], uv[:, 1])))

    colors = color_by_intensity(inten, imin, imax)
    draw_points(out, uv[:, 0], uv[:, 1], colors, radius, alpha)
    return out, len(uv), resid


def pass_timestamps(files, topic, msg_type):
    """(idx, t_ns) for every message on topic -- header stamp only, no heavy
    per-message work (JPEG decode / point array building)."""
    out = []
    for idx, m in enumerate(common.read_deserialized(files, topic, msg_type)):
        out.append((idx, m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec))
    return out


def pick_scenes(cam_ts, lidar_ts, n_scenes, max_dt_ms):
    """Evenly spaced LiDAR scan indices (mirrors project_lidar_to_cam.py's
    --sample), each matched to its nearest-in-time camera frame; pairs whose
    time gap exceeds max_dt_ms are dropped rather than silently mismatched."""
    lidar_idx = np.array([i for i, _ in lidar_ts])
    lidar_t = np.array([t for _, t in lidar_ts], dtype=np.int64)
    cam_idx = np.array([i for i, _ in cam_ts])
    cam_t = np.array([t for _, t in cam_ts], dtype=np.int64)

    n = min(n_scenes, len(lidar_idx))
    sel = np.unique(np.round(np.linspace(0, len(lidar_idx) - 1, n)).astype(int))
    scenes = []
    for s in sel:
        lt = lidar_t[s]
        ci = int(np.argmin(np.abs(cam_t - lt)))
        dt_ms = abs(int(cam_t[ci]) - int(lt)) / 1e6
        if dt_ms > max_dt_ms:
            print(f"  [skip] lidar idx={lidar_idx[s]} t={lt}: nearest camera frame is "
                  f"{dt_ms:.1f}ms away (> {max_dt_ms}ms)")
            continue
        scenes.append({"lidar_idx": int(lidar_idx[s]), "lidar_t": int(lt),
                        "cam_idx": int(cam_idx[ci]), "cam_t": int(cam_t[ci]), "dt_ms": dt_ms})
    return scenes


def speed_profile(rtk_poses_path):
    """(t_mid_ns, speed_mps) from consecutive rtk_poses.npy samples (~50Hz),
    t_mid being the midpoint timestamp of each speed sample."""
    poses = np.load(rtk_poses_path)
    ts = poses[:, 0].astype(np.int64)
    pos = poses[:, 1:4]
    order = np.argsort(ts, kind="stable")
    ts, pos = ts[order], pos[order]
    dt_s = np.diff(ts) / 1e9
    speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.clip(dt_s, 1e-6, None)
    t_mid = (ts[:-1] + ts[1:]) // 2
    return t_mid, speed


def pick_low_speed_scenes(t_mid, speed, cam_ts, lidar_ts, speed_thresh_mps, max_dt_ms,
                           min_episode_dur_s=0.5):
    """One scene per contiguous low-speed episode (speed <= speed_thresh_mps),
    at that episode's minimum-speed instant -- i.e. each stop / crawl the
    vehicle makes yields exactly one scene, timed to its slowest moment
    (least motion blur / lidar-camera sync error). Episodes shorter than
    min_episode_dur_s are dropped: speed noise flickering across the threshold
    for a sample or two is not a real stop/crawl, just estimation jitter."""
    below = speed <= speed_thresh_mps
    episodes = []
    start = None
    for i, b in enumerate(below):
        if b and start is None:
            start = i
        elif not b and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(below) - 1))
    episodes = [(s, e) for s, e in episodes
                if (t_mid[e] - t_mid[s]) / 1e9 >= min_episode_dur_s]

    lidar_idx = np.array([i for i, _ in lidar_ts])
    lidar_t = np.array([t for _, t in lidar_ts], dtype=np.int64)
    cam_idx = np.array([i for i, _ in cam_ts])
    cam_t = np.array([t for _, t in cam_ts], dtype=np.int64)

    scenes = []
    seen_lidar_idx = set()
    for ep, (s, e) in enumerate(episodes):
        j = s + int(np.argmin(speed[s:e + 1]))
        t_target = int(t_mid[j])
        v_min_kmh = float(speed[j] * 3.6)
        dur_s = float((t_mid[e] - t_mid[s]) / 1e9)
        li = int(np.argmin(np.abs(lidar_t - t_target)))
        lt = int(lidar_t[li])
        if lidar_idx[li] in seen_lidar_idx:
            continue
        ci = int(np.argmin(np.abs(cam_t - lt)))
        dt_ms = abs(int(cam_t[ci]) - lt) / 1e6
        if dt_ms > max_dt_ms:
            print(f"  [skip] episode{ep} v_min={v_min_kmh:.2f}km/h dur={dur_s:.1f}s t={t_target}: "
                  f"nearest camera frame is {dt_ms:.1f}ms away (> {max_dt_ms}ms)")
            continue
        seen_lidar_idx.add(lidar_idx[li])
        scenes.append({"lidar_idx": int(lidar_idx[li]), "lidar_t": lt,
                        "cam_idx": int(cam_idx[ci]), "cam_t": int(cam_t[ci]), "dt_ms": dt_ms,
                        "episode": ep, "v_min_kmh": v_min_kmh, "episode_dur_s": dur_s})
    return scenes


def main(argv=None):
    ap = argparse.ArgumentParser(description="Step 5 verification: before/after projection overlays")
    ap.add_argument("bag")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--camera", type=int, required=True)
    ap.add_argument("--result-json", default=None, help="explicit Step 5 result JSON (auto-detected if omitted)")
    ap.add_argument("--tf-override", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--scene-mode", choices=["low-speed", "even"], default="low-speed",
                     help="low-speed: one scene per contiguous low-speed episode, at its minimum-speed "
                          "instant (default). even: n-scenes evenly spaced across the whole bag")
    ap.add_argument("--speed-thresh-kmh", type=float, default=3.0,
                     help="low-speed mode: episode = contiguous stretch with speed <= this [km/h]")
    ap.add_argument("--min-episode-dur-s", type=float, default=0.5,
                     help="low-speed mode: drop episodes shorter than this (speed-noise flicker)")
    ap.add_argument("--rtk-poses", default=None, help="low-speed mode: default <out-dir>/rtk_poses.npy")
    ap.add_argument("--n-scenes", type=int, default=5, help="even mode: number of evenly spaced scenes")
    ap.add_argument("--max-cam-lidar-dt-ms", type=float, default=60.0,
                     help="drop a scene if no camera frame is within this of the lidar scan")
    ap.add_argument("--point-radius-px", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.45, help="point overlay opacity (0-1)")
    ap.add_argument("--imin", type=float, default=0.0, help="intensity colormap min")
    ap.add_argument("--imax", type=float, default=40.0, help="intensity colormap max")
    ap.add_argument("--output-subdir", default=None)
    args = ap.parse_args(argv)

    import cv2
    from sensor_msgs.msg import CompressedImage, PointCloud2

    cfg = common.load_config(args.config)
    files = common.resolve_bag_files(args.bag)
    override = common.load_tf_override(args.tf_override)
    cam = args.camera
    lidar = s5c.CAM_LIDAR[cam]

    if args.result_json:
        with open(args.result_json) as f:
            result = json.load(f)
        result_path = args.result_json
    else:
        result_path, result = find_result(args.out_dir, cam)
    entry = result[f"camera{cam}"]
    x0 = np.asarray(entry.get("x0") or s5c.init_cam_extrinsic(files, cam, lidar, override), np.float64)
    final = np.asarray(entry["params"], np.float64)
    print(f"Step 5 verify: camera{cam} <- lidar_{lidar}, result <- {result_path}")
    print(f"  x0    = {np.round(x0, 4).tolist()}")
    print(f"  final = {np.round(final, 4).tolist()}")

    K, D, w, h, model = s5c.load_camera_info(files, cam)
    project_fn = s5c.make_project_fn(K, D, model)

    cam_topic = f"/sensing/camera/camera{cam}/image_raw/compressed"
    lidar_topic = common.LIDAR_TOPICS[lidar]
    print(f"  pass 1/4: scanning timestamps on {cam_topic}")
    cam_ts = pass_timestamps(files, cam_topic, CompressedImage)
    print(f"  pass 2/4: scanning timestamps on {lidar_topic}")
    lidar_ts = pass_timestamps(files, lidar_topic, PointCloud2)

    if args.scene_mode == "low-speed":
        rtk_poses_path = args.rtk_poses or os.path.join(args.out_dir, "rtk_poses.npy")
        t_mid, speed = speed_profile(rtk_poses_path)
        scenes = pick_low_speed_scenes(t_mid, speed, cam_ts, lidar_ts,
                                        args.speed_thresh_kmh / 3.6, args.max_cam_lidar_dt_ms,
                                        args.min_episode_dur_s)
        if not scenes:
            raise SystemExit(f"no low-speed episode (<= {args.speed_thresh_kmh} km/h) in "
                              f"{rtk_poses_path} yielded a usable scene")
        for sc in scenes:
            print(f"  episode{sc['episode']}: v_min={sc['v_min_kmh']:.2f}km/h "
                  f"dur={sc['episode_dur_s']:.1f}s -> lidar_t={sc['lidar_t']}")
    else:
        scenes = pick_scenes(cam_ts, lidar_ts, args.n_scenes, args.max_cam_lidar_dt_ms)
        if not scenes:
            raise SystemExit("no scene had a camera frame within --max-cam-lidar-dt-ms of a lidar scan")
    print(f"  {len(scenes)} scenes selected")

    want_lidar = {s["lidar_idx"] for s in scenes}
    want_cam = {s["cam_idx"] for s in scenes}

    print("  pass 3/4: reading selected lidar scans")
    lidar_data = {}
    for idx, m in enumerate(common.read_deserialized(files, lidar_topic, PointCloud2)):
        if idx not in want_lidar:
            continue
        arr = common.pointcloud2_to_array(m)
        xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)
        inten = (arr["intensity"].astype(np.float64) if "intensity" in arr.dtype.names
                  else np.zeros(len(arr)))
        valid = np.all(np.isfinite(xyz), axis=1)
        lidar_data[idx] = (xyz[valid], inten[valid])
        if len(lidar_data) == len(want_lidar):
            break

    print("  pass 4/4: reading selected camera frames")
    cam_data = {}
    for idx, m in enumerate(common.read_deserialized(files, cam_topic, CompressedImage)):
        if idx not in want_cam:
            continue
        img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            cam_data[idx] = img
        if len(cam_data) == len(want_cam):
            break

    out_dir = os.path.join(args.out_dir, args.output_subdir or f"step5_verify_camera{cam}")
    os.makedirs(out_dir, exist_ok=True)

    n_written = 0
    for i, sc in enumerate(scenes):
        img = cam_data.get(sc["cam_idx"])
        xyz_inten = lidar_data.get(sc["lidar_idx"])
        if img is None or xyz_inten is None:
            print(f"  [skip] scene{i:02d}: missing frame/scan data")
            continue
        xyz, inten = xyz_inten

        before, n_before, resid_before = render_overlay(
            img, xyz, inten, x0, project_fn, w, h, cfg, args.imin, args.imax,
            args.point_radius_px, args.alpha)
        after, n_after, resid_after = render_overlay(
            img, xyz, inten, final, project_fn, w, h, cfg, args.imin, args.imax,
            args.point_radius_px, args.alpha)

        tag = f"scene{i:02d}_lidar_t{sc['lidar_t']}"
        cv2.imwrite(os.path.join(out_dir, f"{tag}_before.png"), before)
        cv2.imwrite(os.path.join(out_dir, f"{tag}_after.png"), after)
        before_lb = np.vstack([
            label_bar(w, f"BEFORE (init TF)  n={n_before}  DT_resid={resid_before:.2f}px"), before])
        after_lb = np.vstack([
            label_bar(w, f"AFTER (optimized TF)  n={n_after}  DT_resid={resid_after:.2f}px"), after])
        cv2.imwrite(os.path.join(out_dir, f"{tag}_compare.png"), np.vstack([before_lb, after_lb]))

        print(f"  scene{i:02d} lidar_t={sc['lidar_t']} cam_dt={sc['dt_ms']:.1f}ms "
              f"before: n={n_before} resid={resid_before:.2f}px | "
              f"after: n={n_after} resid={resid_after:.2f}px")
        n_written += 1

    print(f"wrote {n_written} scene(s) x {{before,after,compare}}.png to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
