#!/usr/bin/env python3
"""MCAP の CompressedImage(jpeg) を sensor 別ディレクトリに JPG 保存する。

ファイル名は header.stamp のナノ秒(エポック)。 {sec}{nanosec:09d}.jpg
--cams 未指定時は MCAP 内の全カメラトピックを自動検出する。
各カメラディレクトリに camera_info.json も保存する。
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, CompressedImage

_CAM_TOPIC_RE = re.compile(r"^/sensing/camera/(camera\d+)/image_raw/compressed$")


def build_cam_map(cam_ids):
    return {
        f"/sensing/camera/camera{i}/image_raw/compressed": f"camera{i}"
        for i in cam_ids
    }


def discover_cam_map(mcap_path):
    """MCAP のチャンネル一覧からカメラトピックを自動検出して cam_map を返す。"""
    with open(mcap_path, "rb") as f:
        summary = make_reader(f).get_summary()
    cam_map = {}
    if summary:
        for ch in summary.channels.values():
            m = _CAM_TOPIC_RE.match(ch.topic)
            if m:
                cam_map[ch.topic] = m.group(1)
    return cam_map


def extract_camera_infos(mcap_path, cam_names, out_root):
    """各カメラの最初の CameraInfo を {cam_dir}/camera_info.json に保存する。"""
    info_topics = {
        f"/sensing/camera/{name}/camera_info": name
        for name in cam_names
    }
    saved = set()
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages(topics=list(info_topics.keys())):
            name = info_topics.get(channel.topic)
            if name is None or name in saved:
                continue
            msg = deserialize_message(message.data, CameraInfo)
            info = {
                "width":            msg.width,
                "height":           msg.height,
                "K":                np.array(msg.k).reshape(3, 3).tolist(),
                "D":                list(msg.d),
                "distortion_model": msg.distortion_model,
            }
            dst = out_root / name / "camera_info.json"
            with open(dst, "w") as fp:
                json.dump(info, fp, indent=2)
            saved.add(name)
            print(f"[cam] camera_info: {name} {msg.width}x{msg.height} "
                  f"model={msg.distortion_model}", flush=True)
            if saved >= set(cam_names):
                break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mcap")
    ap.add_argument("out_root")
    ap.add_argument("--cams", default=None,
                    help="カメラ番号(カンマ区切り)。省略時はMCAP内の全カメラを自動検出")
    ap.add_argument("--limit", type=int, default=0, help="各カメラ最大枚数(0=無制限/検証用)")
    ap.add_argument("--start", type=float, default=None, help="開始時刻(UNIX秒)")
    ap.add_argument("--end", type=float, default=None, help="終了時刻(UNIX秒)")
    args = ap.parse_args()

    if args.cams is not None:
        cam_map = build_cam_map([int(x) for x in args.cams.split(",")])
    else:
        cam_map = discover_cam_map(args.mcap)
        if not cam_map:
            print("[cam] ERROR: no camera topics found in MCAP", file=sys.stderr)
            return 1
        print(f"[cam] auto-detected cameras: {sorted(cam_map.values())}", flush=True)

    out_root = Path(args.out_root)
    for name in cam_map.values():
        (out_root / name).mkdir(parents=True, exist_ok=True)

    extract_camera_infos(args.mcap, list(cam_map.values()), out_root)

    counts = {name: 0 for name in cam_map.values()}
    collisions = 0
    t0 = time.time()
    total = 0

    start_ns = int(args.start * 1e9) if args.start is not None else None
    end_ns = int(args.end * 1e9) if args.end is not None else None

    with open(args.mcap, "rb") as f:
        reader = make_reader(f)
        for _schema, channel, message in reader.iter_messages(
                topics=list(cam_map.keys()), start_time=start_ns, end_time=end_ns):
            name = cam_map.get(channel.topic)
            if name is None:
                continue
            if args.limit and counts[name] >= args.limit:
                if all(counts[n] >= args.limit for n in cam_map.values()):
                    break
                continue
            msg = deserialize_message(message.data, CompressedImage)
            stamp = msg.header.stamp
            fname = f"{stamp.sec}{stamp.nanosec:09d}.jpg"
            dst = out_root / name / fname
            if dst.exists():
                collisions += 1
                dst = out_root / name / f"{stamp.sec}{stamp.nanosec:09d}_{counts[name]}.jpg"
            with open(dst, "wb") as g:
                g.write(bytes(msg.data))
            counts[name] += 1
            total += 1
            if total % 2000 == 0:
                dt = time.time() - t0
                print(f"[cam] {total} 枚 / {dt:.0f}s "
                      + " ".join(f"{n}:{c}" for n, c in counts.items()), flush=True)

    dt = time.time() - t0
    print(f"[cam] DONE total={total} collisions={collisions} elapsed={dt:.0f}s", flush=True)
    for n, c in counts.items():
        print(f"  {n}: {c}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
