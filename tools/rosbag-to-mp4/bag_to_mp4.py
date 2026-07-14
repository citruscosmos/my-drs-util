#!/usr/bin/env python3
"""rosbag (MCAP) の指定カメラを MP4 に変換する。

単一 .mcap ファイルと、分割 rosbag2 ディレクトリ (*_0.mcap, *_1.mcap, ...) の
両方を入力として受け付ける。カメラを複数指定した場合はカメラごとに 1 本の
MP4 を生成する。解像度を省略した場合はカメラの元解像度をそのまま使用する。

出力ファイル名: {rosbag名}_{camera}_{width}x{height}.mp4
  rosbag名は、単一ファイルなら拡張子抜きのファイル名、分割ディレクトリなら
  ディレクトリ名。

CompressedImage (JPEG) を再デコードせず、ffmpeg の mjpeg デマルチプレクサに
直接ストリーミングして H.264/MP4 にエンコードする (システムの ffmpeg が必要)。
"""
import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path

from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, CompressedImage

_CAM_TOPIC_RE = re.compile(r"^/sensing/camera/(camera\d+)/image_raw/compressed$")


def resolve_bag_files(path: str) -> list[str]:
    """単一 .mcap ファイル、または分割 rosbag2 ディレクトリを .mcap のリストに解決する。"""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.mcap")))
        if not files:
            raise FileNotFoundError(f"no .mcap files under directory: {path}")
        return files
    if not os.path.isfile(path):
        raise FileNotFoundError(f"bag not found: {path}")
    return [path]


def bag_name(path: str) -> str:
    p = Path(path)
    return p.stem if p.is_file() else p.name


def discover_cam_topics(files: list[str]) -> dict[str, str]:
    """bag 内のカメラトピックを自動検出して {topic: camera_name} を返す。"""
    topics = {}
    for fp in files:
        with open(fp, "rb") as f:
            summary = make_reader(f).get_summary()
        if not summary:
            continue
        for ch in summary.channels.values():
            m = _CAM_TOPIC_RE.match(ch.topic)
            if m:
                topics[ch.topic] = m.group(1)
    return topics


def bag_stats(files: list[str]) -> tuple[dict[str, int], float]:
    """全ファイルを跨いだトピック別メッセージ数と、bag 全体の長さ(秒)を返す。

    fps 推定用。get_summary() はインデックスの読み出しのみで、メッセージ本体を
    デコードしないため軽量。
    """
    topic_count: dict[str, int] = {}
    tmin = tmax = None
    for fp in files:
        with open(fp, "rb") as f:
            summary = make_reader(f).get_summary()
        if not summary:
            continue
        counts = summary.statistics.channel_message_counts if summary.statistics else {}
        if summary.statistics:
            a = summary.statistics.message_start_time
            b = summary.statistics.message_end_time
            tmin = a if tmin is None else min(tmin, a)
            tmax = b if tmax is None else max(tmax, b)
        for ch in summary.channels.values():
            topic_count[ch.topic] = topic_count.get(ch.topic, 0) + counts.get(ch.id, 0)
    duration_s = (tmax - tmin) / 1e9 if tmin is not None else 0.0
    return topic_count, duration_s


def get_source_resolution(files: list[str], cam_name: str) -> tuple[int, int] | None:
    """CameraInfo からカメラの元解像度を取得する。無ければ最初の1枚をデコードして推定する。"""
    info_topic = f"/sensing/camera/{cam_name}/camera_info"
    for fp in files:
        with open(fp, "rb") as f:
            reader = make_reader(f)
            for _schema, _channel, message in reader.iter_messages(topics=[info_topic]):
                msg = deserialize_message(message.data, CameraInfo)
                return msg.width, msg.height

    img_topic = f"/sensing/camera/{cam_name}/image_raw/compressed"
    import cv2
    import numpy as np
    for fp in files:
        with open(fp, "rb") as f:
            reader = make_reader(f)
            for _schema, _channel, message in reader.iter_messages(topics=[img_topic]):
                msg = deserialize_message(message.data, CompressedImage)
                img = cv2.imdecode(np.frombuffer(bytes(msg.data), dtype=np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    h, w = img.shape[:2]
                    return w, h
    return None


def parse_resolution(s: str) -> tuple[int, int]:
    m = re.match(r"^(\d+)x(\d+)$", s)
    if not m:
        raise argparse.ArgumentTypeError(f"invalid resolution '{s}' (expected WIDTHxHEIGHT, e.g. 1280x720)")
    w, h = int(m.group(1)), int(m.group(2))
    if w % 2 or h % 2:
        raise argparse.ArgumentTypeError(f"resolution must have even width/height: {s}")
    return w, h


def start_ffmpeg(out_path: Path, fps: float, src_wh: tuple[int, int], dst_wh: tuple[int, int],
                  crf: int, preset: str) -> subprocess.Popen:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "mjpeg", "-framerate", f"{fps:.6f}",
        "-i", "-",
    ]
    if dst_wh != src_wh:
        cmd += ["-vf", f"scale={dst_wh[0]}:{dst_wh[1]}"]
    cmd += [
        "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="rosbag (MCAP) の指定カメラを MP4 に変換する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("rosbag", help="入力 .mcap ファイル、または分割 rosbag2 ディレクトリ")
    ap.add_argument("out_dir", help="出力ディレクトリ")
    ap.add_argument("--cameras", default=None,
                    help="カメラ番号 (カンマ区切り、例: 0,2,4)。省略時は bag 内の全カメラを自動検出")
    ap.add_argument("--resolution", type=parse_resolution, default=None,
                    help="出力解像度 WIDTHxHEIGHT (例: 1280x720)。省略時はソース解像度のまま")
    ap.add_argument("--fps", type=float, default=None,
                    help="出力フレームレート。省略時はメッセージ数と bag 長から自動推定")
    ap.add_argument("--start", type=float, default=None, help="開始時刻 (UNIX秒)")
    ap.add_argument("--end", type=float, default=None, help="終了時刻 (UNIX秒)")
    ap.add_argument("--crf", type=int, default=18, help="libx264 CRF (小さいほど高画質・大サイズ)")
    ap.add_argument("--preset", default="medium", help="libx264 preset (ultrafast..veryslow)")
    args = ap.parse_args()

    files = resolve_bag_files(args.rosbag)
    name = bag_name(args.rosbag)

    if args.cameras is not None:
        cam_names = [f"camera{int(x)}" for x in args.cameras.split(",")]
    else:
        cam_topics = discover_cam_topics(files)
        if not cam_topics:
            print("[bag2mp4] ERROR: no camera topics found in bag", file=sys.stderr)
            return 1
        cam_names = sorted(cam_topics.values(), key=lambda s: int(s.replace("camera", "")))
        print(f"[bag2mp4] auto-detected cameras: {cam_names}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_ns = int(args.start * 1e9) if args.start is not None else None
    end_ns = int(args.end * 1e9) if args.end is not None else None

    topic_count, bag_duration_s = bag_stats(files)

    procs: dict[str, subprocess.Popen] = {}
    out_paths: dict[str, Path] = {}
    counts = {c: 0 for c in cam_names}

    for c in cam_names:
        src_wh = get_source_resolution(files, c)
        if src_wh is None:
            print(f"[bag2mp4] WARNING: no image/camera_info found for {c}, skipping", file=sys.stderr)
            continue
        dst_wh = args.resolution if args.resolution else src_wh

        img_topic = f"/sensing/camera/{c}/image_raw/compressed"
        n = topic_count.get(img_topic, 0)
        if n == 0:
            print(f"[bag2mp4] WARNING: no messages on {img_topic}, skipping", file=sys.stderr)
            continue
        fps = args.fps if args.fps is not None else (n / bag_duration_s if bag_duration_s > 0 else 10.0)

        out_path = out_dir / f"{name}_{c}_{dst_wh[0]}x{dst_wh[1]}.mp4"
        print(f"[bag2mp4] {c}: {src_wh[0]}x{src_wh[1]} -> {dst_wh[0]}x{dst_wh[1]} "
              f"@ {fps:.2f}fps -> {out_path}", flush=True)
        procs[c] = start_ffmpeg(out_path, fps, src_wh, dst_wh, args.crf, args.preset)
        out_paths[c] = out_path

    if not procs:
        print("[bag2mp4] ERROR: nothing to encode", file=sys.stderr)
        return 1

    topic_to_cam = {f"/sensing/camera/{c}/image_raw/compressed": c for c in procs}

    total = 0
    for fp in files:
        with open(fp, "rb") as f:
            reader = make_reader(f)
            for _schema, channel, message in reader.iter_messages(
                    topics=list(topic_to_cam.keys()), start_time=start_ns, end_time=end_ns):
                c = topic_to_cam.get(channel.topic)
                if c is None:
                    continue
                msg = deserialize_message(message.data, CompressedImage)
                try:
                    procs[c].stdin.write(bytes(msg.data))
                except BrokenPipeError:
                    print(f"[bag2mp4] ERROR: ffmpeg for {c} exited early", file=sys.stderr)
                    continue
                counts[c] += 1
                total += 1
                if total % 2000 == 0:
                    print(f"[bag2mp4] {total} frames " + " ".join(f"{k}:{v}" for k, v in counts.items()),
                          flush=True)

    ret = 0
    for c, proc in procs.items():
        proc.stdin.close()
        rc = proc.wait()
        if rc != 0:
            print(f"[bag2mp4] ERROR: ffmpeg failed for {c} (exit {rc}) -> {out_paths[c]}", file=sys.stderr)
            ret = 1
        else:
            print(f"[bag2mp4] DONE {c}: {counts[c]} frames -> {out_paths[c]}", flush=True)

    return ret


if __name__ == "__main__":
    sys.exit(main())
