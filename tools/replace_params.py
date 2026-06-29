#!/usr/bin/env python3
import argparse
import math
import re
import sys
from pathlib import Path

import yaml
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion, TransformStamped, Vector3
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py
from sensor_msgs.msg import CameraInfo
from tf2_msgs.msg import TFMessage


CAMERA_INFO_RE = re.compile(r'^/sensing/camera/(camera\d+)/camera_info$')
CAMERA_LINK_RE = re.compile(r'^(camera\d+)/camera_link$')


def rpy_to_quaternion(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def load_tf_message(params_dir: Path, bag_timestamp_ns: int) -> TFMessage:
    tf_yaml_path = params_dir / 'multi_tf_static.yaml'
    with open(tf_yaml_path) as f:
        data = yaml.safe_load(f)

    stamp = Time(
        sec=bag_timestamp_ns // 1_000_000_000,
        nanosec=bag_timestamp_ns % 1_000_000_000,
    )
    transforms = []
    for parent_frame, children in data.items():
        if not isinstance(children, dict):
            continue
        for child_frame, vals in children.items():
            if not isinstance(vals, dict):
                continue
            x = float(vals.get('x', 0.0))
            y = float(vals.get('y', 0.0))
            z = float(vals.get('z', 0.0))
            roll = float(vals.get('roll', 0.0))
            pitch = float(vals.get('pitch', 0.0))
            yaw = float(vals.get('yaw', 0.0))
            qx, qy, qz, qw = rpy_to_quaternion(roll, pitch, yaw)

            ts = TransformStamped()
            ts.header.stamp = stamp
            ts.header.frame_id = parent_frame
            ts.child_frame_id = child_frame
            ts.transform.translation = Vector3(x=x, y=y, z=z)
            ts.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
            transforms.append(ts)

    if not transforms:
        sys.exit('Error: multi_tf_static.yaml parsed to 0 transforms')

    # Auto-generate camera_optical_link for each camera_link.
    # The optical frame is always at a fixed rotation from camera_link:
    # RPY(-pi/2, 0, -pi/2) → quaternion(-0.5, 0.5, -0.5, 0.5)
    optical_transforms = []
    for ts in transforms:
        m = CAMERA_LINK_RE.match(ts.child_frame_id)
        if m:
            cam_prefix = m.group(1)
            opt = TransformStamped()
            opt.header.stamp = stamp
            opt.header.frame_id = ts.child_frame_id
            opt.child_frame_id = f'{cam_prefix}/camera_optical_link'
            opt.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
            opt.transform.rotation = Quaternion(x=-0.5, y=0.5, z=-0.5, w=0.5)
            optical_transforms.append(opt)
    transforms.extend(optical_transforms)

    return TFMessage(transforms=transforms)


def load_camera_infos(params_dir: Path) -> dict:
    """Return dict mapping camera_id (e.g. 'camera0') to CameraInfo template (no header)."""
    camera_infos = {}
    for cam_yaml in sorted(params_dir.glob('camera*/camera_info.yaml')):
        cam_id = cam_yaml.parent.name
        with open(cam_yaml) as f:
            data = yaml.safe_load(f)
        info = CameraInfo()
        info.width = int(data['image_width'])
        info.height = int(data['image_height'])
        info.distortion_model = str(data['distortion_model'])
        info.k = [float(v) for v in data['camera_matrix']['data']]
        info.d = [float(v) for v in data['distortion_coefficients']['data']]
        info.p = [float(v) for v in data['projection_matrix']['data']]
        info.r = [float(v) for v in data['rectification_matrix']['data']]
        camera_infos[cam_id] = info
    return camera_infos


def build_camera_info(template: CameraInfo, orig_header) -> CameraInfo:
    msg = CameraInfo()
    msg.header = orig_header
    msg.width = template.width
    msg.height = template.height
    msg.distortion_model = template.distortion_model
    msg.k = template.k
    msg.d = template.d
    msg.p = template.p
    msg.r = template.r
    return msg


def main():
    parser = argparse.ArgumentParser(
        description='Replace /tf_static and camera_info payloads in an MCAP bag with calibration data.'
    )
    parser.add_argument('--input', required=True, help='Input bag path (.mcap)')
    parser.add_argument('--output', required=True, help='Output bag path (.mcap)')
    parser.add_argument(
        '--params', required=True,
        help='Config directory containing multi_tf_static.yaml and camera*/camera_info.yaml',
    )
    parser.add_argument('--force', action='store_true', help='Overwrite output if it exists')
    parser.add_argument('--compress', action='store_true', help='Compress output with zstd')
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        if not args.force:
            sys.exit(f'Output exists: {args.output}. Use --force to overwrite.')
        output_path.unlink()

    params_dir = Path(args.params)
    camera_infos = load_camera_infos(params_dir)

    storage_id = 'mcap' if args.input.endswith('.mcap') else ''
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=args.input, storage_id=storage_id),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )

    out_storage = rosbag2_py.StorageOptions(uri=args.output, storage_id='mcap')
    if args.compress:
        out_storage.storage_config_yaml = 'compression: zstd'

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        out_storage,
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )

    for topic in reader.get_all_topics_and_types():
        writer.create_topic(topic)

    warned_cameras = set()

    while reader.has_next():
        topic, raw, timestamp = reader.read_next()

        if topic == '/tf_static':
            tf_msg = load_tf_message(params_dir, timestamp)
            writer.write(topic, serialize_message(tf_msg), timestamp)
            continue

        m = CAMERA_INFO_RE.match(topic)
        if m:
            cam_id = m.group(1)
            if cam_id not in camera_infos:
                if cam_id not in warned_cameras:
                    print(
                        f'Warning: no calibration config for {cam_id}, passing through original',
                        file=sys.stderr,
                    )
                    warned_cameras.add(cam_id)
                writer.write(topic, raw, timestamp)
            else:
                orig = deserialize_message(raw, CameraInfo)
                new_msg = build_camera_info(camera_infos[cam_id], orig.header)
                writer.write(topic, serialize_message(new_msg), timestamp)
            continue

        writer.write(topic, raw, timestamp)

    print(f'Done: {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
