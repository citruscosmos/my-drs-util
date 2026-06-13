"""Tests for tools/replace_params.py"""
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Quaternion, TransformStamped, Vector3
from rclpy.serialization import deserialize_message, serialize_message
import rosbag2_py
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage


TOOLS_DIR = Path(__file__).parent.parent / 'tools'
REPLACE_PARAMS = TOOLS_DIR / 'replace_params.py'


# ── Bag helpers ───────────────────────────────────────────────────────────────

def write_bag(path: Path, messages: list) -> None:
    """Write a minimal MCAP bag. messages: list of (topic, type_str, raw, timestamp_ns)."""
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )
    registered = set()
    for topic, type_str, raw, timestamp in messages:
        if topic not in registered:
            writer.create_topic(rosbag2_py.TopicMetadata(
                name=topic,
                type=type_str,
                serialization_format='cdr',
            ))
            registered.add(topic)
        writer.write(topic, raw, timestamp)


def read_bag_messages(path: Path) -> list:
    """Read all messages from a bag. Returns list of (topic, raw, timestamp)."""
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(path), storage_id='mcap'),
        rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        ),
    )
    msgs = []
    while reader.has_next():
        msgs.append(reader.read_next())
    return msgs


# ── Message builders ──────────────────────────────────────────────────────────

def make_camera_info_raw(frame_id: str, k_val: float = 999.0) -> bytes:
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.width = 100
    msg.height = 100
    msg.distortion_model = 'plumb_bob'
    msg.k = [k_val] * 9
    msg.d = [0.0] * 5
    msg.p = [k_val] * 12
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    return serialize_message(msg)


def make_tf_static_raw() -> bytes:
    ts = TransformStamped()
    ts.header.stamp = Time(sec=1, nanosec=0)
    ts.header.frame_id = 'old_parent'
    ts.child_frame_id = 'old_child'
    ts.transform.translation = Vector3(x=99.0, y=99.0, z=99.0)
    ts.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return serialize_message(TFMessage(transforms=[ts]))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def params_dir(tmp_path) -> Path:
    """Minimal params dir: multi_tf_static.yaml with 3 known transforms, camera0–2 infos."""
    d = tmp_path / 'params'
    d.mkdir()

    tf_data = {
        'base_link': {
            'child_a': {'x': 1.0, 'y': 2.0, 'z': 3.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0},
        },
        'parent_b': {
            'child_b': {'x': 4.0, 'y': 5.0, 'z': 6.0, 'roll': 0.1, 'pitch': 0.2, 'yaw': 0.3},
        },
        'parent_c': {
            'child_c': {'x': 7.0, 'y': 8.0, 'z': 9.0, 'roll': 0.0, 'pitch': 0.5, 'yaw': 1.0},
        },
    }
    (d / 'multi_tf_static.yaml').write_text(yaml.dump(tf_data))

    # camera0=10.0, camera1=20.0, camera2=30.0  (K[0] is the distinguishing value)
    for i, cam_id in enumerate(['camera0', 'camera1', 'camera2']):
        k_val = float((i + 1) * 10)
        cam_dir = d / cam_id
        cam_dir.mkdir()
        cam_data = {
            'image_width': 640,
            'image_height': 480,
            'camera_name': cam_id,
            'camera_matrix': {'rows': 3, 'cols': 3, 'data': [k_val] * 9},
            'distortion_model': 'plumb_bob',
            'distortion_coefficients': {'rows': 1, 'cols': 5, 'data': [0.0] * 5},
            'projection_matrix': {'rows': 3, 'cols': 4, 'data': [k_val] * 12},
            'rectification_matrix': {'rows': 3, 'cols': 3,
                                      'data': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]},
        }
        (cam_dir / 'camera_info.yaml').write_text(yaml.dump(cam_data))

    return d


@pytest.fixture
def input_bag(tmp_path) -> Path:
    """Synthetic bag: /tf_static, camera0, camera1, camera9 (absent from config), /imu/data."""
    bag_path = tmp_path / 'input.mcap'
    write_bag(bag_path, [
        ('/tf_static', 'tf2_msgs/msg/TFMessage',
         make_tf_static_raw(), 1_000_000_000),
        ('/sensing/camera/camera0/camera_info', 'sensor_msgs/msg/CameraInfo',
         make_camera_info_raw('camera0_optical_frame'), 1_100_000_000),
        ('/sensing/camera/camera1/camera_info', 'sensor_msgs/msg/CameraInfo',
         make_camera_info_raw('camera1_optical_frame'), 1_200_000_000),
        ('/sensing/camera/camera9/camera_info', 'sensor_msgs/msg/CameraInfo',
         make_camera_info_raw('camera9_optical_frame'), 1_300_000_000),
        ('/imu/data', 'std_msgs/msg/String',
         serialize_message(String(data='passthrough')), 1_400_000_000),
    ])
    return bag_path


def run_replace(input_path, output_path, params, extra_args=None):
    cmd = [
        sys.executable, str(REPLACE_PARAMS),
        '--input', str(input_path),
        '--output', str(output_path),
        '--params', str(params),
    ]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


# ── Test 1: /tf_static replacement ───────────────────────────────────────────

class TestTfStaticReplacement:
    def test_three_transform_pairs(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        result = run_replace(input_bag, output, params_dir)
        assert result.returncode == 0, result.stderr

        msgs = read_bag_messages(output)
        tf_msgs = [(t, r) for t, r, _ in msgs if t == '/tf_static']
        assert len(tf_msgs) == 1, 'Expected exactly one /tf_static message in output'

        tf_msg = deserialize_message(tf_msgs[0][1], TFMessage)
        by_child = {ts.child_frame_id: ts for ts in tf_msg.transforms}

        # All three pairs from multi_tf_static.yaml must be present
        assert 'child_a' in by_child
        assert 'child_b' in by_child
        assert 'child_c' in by_child

        # spot-check child_a translation and zero-rotation
        ts_a = by_child['child_a']
        assert ts_a.header.frame_id == 'base_link'
        assert abs(ts_a.transform.translation.x - 1.0) < 1e-6
        assert abs(ts_a.transform.translation.y - 2.0) < 1e-6
        assert abs(ts_a.transform.translation.z - 3.0) < 1e-6
        assert abs(ts_a.transform.rotation.w - 1.0) < 1e-6  # identity

        # spot-check child_b parent frame and translation
        ts_b = by_child['child_b']
        assert ts_b.header.frame_id == 'parent_b'
        assert abs(ts_b.transform.translation.x - 4.0) < 1e-6

        # spot-check child_c translation
        ts_c = by_child['child_c']
        assert abs(ts_c.transform.translation.z - 9.0) < 1e-6

    def test_old_tf_static_values_not_present(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        run_replace(input_bag, output, params_dir)

        msgs = read_bag_messages(output)
        tf_msgs = [(t, r) for t, r, _ in msgs if t == '/tf_static']
        tf_msg = deserialize_message(tf_msgs[0][1], TFMessage)

        # original bag had translation (99, 99, 99) — must not appear
        for ts in tf_msg.transforms:
            assert abs(ts.transform.translation.x - 99.0) > 1.0

    def test_tf_static_stamp_from_bag_timestamp(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        run_replace(input_bag, output, params_dir)

        msgs = read_bag_messages(output)
        tf_msgs = [(t, r, ts) for t, r, ts in msgs if t == '/tf_static']
        tf_msg = deserialize_message(tf_msgs[0][1], TFMessage)

        # bag /tf_static timestamp was 1_000_000_000 ns = sec=1, nanosec=0
        for ts in tf_msg.transforms:
            assert ts.header.stamp.sec == 1
            assert ts.header.stamp.nanosec == 0


# ── Test 2: camera_info K matrix replacement ─────────────────────────────────

class TestCameraInfoReplacement:
    def test_k_matrix_from_config_frame_id_preserved(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        result = run_replace(input_bag, output, params_dir)
        assert result.returncode == 0, result.stderr

        msgs = read_bag_messages(output)
        camera_msgs = {t: deserialize_message(r, CameraInfo)
                       for t, r, _ in msgs if t.endswith('/camera_info')
                       and 'camera9' not in t}

        # camera0: K[0] from config = 10.0, frame_id preserved from original
        cam0 = camera_msgs['/sensing/camera/camera0/camera_info']
        assert cam0.k[0] == pytest.approx(10.0)
        assert cam0.header.frame_id == 'camera0_optical_frame'

        # camera1: K[0] from config = 20.0
        cam1 = camera_msgs['/sensing/camera/camera1/camera_info']
        assert cam1.k[0] == pytest.approx(20.0)
        assert cam1.header.frame_id == 'camera1_optical_frame'

    def test_original_k_values_overwritten(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        run_replace(input_bag, output, params_dir)

        msgs = read_bag_messages(output)
        cam0 = next(
            deserialize_message(r, CameraInfo)
            for t, r, _ in msgs if t == '/sensing/camera/camera0/camera_info'
        )
        # original K was 999.0, config K is 10.0
        assert cam0.k[0] != pytest.approx(999.0)

    def test_width_height_from_config(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        run_replace(input_bag, output, params_dir)

        msgs = read_bag_messages(output)
        cam0 = next(
            deserialize_message(r, CameraInfo)
            for t, r, _ in msgs if t == '/sensing/camera/camera0/camera_info'
        )
        assert cam0.width == 640
        assert cam0.height == 480


# ── Test 3: Camera absent from config → passthrough with warning ──────────────

class TestCameraAbsentFromConfig:
    def test_unknown_camera_passes_through_unchanged(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        result = run_replace(input_bag, output, params_dir)
        assert result.returncode == 0, result.stderr

        msgs = read_bag_messages(output)
        cam9_msgs = [(t, r) for t, r, _ in msgs
                     if t == '/sensing/camera/camera9/camera_info']
        assert len(cam9_msgs) == 1, 'camera9 topic must be present in output'

        cam9 = deserialize_message(cam9_msgs[0][1], CameraInfo)
        assert cam9.k[0] == pytest.approx(999.0)  # original value preserved

    def test_unknown_camera_warning_printed(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        result = run_replace(input_bag, output, params_dir)
        assert 'camera9' in result.stderr

    def test_other_topics_still_pass_through(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        run_replace(input_bag, output, params_dir)

        msgs = read_bag_messages(output)
        imu_msgs = [(t, r) for t, r, _ in msgs if t == '/imu/data']
        assert len(imu_msgs) == 1
        assert deserialize_message(imu_msgs[0][1], String).data == 'passthrough'


# ── Test 4: Output collision guard ───────────────────────────────────────────

class TestOutputCollisionGuard:
    def test_error_without_force(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        output.touch()
        result = run_replace(input_bag, output, params_dir)
        assert result.returncode != 0
        assert 'force' in result.stderr.lower()

    def test_success_with_force(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'out.mcap'
        output.touch()
        result = run_replace(input_bag, output, params_dir, extra_args=['--force'])
        assert result.returncode == 0, result.stderr
        assert output.stat().st_size > 0

    def test_no_error_when_output_does_not_exist(self, tmp_path, params_dir, input_bag):
        output = tmp_path / 'fresh.mcap'
        result = run_replace(input_bag, output, params_dir)
        assert result.returncode == 0, result.stderr


# ── Test 5: Config has camera, bag does not ───────────────────────────────────

class TestConfigHasCameraBagDoesNot:
    def test_no_crash_when_bag_missing_configured_camera(self, tmp_path, params_dir):
        # params has camera0, camera1, camera2 — bag only has camera0
        bag_path = tmp_path / 'input_partial.mcap'
        write_bag(bag_path, [
            ('/tf_static', 'tf2_msgs/msg/TFMessage',
             make_tf_static_raw(), 1_000_000_000),
            ('/sensing/camera/camera0/camera_info', 'sensor_msgs/msg/CameraInfo',
             make_camera_info_raw('camera0_optical_frame'), 1_100_000_000),
        ])

        output = tmp_path / 'out.mcap'
        result = run_replace(bag_path, output, params_dir)
        assert result.returncode == 0, result.stderr

    def test_missing_camera_topic_not_injected(self, tmp_path, params_dir):
        bag_path = tmp_path / 'input_partial.mcap'
        write_bag(bag_path, [
            ('/tf_static', 'tf2_msgs/msg/TFMessage',
             make_tf_static_raw(), 1_000_000_000),
            ('/sensing/camera/camera0/camera_info', 'sensor_msgs/msg/CameraInfo',
             make_camera_info_raw('camera0_optical_frame'), 1_100_000_000),
        ])

        output = tmp_path / 'out.mcap'
        run_replace(bag_path, output, params_dir)

        msgs = read_bag_messages(output)
        topics = {t for t, _, _ in msgs}
        # camera1 and camera2 are in config but not in bag — must NOT appear in output
        assert '/sensing/camera/camera1/camera_info' not in topics
        assert '/sensing/camera/camera2/camera_info' not in topics
