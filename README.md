# my-rosbag-util

A utility toolbox for post-processing DRS rosbag (MCAP) files.

## Tools

### `tools/replace_params.py` — Calibration replacer

Replaces `/tf_static` and `/sensing/camera/camera*/camera_info` topic payloads in an existing MCAP bag with calibration data from a config directory, producing a corrected output bag.

**Use case:** When calibration is refined after a recording session, re-recording is expensive. This tool applies the corrected calibration as a post-process step.

#### Usage

```bash
python3 tools/replace_params.py \
    --input  bag.mcap \
    --output bag_updated.mcap \
    --params /opt/drs/config/params/default \
    [--force] \
    [--compress]
```

| Option | Description |
|---|---|
| `--input` | Input MCAP bag path |
| `--output` | Output MCAP bag path |
| `--params` | Config directory containing `multi_tf_static.yaml` and `camera*/camera_info.yaml` |
| `--force` | Overwrite output if it already exists |
| `--compress` | Compress output with zstd |

#### Config directory structure

```
default/
├── multi_tf_static.yaml          # Full vehicle TF tree (RPY radians, extrinsic XYZ)
├── camera0/
│   └── camera_info.yaml          # camera_calibration_parsers format
├── camera1/
│   └── camera_info.yaml
└── ...
```

Local reference: `~/data_recording_system/src/individual_params/config/default/`

#### Behavior

- `/tf_static`: Rebuilt from `multi_tf_static.yaml` and written once. Each `TransformStamped` stamp is copied from the original bag message timestamp.
- `/sensing/camera/cameraX/camera_info`: K, D, P, R matrices replaced from `cameraX/camera_info.yaml`. The `frame_id` is preserved from the original message.
- Camera topics with no matching config entry: passed through unchanged with a warning on stderr.
- All other topics: passed through unchanged.

#### Processing multiple files

Use a shell loop to convert all MCAP files in a directory:

```bash
for f in /path/to/input/*.mcap; do
    python3 tools/replace_params.py \
        --input  "$f" \
        --output "/path/to/output/$(basename "$f")" \
        --params /opt/drs/config/params/default
done
```

#### Dependencies

Requires a ROS 2 Humble environment.

```
rosbag2_py
rclpy
sensor_msgs
tf2_msgs
geometry_msgs
builtin_interfaces
pyyaml
```

## Installation

1. Source your ROS 2 Humble workspace:

```bash
source /opt/ros/humble/setup.bash
```

2. Install the Python dependency:

```bash
pip install pyyaml
```

3. Clone this repository:

```bash
git clone https://github.com/citruscosmos/my-rosbag-util.git
cd my-rosbag-util
```

## Tests

```bash
pytest tests/
```

Unit tests with a synthetic MCAP fixture are in `tests/test_replace_params.py`.

## Third-Party Dependencies

This repository is [MIT licensed](LICENSE). All code in this repository is original; no third-party source code is vendored or embedded here. Third-party functionality is used only via standard package imports — either pip-installed packages or a sourced ROS 2 environment — so none of the licenses below impose any additional restriction on this project's own license.

| Package | License | Used by |
|---|---|---|
| numpy | BSD-3-Clause | all tools |
| scipy | BSD-3-Clause | `tools/auto-calib` |
| open3d | MIT | `tools/auto-calib` (Step 4 ICP) |
| opencv-python (OpenCV) | Apache-2.0 | `tools/auto-calib` (Step 5), `tools/lidar-camera` |
| mcap / mcap-ros2-support | Apache-2.0 | MCAP bag reading (all tools) |
| pyyaml | MIT | config / TF YAML parsing |
| matplotlib | Matplotlib License (BSD-style) | `tools/auto-calib`, `tools/lidar-camera` (`tune_extrinsic.py`) |
| PySide6 (Qt for Python) | LGPLv3 (or commercial) | `tools/lidar-camera/tune_extrinsic.py` only |
| ROS 2 Humble (`rclpy`, `rosbag2_py`, `sensor_msgs`, `tf2_msgs`, `geometry_msgs`, `builtin_interfaces`, `nav_msgs`) | Apache-2.0 | bag/message I/O (all tools); not pip-installed — provided by a sourced ROS 2 Humble environment |

Note on PySide6: it is used only as a dynamically imported library (pip package) in one optional GUI tool; no Qt source is bundled in this repository, so LGPLv3's linking terms are satisfied without any special build steps.
