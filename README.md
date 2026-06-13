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

## Tests

```bash
pytest tests/
```

Unit tests with a synthetic MCAP fixture are in `tests/test_replace_params.py`.
