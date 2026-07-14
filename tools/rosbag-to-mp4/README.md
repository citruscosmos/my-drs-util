# rosbag-to-mp4

Convert camera topics inside a rosbag (MCAP) into MP4 video files.

## Dependencies

- `mcap` — MCAP file reading
- `rclpy` / `sensor_msgs` — ROS 2 message deserialization (requires a sourced ROS 2 environment)
- `numpy`, `opencv-python` — used only as a fallback to detect source resolution when a camera has no `camera_info` topic
- `ffmpeg` (system binary, e.g. `apt install ffmpeg`) — does the actual JPEG→H.264/MP4 encoding

## Usage

```bash
python3 bag_to_mp4.py <rosbag> <out_dir> [--cameras 0,2,4] [--resolution WIDTHxHEIGHT] [--fps FPS] [--start SEC] [--end SEC] [--crf N] [--preset NAME]
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `rosbag` | ✓ | — | Input `.mcap` file, or a split rosbag2 directory (`*_0.mcap`, `*_1.mcap`, ...) |
| `out_dir` | ✓ | — | Output directory for the MP4 files |
| `--cameras` | | auto-detect | Comma-separated camera IDs (e.g. `0,2,4`); omit to convert every camera found in the bag |
| `--resolution` | | source resolution | `WIDTHxHEIGHT` (must be even numbers, e.g. `1280x720`); omit to keep the camera's native resolution |
| `--fps` | | auto-estimated | Output frame rate; omit to estimate from message count / bag duration |
| `--start` / `--end` | | full bag | Trim range in UNIX seconds |
| `--crf` | | `18` | libx264 CRF (lower = higher quality, larger file) |
| `--preset` | | `medium` | libx264 preset (`ultrafast` .. `veryslow`) |

**Topics:** `/sensing/camera/camera{N}/image_raw/compressed` (JPEG), `/sensing/camera/camera{N}/camera_info` (for native resolution)

**Output file name:** `{rosbag_name}_{camera}_{width}x{height}.mp4`
placed directly under `out_dir`. `rosbag_name` is the input file's stem for a
single `.mcap`, or the directory name for a split rosbag2 directory.

**Examples:**

```bash
# All cameras, native resolution, single mcap
python3 bag_to_mp4.py sample.mcap ./mp4_out

# camera0 and camera2 only, downscaled to 1280x720
python3 bag_to_mp4.py sample.mcap ./mp4_out --cameras 0,2 --resolution 1280x720

# Split rosbag2 directory, trimmed to a 10s window
python3 bag_to_mp4.py /path/to/split_bag_dir ./mp4_out --start 1782377330 --end 1782377340
```

## Notes

- CompressedImage (JPEG) frames are streamed directly into `ffmpeg`'s `mjpeg`
  demuxer without a Python-side decode/re-encode step, so this scales to very
  large (multi-GB) bags with low memory overhead.
- The frame rate is estimated once from the *whole* bag's message count and
  duration (from the MCAP summary, no full scan needed), even when `--start`/
  `--end` trims the output — this keeps playback speed matching real time.
  Pass `--fps` explicitly if you need a precise rate.
