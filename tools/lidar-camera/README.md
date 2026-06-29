# lidar-camera

Tools for extracting camera images and LiDAR point clouds from MCAP bags and projecting LiDAR onto camera images.

## Dependencies

- `mcap` — MCAP file reading
- `rclpy` / `sensor_msgs` — ROS 2 message deserialization (requires a ROS 2 environment)
- `numpy`
- `opencv-python`

## Scripts

| Script | Role |
|--------|------|
| `extract_cameras.py` | Extract camera images (JPEG) from MCAP |
| `extract_lidar_pcd.py` | Extract LiDAR point clouds (PCD) from MCAP |
| `project_lidar_to_cam.py` | Project LiDAR point clouds onto camera images |

---

## Command Flow

### Setup: Create output directories

```bash
mkdir -p output/camera{2,3,6,7,8,9,10,11}
mkdir -p output/lidar_{front,left,rear,right}
```

---

### Step 1: Extract camera images

Extracts compressed images (JPEG) from an MCAP file for the specified cameras.
File names are nanosecond timestamps from `header.stamp` (`{sec}{nanosec:09d}.jpg`).

```bash
python3 extract_cameras.py <mcap> <out_root> [--cams <camera IDs>] [--limit <N>]
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `mcap` | ✓ | — | Input MCAP file path |
| `out_root` | ✓ | — | Output root directory |
| `--cams` | | auto-detect | Comma-separated camera IDs to extract; omit to use all cameras found in the MCAP |
| `--limit` | | `0` (unlimited) | Maximum frames per camera |
| `--start` | | `None` | Start time in Unix seconds (e.g. `1751234567.0`) |
| `--end` | | `None` | End time in Unix seconds |

**Topics:** `/sensing/camera/camera{N}/image_raw/compressed`

**Examples:**

```bash
# Extract all frames from camera8–11 (default)
python3 extract_cameras.py sample.mcap ./output

# Extract 10 frames each from camera2,3,6,7 (for validation)
python3 extract_cameras.py sample.mcap ./output --cams 2,3,6,7 --limit 10
```

**Output structure:**

```
output/
  camera2/
    1751234567000000000.jpg
    ...
  camera3/
    ...
  camera8/
    ...
```

---

### Step 2: Extract LiDAR point clouds

Extracts point clouds from four LiDAR sensors in binary PCD format.
Fields are `x y z intensity` (float32). Non-finite points (NaN/Inf) are removed.

```bash
python3 extract_lidar_pcd.py <mcap> <out_root> [--limit <N>]
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `mcap` | ✓ | — | Input MCAP file path |
| `out_root` | ✓ | — | Output root directory |
| `--limit` | | `0` (unlimited) | Maximum frames per LiDAR |
| `--start` | | `None` | Start time in Unix seconds (e.g. `1751234567.0`) |
| `--end` | | `None` | End time in Unix seconds |

**Topics:**

| Topic | Output directory |
|-------|-----------------|
| `/sensing/lidar/front/seyond_points` | `lidar_front/` |
| `/sensing/lidar/left/seyond_points` | `lidar_left/` |
| `/sensing/lidar/rear/seyond_points` | `lidar_rear/` |
| `/sensing/lidar/right/seyond_points` | `lidar_right/` |

**Examples:**

```bash
# Extract all frames
python3 extract_lidar_pcd.py sample.mcap ./output

# Extract 5 frames per LiDAR (for validation)
python3 extract_lidar_pcd.py sample.mcap ./output --limit 5
```

**Output structure:**

```
output/
  lidar_front/
    1751234567000000000.pcd
    ...
  lidar_left/
    ...
  lidar_rear/
    ...
  lidar_right/
    ...
```

---

### Step 3: Project LiDAR onto camera images

Uses the images and point clouds from Steps 1 and 2 to overlay LiDAR points on camera images.
For each LiDAR frame, the nearest-timestamp camera image is matched automatically.

```bash
python3 project_lidar_to_cam.py <cam> <out_root> [--sample <N>] [--limit <N>] [--alpha <float>]
```

**Arguments:**

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `cam` | ✓ | — | Target camera (`camera2` / `camera3` / `camera6` / `camera7`) |
| `out_root` | ✓ | — | Output root directory |
| `--sample` | | `0` (disabled) | Process N evenly-spaced frames (for validation) |
| `--limit` | | `0` (unlimited) | Stop after N frames from the beginning |
| `--alpha` | | `0.45` | Point opacity (0 = fully transparent, 1 = opaque) |
| `--start` | | `None` | Start time in Unix seconds (e.g. `1751234567.0`) |
| `--end` | | `None` | End time in Unix seconds |
| `--cam-dir` | | `CAM_CONFIGS` | Camera image directory (overrides hardcoded path) |
| `--lidar-dir` | | `CAM_CONFIGS` | LiDAR PCD directory (overrides hardcoded path) |

**Camera–LiDAR pairs:**

| Camera | LiDAR |
|--------|-------|
| camera2, camera3 | lidar_right |
| camera6, camera7 | lidar_left |

**Output modes:**

| Directory | Processing |
|-----------|-----------|
| `distort/` | Original image with points projected using the distortion model (`cv2.projectPoints`) |
| `undistort/` | Undistorted image with pinhole projection (`cv2.remap` + new camera matrix) |

Point colors are mapped via JET colormap with intensity normalized to 0–40.

**Examples:**

```bash
# Process all frames for camera2
python3 project_lidar_to_cam.py camera2 ./proj_output

# Process 10 evenly-spaced frames for camera3 (validation)
python3 project_lidar_to_cam.py camera3 ./proj_output --sample 10

# Increase point opacity for visual inspection
python3 project_lidar_to_cam.py camera2 ./proj_output --sample 10 --alpha 0.8
```

> **Note:** Input directory paths are hardcoded in `CAM_CONFIGS` inside `project_lidar_to_cam.py`.
> If your Step 1/2 output location differs, update `CAM_CONFIGS` accordingly.

**Output structure:**

```
proj_output/
  distort/
    1751234567000000000.jpg
    ...
  undistort/
    1751234567000000000.jpg
    ...
```

---

## Overall Flow

```
sample.mcap
    │
    ├─ extract_cameras.py   ──→  output/camera{N}/*.jpg
    │
    └─ extract_lidar_pcd.py ──→  output/lidar_{front,left,rear,right}/*.pcd
                                           │
                                project_lidar_to_cam.py
                                           │
                                proj_output/distort/*.jpg
                                proj_output/undistort/*.jpg
```

## Known Limitations

- Camera intrinsics are shared across all cameras (camera2 values are reused for camera3, 6, and 7).
