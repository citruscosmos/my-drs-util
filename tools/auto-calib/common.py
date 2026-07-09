"""Auto-calibration pipeline — shared utilities.

Reads single or split (rosbag2 directory) mcap bags, interpolates RTK/INS
trajectories (linear position + quaternion slerp), and provides small numpy
rotation helpers. The rest of the codebase (tools/lidar-camera) avoids scipy
because the system scipy is binary-incompatible with numpy 2.x, so all
rotation math here is implemented with numpy only.

Quaternion convention throughout: [x, y, z, w] (ROS / tf_static order).
"""
from __future__ import annotations

import glob
import os

import numpy as np

# PointField datatype enum -> numpy dtype
POINTFIELD_DTYPE = {
    1: "int8", 2: "uint8", 3: "int16", 4: "uint16",
    5: "int32", 6: "uint32", 7: "float32", 8: "float64",
}

# Topic names (fixed by the vehicle's Autoware config).
LIDAR_NAMES = ("front", "left", "rear", "right")
LIDAR_TOPICS = {n: f"/sensing/lidar/{n}/seyond_points" for n in LIDAR_NAMES}
ODOM_TOPIC = "/sensing/ins/oxts/odometry"
NAVSATFIX_TOPIC = "/sensing/ins/oxts/nav_sat_fix"
TF_STATIC_TOPIC = "/tf_static"

# Candidate per-point time field names, in priority order. Unit is inferred
# from the value range at read time (see infer_time_unit_ns), because it varies
# by bag: some bags expose t_us (microseconds) + timestamp (ns), others expose a
# single time_stamp (ns).
LIDAR_TIME_FIELDS = ("time_stamp", "t_us", "timestamp")

# drs_base_link -> oxts_link is a 180 deg rotation about X (q_xyzw = (1,0,0,0)),
# NOT identity. oxts is FRD, drs is FLU.
DRS_T_OXTS_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # [x,y,z,w]


# --------------------------------------------------------------------------- #
# Bag file resolution / message iteration
# --------------------------------------------------------------------------- #
def resolve_bag_files(path: str) -> list[str]:
    """Return an ordered list of .mcap files for a single file or a split dir.

    For a directory (split rosbag2), files are returned in sorted name order,
    which matches the recording order for the ``*_0.mcap, *_1.mcap, ...``
    naming scheme.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.mcap")))
        if not files:
            raise FileNotFoundError(f"no .mcap files under directory: {path}")
        return files
    if not os.path.isfile(path):
        raise FileNotFoundError(f"bag not found: {path}")
    return [path]


def iter_messages(files: list[str], topics: list[str] | None = None):
    """Yield (topic, t_ns, raw_bytes) across all files in order.

    ``t_ns`` is the log time from the mcap message record. Raw bytes are left
    undeserialized so callers pay only for the topics/messages they need.
    """
    from mcap.reader import make_reader

    for fp in files:
        with open(fp, "rb") as f:
            reader = make_reader(f)
            for _schema, channel, message in reader.iter_messages(topics=topics):
                yield channel.topic, message.log_time, message.data


def read_deserialized(files: list[str], topic: str, msg_type):
    """Yield deserialized messages for a single topic across all files."""
    from rclpy.serialization import deserialize_message

    for tp, _tns, data in iter_messages(files, [topic]):
        if tp == topic:
            yield deserialize_message(data, msg_type)


def bag_summary(files: list[str]) -> dict:
    """Aggregate per-topic message counts, schemas, and time span across files."""
    from mcap.reader import make_reader

    topic_count: dict[str, int] = {}
    topic_schema: dict[str, str] = {}
    tmin = tmax = None
    for fp in files:
        with open(fp, "rb") as f:
            summary = make_reader(f).get_summary()
        schemas = {s.id: s for s in summary.schemas.values()}
        counts = summary.statistics.channel_message_counts if summary.statistics else {}
        if summary.statistics:
            a = summary.statistics.message_start_time
            b = summary.statistics.message_end_time
            tmin = a if tmin is None else min(tmin, a)
            tmax = b if tmax is None else max(tmax, b)
        for ch in summary.channels.values():
            sc = schemas.get(ch.schema_id)
            topic_count[ch.topic] = topic_count.get(ch.topic, 0) + counts.get(ch.id, 0)
            topic_schema[ch.topic] = sc.name if sc else "?"
    return {
        "topic_count": topic_count,
        "topic_schema": topic_schema,
        "t_start_ns": tmin,
        "t_end_ns": tmax,
        "duration_s": (tmax - tmin) / 1e9 if tmin is not None else 0.0,
    }


# --------------------------------------------------------------------------- #
# PointCloud2 helpers
# --------------------------------------------------------------------------- #
def pointcloud2_to_array(msg) -> np.ndarray:
    """Return a structured numpy array view over a PointCloud2's fields."""
    names = [f.name for f in msg.fields]
    formats = [np.dtype(POINTFIELD_DTYPE[f.datatype]) for f in msg.fields]
    offsets = [f.offset for f in msg.fields]
    dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": msg.point_step}
    )
    n = msg.width * msg.height
    return np.frombuffer(bytes(msg.data), dtype=dtype, count=n)


def find_time_field(field_names) -> str | None:
    for name in LIDAR_TIME_FIELDS:
        if name in field_names:
            return name
    return None


def infer_time_unit_ns(values: np.ndarray) -> float:
    """Return the multiplier that converts per-point time values to nanoseconds.

    Both microsecond and nanosecond encodings span ~92 ms for one Seyond scan.
    In microseconds that span is ~9.2e4, in nanoseconds ~9.2e7. Threshold at
    1e6: any span at or below that must be microseconds.
    """
    v = np.asarray(values, dtype=np.float64)
    span = float(v.max() - v.min()) if v.size else 0.0
    return 1000.0 if span <= 1e6 else 1.0


# --------------------------------------------------------------------------- #
# Quaternion / transform math (numpy only, quaternion = [x,y,z,w])
# --------------------------------------------------------------------------- #
def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) [x,y,z,w] to rotation matrix. Supports batching."""
    q = quat_normalize(np.asarray(q, dtype=np.float64))
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a single 3x3 rotation matrix to quaternion [x,y,z,w]."""
    R = np.asarray(R, dtype=np.float64)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([x, y, z, w]))


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions [x,y,z,w] (supports batching)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast(a, b).shape, dtype=np.float64)
    out[..., 0] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 1] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 2] = aw * bz + ax * by - ay * bx + az * bw
    out[..., 3] = aw * bw - ax * bx - ay * by - az * bz
    return out


def make_transform(pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position + quaternion [x,y,z,w]."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_to_matrix(quat)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T


def euler_to_matrix_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX extrinsic RPY (rad) -> 3x3 (= Rz @ Ry @ Rx), matching tools/lidar-camera."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def tf_params_to_matrix(d: dict) -> np.ndarray:
    """{x,y,z,roll,pitch,yaw} -> 4x4 (ZYX extrinsic RPY, radians)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = euler_to_matrix_zyx(d.get("roll", 0.0), d.get("pitch", 0.0), d.get("yaw", 0.0))
    T[:3, 3] = [d.get("x", 0.0), d.get("y", 0.0), d.get("z", 0.0)]
    return T


def load_tf_override(path: str | None) -> dict[str, np.ndarray]:
    """Load drs_base_link->lidar_* extrinsic overrides from a multi_tf_static YAML.

    Format:  drs_base_link: {lidar_front: {x,y,z,roll,pitch,yaw}, ...}
    Returns {child_frame: 4x4}. Used to override a bag's /tf_static when it is
    wrong (e.g. the recorded front extrinsic disagrees with the accepted one).
    """
    if not path:
        return {}
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    out: dict[str, np.ndarray] = {}
    for _parent, children in data.items():
        if not isinstance(children, dict):
            continue
        for child, params in children.items():
            if isinstance(params, dict):
                out[child] = tf_params_to_matrix(params)
    return out


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def quat_slerp_batch(q0: np.ndarray, q1: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Vectorized slerp between paired quaternions [x,y,z,w].

    q0, q1: (M,4); frac: (M,) in [0,1]. Returns (M,4).
    """
    q0 = quat_normalize(np.atleast_2d(np.asarray(q0, dtype=np.float64)))
    q1 = quat_normalize(np.atleast_2d(np.asarray(q1, dtype=np.float64)))
    frac = np.asarray(frac, dtype=np.float64).reshape(-1)

    dot = np.sum(q0 * q1, axis=1)
    # Take the shorter arc.
    flip = dot < 0.0
    q1 = np.where(flip[:, None], -q1, q1)
    dot = np.clip(np.abs(dot), -1.0, 1.0)

    out = np.empty_like(q0)
    # Near-parallel: fall back to normalized linear interpolation (numerically stable).
    lin = dot > 0.9995
    if np.any(lin):
        q = q0[lin] + frac[lin, None] * (q1[lin] - q0[lin])
        out[lin] = quat_normalize(q)
    sl = ~lin
    if np.any(sl):
        theta0 = np.arccos(dot[sl])
        sin0 = np.sin(theta0)
        f = frac[sl]
        s0 = np.sin((1.0 - f) * theta0) / sin0
        s1 = np.sin(f * theta0) / sin0
        out[sl] = s0[:, None] * q0[sl] + s1[:, None] * q1[sl]
    return quat_normalize(out)


# --------------------------------------------------------------------------- #
# Trajectory interpolation
# --------------------------------------------------------------------------- #
class TrajectoryInterpolator:
    """Interpolates a time-ordered pose sequence (position linear, quat slerp).

    Stores positions in a local frame: the first sample's position is subtracted
    and returned via ``local_origin`` so callers can keep float64 precision even
    for ECEF-scale inputs.
    """

    def __init__(self, ts_ns: np.ndarray, pos: np.ndarray, quat: np.ndarray):
        order = np.argsort(ts_ns, kind="stable")
        self.ts = np.asarray(ts_ns, dtype=np.float64)[order]
        pos = np.asarray(pos, dtype=np.float64)[order]
        self.local_origin = pos[0].copy()
        self.pos = pos - self.local_origin
        self.quat = quat_normalize(np.asarray(quat, dtype=np.float64)[order])
        self.t_min = float(self.ts[0])
        self.t_max = float(self.ts[-1])

    def covers(self, t_ns) -> bool:
        t = np.asarray(t_ns, dtype=np.float64)
        return bool(np.all((t >= self.t_min) & (t <= self.t_max)))

    def max_gap_s(self) -> float:
        return float(np.max(np.diff(self.ts)) / 1e9) if self.ts.size > 1 else 0.0

    def interpolate(self, query_ns: np.ndarray):
        """Return (pos (M,3) local-frame, quat (M,4)) for query timestamps.

        Query times are clamped to the trajectory span before interpolation.
        """
        q = np.clip(np.asarray(query_ns, dtype=np.float64).reshape(-1), self.t_min, self.t_max)
        idx = np.searchsorted(self.ts, q, side="right") - 1
        idx = np.clip(idx, 0, len(self.ts) - 2)
        t0 = self.ts[idx]
        t1 = self.ts[idx + 1]
        denom = t1 - t0
        frac = np.where(denom > 0, (q - t0) / denom, 0.0)
        p0 = self.pos[idx]
        p1 = self.pos[idx + 1]
        pos = p0 + frac[:, None] * (p1 - p0)
        quat = quat_slerp_batch(self.quat[idx], self.quat[idx + 1], frac)
        return pos, quat


def load_odometry(files: list[str]):
    """Read /sensing/ins/oxts/odometry into arrays.

    Returns dict with ts_ns (N,), pos (N,3), quat (N,4)[xyzw], cov_xyz (N,3),
    frame_id, child_frame_id.
    """
    from nav_msgs.msg import Odometry

    ts, pos, quat, cov = [], [], [], []
    parent = child = None
    for m in read_deserialized(files, ODOM_TOPIC, Odometry):
        if parent is None:
            parent, child = m.header.frame_id, m.child_frame_id
        ts.append(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec)
        p = m.pose.pose.position
        o = m.pose.pose.orientation
        c = m.pose.covariance
        pos.append((p.x, p.y, p.z))
        quat.append((o.x, o.y, o.z, o.w))
        cov.append((c[0], c[7], c[14]))
    return {
        "ts_ns": np.array(ts, dtype=np.float64),
        "pos": np.array(pos, dtype=np.float64),
        "quat": np.array(quat, dtype=np.float64),
        "cov_xyz": np.array(cov, dtype=np.float64),
        "frame_id": parent,
        "child_frame_id": child,
    }


def detect_frame_kind(pos: np.ndarray) -> str:
    """'ECEF' if positions are geocentric-scale (~1e6 m), else 'LOCAL'."""
    mag = float(np.linalg.norm(pos, axis=1).mean()) if len(pos) else 0.0
    return "ECEF" if mag > 1e5 else "LOCAL"


def load_tf_static(files: list[str]) -> dict[tuple[str, str], np.ndarray]:
    """Return {(parent, child): 4x4 transform} from /tf_static (last wins)."""
    from tf2_msgs.msg import TFMessage

    out: dict[tuple[str, str], np.ndarray] = {}
    for m in read_deserialized(files, TF_STATIC_TOPIC, TFMessage):
        for t in m.transforms:
            tr = t.transform.translation
            q = t.transform.rotation
            out[(t.header.frame_id, t.child_frame_id)] = make_transform(
                (tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w)
            )
    return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    # Step 0 (validation gate)
    "rtk_cov_warn_m2": 0.1,
    "rtk_max_gap_sec": 0.5,
    "min_yaw_coverage_deg": 360.0,
    "require_cameras": False,
    # Step 1 (shared physical-plausibility gate)
    "rtk_max_speed_mps": 30.0,
    "rtk_max_reject_ratio": 0.1,
    "rtk_smoothing_window": 0,
    # Step 2
    "undistort_enabled": True,
    "lidar_scan_stride": 1,
    # GPU acceleration (currently used by Step 3's visibility range-filter,
    # the O(n_voxels x n_scans) bottleneck; auto-disables if no CUDA device).
    "gpu_enabled": True,
    "gpu_visibility_chunk_scans": 32,  # scans batched per GPU call (bounds VRAM use)
    # Step 3 (front map build + visibility-normalized voxel voting)
    "map_voxel_size": 0.1,
    "vote_threshold_ratio": 0.3,
    "map_scan_stride": 3,
    # Approximate Seyond visibility model for vote-denominator normalization.
    # A voxel counts toward a scan's denominator only if its center falls in the
    # sensor's range + FOV cone for that scan (so static structure the vehicle
    # was merely *near* but never faced is not penalized).
    "vote_max_range": 60.0,
    "vote_min_range": 1.5,
    "vote_hfov_deg": 120.0,
    "vote_vfov_deg": 40.0,
    # Occlusion-aware visibility (2026-07-04): the plain range+FOV cone has no
    # line-of-sight model, so on a tight-loop path it counts a near-path voxel
    # as "visible" from dozens of other loop positions up to vote_max_range
    # away even though the ground/other structure actually blocked the view
    # from there. That inflated visible denominator votes out nearly all
    # near-path structure (confirmed empirically: median occ 2-8 vs. median
    # visible 120-150 within 40m of the driven path). Fix: build a per-scan
    # min-range lookup from that scan's own real returns (in the sensor
    # frame, pre-world-transform), keyed by metric arc-length bins (azimuth/
    # elevation angle * that point's own range, so the bin footprint is a
    # constant ~vote_occlusion_footprint_m at any range -- a fixed angular
    # bin width was tried first, but couples near-field recovery to far-field
    # filter strength since its physical footprint scales with range; see
    # docs/architecture.md Step 3 notes). A voxel only counts as "visible"
    # for a scan if that scan actually measured a return in the same bin at
    # >= the voxel's own range (nothing closer was hit there) -- not merely
    # "geometrically within range+FOV of the origin". A bin with no return at
    # all is treated as no evidence (not visible), since a directional gap is
    # more often sensor scan-pattern sparsity than true occlusion.
    "vote_occlusion_enabled": True,
    "vote_occlusion_footprint_m": 0.15,  # target per-bin physical footprint [m]
    "vote_occlusion_tol": 0.5,          # [m] slack: visible if measured range >= voxel range - tol
    # Step 4 (lidar-to-lidar extrinsic; multi-scan point-to-plane Gauss-Newton
    # against the front map, coarse-to-fine over icp_multiscale_voxels).
    "icp_multiscale_voxels": [0.5, 0.2, 0.1],
    "icp_corr_dist_scale": 3.0,        # correspondence gate = voxel * this
    "icp_max_correspondence_dist": 2.0,  # absolute upper cap on the gate
    "icp_max_iterations": 30,          # GN/LM iters per scale
    "icp_rmse_threshold": 0.02,        # early-exit RMSE [m] (often below the map's own
                                       # smear floor, so convergence is step-based too)
    "icp_step_tol": 1e-4,              # converged when max(|rot|,|trans|) step < this
    "icp_huber_scale": 1.0,            # Huber delta = voxel * this (robust weighting)
    "icp_degeneracy_rel": 1.0e-4,      # project out Hessian dirs below eig_max*this
    "icp_method": "point_to_plane",
    "l2l_scan_stride": 12,             # other-lidar scan subsampling (2026-07-04: raised
                                       # 3->12 after verifying near-zero rmse impact at full
                                       # scale on the reference bag; ~4x fewer map/inter blocks)
    "l2l_points_per_scan": 1000,       # random subsample per scan (2026-07-04: 3000->1000,
                                       # same verification; still ample given ~120k pts/scan)
    "l2l_max_range": 60.0,             # crop other-scan points beyond this [m]
    "l2l_normal_radius_scale": 3.0,    # front-map normal radius = voxel * this
    # Step 4 term A (inter-lidar near-simultaneous overlap)
    "l2l_pair_dt_max": 0.06,           # max time gap for a co-observed scan pair [s]
    "l2l_pair_ref_scans": 3,           # neighbor scans aggregated into the local ref
    "l2l_interlidar_weight": 1.0,      # term A weight
    "l2l_map_weight": 1.0,             # term B weight
    # Step 5 (camera-to-lidar edge alignment)
    "cam_frame_stride": 5,             # use every Nth camera frame
    "cam_reproj_accept_px": 2.0,       # accept if median DT residual <= this [px]
    "canny_low": 50,
    "canny_high": 150,
    "dist_transform_max": 20.0,        # clamp the distance field [px]
    "image_scale": 0.5,                # downscale raw images before Canny/DT [0-1]; whole camera's frames are held in RAM at once, native 4K OOMs
    "lbfgsb_max_iter": 200,
    "lbfgsb_ftol": 1e-6,
    "cam_proj_max_points": 8000,       # projected map points per frame (subsample)
    "cam_trans_bound_m": 0.5,          # L-BFGS-B translation bound around init [m]
    "cam_rot_bound_rad": 0.3,          # L-BFGS-B rotation bound around init [rad]
    "cam_parallel_workers": 4,         # cameras optimized in parallel (multiprocessing)
    "cam_lbfgsb_eps": 1.0e-3,          # numerical-gradient step (pixel-scale cost)
    # Coarse-to-fine: Gaussian-blur the DT per stage (wide smooth basin -> sharpen).
    # Use with sparser/stronger Canny so the DT has contrast (dense edges -> flat DT).
    "cam_dt_blur_sigmas": [16.0, 8.0, 4.0, 0.0],
    # Step 5 LiDAR edge extraction (project structure edges, not the dense cloud;
    # the dense cloud aliases against cluttered image edges).
    "cam_use_edges": True,
    "cam_edge_ratio": 0.15,            # keep top fraction by edge score
    "cam_edge_knn": 15,                # neighbors for surface-variation / intensity
    "cam_edge_max_points": 600000,     # subsample the map before edge scoring
    # Step 5 v2 M0 (identifiability sweep check; step5_m0_check.py). Decoupled
    # from Step 4 / RTK: uses a single motion-compensated LiDAR scan per
    # camera frame, not the accumulated map.
    "m0_num_pairs": 8,                 # scan-image pairs per camera
    "m0_max_time_gap_s": 0.05,         # max |t_scan0 - t_img| to accept a pairing
    "m0_blur_var_min": 30.0,           # reject camera frames with Laplacian var below this
    "m0_sweep_trans_m": 0.20,          # +/- translation sweep span per axis [m]
    "m0_sweep_rot_deg": 3.0,           # +/- rotation sweep span per axis [deg]
    "m0_sweep_steps": 21,              # samples per axis sweep
    "m0_az_res_deg": 0.15,             # range-image azimuth bin size
    "m0_el_res_deg": 0.15,             # range-image elevation bin size
    "m0_occ_jump_abs_m": 0.3,          # occlusion-edge min absolute depth jump [m]
    "m0_occ_jump_rel": 0.03,           # occlusion-edge min relative depth jump (* range)
    "m0_planar_knn": 15,               # neighbors for single-scan surface-variation
    "m0_planar_ratio": 0.15,           # keep top fraction by surface-variation score
    "m0_planar_max_points": 40000,     # subsample scan before planar-edge scoring
    "m0_grad_blur_sigma": 2.0,         # blur sigma for the image gradient-magnitude map
    "m0_mi_bins": 16,                  # histogram bins for the MI cross-check
    "m0_mi_max_points": 5000,          # subsample per frame for the MI cross-check
    "m0_proj_margin": 80.0,            # frustum-crop margin [px]
    "m0_unimodal_disp_trans_m": 0.03,  # accept: translation-axis peak within this of init
    "m0_unimodal_disp_rot_deg": 0.3,   # accept: rotation-axis peak within this of init
    "m0_go_axes_min": 4,               # >= this many of 6 axes must pass for GO
    # Step 5 v3 (whole-trajectory accumulated camera point cloud -> LiDAR map
    # registration; RTK-fix prerequisite). step5_v3_cam_cloud.py / step5_v3_register.py.
    "v3_frame_stride": 2,              # camera frames used for KLT tracking (dense; NOT cam_frame_stride)
    "v3_klt_max_corners": 600,         # goodFeaturesToTrack cap per re-seed
    "v3_klt_quality": 0.01,            # goodFeaturesToTrack qualityLevel
    "v3_klt_min_dist": 12.0,           # goodFeaturesToTrack minDistance [px]
    "v3_klt_reseed_below": 300,        # re-seed new corners when active count drops below this
    "v3_track_min_len": 5,             # drop tracks shorter than this (too little parallax)
    "v3_track_max_len": 60,            # force-close a track at this length (bound triangulation cost)
    "v3_min_parallax_deg": 2.0,        # reject a triangulated point below this max ray-angle spread
    "v3_max_reproj_norm": 0.02,        # reject triangulation if mean normalized-coord reproj error exceeds this
    "v3_max_tracks_per_cam": 20000,    # cap accumulated tracks per camera (perf)
    "v3_triangulate_obs_stride": 4,    # use every Nth track observation for ray intersection
                                        # (widens baseline; final refinement still uses all obs)
    # Coarse-to-fine snap distance per outer iter (holds at the last value beyond the
    # schedule length); a fixed tight threshold can't recover rotation error beyond
    # ~1deg at typical triangulation depths (depth * tan(theta) blows past it).
    "v3_snap_max_dist_schedule": [3.0, 1.5, 0.75, 0.5],
    "v3_outer_iters": 5,               # reconstruct<->register outer iterations
    "v3_huber_px": 3.0,                # robust loss scale for the reprojection GN [px]
    "v3_trans_bound_m": 0.5,           # search bound around tf_static init [m]
    "v3_rot_bound_rad": 0.3,           # search bound around tf_static init [rad]
    "v3_ls_max_nfev": 60,              # least_squares max function evals per outer iter
    # Visualization / logging (shared across Steps 0-4; see viz.py). Every step
    # tees its stdout to <out_dir>/stepN_log.txt regardless of this flag; only
    # the matplotlib PNG output is gated by viz_enabled (kept separate so a
    # headless/CI run can skip plotting cost while still getting the log file).
    "viz_enabled": True,
    "viz_dpi": 110,
    "viz_max_points": 150000,          # cap points rendered per scatter plot
    "viz_max_scans_overlay": 60,       # cap scans concatenated for Step 4 overlay plots
    "viz_point_size": 0.15,            # marker size (pt^2) for dense point-cloud overlay plots
                                       # (Step2 sample heatmap, Step3 map/removed, Step4 overlay);
                                       # kept small + semi-transparent (viz_point_alpha) so dense
                                       # clusters don't merge into solid blobs that hide fine
                                       # misalignment
    "viz_point_alpha": 0.4,
}


def load_config(path: str | None = None) -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        import yaml

        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg.update(user)
    return cfg
