#!/usr/bin/env python3
"""MCAP 内の sensor_msgs/msg/NavSatFix トピックを検出し、GPS経路マップをPNG保存する。

単独の .mcap ファイル、または複数の *.mcap + metadata.yaml が入った
split rosbag2 ディレクトリのどちらも指定できる。トピック名は問わず、
スキーマが sensor_msgs/msg/NavSatFix であるトピックを自動検出する。

oxts_msgs/msg/Ncom トピックがあれば、そこから OxTS の GPS position mode
(RTK Float / RTK Integer 等、NavSatFix.status では区別できない詳細な
ステータス) を復号し、それで色分けする。無ければ NavSatFix.status.status
にフォールバックする。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import NavSatFix

NAVSATFIX_TYPE = "sensor_msgs/msg/NavSatFix"
NCOM_TYPE = "oxts_msgs/msg/Ncom"
WGS84_R = 6378137.0  # equatorial radius [m]

# sensor_msgs/msg/NavSatStatus.status values -> (color, label). Fallback
# coloring, used only when the bag has no oxts_msgs/msg/Ncom topic to decode
# the richer GPS position mode from (see GPS_POS_MODE_STYLE below).
STATUS_STYLE = {
    -1: ("red", "NO_FIX"),
    0: ("orange", "FIX"),
    1: ("gold", "SBAS_FIX"),
    2: ("green", "GBAS_FIX"),
}

# OxTS NCOM "GPS position mode" enum (COM_GPS_XMODE_TYPE_NAME in OxTS's
# reference NComRxC.c decoder), distinguishing RTK float vs. RTK integer
# (ambiguity-resolved) solutions that sensor_msgs/msg/NavSatStatus cannot.
GPS_POS_MODE_NAMES = [
    "None", "Search", "Doppler", "SPS", "Differential", "RTK Float", "RTK Integer",
    "WAAS", "Omnistar", "Omnistar HP", "No Data", "Blanked", "Doppler (PP)", "SPS (PP)",
    "Differential (PP)", "RTK Float (PP)", "RTK Integer (PP)", "Omnistar XP", "CDGPS",
    "Not Recognised", "gxDoppler", "gxSPS", "gxDifferential", "gxFloat", "gxInteger",
    "ixDoppler", "ixSPS", "ixDifferential", "ixFloat", "ixInteger", "Unknown",
]
_GPS_POS_MODE_NO_FIX = {0, 1, 10, 11, 19, 30}
_GPS_POS_MODE_COARSE = {2, 3, 12, 13, 20, 21, 25, 26}
_GPS_POS_MODE_DIFFERENTIAL = {4, 7, 8, 9, 14, 17, 18, 22, 27}
_GPS_POS_MODE_RTK_FLOAT = {5, 15, 23, 28}
_GPS_POS_MODE_RTK_INTEGER = {6, 16, 24, 29}


def _gps_pos_mode_color(idx: int) -> str:
    if idx in _GPS_POS_MODE_RTK_INTEGER:
        return "green"
    if idx in _GPS_POS_MODE_RTK_FLOAT:
        return "dodgerblue"
    if idx in _GPS_POS_MODE_DIFFERENTIAL:
        return "gold"
    if idx in _GPS_POS_MODE_COARSE:
        return "orange"
    return "red"  # no-fix / unrecognised / unknown


GPS_POS_MODE_STYLE = {i: (_gps_pos_mode_color(i), name) for i, name in enumerate(GPS_POS_MODE_NAMES)}
GPS_POS_MODE_UNKNOWN = -1  # sentinel: no Ncom pos-mode update seen yet at this timestamp
GPS_POS_MODE_STYLE[GPS_POS_MODE_UNKNOWN] = ("gray", "Unknown (pre-Ncom)")

# Byte offsets into the 72-byte raw OxTS NCOM packet (oxts_msgs/msg/Ncom.raw_packet),
# per OxTS's NComRxC.h PacketIndexes / NComRxC.c DecodeExtra0. The GPS position mode
# lives in the "channel status" block, but only when the packet's rotating channel
# index equals 0 ("time, num sats, position, velocity and dual-antenna modes").
NCOM_CHANNEL_INDEX_OFFSET = 62
NCOM_CHANNEL_STATUS_OFFSET = 63
NCOM_POS_MODE_BYTE_OFFSET = NCOM_CHANNEL_STATUS_OFFSET + 5  # = 68


def resolve_mcap_files(path: str) -> list[str]:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.mcap")))
        if not files:
            raise FileNotFoundError(f"no .mcap files under directory: {path}")
        return files
    if not os.path.isfile(path):
        raise FileNotFoundError(f"file not found: {path}")
    return [path]


def discover_topics_by_schema(files: list[str], schema_name: str) -> list[str]:
    """Return topic names whose schema matches schema_name, across all files."""
    topics = set()
    for fp in files:
        with open(fp, "rb") as f:
            summary = make_reader(f).get_summary()
        if not summary:
            continue
        schemas = {s.id: s for s in summary.schemas.values()}
        for ch in summary.channels.values():
            sc = schemas.get(ch.schema_id)
            if sc and sc.name == schema_name:
                topics.add(ch.topic)
    return sorted(topics)


def discover_navsatfix_topics(files: list[str]) -> list[str]:
    """Return topic names whose schema is sensor_msgs/msg/NavSatFix, across all files."""
    return discover_topics_by_schema(files, NAVSATFIX_TYPE)


def read_navsatfix(files: list[str], topic: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (N,5) [t_ns, lat, lon, alt, cov_horiz_std_m] sorted by time, and (N,) status.status.

    cov_horiz_std_m is sqrt(mean(cov_xx, cov_yy)) from position_covariance, i.e. an
    isotropic 1-sigma horizontal position uncertainty estimate in meters. It is 0
    when position_covariance_type is UNKNOWN (covariance not populated).
    """
    ts, lat, lon, alt, cov_std, status = [], [], [], [], [], []
    for fp in files:
        with open(fp, "rb") as f:
            reader = make_reader(f)
            for _schema, channel, message in reader.iter_messages(topics=[topic]):
                msg = deserialize_message(message.data, NavSatFix)
                ts.append(message.log_time)
                lat.append(msg.latitude)
                lon.append(msg.longitude)
                alt.append(msg.altitude)
                status.append(msg.status.status)
                if msg.position_covariance_type == 0:  # COVARIANCE_TYPE_UNKNOWN
                    cov_std.append(0.0)
                else:
                    cov_xx, cov_yy = msg.position_covariance[0], msg.position_covariance[4]
                    cov_std.append(np.sqrt(max((cov_xx + cov_yy) / 2.0, 0.0)))

    data = np.column_stack([
        np.asarray(ts, dtype=np.int64),
        np.asarray(lat, dtype=np.float64),
        np.asarray(lon, dtype=np.float64),
        np.asarray(alt, dtype=np.float64),
        np.asarray(cov_std, dtype=np.float64),
    ])
    status_arr = np.asarray(status, dtype=np.int64)
    order = np.argsort(data[:, 0], kind="stable")
    return data[order], status_arr[order]


def read_ncom_gps_pos_mode(files: list[str], topic: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_ns, pos_mode_idx) for every valid GPS-position-mode update on `topic`.

    Decodes the raw 72-byte OxTS NCOM packet directly from its embedded
    ros2msg schema (via mcap_ros2), so this needs no oxts_msgs package
    installed. Only packets whose rotating channel index is 0 carry the GPS
    position mode; other channels are skipped. A packet marking the field
    invalid (top bit of the status byte set) is also skipped, since the
    caller is expected to forward-fill the last known value across time.
    """
    from mcap_ros2.decoder import DecoderFactory

    ts, modes = [], []
    for fp in files:
        with open(fp, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _schema, _channel, message, ros_msg in reader.iter_decoded_messages(topics=[topic]):
                raw = ros_msg.raw_packet
                if len(raw) <= NCOM_POS_MODE_BYTE_OFFSET:
                    continue
                if (raw[NCOM_CHANNEL_INDEX_OFFSET] & 0xFF) != 0:
                    continue
                status_byte = raw[NCOM_POS_MODE_BYTE_OFFSET]
                if status_byte & 0x80:
                    continue  # marked invalid for this update
                idx = status_byte & 0x7F
                if idx >= len(GPS_POS_MODE_NAMES):
                    continue
                ts.append(message.log_time)
                modes.append(idx)

    t_ns = np.asarray(ts, dtype=np.int64)
    mode = np.asarray(modes, dtype=np.int64)
    order = np.argsort(t_ns, kind="stable")
    return t_ns[order], mode[order]


def asof_forward_fill(query_ts: np.ndarray, ref_ts: np.ndarray, ref_val: np.ndarray,
                       fill_value):
    """For each t in query_ts, return ref_val at the latest ref_ts <= t.

    ref_ts must be sorted. Queries before the first ref_ts (or when ref_ts is
    empty) get fill_value.
    """
    if len(ref_ts) == 0:
        return np.full(len(query_ts), fill_value)
    idx = np.searchsorted(ref_ts, query_ts, side="right") - 1
    return np.where(idx >= 0, ref_val[np.clip(idx, 0, len(ref_val) - 1)], fill_value)


def latlon_to_enu(lat, lon, lat0, lon0):
    """Equirectangular approx local tangent-plane meters; fine at vehicle/parking-lot scale."""
    x = np.radians(lon - lon0) * WGS84_R * np.cos(np.radians(lat0))
    y = np.radians(lat - lat0) * WGS84_R
    return x, y


def enu_to_latlon(x, y, lat0, lon0):
    """Inverse of latlon_to_enu."""
    lat = lat0 + np.degrees(y / WGS84_R)
    lon = lon0 + np.degrees(x / (WGS84_R * np.cos(np.radians(lat0))))
    return lat, lon


def latlon_to_webmercator(lat, lon):
    """WGS84 lat/lon (deg) -> EPSG:3857 Web Mercator x/y (meters), for basemap tile alignment."""
    x = np.radians(lon) * WGS84_R
    y = WGS84_R * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y


def covariance_corridor(lat, lon, cov_std, sigma):
    """Offset curves parallel to the (lat, lon) path at +-sigma*cov_std meters.

    Offsets are computed in a local ENU tangent plane (true meters) so the
    corridor width is not distorted by the Web Mercator projection used for
    display; the result is converted back to lat/lon.
    """
    lat0, lon0 = lat[0], lon[0]
    east, north = latlon_to_enu(lat, lon, lat0, lon0)

    dx, dy = np.gradient(east), np.gradient(north)
    seg = np.hypot(dx, dy)
    seg[seg < 1e-6] = 1e-6
    tx, ty = dx / seg, dy / seg
    px, py = -ty, tx  # unit vector perpendicular to the direction of travel

    r = sigma * cov_std
    left_lat, left_lon = enu_to_latlon(east + px * r, north + py * r, lat0, lon0)
    right_lat, right_lon = enu_to_latlon(east - px * r, north - py * r, lat0, lon0)
    return (left_lat, left_lon), (right_lat, right_lon)


def detect_speed_jumps(t_ns: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                        max_speed_mps: float) -> np.ndarray:
    """Flag points implying an impossible vehicle speed to/from their neighbors.

    Assumes (t_ns, lat, lon) are already time-sorted and jump-free otherwise
    (e.g. zero-jump points already removed). A point is flagged only when both
    the speed into it and the speed out of it exceed max_speed_mps, so an
    isolated GPS teleport is caught without also flagging its legitimate
    neighbors. Sequence endpoints use whichever single side is available.
    """
    n = len(t_ns)
    bad = np.zeros(n, dtype=bool)
    if n < 2 or max_speed_mps <= 0:
        return bad

    east, north = latlon_to_enu(lat, lon, lat[0], lon[0])
    dt = np.diff(t_ns) / 1e9
    dist = np.hypot(np.diff(east), np.diff(north))
    with np.errstate(divide="ignore", invalid="ignore"):
        seg_speed = np.where(dt > 0, dist / dt, np.inf)

    speed_in = np.concatenate(([np.nan], seg_speed))
    speed_out = np.concatenate((seg_speed, [np.nan]))

    too_fast_in = speed_in > max_speed_mps
    too_fast_out = speed_out > max_speed_mps
    bad[1:-1] = too_fast_in[1:-1] & too_fast_out[1:-1]
    bad[0] = too_fast_out[0]
    bad[-1] = too_fast_in[-1]
    return bad


def plot_route(data: np.ndarray, status: np.ndarray, color_status: np.ndarray,
                style_map: dict, legend_title: str, out_path: str, title: str,
                require_fix: bool, cov_sigma: float, basemap: bool,
                max_speed_mps: float) -> bool:
    if len(data) == 0:
        print(f"  [skip] no data: {title}", file=sys.stderr)
        return False

    # Never let automatic jump detection exclude every single point: always
    # keep the very first (finite) fix so a map is still produced even when
    # the whole bag looks like jumps/outliers (e.g. a bag with no real GPS
    # fix at all, where every consecutive pair implies an impossible speed).
    keep_first = bool(np.isfinite(data[0, 1]) and np.isfinite(data[0, 2]))

    zero_jump = (data[:, 1] == 0.0) & (data[:, 2] == 0.0)
    if keep_first:
        zero_jump[0] = False
    n_zero_jump = int(zero_jump.sum())
    if n_zero_jump > 0:
        print(f"  [note] {title}: excluded {n_zero_jump} 0 Jump point(s) (lat=lon=0)")

    candidate = np.isfinite(data[:, 1]) & np.isfinite(data[:, 2]) & ~zero_jump
    idx = np.where(candidate)[0]
    speed_jump = np.zeros(len(data), dtype=bool)
    speed_jump[idx] = detect_speed_jumps(data[idx, 0], data[idx, 1], data[idx, 2], max_speed_mps)
    if keep_first:
        speed_jump[0] = False
    n_speed_jump = int(speed_jump.sum())
    if n_speed_jump > 0:
        print(f"  [note] {title}: excluded {n_speed_jump} speed Jump point(s) "
              f"(implied speed > {max_speed_mps:g} m/s)")

    valid = candidate & ~speed_jump
    if require_fix:
        valid &= status >= 0
    if not np.any(valid):
        print(f"  [skip] no valid fixes: {title}", file=sys.stderr)
        return False

    d, st_color = data[valid], color_status[valid]
    lat, lon, cov_std = d[:, 1], d[:, 2], d[:, 4]
    mx, my = latlon_to_webmercator(lat, lon)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 9))

    if len(d) >= 2 and cov_sigma > 0 and np.any(cov_std > 0):
        (l_lat, l_lon), (r_lat, r_lon) = covariance_corridor(lat, lon, cov_std, cov_sigma)
        lmx, lmy = latlon_to_webmercator(l_lat, l_lon)
        rmx, rmy = latlon_to_webmercator(r_lat, r_lon)
        ax.plot(lmx, lmy, "--", color="blue", linewidth=1, alpha=0.8,
                label=f"±{cov_sigma:g}σ cov. radius", zorder=3)
        ax.plot(rmx, rmy, "--", color="blue", linewidth=1, alpha=0.8, zorder=3)

    for s in sorted(set(st_color.tolist())):
        color, label = style_map.get(s, ("gray", f"value={s}"))
        m = st_color == s
        ax.scatter(mx[m], my[m], c=color, s=6, label=label, zorder=4)

    ax.scatter(mx[0], my[0], facecolor="none", edgecolor="blue", s=250, marker="*",
               linewidth=1.5, zorder=5, label="start")
    ax.scatter(mx[-1], my[-1], facecolor="none", edgecolor="blue", s=150, marker="X",
               linewidth=1.5, zorder=5, label="end")

    pad_x = 0.15 * max(np.ptp(mx), 1.0)
    pad_y = 0.15 * max(np.ptp(my), 1.0)
    ax.set_xlim(mx.min() - pad_x, mx.max() + pad_x)
    ax.set_ylim(my.min() - pad_y, my.max() + pad_y)
    ax.set_aspect("equal")

    if basemap:
        try:
            import contextily as ctx
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
        except Exception as e:
            print(f"  [warn] basemap fetch failed ({e}); plotting without it", file=sys.stderr)

    jump_notes = []
    if n_zero_jump > 0:
        jump_notes.append(f"0 Jump: {n_zero_jump} pts excluded (lat=lon=0)")
    if n_speed_jump > 0:
        jump_notes.append(f"Speed Jump: {n_speed_jump} pts excluded (> {max_speed_mps:g} m/s)")
    if jump_notes:
        ax.text(0.02, 0.02, "\n".join(jump_notes),
                transform=ax.transAxes, fontsize=9, color="red", va="bottom", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("Web Mercator X [m]")
    ax.set_ylabel("Web Mercator Y [m]")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, title=legend_title, title_fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def sanitize_topic(topic: str) -> str:
    return topic.strip("/").replace("/", "_")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input",
                     help="single .mcap file, or a split rosbag2 directory (*.mcap + metadata.yaml)")
    ap.add_argument("--out-dir", default=None,
                     help="output directory for PNGs (default: alongside input)")
    ap.add_argument("--topics", default=None,
                     help="comma-separated NavSatFix topics to restrict to (default: auto-detect)")
    ap.add_argument("--require-fix", action="store_true",
                     help="drop points with status.status < 0 (NO_FIX)")
    ap.add_argument("--cov-sigma", type=float, default=1.0,
                     help="covariance corridor width in sigma multiples of "
                          "sqrt(mean(cov_xx, cov_yy)) (default: 1.0, i.e. 1-sigma). "
                          "Set to 0 to disable the corridor.")
    ap.add_argument("--no-basemap", action="store_true",
                     help="skip fetching the OpenStreetMap tile background (offline use)")
    ap.add_argument("--max-speed", type=float, default=50.0,
                     help="max plausible implied speed in m/s between consecutive fixes "
                          "(default: 50.0, i.e. 180 km/h); a point is excluded as a "
                          "'Speed Jump' only when the speed both into and out of it "
                          "exceeds this. Set to 0 to disable the check.")
    args = ap.parse_args(argv)

    files = resolve_mcap_files(args.input)
    out_dir = args.out_dir or (
        args.input if os.path.isdir(args.input) else os.path.dirname(os.path.abspath(args.input))
    )
    os.makedirs(out_dir, exist_ok=True)

    if args.topics:
        topics = [t.strip() for t in args.topics.split(",") if t.strip()]
    else:
        topics = discover_navsatfix_topics(files)

    if not topics:
        print(f"no {NAVSATFIX_TYPE} topics found in {args.input}", file=sys.stderr)
        return 1

    print(f"NavSatFix topics: {topics}")

    # Prefer coloring by the OxTS NCOM GPS position mode (distinguishes RTK
    # Float / RTK Integer etc.) over NavSatFix's coarse 4-level status, when
    # the bag has an Ncom topic to decode it from.
    ncom_topics = discover_topics_by_schema(files, NCOM_TYPE)
    ncom_t_ns, ncom_mode = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    if ncom_topics:
        ncom_t_ns, ncom_mode = read_ncom_gps_pos_mode(files, ncom_topics[0])
        print(f"Ncom topic: {ncom_topics[0]} ({len(ncom_t_ns)} GPS pos-mode updates decoded)")
    else:
        print(f"no {NCOM_TYPE} topic found; falling back to NavSatFix.status.status for coloring")

    ok = 0
    for topic in topics:
        data, status = read_navsatfix(files, topic)
        if len(data) == 0:
            print(f"  [skip] {topic}: no messages", file=sys.stderr)
            continue

        if len(ncom_t_ns) > 0:
            color_status = asof_forward_fill(data[:, 0], ncom_t_ns, ncom_mode, GPS_POS_MODE_UNKNOWN)
            style_map, legend_title = GPS_POS_MODE_STYLE, "GPS pos mode (Ncom)"
        else:
            color_status, style_map, legend_title = status, STATUS_STYLE, "status.status (NavSatFix)"

        out_path = os.path.join(out_dir, f"gps_route_{sanitize_topic(topic)}.png")
        title = f"GPS route: {topic} ({len(data)} fixes)"
        if plot_route(data, status, color_status, style_map, legend_title, out_path, title,
                       args.require_fix, args.cov_sigma, not args.no_basemap, args.max_speed):
            print(f"  wrote {out_path}  ({len(data)} points)")
            ok += 1

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
