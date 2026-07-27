#!/usr/bin/env python3
"""車両操作(レーンチェンジ・急加速・急減速・停止)を GNSS/INS 情報から検出する。

INS の `sensor_msgs/msg/Imu` トピック(角速度 z 成分 = ヨーレート、姿勢クォータニオン
= heading、linear_acceleration.x = 前後加速度)から検出する。トピック名は問わず、
スキーマが `sensor_msgs/msg/Imu` であるトピックを自動検出する(plot_gps_route.py の
NavSatFix 自動検出と同じ手法)。停止検出のみはGPS位置列だけで完結するため、Imu
トピックの有無に関わらず常に動作する。

Imu トピックが無いbagでは、GPS 位置列(lat/lon)の平滑微分からヨーレート/heading/
加速度を推定するフォールバックに切り替える(精度は落ちる)。

設計ドック(office-hours, 2026-07-18)の Implementation-time pivot を参照:
当初は oxts_msgs/msg/Ncom の生バイトデコードを予定していたが、確認済みの
グラウンドトゥルースbagに既に Imu トピックが存在することが分かり、生バイト
解析より遥かに安全なこちらの方式に切り替えた。
"""
from __future__ import annotations

import csv
import os

import numpy as np
from mcap.reader import make_reader
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu

IMU_TYPE = "sensor_msgs/msg/Imu"

# Defaults for the hysteresis pulse classifier — calibrated against the
# confirmed ground-truth bag (design doc "The Assignment"):
# HRdqo3pf_dmcYiVVH_2026-07-16T16-36-06+0900. Real yaw-rate range observed:
# a confirmed lane change ~150-200s in (peaks ~1.7-3.9 deg/s, net heading
# change ~10 deg over the window) vs. a real turn ~0-15s in (peak ~29 deg/s,
# net heading change ~79 deg). 2 deg/s sits comfortably below the lane-change
# peaks and >10x below the turn's peak, so it separates the two cleanly on
# this bag. See tests/fixtures/ground_truth_yaw_rate.npz for the pinned
# regression fixture (extracted from both windows) and
# TestCalibrationAgainstGroundTruthBag in tests/test_maneuver_detection.py.
DEFAULT_YAW_RATE_THRESHOLD_RAD_S = np.radians(2.0)  # enter-pulse threshold
DEFAULT_YAW_RATE_HYSTERESIS_RAD_S = np.radians(0.5)  # exit-pulse threshold (lower, avoids chatter)
DEFAULT_MIN_PULSE_S = 0.5
DEFAULT_MAX_PULSE_S = 6.0
DEFAULT_MAX_NET_HEADING_CHANGE_RAD = np.radians(20)  # distinguishes lane change from sustained turn

# A real lane change is a bipolar S-curve (steer one way, then steer back to
# straighten out) — yaw rate crosses through zero between the two halves,
# tripping the hysteresis exit and splitting one real maneuver into two raw
# pulses of opposite sign. Calibrated against the same ground-truth bag: the
# two known same-maneuver pairs were 0.4s and 1.2s apart; the nearest
# non-pair (same-sign, unrelated) gap was 4.0s, so 2.0s separates them with
# margin on both sides.
DEFAULT_MERGE_GAP_S = 2.0

# A real lane change steers back to (near) its pre-maneuver heading once the
# vehicle settles into the new lane. A vehicle merely following a gradual
# road curve does NOT return — heading keeps drifting the same direction
# after the brief yaw-rate blip that crossed the entry threshold. Calibrated
# against the same ground-truth bag: 3 confirmed lane changes had <=1.2 deg
# of heading drift 3s after their pulse ended; one road-curvature false
# positive had 4.7 deg drift at the same check point, so 3.0 deg separates
# them with margin.
DEFAULT_RETURN_CHECK_S = 3.0
DEFAULT_MAX_RETURN_DRIFT_RAD = np.radians(3.0)

# Harsh acceleration/braking: commonly-cited telematics default (~0.35g —
# see design doc Landscape Check). Sign/scale cross-validated on the
# ground-truth bag against GPS-derived acceleration (correlation 0.84,
# same sign — no frame flip for this signal). NOT re-calibrated against a
# confirmed harsh-event bag the way the lane-change thresholds were: this
# bag's max observed |accel_x| is only ~0.26g, a careful test drive with no
# harsh events, so "0 detected" here is an honest result, not a bug.
_GRAVITY_MPS2 = 9.80665
DEFAULT_HARSH_ACCEL_THRESHOLD_MPS2 = 0.35 * _GRAVITY_MPS2
DEFAULT_HARSH_ACCEL_HYSTERESIS_MPS2 = 0.15 * _GRAVITY_MPS2
DEFAULT_HARSH_MIN_DURATION_S = 0.3
DEFAULT_HARSH_MAX_DURATION_S = 5.0

# Stop detection: below normal walking pace, sustained — distinguishes an
# actual stop from momentary low-speed GPS noise between fixes.
DEFAULT_STOP_SPEED_THRESHOLD_MPS = 0.5
DEFAULT_STOP_MIN_DURATION_S = 2.0


def discover_imu_topics(files: list[str]) -> list[str]:
    """Return topic names whose schema is sensor_msgs/msg/Imu, across all files."""
    from plot_gps_route import discover_topics_by_schema  # local import: avoid a
    # module-level import cycle (plot_gps_route.py imports this module for CLI
    # integration; this module only needs plot_gps_route's helpers at call time).
    return discover_topics_by_schema(files, IMU_TYPE)


def _quaternion_to_yaw(q) -> float:
    """REP-103 yaw extraction from a geometry_msgs/msg/Quaternion (radians)."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def read_imu_signals(files: list[str], topic: str
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_ns, yaw_rate, heading, accel_x) sorted by time, from an
    Imu-schema topic, in a single pass over the topic (DRY — avoids
    re-scanning the same messages once per signal).

    yaw_rate is angular_velocity.z [rad/s]. heading is derived from the
    orientation quaternion [rad], REP-103 convention. accel_x is
    linear_acceleration.x [m/s^2] (forward/longitudinal, body-frame) — sign
    cross-checked against GPS-derived acceleration on the ground-truth bag
    (correlation 0.84, same sign): positive = accelerating forward, negative
    = braking.
    """
    ts, yaw_rate, heading, accel_x = [], [], [], []
    for fp in files:
        with open(fp, "rb") as f:
            reader = make_reader(f)
            for _schema, _channel, message in reader.iter_messages(topics=[topic]):
                msg = deserialize_message(message.data, Imu)
                ts.append(message.log_time)
                yaw_rate.append(msg.angular_velocity.z)
                heading.append(_quaternion_to_yaw(msg.orientation))
                accel_x.append(msg.linear_acceleration.x)

    t_ns = np.asarray(ts, dtype=np.int64)
    yaw_rate_arr = np.asarray(yaw_rate, dtype=np.float64)
    heading_arr = np.asarray(heading, dtype=np.float64)
    accel_x_arr = np.asarray(accel_x, dtype=np.float64)
    order = np.argsort(t_ns, kind="stable")
    return t_ns[order], yaw_rate_arr[order], heading_arr[order], accel_x_arr[order]


def wrap_angle(angle_rad):
    """Wrap an angle (scalar or array) to [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi


def derive_yaw_rate_heading_from_position(t_ns: np.ndarray, lat: np.ndarray, lon: np.ndarray
                                           ) -> tuple[np.ndarray, np.ndarray]:
    """Approach A fallback: estimate yaw rate + heading by differentiating the
    GPS lat/lon position series in a local ENU tangent plane.

    Lower fidelity than the Imu-based path — double differentiation
    amplifies GPS position noise (especially on RTK-float-quality fixes).
    Used only when no Imu-schema topic is available. Returns (yaw_rate,
    heading), both length len(t_ns); the first sample's yaw_rate is 0
    (no prior sample to differentiate against).
    """
    from plot_gps_route import latlon_to_enu  # local import: see discover_imu_topics

    n = len(t_ns)
    if n < 3:
        return np.zeros(n), np.zeros(n)

    east, north = latlon_to_enu(lat, lon, lat[0], lon[0])
    heading = np.arctan2(np.gradient(east), np.gradient(north))
    dt = np.gradient(t_ns.astype(np.float64)) / 1e9
    dt[dt == 0] = np.nan
    yaw_rate = np.zeros(n)
    yaw_rate[1:] = wrap_angle(np.diff(heading)) / (dt[1:])
    yaw_rate = np.nan_to_num(yaw_rate, nan=0.0, posinf=0.0, neginf=0.0)
    return yaw_rate, heading


def _merge_opposite_sign_pulses(events: list[tuple[int, int, int, float]], merge_gap_s: float
                                 ) -> list[tuple[int, int, int, float]]:
    """Merge adjacent opposite-sign pulses into one lane-change event.

    A real lane change is a bipolar S-curve (steer one way, then steer back
    to straighten out) — yaw rate crosses through zero between the two
    halves, which trips the hysteresis exit in `detect_lane_changes` and
    splits one real maneuver into two raw same-direction pulses. Discovered
    via ground-truth validation: the tool reported 6 events against 3
    manually-confirmed lane changes on the same bag used for T6 (see design
    doc); two of the three were each split into an opposite-sign pair close
    in time. This merges such pairs back into one event, keeping the
    larger-magnitude peak. Pairs of the SAME sign are never merged — that
    would conflate two distinct maneuvers, not un-split one.
    """
    if not events:
        return []
    merged = [events[0]]
    for start, peak_t, end, peak in events[1:]:
        prev_start, prev_peak_t, prev_end, prev_peak = merged[-1]
        gap_s = (start - prev_end) / 1e9
        if gap_s <= merge_gap_s and (prev_peak > 0) != (peak > 0):
            if abs(peak) > abs(prev_peak):
                merged[-1] = (prev_start, peak_t, end, peak)
            else:
                merged[-1] = (prev_start, prev_peak_t, end, prev_peak)
        else:
            merged.append((start, peak_t, end, peak))
    return merged


def _filter_non_returning_events(events: list[tuple[int, int, int, float]],
                                  t_ns: np.ndarray, heading: np.ndarray,
                                  return_check_s: float, max_return_drift_rad: float
                                  ) -> list[tuple[int, int, int, float]]:
    """Reject events where heading keeps drifting the same direction well
    after the pulse ends, instead of returning to its pre-pulse baseline.

    A real lane change steers back straight once the vehicle settles into
    the new lane; a vehicle merely following a gradual road curve does not
    return — discovered via ground-truth validation, where a road-curvature-
    following steer was otherwise indistinguishable from a real lane change
    by the within-pulse net-heading-change check alone.
    """
    n = len(t_ns)
    filtered = []
    for start_ns, peak_ns, end_ns, peak in events:
        start_idx = np.searchsorted(t_ns, start_ns)
        end_idx = np.searchsorted(t_ns, end_ns)
        target_idx = min(np.searchsorted(t_ns, end_ns + int(return_check_s * 1e9)), n - 1)
        if target_idx <= end_idx:
            filtered.append((start_ns, peak_ns, end_ns, peak))  # not enough data to check — accept
            continue
        drift = wrap_angle(heading[target_idx] - heading[start_idx])
        if abs(drift) <= max_return_drift_rad:
            filtered.append((start_ns, peak_ns, end_ns, peak))
    return filtered


def detect_lane_changes(t_ns: np.ndarray, yaw_rate: np.ndarray, heading: np.ndarray, *,
                         yaw_rate_threshold: float = DEFAULT_YAW_RATE_THRESHOLD_RAD_S,
                         yaw_rate_hysteresis: float = DEFAULT_YAW_RATE_HYSTERESIS_RAD_S,
                         min_pulse_s: float = DEFAULT_MIN_PULSE_S,
                         max_pulse_s: float = DEFAULT_MAX_PULSE_S,
                         max_net_heading_change_rad: float = DEFAULT_MAX_NET_HEADING_CHANGE_RAD,
                         merge_gap_s: float = DEFAULT_MERGE_GAP_S,
                         return_check_s: float = DEFAULT_RETURN_CHECK_S,
                         max_return_drift_rad: float = DEFAULT_MAX_RETURN_DRIFT_RAD,
                         ) -> list[tuple[int, int, int, float]]:
    """Hand-rolled hysteresis threshold-crossing pulse detector.

    Distinguishes a lane-change (brief S-curve yaw-rate pulse that returns to
    the original heading) from a sustained turn (large net heading change) per
    design doc Premise 2. No scipy dependency (eng-review decision). Adjacent
    opposite-sign pulses close in time are merged into one event (see
    `_merge_opposite_sign_pulses`) — a real lane change's yaw rate crosses
    zero between its two steering halves, which would otherwise double-count
    it as two events. Merged events are then filtered to reject ones where
    heading keeps drifting after the pulse ends instead of returning to
    baseline (see `_filter_non_returning_events`) — distinguishes a real lane
    change from settling into a gradual road curve.

    NOTE: threshold defaults are calibrated against one ground-truth bag —
    see design doc Open Questions / The Assignment. Validate again before
    trusting on a different vehicle/ECU setup.

    Returns a list of (t_start_ns, t_peak_ns, t_end_ns, peak_yaw_rate) events.
    The "lane change during road curvature" compound case (Open Questions) —
    a lane change performed WHILE the road is also curving, at the same
    time — is still NOT handled; `_filter_non_returning_events` only rejects
    curvature with no lane change at all, not curvature superimposed on a
    real one. Explicitly out of scope for v1.
    """
    n = len(t_ns)
    if n < 3:
        return []

    events: list[tuple[int, int, int, float]] = []
    in_pulse = False
    pulse_start_idx = None
    peak_idx = None

    for i in range(n):
        v = yaw_rate[i]
        if not in_pulse:
            if abs(v) >= yaw_rate_threshold:
                in_pulse = True
                pulse_start_idx = i
                peak_idx = i
        else:
            if abs(v) > abs(yaw_rate[peak_idx]):
                peak_idx = i
            if abs(v) < yaw_rate_hysteresis:
                end_idx = i
                duration_s = (t_ns[end_idx] - t_ns[pulse_start_idx]) / 1e9
                net_heading = wrap_angle(heading[end_idx] - heading[pulse_start_idx])
                if (min_pulse_s <= duration_s <= max_pulse_s
                        and abs(net_heading) <= max_net_heading_change_rad):
                    events.append((int(t_ns[pulse_start_idx]), int(t_ns[peak_idx]),
                                   int(t_ns[end_idx]), float(yaw_rate[peak_idx])))
                in_pulse = False
                pulse_start_idx = None
                peak_idx = None

    merged = _merge_opposite_sign_pulses(events, merge_gap_s)
    filtered = _filter_non_returning_events(merged, t_ns, heading, return_check_s, max_return_drift_rad)
    return [(s, p, e, m, "lane_change") for s, p, e, m in filtered]


def detect_harsh_accel_decel(t_ns: np.ndarray, accel_x: np.ndarray, *,
                              threshold_mps2: float = DEFAULT_HARSH_ACCEL_THRESHOLD_MPS2,
                              hysteresis_mps2: float = DEFAULT_HARSH_ACCEL_HYSTERESIS_MPS2,
                              min_duration_s: float = DEFAULT_HARSH_MIN_DURATION_S,
                              max_duration_s: float = DEFAULT_HARSH_MAX_DURATION_S,
                              ) -> list[tuple[int, int, int, float, str]]:
    """Hysteresis threshold-crossing detector for harsh acceleration/braking.

    Positive accel_x excursions above threshold_mps2 are tagged
    "harsh_accel"; negative excursions "harsh_decel". Unlike a lane change,
    a single acceleration/braking event is inherently unipolar — no
    opposite-sign merge or return-to-baseline logic needed.

    Returns (t_start_ns, t_peak_ns, t_end_ns, peak_accel_mps2, maneuver_type).
    """
    n = len(t_ns)
    if n < 2:
        return []

    events: list[tuple[int, int, int, float, str]] = []
    in_pulse = False
    pulse_start_idx = None
    peak_idx = None

    for i in range(n):
        v = accel_x[i]
        if not in_pulse:
            if abs(v) >= threshold_mps2:
                in_pulse = True
                pulse_start_idx = i
                peak_idx = i
        else:
            if abs(v) > abs(accel_x[peak_idx]):
                peak_idx = i
            if abs(v) < hysteresis_mps2:
                end_idx = i
                duration_s = (t_ns[end_idx] - t_ns[pulse_start_idx]) / 1e9
                if min_duration_s <= duration_s <= max_duration_s:
                    maneuver_type = "harsh_accel" if accel_x[peak_idx] > 0 else "harsh_decel"
                    events.append((int(t_ns[pulse_start_idx]), int(t_ns[peak_idx]),
                                   int(t_ns[end_idx]), float(accel_x[peak_idx]), maneuver_type))
                in_pulse = False
                pulse_start_idx = None
                peak_idx = None

    return events


def derive_accel_from_position(t_ns: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Approach A fallback: estimate forward acceleration by double-
    differentiating the GPS lat/lon position series. Lower fidelity than
    the Imu-based path — double differentiation of an already-noisy signal
    amplifies GPS position noise considerably more than the single
    differentiation used for speed. Used only when no Imu-schema topic is
    available.
    """
    from plot_gps_route import latlon_to_enu  # local import: see discover_imu_topics

    n = len(t_ns)
    if n < 3:
        return np.zeros(n)

    east, north = latlon_to_enu(lat, lon, lat[0], lon[0])
    dt = np.gradient(t_ns.astype(np.float64)) / 1e9
    dt[dt == 0] = np.nan
    speed = np.hypot(np.gradient(east), np.gradient(north)) / dt
    accel = np.gradient(speed) / dt
    return np.nan_to_num(accel, nan=0.0, posinf=0.0, neginf=0.0)


def detect_stops(fix_t_ns: np.ndarray, lat: np.ndarray, lon: np.ndarray, *,
                  speed_threshold_mps: float = DEFAULT_STOP_SPEED_THRESHOLD_MPS,
                  min_duration_s: float = DEFAULT_STOP_MIN_DURATION_S,
                  ) -> list[tuple[int, int, int, float, str]]:
    """Detect contiguous low-speed segments (vehicle stopped), from GPS
    position differencing. Position-derived speed is fine here — unlike for
    lane-change classification, a stop is inherently slow/low-frequency and
    doesn't need Imu's higher sample rate or lower noise floor, so this
    always runs regardless of whether an Imu-schema topic is present.

    Returns (t_start_ns, t_mid_ns, t_end_ns, duration_s, "stop") events —
    there is no meaningful "peak" for a stop, so the middle field is the
    segment midpoint; magnitude is the stop duration in seconds.
    """
    from plot_gps_route import latlon_to_enu  # local import: see discover_imu_topics

    n = len(fix_t_ns)
    if n < 2:
        return []

    east, north = latlon_to_enu(lat, lon, lat[0], lon[0])
    dt = np.diff(fix_t_ns) / 1e9
    dist = np.hypot(np.diff(east), np.diff(north))
    speed = np.divide(dist, dt, out=np.full_like(dist, np.inf), where=dt > 0)
    stopped = speed < speed_threshold_mps  # length n-1, one flag per between-fix segment

    events: list[tuple[int, int, int, float, str]] = []
    i = 0
    while i < len(stopped):
        if not stopped[i]:
            i += 1
            continue
        j = i
        while j < len(stopped) and stopped[j]:
            j += 1
        start_idx, end_idx = i, j  # fix indices bounding the stopped segment
        duration_s = (fix_t_ns[end_idx] - fix_t_ns[start_idx]) / 1e9
        if duration_s >= min_duration_s:
            mid_idx = (start_idx + end_idx) // 2
            events.append((int(fix_t_ns[start_idx]), int(fix_t_ns[mid_idx]),
                            int(fix_t_ns[end_idx]), float(duration_s), "stop"))
        i = j

    return events


def map_events_to_fixes(events: list[tuple[int, int, int, float, str]],
                         fix_t_ns: np.ndarray, fix_lat: np.ndarray, fix_lon: np.ndarray
                         ) -> list[tuple[int, float, float, str, float]]:
    """Map each event's peak/mid time to the nearest NavSatFix fix (by time),
    for plotting/CSV purposes. Detection itself runs at whatever native rate
    its source signal has (Imu, or GPS fixes directly for stops) — only the
    event *time* gets matched to a (possibly sparser) GPS fix here, not the
    other way around.

    An event time outside the fix range is clamped to the nearest available
    fix rather than dropped, so no detected event silently disappears.

    Returns a list of (t_ns, lat, lon, maneuver_type, magnitude).
    """
    if len(fix_t_ns) == 0:
        return []

    mapped = []
    for t_start_ns, t_peak_ns, t_end_ns, magnitude, maneuver_type in events:
        idx = np.searchsorted(fix_t_ns, t_peak_ns)
        if idx <= 0:
            nearest = 0
        elif idx >= len(fix_t_ns):
            nearest = len(fix_t_ns) - 1
        else:
            before, after = idx - 1, idx
            nearest = before if (t_peak_ns - fix_t_ns[before]) <= (fix_t_ns[after] - t_peak_ns) else after
        mapped.append((t_peak_ns, float(fix_lat[nearest]), float(fix_lon[nearest]),
                        maneuver_type, abs(magnitude)))
    return mapped


def write_maneuver_events_csv(events: list[tuple[int, float, float, str, float]], out_path: str) -> None:
    """Write a (t_ns, lat, lon, maneuver_type, magnitude) CSV sidecar.

    magnitude's unit depends on maneuver_type: rad/s peak yaw rate for
    "lane_change", m/s^2 peak longitudinal accel for "harsh_accel"/
    "harsh_decel", seconds of stopped duration for "stop".

    Always writes a valid file, even with zero events (header-only) — a bag
    with no detected maneuvers is a valid result, not a tool failure, and
    downstream consumers must be able to tell "ran, found nothing" apart from
    "did not run".
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_ns", "lat", "lon", "maneuver_type", "magnitude"])
        for t_ns, lat, lon, maneuver_type, magnitude in events:
            writer.writerow([t_ns, lat, lon, maneuver_type, magnitude])
