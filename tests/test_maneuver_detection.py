"""Tests for tools/gps-route/maneuver_detection.py.

Covers the hysteresis pulse classifier, the GPS-position-derivative
fallback, event-to-fix mapping, and the CSV sidecar writer — all with
synthetic data (no real bag files needed). The one xfail test documents
the "lane change during road curvature" known v1 limitation (design doc
Open Questions), which needs a real fix once addressed, not before.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).parent.parent / 'tools' / 'gps-route'
sys.path.insert(0, str(TOOLS_DIR))

import plot_gps_route  # noqa: E402 (sys.path.insert must come first)
import maneuver_detection  # noqa: E402


# ── Synthetic signal builders ─────────────────────────────────────────────────

def _lane_change_pulse(n=50, dt=0.1, pulse_start=20, pulse_len=10, peak_yaw_rate=0.3):
    """A brief S-curve yaw-rate pulse that returns to the original heading —
    ramps up, holds, ramps back down and slightly negative to cancel the net
    heading change, then settles at zero. Everywhere else, yaw_rate is 0."""
    t_ns = (np.arange(n) * dt * 1e9).astype(np.int64)
    yaw_rate = np.zeros(n)
    half = pulse_len // 2
    yaw_rate[pulse_start:pulse_start + half] = peak_yaw_rate
    yaw_rate[pulse_start + half:pulse_start + pulse_len] = -peak_yaw_rate
    heading = np.concatenate(([0.0], np.cumsum(yaw_rate[1:]) * dt))
    return t_ns, yaw_rate, heading


def _sustained_turn(n=50, dt=0.1, peak_yaw_rate=0.3):
    """A turn: yaw_rate stays at peak_yaw_rate for the whole window — large
    net heading change, must NOT be classified as a lane change."""
    t_ns = (np.arange(n) * dt * 1e9).astype(np.int64)
    yaw_rate = np.full(n, peak_yaw_rate)
    heading = np.cumsum(yaw_rate) * dt
    return t_ns, yaw_rate, heading


# ── detect_lane_changes ────────────────────────────────────────────────────────

class TestDetectLaneChanges:
    def test_isolated_pulse_is_tagged(self):
        t_ns, yaw_rate, heading = _lane_change_pulse()
        events = maneuver_detection.detect_lane_changes(t_ns, yaw_rate, heading)
        assert len(events) == 1

    def test_sustained_turn_not_tagged(self):
        """Core Premise 2 distinction: a real turn must not be misclassified."""
        t_ns, yaw_rate, heading = _sustained_turn()
        events = maneuver_detection.detect_lane_changes(t_ns, yaw_rate, heading)
        assert events == [], 'a sustained turn must not be classified as a lane change'

    def test_flat_signal_no_events(self):
        n = 30
        t_ns = (np.arange(n) * 1e8).astype(np.int64)
        yaw_rate = np.zeros(n)
        heading = np.zeros(n)
        assert maneuver_detection.detect_lane_changes(t_ns, yaw_rate, heading) == []

    def test_empty_and_near_empty_arrays_do_not_crash(self):
        assert maneuver_detection.detect_lane_changes(
            np.array([], dtype=np.int64), np.array([]), np.array([])) == []
        assert maneuver_detection.detect_lane_changes(
            np.array([0, 1], dtype=np.int64), np.array([0.2, 0.2]), np.array([0.0, 0.1])) == []

    def test_pulse_too_long_not_tagged_as_lane_change(self):
        """A yaw-rate excursion lasting far longer than a lane change (e.g. a
        slow, held turn) must fall outside max_pulse_s and not be tagged."""
        t_ns, yaw_rate, heading = _lane_change_pulse(n=200, dt=0.1, pulse_start=20, pulse_len=150)
        events = maneuver_detection.detect_lane_changes(t_ns, yaw_rate, heading, max_pulse_s=6.0)
        assert events == []

    @pytest.mark.xfail(reason=(
        "Known v1 limitation (design doc Open Questions): a lane change performed "
        "while the road itself is curving superimposes a sustained background yaw "
        "rate on the pulse. Premise 2's S-curve-vs-turn distinction alone does not "
        "separate these; needs detrending/baseline-subtraction, deferred out of v1 "
        "scope. This test documents the limitation, not a bug to silently ignore."))
    def test_lane_change_during_curving_road_compound_case(self):
        # Synthetic compound signal: a sustained road-curvature yaw rate above
        # the hysteresis exit threshold (but below the pulse-entry threshold on
        # its own) with a real lane-change pulse superimposed. Because yaw_rate
        # never drops back below yaw_rate_hysteresis after the pulse (the
        # background curvature alone keeps it elevated), the detector's pulse
        # never "closes" and the lane change goes undetected — a false negative.
        n = 80
        dt = 0.1
        t_ns = (np.arange(n) * dt * 1e9).astype(np.int64)
        curvature_yaw_rate = 0.1  # sustained road curve: > hysteresis (0.05), < entry (0.15) alone
        yaw_rate = np.full(n, curvature_yaw_rate)
        yaw_rate[20:25] += 0.3
        yaw_rate[25:30] -= 0.3
        heading = np.cumsum(yaw_rate) * dt
        events = maneuver_detection.detect_lane_changes(t_ns, yaw_rate, heading)
        # Expected (once this is fixed): exactly one lane-change event detected
        # despite the background curvature. Today it is not reliably separated.
        assert len(events) == 1


# ── derive_yaw_rate_heading_from_position (Approach A fallback) ─────────────

class TestDeriveYawRateHeadingFromPosition:
    def test_straight_line_yields_near_zero_yaw_rate(self):
        lat0, lon0 = 35.0, 139.0
        n = 10
        y = np.arange(n) * 10.0  # driving due north at 10 m/s, 1s steps
        t_ns = (np.arange(n) * 1e9).astype(np.int64)
        lat, lon = plot_gps_route.enu_to_latlon(np.zeros(n), y, lat0, lon0)
        yaw_rate, heading = maneuver_detection.derive_yaw_rate_heading_from_position(t_ns, lat, lon)
        assert np.allclose(yaw_rate[1:], 0.0, atol=1e-6)

    def test_insufficient_points_no_crash(self):
        yaw_rate, heading = maneuver_detection.derive_yaw_rate_heading_from_position(
            np.array([0, 1], dtype=np.int64), np.array([35.0, 35.001]), np.array([139.0, 139.0]))
        assert len(yaw_rate) == 2
        assert len(heading) == 2


# ── detect_harsh_accel_decel ──────────────────────────────────────────────────

class TestDetectHarshAccelDecel:
    def _pulse(self, n=30, dt=0.1, start=10, length=5, peak=5.0):
        t_ns = (np.arange(n) * dt * 1e9).astype(np.int64)
        accel_x = np.zeros(n)
        accel_x[start:start + length] = peak
        return t_ns, accel_x

    def test_harsh_accel_tagged_by_positive_sign(self):
        t_ns, accel_x = self._pulse(peak=5.0)
        events = maneuver_detection.detect_harsh_accel_decel(t_ns, accel_x)
        assert len(events) == 1
        assert events[0][4] == 'harsh_accel'
        assert events[0][3] > 0

    def test_harsh_decel_tagged_by_negative_sign(self):
        t_ns, accel_x = self._pulse(peak=-5.0)
        events = maneuver_detection.detect_harsh_accel_decel(t_ns, accel_x)
        assert len(events) == 1
        assert events[0][4] == 'harsh_decel'
        assert events[0][3] < 0

    def test_below_threshold_not_tagged(self):
        t_ns, accel_x = self._pulse(peak=1.0)  # well under the ~3.43 m/s^2 default
        assert maneuver_detection.detect_harsh_accel_decel(t_ns, accel_x) == []

    def test_empty_and_near_empty_arrays_do_not_crash(self):
        assert maneuver_detection.detect_harsh_accel_decel(
            np.array([], dtype=np.int64), np.array([])) == []
        assert maneuver_detection.detect_harsh_accel_decel(
            np.array([0], dtype=np.int64), np.array([10.0])) == []


# ── detect_stops ───────────────────────────────────────────────────────────────

class TestDetectStops:
    def test_stationary_segment_tagged_as_stop(self):
        lat0, lon0 = 35.0, 139.0
        # 5s driving, then 3s stopped (same position), then 5s driving again.
        n_drive1, n_stop, n_drive2 = 5, 3, 5
        t_ns = (np.arange(n_drive1 + n_stop + n_drive2) * 1e9).astype(np.int64)
        y = np.concatenate([
            np.arange(n_drive1) * 10.0,
            np.full(n_stop, (n_drive1 - 1) * 10.0),
            (n_drive1 - 1) * 10.0 + np.arange(1, n_drive2 + 1) * 10.0,
        ])
        lat, lon = plot_gps_route.enu_to_latlon(np.zeros(len(y)), y, lat0, lon0)
        events = maneuver_detection.detect_stops(t_ns, lat, lon, min_duration_s=2.0)
        assert len(events) == 1
        assert events[0][4] == 'stop'
        assert events[0][3] >= 2.0

    def test_continuous_driving_produces_no_stop(self):
        lat0, lon0 = 35.0, 139.0
        n = 20
        t_ns = (np.arange(n) * 1e9).astype(np.int64)
        y = np.arange(n) * 10.0  # steady 10 m/s throughout
        lat, lon = plot_gps_route.enu_to_latlon(np.zeros(n), y, lat0, lon0)
        assert maneuver_detection.detect_stops(t_ns, lat, lon) == []

    def test_brief_low_speed_below_min_duration_not_tagged(self):
        lat0, lon0 = 35.0, 139.0
        # Only 1s of near-zero speed — below the 2.0s default min_duration_s.
        t_ns = np.array([0, 1, 2, 3], dtype=np.int64) * int(1e9)
        y = np.array([0.0, 10.0, 10.0, 20.0])
        lat, lon = plot_gps_route.enu_to_latlon(np.zeros(4), y, lat0, lon0)
        assert maneuver_detection.detect_stops(t_ns, lat, lon) == []

    def test_empty_and_near_empty_no_crash(self):
        assert maneuver_detection.detect_stops(
            np.array([], dtype=np.int64), np.array([]), np.array([])) == []
        assert maneuver_detection.detect_stops(
            np.array([0], dtype=np.int64), np.array([35.0]), np.array([139.0])) == []


# ── map_events_to_fixes ────────────────────────────────────────────────────────

class TestMapEventsToFixes:
    def test_event_within_range_maps_to_nearest_fix(self):
        fix_t_ns = np.array([0, 10, 20, 30], dtype=np.int64)
        fix_lat = np.array([35.0, 35.001, 35.002, 35.003])
        fix_lon = np.array([139.0, 139.0, 139.0, 139.0])
        events = [(9, 11, 13, 0.2, 'lane_change')]  # peak at t=11, closer to fix idx 1 (t=10)
        mapped = maneuver_detection.map_events_to_fixes(events, fix_t_ns, fix_lat, fix_lon)
        assert len(mapped) == 1
        assert mapped[0][1] == pytest.approx(35.001)
        assert mapped[0][3] == 'lane_change'

    def test_event_before_range_clamps_to_first_fix(self):
        fix_t_ns = np.array([100, 200], dtype=np.int64)
        fix_lat = np.array([35.0, 35.001])
        fix_lon = np.array([139.0, 139.0])
        events = [(0, 5, 10, 0.2, 'harsh_decel')]  # well before the fix range
        mapped = maneuver_detection.map_events_to_fixes(events, fix_t_ns, fix_lat, fix_lon)
        assert len(mapped) == 1, 'event outside fix range must be clamped, not dropped'
        assert mapped[0][1] == pytest.approx(35.0)

    def test_event_after_range_clamps_to_last_fix(self):
        fix_t_ns = np.array([100, 200], dtype=np.int64)
        fix_lat = np.array([35.0, 35.001])
        fix_lon = np.array([139.0, 139.0])
        events = [(900, 950, 1000, 0.2, 'stop')]
        mapped = maneuver_detection.map_events_to_fixes(events, fix_t_ns, fix_lat, fix_lon)
        assert len(mapped) == 1
        assert mapped[0][1] == pytest.approx(35.001)

    def test_no_fixes_returns_empty(self):
        events = [(0, 1, 2, 0.2, 'lane_change')]
        mapped = maneuver_detection.map_events_to_fixes(
            events, np.array([], dtype=np.int64), np.array([]), np.array([]))
        assert mapped == []


# ── Pinned regression fixture: real ground-truth bag (T6 validation) ────────

FIXTURE_PATH = Path(__file__).parent / 'fixtures' / 'ground_truth_yaw_rate.npz'


class TestCalibrationAgainstGroundTruthBag:
    """Pinned regression test for the T6 validation spike (design doc "The
    Assignment"). The fixture holds real angular_velocity.z/orientation data
    sliced from two confirmed windows of
    HRdqo3pf_dmcYiVVH_2026-07-16T16-36-06+0900: a real turn (~0-15s, peak
    ~29 deg/s, ~79 deg net heading change) and a window (~145-200s) with
    exactly 3 user-confirmed lane changes at ~153-157s, ~172-178s, and
    ~182-183s, plus one road-curvature false positive at ~194-196s (the user
    reported the tool initially found 6 events here — 2 real lane changes
    were each double-counted as opposite-sign pulse pairs, and the tail-end
    road curvature was misclassified as a 4th lane change). This pins the
    calibration (thresholds, merge-gap, and return-to-baseline check) so a
    future change to the classifier can't silently regress on real data —
    not just a manual check that gets forgotten, per the eng-review's
    critical-gap finding.
    """

    def test_confirmed_turn_window_produces_no_lane_change_events(self):
        fx = np.load(FIXTURE_PATH)
        events = maneuver_detection.detect_lane_changes(
            fx['turn_t_ns'], fx['turn_yaw_rate'], fx['turn_heading'])
        assert events == [], (
            'the confirmed real-turn window must not be misclassified as a lane change')

    def test_confirmed_lane_change_window_produces_exactly_the_three_confirmed_events(self):
        fx = np.load(FIXTURE_PATH)
        events = maneuver_detection.detect_lane_changes(
            fx['lc_t_ns'], fx['lc_yaw_rate'], fx['lc_heading'])
        assert len(events) == 3, (
            'expected exactly the 3 user-confirmed lane changes (2 merged S-curve '
            'pairs + 1 standalone), with the road-curvature false positive at '
            '~194-196s rejected by the return-to-baseline check')


# ── write_maneuver_events_csv ─────────────────────────────────────────────────

class TestWriteManeuverEventsCsv:
    def test_writes_rows_for_each_event(self, tmp_path):
        out = tmp_path / 'events.csv'
        events = [(123, 35.0, 139.0, 'lane_change', 0.3)]
        maneuver_detection.write_maneuver_events_csv(events, str(out))
        with open(out) as f:
            rows = list(csv.reader(f))
        assert rows[0] == ['t_ns', 'lat', 'lon', 'maneuver_type', 'magnitude']
        assert rows[1][3] == 'lane_change'

    def test_zero_events_writes_header_only_valid_file(self, tmp_path):
        out = tmp_path / 'events.csv'
        maneuver_detection.write_maneuver_events_csv([], str(out))
        assert out.exists(), 'a bag with zero detected maneuvers must still produce a valid CSV'
        with open(out) as f:
            rows = list(csv.reader(f))
        assert rows == [['t_ns', 'lat', 'lon', 'maneuver_type', 'magnitude']]

    def test_creates_output_directory_if_missing(self, tmp_path):
        out = tmp_path / 'nested' / 'dir' / 'events.csv'
        maneuver_detection.write_maneuver_events_csv([], str(out))
        assert out.exists()
