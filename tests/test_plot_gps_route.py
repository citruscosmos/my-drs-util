"""Baseline regression tests for tools/gps-route/plot_gps_route.py.

Covers the pure math/utility functions that existed before the maneuver-
tagging feature was added, so future changes to the shared NCOM/ENU
scaffolding (reused by tools/gps-route/maneuver_detection.py) can't silently
break this tool's existing GPS-route plotting behavior.

`tools/gps-route` has a hyphen in its name, so it can't be `import`ed as a
normal package; load the script directly via importlib instead.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS_DIR = Path(__file__).parent.parent / 'tools' / 'gps-route'
MODULE_PATH = TOOLS_DIR / 'plot_gps_route.py'

# plot_gps_route.py imports maneuver_detection at module level (sibling
# script, resolved via sys.path when run directly) — add the directory so
# that import resolves the same way here.
sys.path.insert(0, str(TOOLS_DIR))

_spec = importlib.util.spec_from_file_location('plot_gps_route', MODULE_PATH)
plot_gps_route = importlib.util.module_from_spec(_spec)
sys.modules['plot_gps_route'] = plot_gps_route
_spec.loader.exec_module(plot_gps_route)


# ── latlon_to_enu / enu_to_latlon round-trip ─────────────────────────────────

class TestEnuRoundTrip:
    def test_round_trip_at_origin(self):
        lat0, lon0 = 35.681236, 139.767125  # Tokyo Station, arbitrary origin
        lat, lon = np.array([lat0]), np.array([lon0])
        x, y = plot_gps_route.latlon_to_enu(lat, lon, lat0, lon0)
        assert x[0] == pytest.approx(0.0, abs=1e-6)
        assert y[0] == pytest.approx(0.0, abs=1e-6)

    def test_round_trip_offset_point(self):
        lat0, lon0 = 35.681236, 139.767125
        lat = np.array([35.682236, 35.680236])
        lon = np.array([139.768125, 139.766125])
        x, y = plot_gps_route.latlon_to_enu(lat, lon, lat0, lon0)
        lat2, lon2 = plot_gps_route.enu_to_latlon(x, y, lat0, lon0)
        assert lat2 == pytest.approx(lat, abs=1e-9)
        assert lon2 == pytest.approx(lon, abs=1e-9)

    def test_north_is_positive_y(self):
        """A point due north of the origin must have positive y, ~zero x."""
        lat0, lon0 = 35.0, 139.0
        lat, lon = np.array([35.001]), np.array([139.0])
        x, y = plot_gps_route.latlon_to_enu(lat, lon, lat0, lon0)
        assert y[0] > 0
        assert x[0] == pytest.approx(0.0, abs=1e-3)

    def test_east_is_positive_x(self):
        lat0, lon0 = 35.0, 139.0
        lat, lon = np.array([35.0]), np.array([139.001])
        x, y = plot_gps_route.latlon_to_enu(lat, lon, lat0, lon0)
        assert x[0] > 0
        assert y[0] == pytest.approx(0.0, abs=1e-3)


# ── detect_speed_jumps ────────────────────────────────────────────────────────

class TestDetectSpeedJumps:
    def _straight_line_track(self, n=10, speed_mps=10.0, dt=1.0, lat0=35.0, lon0=139.0):
        """n fixes driving due north at a constant speed_mps, dt seconds apart."""
        t_ns = (np.arange(n) * dt * 1e9).astype(np.int64)
        dy_per_step = speed_mps * dt
        y = np.arange(n) * dy_per_step
        lat, lon = plot_gps_route.enu_to_latlon(np.zeros(n), y, lat0, lon0)
        return t_ns, lat, lon

    def test_no_jumps_on_steady_track(self):
        t_ns, lat, lon = self._straight_line_track(speed_mps=10.0)
        bad = plot_gps_route.detect_speed_jumps(t_ns, lat, lon, max_speed_mps=50.0)
        assert not bad.any()

    def test_isolated_teleport_flagged(self):
        t_ns, lat, lon = self._straight_line_track(n=10, speed_mps=10.0)
        # Teleport fix #5 far away — implies an impossible speed both in and out.
        lat[5], lon[5] = 40.0, 145.0
        bad = plot_gps_route.detect_speed_jumps(t_ns, lat, lon, max_speed_mps=50.0)
        assert bad[5]
        assert not bad[4] and not bad[6], 'legitimate neighbors must not be flagged'

    def test_max_speed_zero_disables_check(self):
        t_ns, lat, lon = self._straight_line_track(n=5, speed_mps=100.0)
        bad = plot_gps_route.detect_speed_jumps(t_ns, lat, lon, max_speed_mps=0.0)
        assert not bad.any()

    def test_single_point_no_crash(self):
        bad = plot_gps_route.detect_speed_jumps(
            np.array([0], dtype=np.int64), np.array([35.0]), np.array([139.0]),
            max_speed_mps=50.0)
        assert bad.shape == (1,)
        assert not bad.any()

    def test_hard_braking_not_mistaken_for_jump(self):
        """A single point with fast speed in but slow speed out (real deceleration)
        must NOT be flagged — only points fast on BOTH sides are glitches."""
        lat0, lon0 = 35.0, 139.0
        t_ns = np.array([0, 1, 2], dtype=np.int64) * int(1e9)
        # y (north, meters): 0 -> 60 (60 m/s approach, exceeds threshold) -> 61 (1 m/s after braking).
        y = np.array([0.0, 60.0, 61.0])
        lat, lon = plot_gps_route.enu_to_latlon(np.zeros(3), y, lat0, lon0)
        bad = plot_gps_route.detect_speed_jumps(t_ns, lat, lon, max_speed_mps=50.0)
        assert not bad[1], 'one-sided fast approach (braking) must not be flagged as a jump'


# ── time_tick_points ──────────────────────────────────────────────────────────

class TestTimeTickPoints:
    def test_ticks_at_every_interval_excluding_zero(self):
        t_ns = (np.arange(0, 100, 1) * 1e9).astype(np.int64)  # 0..99s, 1 point/s
        lat = np.linspace(35.0, 35.001, len(t_ns))
        lon = np.full(len(t_ns), 139.0)
        points = plot_gps_route.time_tick_points(t_ns, lat, lon, interval_s=30.0)
        assert [p[0] for p in points] == [30.0, 60.0, 90.0]

    def test_zero_or_negative_interval_disables(self):
        t_ns = (np.arange(0, 100, 1) * 1e9).astype(np.int64)
        lat = lon = np.zeros(len(t_ns))
        assert plot_gps_route.time_tick_points(t_ns, lat, lon, interval_s=0) == []
        assert plot_gps_route.time_tick_points(t_ns, lat, lon, interval_s=-5) == []

    def test_track_shorter_than_interval_produces_no_ticks(self):
        t_ns = (np.arange(0, 10, 1) * 1e9).astype(np.int64)  # only 9s long
        lat = lon = np.zeros(len(t_ns))
        assert plot_gps_route.time_tick_points(t_ns, lat, lon, interval_s=30.0) == []

    def test_empty_input_no_crash(self):
        assert plot_gps_route.time_tick_points(
            np.array([], dtype=np.int64), np.array([]), np.array([]), interval_s=30.0) == []

    def test_tick_position_matches_nearest_fix(self):
        t_ns = np.array([0, 29, 31, 60], dtype=np.int64) * int(1e9)
        lat = np.array([35.0, 35.001, 35.002, 35.003])
        lon = np.full(4, 139.0)
        points = plot_gps_route.time_tick_points(t_ns, lat, lon, interval_s=30.0)
        assert points[0][0] == 30.0
        assert points[0][1] == pytest.approx(35.002)  # nearest to t=30s is t=31s (idx 2)


# ── asof_forward_fill ─────────────────────────────────────────────────────────

class TestAsofForwardFill:
    def test_forward_fills_last_known_value(self):
        ref_ts = np.array([0, 10, 20], dtype=np.int64)
        ref_val = np.array([1, 2, 3])
        query_ts = np.array([5, 15, 25])
        result = plot_gps_route.asof_forward_fill(query_ts, ref_ts, ref_val, fill_value=-1)
        assert list(result) == [1, 2, 3]

    def test_query_before_first_ref_uses_fill_value(self):
        ref_ts = np.array([10, 20], dtype=np.int64)
        ref_val = np.array([1, 2])
        query_ts = np.array([0, 15])
        result = plot_gps_route.asof_forward_fill(query_ts, ref_ts, ref_val, fill_value=-1)
        assert list(result) == [-1, 1]

    def test_empty_ref_returns_all_fill_value(self):
        result = plot_gps_route.asof_forward_fill(
            np.array([1, 2, 3]), np.array([], dtype=np.int64), np.array([]), fill_value=-1)
        assert list(result) == [-1, -1, -1]

    def test_exact_match_uses_that_value(self):
        ref_ts = np.array([0, 10, 20], dtype=np.int64)
        ref_val = np.array([1, 2, 3])
        query_ts = np.array([10])
        result = plot_gps_route.asof_forward_fill(query_ts, ref_ts, ref_val, fill_value=-1)
        assert list(result) == [2]


# ── covariance_corridor ───────────────────────────────────────────────────────

class TestCovarianceCorridor:
    def test_corridor_straddles_straight_path(self):
        lat = np.array([35.0, 35.001, 35.002])
        lon = np.array([139.0, 139.0, 139.0])
        cov_std = np.array([1.0, 1.0, 1.0])
        (l_lat, l_lon), (r_lat, r_lon) = plot_gps_route.covariance_corridor(
            lat, lon, cov_std, sigma=1.0)
        # Path runs due north (dlat > 0); corridor edges should be offset in longitude
        # (east/west), not latitude, for a north-south path.
        assert np.all(np.abs(l_lon - lon) > 1e-6)
        assert np.all(np.abs(r_lon - lon) > 1e-6)
        # Left and right edges must be on opposite sides.
        assert np.sign(l_lon[1] - lon[1]) != np.sign(r_lon[1] - lon[1])

    def test_zero_sigma_collapses_corridor_to_path(self):
        lat = np.array([35.0, 35.001, 35.002])
        lon = np.array([139.0, 139.0, 139.0])
        cov_std = np.array([1.0, 1.0, 1.0])
        (l_lat, l_lon), (r_lat, r_lon) = plot_gps_route.covariance_corridor(
            lat, lon, cov_std, sigma=0.0)
        assert l_lat == pytest.approx(lat, abs=1e-9)
        assert r_lon == pytest.approx(lon, abs=1e-9)


# ── sanitize_topic / input_basename ──────────────────────────────────────────

class TestSanitizeTopic:
    def test_strips_leading_slash_and_replaces_internal_slashes(self):
        assert plot_gps_route.sanitize_topic('/sensing/gnss/fix') == 'sensing_gnss_fix'

    def test_no_leading_slash(self):
        assert plot_gps_route.sanitize_topic('gnss/fix') == 'gnss_fix'


class TestInputBasename:
    def test_single_mcap_file_strips_extension(self, tmp_path):
        f = tmp_path / 'recording.mcap'
        f.write_bytes(b'')
        assert plot_gps_route.input_basename(str(f)) == 'recording'

    def test_directory_keeps_folder_name(self, tmp_path):
        d = tmp_path / 'my_recording_dir'
        d.mkdir()
        assert plot_gps_route.input_basename(str(d)) == 'my_recording_dir'

    def test_directory_with_trailing_slash(self, tmp_path):
        d = tmp_path / 'my_recording_dir'
        d.mkdir()
        assert plot_gps_route.input_basename(str(d) + '/') == 'my_recording_dir'


# ── resolve_mcap_files ────────────────────────────────────────────────────────

class TestResolveMcapFiles:
    def test_single_file(self, tmp_path):
        f = tmp_path / 'a.mcap'
        f.write_bytes(b'')
        assert plot_gps_route.resolve_mcap_files(str(f)) == [str(f)]

    def test_directory_returns_sorted_mcap_files(self, tmp_path):
        (tmp_path / 'b.mcap').write_bytes(b'')
        (tmp_path / 'a.mcap').write_bytes(b'')
        (tmp_path / 'metadata.yaml').write_bytes(b'')
        files = plot_gps_route.resolve_mcap_files(str(tmp_path))
        assert [Path(f).name for f in files] == ['a.mcap', 'b.mcap']

    def test_directory_with_no_mcap_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            plot_gps_route.resolve_mcap_files(str(tmp_path))

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            plot_gps_route.resolve_mcap_files(str(tmp_path / 'does_not_exist.mcap'))
