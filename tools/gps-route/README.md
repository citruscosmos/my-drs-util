# gps-route

Finds `sensor_msgs/msg/NavSatFix` topics in an MCAP bag (by schema, not by
hardcoded topic name) and plots each one's GPS route as a PNG.

| Script | Role |
|---|---|
| `plot_gps_route.py` | Plot the GPS route for a single `.mcap` file or split rosbag2 directory |
| `batch_plot_gps_route.py` | Run `plot_gps_route.py` over every immediate subfolder of a parent directory |

## Dependencies

- `mcap` — MCAP file reading
- `mcap-ros2-support` — decodes `oxts_msgs/msg/Ncom` directly from its schema embedded in the bag (no `oxts_msgs` ROS package needed)
- `rclpy` / `sensor_msgs` — ROS 2 message deserialization for `NavSatFix` (requires a sourced ROS 2 environment)
- `numpy`
- `matplotlib`
- `contextily` — OpenStreetMap tile basemap (requires network access to `tile.openstreetmap.org`)

## Usage

```bash
python3 plot_gps_route.py <mcap-or-dir> [--out-dir OUT] [--topics t1,t2] \
    [--require-fix] [--cov-sigma N] [--no-basemap] [--max-speed N]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `input` | ✓ | — | Single `.mcap` file, or a split rosbag2 directory (`*.mcap` + `metadata.yaml`) |
| `--out-dir` | | alongside `input` | Output directory for PNGs |
| `--topics` | | auto-detect | Comma-separated NavSatFix topics to restrict to; omit to auto-detect all topics whose schema is `sensor_msgs/msg/NavSatFix` |
| `--require-fix` | | off | Drop points with `status.status < 0` (`NO_FIX`) before plotting |
| `--cov-sigma` | | `1.0` | Covariance corridor half-width, in multiples of the 1-sigma horizontal std dev (`sqrt(mean(cov_xx, cov_yy))`). `0` disables the corridor. |
| `--no-basemap` | | off | Skip fetching the OpenStreetMap tile background (for offline use) |
| `--max-speed` | | `50.0` | Max plausible implied speed in m/s between consecutive fixes (180 km/h). Points implying a faster speed are excluded as a "Speed Jump" (see below). `0` disables the check. |

One PNG is written per matched topic: `gps_route_<sanitized_topic>.png`, plotted
over an OpenStreetMap tile background (Web Mercator projection), with the
start (☆) and end (X) marked in blue. A pair of blue dashed lines runs
parallel to the trajectory at ±`cov-sigma` sigma, showing the reported
horizontal position uncertainty as a corridor around the path.

**Coloring — GPS position mode vs. NavSatFix status:** `sensor_msgs/msg/NavSatFix.status.status`
only distinguishes 4 coarse levels (`NO_FIX`/`FIX`/`SBAS_FIX`/`GBAS_FIX`) and
cannot tell an RTK **float** solution (ambiguity unresolved, decimeter-level)
from an RTK **integer** solution (ambiguity resolved, centimeter-level). If
the bag also has an `oxts_msgs/msg/Ncom` topic (OxTS's raw 72-byte NCOM
packet), this tool decodes the richer OxTS "GPS position mode" from it
instead and colors by that:

| Color | Modes |
|---|---|
| red | `None`, `Search`, `No Data`, `Blanked`, `Not Recognised`, `Unknown` |
| orange | `Doppler`, `SPS` (incl. `(PP)`, `gx*`, `ix*` variants) — autonomous, no correction |
| gold | `Differential`, `WAAS`, `Omnistar*`, `CDGPS` (incl. `(PP)`, `gx*`, `ix*` variants) — meter-level, externally corrected |
| **dodgerblue** | **`RTK Float`** (incl. `(PP)`, `gxFloat`, `ixFloat`) — ambiguity unresolved |
| **green** | **`RTK Integer`** (incl. `(PP)`, `gxInteger`, `ixInteger`) — ambiguity resolved, best precision |
| gray | `Unknown (pre-Ncom)` — before the first decoded GPS-position-mode update (see below) |

The legend title ("GPS pos mode (Ncom)" vs. "status.status (NavSatFix)")
shows which source was used. If no `oxts_msgs/msg/Ncom` topic is found in the
bag, coloring automatically falls back to the plain `NavSatFix.status.status`
4-level scheme (red=NO_FIX, orange=FIX, gold=SBAS_FIX, green=GBAS_FIX).

The GPS position mode is decoded straight from the NCOM packet's embedded
schema (via `mcap_ros2`), without depending on the vehicle-specific
`oxts_msgs` ROS package. NCOM cycles different kinds of status data across
packets via a rotating "channel" byte; the position mode only appears in
channel 0, at a fixed byte offset, and is forward-filled onto each
`NavSatFix` fix's timestamp (last known mode at-or-before that time). Fixes
timestamped before the very first decoded update are colored gray/"Unknown
(pre-Ncom)". See OxTS's own `NComRxC.c`/`.h` reference decoder
(`DecodeExtra0`, `PacketIndexes`, `COM_GPS_XMODE_TYPE_NAME`) for the
authoritative packet layout this reimplements.

Two kinds of glitch points are detected and excluded from the plotted route,
each reported with a red note at the bottom-left of the plot (and printed to
stdout):

- **0 Jump** — fixes reporting exactly `latitude == 0` and `longitude == 0`
  ("null island", a common GPS/INS sentinel for an invalid fix), which would
  otherwise draw a spurious jump to the equator/prime meridian.
- **Speed Jump** — fixes implying a vehicle speed faster than `--max-speed`
  both to and from their neighbors (an isolated GPS teleport). Speed is
  computed from consecutive fixes in a local ENU tangent-plane, using the
  already-jump-filtered sequence. A point is excluded only when *both* the
  speed into it and the speed out of it exceed the threshold, so ordinary
  hard acceleration/deceleration is not mistaken for a glitch; sequence
  endpoints use whichever single side is available.

The very first (finite) fix is never excluded by either jump check, even if
it would otherwise qualify — so a map is always produced, with at least that
one point, even for a bag where the entire GPS trace looks like jumps/outliers
(e.g. a bag with no real fix at all, where every consecutive pair implies an
impossible speed).

**Examples:**

```bash
# Single mcap file
python3 plot_gps_route.py recording.mcap --out-dir ./out

# Split rosbag2 directory (multiple *.mcap + metadata.yaml)
python3 plot_gps_route.py /path/to/recording_dir --out-dir ./out

# Offline / no network access to the OSM tile server
python3 plot_gps_route.py recording.mcap --out-dir ./out --no-basemap

# Wider 2-sigma (~95%) corridor
python3 plot_gps_route.py recording.mcap --out-dir ./out --cov-sigma 2

# Stricter speed-jump threshold (36 km/h) for slow-speed parking-lot tests
python3 plot_gps_route.py recording.mcap --out-dir ./out --max-speed 10
```

## Batch processing: `batch_plot_gps_route.py`

Runs `plot_gps_route.py` once per immediate subfolder of a parent directory —
e.g. an `ecu0/` folder containing one split-rosbag2 directory per recording.
A failure in one subfolder (no `NavSatFix` topic, no `.mcap` files, etc.) is
logged and skipped; the batch continues with the rest.

```bash
python3 batch_plot_gps_route.py <parent_dir> [--out-dir OUT] [-- <plot_gps_route.py args>]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `parent_dir` | ✓ | — | Directory whose immediate subfolders are each processed as one bag |
| `--out-dir` | | write into each subfolder itself | Output root; when given, each subfolder's PNGs are mirrored to `<out-dir>/<subfolder name>/` instead of writing back into the (possibly read-only) source tree |
| `-- ...` | | — | Anything after `--` is forwarded verbatim to `plot_gps_route.py` (e.g. `--no-basemap`, `--cov-sigma`, `--max-speed`, `--require-fix`) |

**Examples:**

```bash
# Process every recording under ecu0/, writing PNGs back into each recording's own folder
python3 batch_plot_gps_route.py /mnt/lstr/rosbag/.../ecu0

# Mirror output to a separate tree instead, and skip the OSM basemap fetch
python3 batch_plot_gps_route.py /mnt/lstr/rosbag/.../ecu0 --out-dir ./out -- --no-basemap
```

## Notes

- Topic discovery is schema-based, so it works regardless of the topic name.
- The covariance corridor is computed in a local ENU tangent-plane (true
  meters, from `position_covariance`) and only reprojected to Web Mercator for
  display, so its width is not distorted by the map projection. It collapses
  to zero at points where `position_covariance_type` is `UNKNOWN`.
- `status.status` and `position_covariance` are not always reliable indicators
  of fix quality on this vehicle's INS unit — a bag with no real GPS fix can
  still report `status == STATUS_FIX` while `latitude`/`longitude`/`altitude`
  are physically implausible (e.g. altitude in the thousands of km). If the
  route lands somewhere absurd on the map (e.g. open ocean), that indicates
  invalid fixes rather than a tool bug.
- Basemap tiles are fetched live from the public OpenStreetMap tile server on
  every run (no local caching); heavy/repeated batch use should point
  `contextily` at a different tile provider or a local tile cache instead, per
  [OSM's tile usage policy](https://operations.osmfoundation.org/policies/tiles/).
