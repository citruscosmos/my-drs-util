---
name: gps-route-batch
description: Run tools/gps-route over every immediate subfolder of a given parent directory (e.g. an ecu0/ recordings folder), tagging vehicle maneuvers (lane change, harsh accel/decel, stop) and elapsed-time ticks on each map, then collect all generated *_gps-route.png and *_maneuvers.csv files into a sibling "gps-route" folder next to the parent directory. Use when the user asks to run gps-route / generate GPS route maps for "all folders under X" or similar batch requests.
---

# gps-route-batch

Batch-runs `tools/gps-route/batch_plot_gps_route.py` over every immediate
subfolder of a given directory, then consolidates the resulting PNGs (and
maneuver CSV sidecars) into a `gps-route/` folder that sits **next to** (same
hierarchy level as) the input directory — not inside it.

Example: input `/media/autoware/COMLOPS/ecu0` → output PNGs end up in
`/media/autoware/COMLOPS/gps-route/`.

## Step 1 — Determine the input directory

The user's request names the parent directory (e.g. "ecu0 配下のフォルダ全てに
gps-route を実施したい"). If not given, ask for the absolute path. Verify it
exists and contains subfolders:

```bash
ls -la <input_dir>
```

## Step 2 — Run the batch tool

Each subfolder is expected to be a single `.mcap` file or a split rosbag2
directory (`*.mcap` files, `metadata.yaml` optional — the tool globs
`*.mcap`). The system `python3` does NOT have the required deps (`mcap`,
`rclpy`, `sensor_msgs`) — use the repo's `.venv` instead:

```bash
cd <repo_root>/tools/gps-route
<repo_root>/.venv/bin/python3 batch_plot_gps_route.py <input_dir> -- --tag-maneuvers
```

**Before running at scale, confirm the OSM basemap actually renders** — the
map is otherwise blank (no roads/imagery, just the plotted route on a white
background), which defeats the point of a route map. Check for
`contextily`, the tile-fetching lib, in the venv first:

```bash
<repo_root>/.venv/bin/python3 -c "import contextily" || <repo_root>/.venv/bin/python3 -m pip install contextily
```

If missing, `plot_gps_route.py` silently degrades — it prints
`[warn] basemap fetch failed (No module named 'contextily'); plotting
without it` to stderr and still exits 0, so a naive glance at "N ok, 0
failed" will not catch a batch that ran with no basemap at all. Do one
single-folder run first and open the resulting PNG to confirm the map tiles
are actually there before committing to a long batch run.

`--tag-maneuvers` is opt-in on `plot_gps_route.py` (default off) and is what
enables the operation-event overlay: it detects lane-change / harsh-accel /
harsh-decel / stop events (from a `sensor_msgs/msg/Imu` topic, falling back
to a lower-accuracy GPS-position-derivative estimate when no Imu topic
exists) and marks them on the map, plus writes a
`<subfolder-name>_maneuvers.csv` sidecar per subfolder. Everything after
`--` is forwarded verbatim to `plot_gps_route.py`, so other flags
(`--no-basemap`, `--cov-sigma`, etc.) can be appended the same way if asked
for. Elapsed-time tick labels (every 30s along the route, `--time-tick-interval`
to change) are always on — no flag needed unless the user wants a different
interval or to disable them (`--time-tick-interval 0`).

This writes `<subfolder-name>_gps-route.png` (and, with `--tag-maneuvers`,
`<subfolder-name>_maneuvers.csv`) back into each subfolder itself (default
`--out-dir` behavior — no `--out-dir` flag needed). A failure in one
subfolder (no NavSatFix topic, no .mcap files, etc.) is logged and skipped;
the batch continues. The final line reports `[batch] done: N ok, M failed,
out of T subfolder(s)` — this run can take a while for large bags (roughly
15-20s per subfolder observed in practice), so prefer running it in the
background and polling the log rather than blocking.

## Step 3 — Consolidate PNGs and maneuver CSVs into the sibling gps-route folder

The destination is the parent of `<input_dir>`, with a `gps-route` folder
alongside it — i.e. `dirname(<input_dir>)/gps-route`, NOT
`<input_dir>/gps-route`.

```bash
dest="$(dirname "<input_dir>")/gps-route"
mkdir -p "$dest"
find "<input_dir>" -maxdepth 2 \( -name "*_gps-route.png" -o -name "*_maneuvers.csv" \) -exec mv -t "$dest" {} +
```

## Step 4 — Verify and report

```bash
ls "$dest"/*_gps-route.png | wc -l
ls "$dest"/*_maneuvers.csv | wc -l
find "<input_dir>" -maxdepth 2 \( -name "*_gps-route.png" -o -name "*_maneuvers.csv" \) | wc -l   # should be 0 after the move
```

Also grep the batch log for basemap failures — a per-tile fetch failure
(e.g. a transient DNS blip resolving `tile.openstreetmap.org`) doesn't fail
the subfolder, it just silently produces a blank-background map for it:

```bash
grep -B8 "basemap fetch failed" <batch_log>   # shows which "[batch] (i/N) <subfolder>" runs were affected
```

For any affected subfolder, re-run just that one directly (not through the
batch script) once network access is confirmed working, e.g.:

```bash
<repo_root>/.venv/bin/python3 plot_gps_route.py <input_dir>/<subfolder> --out-dir <input_dir>/<subfolder> --tag-maneuvers
```

If the destination `gps-route` folder is owned by `root` with no write
access for the current user (seen on NAS-mounted parents), `mv` in Step 3
fails per-file with "permission denied" while still reporting nonzero
counts moved for others — ask the user to `chmod`/`chown` it, then re-run
Step 3.

Report to the user: how many subfolders were processed, success/fail counts
from the batch tool's summary line, and the final PNG/CSV counts in the
destination folder. Raw recording subfolders under `<input_dir>` are never
moved or deleted by this skill — only the generated PNGs and CSV sidecars.
