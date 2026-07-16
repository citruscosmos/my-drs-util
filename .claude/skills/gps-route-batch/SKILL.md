---
name: gps-route-batch
description: Run tools/gps-route over every immediate subfolder of a given parent directory (e.g. an ecu0/ recordings folder), then collect all generated *_gps-route.png files into a sibling "gps-route" folder next to the parent directory. Use when the user asks to run gps-route / generate GPS route maps for "all folders under X" or similar batch requests.
---

# gps-route-batch

Batch-runs `tools/gps-route/batch_plot_gps_route.py` over every immediate
subfolder of a given directory, then consolidates the resulting PNGs into a
`gps-route/` folder that sits **next to** (same hierarchy level as) the input
directory — not inside it.

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
`*.mcap`). Run from the repo root (no special venv activation needed; deps
are importable from the system `python3`):

```bash
cd <repo_root>/tools/gps-route
python3 batch_plot_gps_route.py <input_dir>
```

This writes `<subfolder-name>_gps-route.png` back into each subfolder itself
(default `--out-dir` behavior — no `--out-dir` flag needed). A failure in one
subfolder (no NavSatFix topic, no .mcap files, etc.) is logged and skipped;
the batch continues. The final line reports `[batch] done: N ok, M failed,
out of T subfolder(s)` — this run can take a while for large bags (roughly
15-20s per subfolder observed in practice), so prefer running it in the
background and polling the log rather than blocking.

## Step 3 — Consolidate PNGs into the sibling gps-route folder

The destination is the parent of `<input_dir>`, with a `gps-route` folder
alongside it — i.e. `dirname(<input_dir>)/gps-route`, NOT
`<input_dir>/gps-route`.

```bash
dest="$(dirname "<input_dir>")/gps-route"
mkdir -p "$dest"
find "<input_dir>" -maxdepth 2 -name "*_gps-route.png" -exec mv -t "$dest" {} +
```

## Step 4 — Verify and report

```bash
ls "$dest" | wc -l
find "<input_dir>" -maxdepth 2 -name "*_gps-route.png" | wc -l   # should be 0 after the move
```

Report to the user: how many subfolders were processed, success/fail counts
from the batch tool's summary line, and the final PNG count in the
destination folder. Raw recording subfolders under `<input_dir>` are never
moved or deleted by this skill — only the generated PNGs.
