---
name: drs-ingest
description: End-to-end pipeline for a new jt10 recording group — detect the two USB SSDs (ecu0/ecu1, plugged into the fixed portA/portB slots), convert them to a T4/NuScenes dataset in local chunks via rosbag-data-pipeline while uploading the raw data to jt10/<group>/{ecu0,ecu1} over NFS (size-verified, SAFE TO WIPE only when verified), and generate GPS route maps + maneuver tags straight from the SSD. Use when the operator says they've plugged in the two recording SSDs and want to ingest a new jt10 group, or asks to run the full USB-to-t4dataset pipeline.
---

# drs-ingest

Chains three stages for one new jt10 recording group, triggered manually
after an operator physically plugs in both USB SSDs:

1. **Detect + upload** raw `ecu0`/`ecu1` data from the two USB SSDs to
   `raw/jt10/<group>/{ecu0,ecu1}` via Alluxio's S3 API, with persistence
   verification and a permission fix — `tools/drs-ingest/detect_and_upload.py`.
2. **GPS route generation** from the uploaded `ecu0` data — an ephemeral ROS2
   container running `tools/gps-route`, wrapped by
   `tools/drs-ingest/run_gps_route_ephemeral.sh`.
3. **T4/NuScenes conversion** from `ecu0`+`ecu1` — a direct call into the
   sibling repo's `rosbag-data-pipeline/python/run_batch_t4_pipeline.py`.

Design notes (verified 2026-09-16, see this session's history for the raw
investigation if anything below needs re-deriving):

- **NAS layout (2026-10-05):** raw data lives in `raw/<vehicle>/<group>/`
  (`/mnt/DataPool/SharedData/raw/jt10/<group>/{ecu0,ecu1,gps-route}`); the converted
  datasets live in `datasets/<group>/` (see `docs/t4dataset-v1-shard-ext.md`). The old
  top-level vehicle dirs (`jt10/`, `id1/`, ...) no longer exist.
- The raw data is reachable via Alluxio's S3 API as bucket `raw`
  (`s3://raw/jt10/<group>/...`; the bucket is the top-level NAS dir). It may not
  appear in a plain `aws s3 ls` bucket listing — addressing `s3://raw/...`
  directly works. Note: with no Alluxio *worker* registered (observed 2026-10-05:
  `alluxio fsadmin report capacity` says "No workers found"), S3 reads of metadata and
  listings work but S3 uploads fail with "No available Alluxio worker found" —
  use `--transfer-mode nfs`. The worker container
  `petadata-alluxio-worker-warm-1` bind-mounts the host's
  `/mnt/DataPool/SharedData` (the NAS's NFS export) at
  `/opt/alluxio/underFSStorage` — S3 writes and direct-NFS writes land on
  the exact same files, just via two access paths.
- S3-written objects come back `root:root` / `rwxrwx---` (no "other"
  access at all) — fixed via `docker exec petadata-alluxio-worker-warm-1
  chmod -R o+rwX ...` (that container's root has no NFS root-squash issue; a
  plain `shimaki`-user `chmod`/`rm` over the direct NFS path gets refused).
  **Other-write, not just other-read**, is needed — every pre-existing jt10
  group directory is `drwxrwxrwx`, and Stage 2's ephemeral (unprivileged)
  container must be able to create new files under `gps-route/` without any
  further root trick. `detect_and_upload.py` also pre-creates `gps-route/`
  (empty) at this step, with this open permission, so
  downstream stages never need to write into the (deliberately left
  more-restrictive) `ecu0/`/`ecu1/` raw-data tree at all — confirmed
  directly: `batch_plot_gps_route.py`'s default "write output alongside the
  input" behavior gets a `PermissionError` there, which is why Stage 2 always
  passes `--out-dir <group>/gps-route` instead (nests one level per
  recording; `run_gps_route_ephemeral.sh` flattens it back to match every
  other group's layout).
  A chmod immediately after a *large* (multi-GB) upload can also be silently
  lost — Alluxio's async persist-to-UFS finalization for a big file can still
  be in flight for a few seconds after the S3 API reports the object present,
  and a chmod racing that finalization gets clobbered when the finalization
  completes (confirmed directly: reproducible on a 2.3GB test upload, not on
  a 37-byte one). `fix_permissions()` retries+re-verifies (`find ... !
  -perm -o+rw`) rather than firing the chmod once.
- Alluxio only **discovers** a child entry that was created out-of-band
  (`docker exec mkdir`, or Stage 2's direct-NFS-equivalent write into
  `gps-route/`) once its **parent directory has itself been listed** — a
  listing of the nested path alone can come back empty even though the files
  are already on disk (confirmed directly: listing `.../gps-route/` before
  ever listing the group root returned nothing; listing the group root first,
  then retrying, fixed it immediately). `run_gps_route_ephemeral.sh`'s
  verification step always lists the group root once before checking
  `gps-route/`'s contents.
- Whichever USB port a drive lands on is **not fixed** — `ecu0`/`ecu1` is
  determined by scanning folder names after mount, not by port number.
- Write-type safety: a sibling Alluxio proxy had a real data-loss incident
  from `ASYNC_THROUGH` writes stuck un-persisted (fixed 2026-09-16). Since
  the client can't force a write-type on this endpoint, Stage 1 always
  verifies persistence by object-count after upload rather than trusting
  the upload call's exit code alone — only recordings that verify are
  reported safe to wipe the source SSD.
- Only Stage 3 (T4 conversion) does a real "bring the full raw data
  locally" step, and it's internal to `run_batch_t4_pipeline.py` itself
  (unconditional `rsync`-to-`workspace/` staging when `--docker-container`
  is used) — not duplicated by this skill. Stage 2 never stages a local
  copy: `tools/gps-route` reads only a few narrow topics (NavSatFix/Imu/
  Ncom) via mcap's indexed reader, not the full multi-GB payload — this was
  empirically confirmed this session (a 93GB/27-file recording still
  processed in the usual ~15-20s).
- Stage 2 writes its PNG/CSV output directly to the jt10 NFS mount (not via
  S3), so `run_gps_route_ephemeral.sh` explicitly re-verifies the result is
  visible via the S3 API afterward (Alluxio's own UFS metadata sync was
  confirmed to pick up direct-NFS writes within seconds in testing, but the
  script checks rather than assumes).
- Stage 3 relies on `run_batch_t4_pipeline.py`'s own lock file
  (`workspace/.batch_pipeline.lock`) for mutual exclusion — do **not** add
  a second lock. Another session may already be converting jt10 groups
  (observed directly this session); an `exit(1)` from an already-held lock
  means "did not run because another batch is active," not a failure.
- Stage 1 always checks each mounted USB port's link speed before scanning
  (`usb_link_speed_mbps()`, sysfs-based) and warns — non-fatally — if it's
  below USB 3 (default threshold 5000 Mbps). Confirmed directly (2026-09-24):
  two identical SSDs can differ by ~23x in real read throughput (43.6 MB/s
  USB 2.0 vs 984 MB/s USB 3) purely from a port/cable link-speed difference
  that mounting doesn't otherwise surface — a `[warn] portX is linked below
  USB 3 speed` line means reseat the cable/port before a large upload, not
  that anything failed.
- Stage 1 also supports `--transfer-mode nfs` as an opt-in alternative to
  the default `--transfer-mode s3`: writes directly to `--nas-root` via
  rsync (same approach as Stage 3's tar+rsync copy-to-jt10) instead of
  through Alluxio's S3 API. Measured ~3x faster (397 MB/s NFS-direct vs
  138 MB/s through the S3 gateway on this deployment — the S3 gateway
  itself is the bottleneck, not the NAS). Because this bypasses Alluxio's
  own write path, it runs a one-time `alluxio fs loadMetadata -F
  /<bucket>/<group>` at the end so the group becomes visible via the S3
  API / to Stage 2 and 3 (same mechanism `run_batch_t4_pipeline.py` added
  for its own tar+rsync mode). Combined with fixing a USB 2.0 link (see
  above), a 2.1TB group's Stage 1 was measured/projected to drop from
  ~12h27m to ~1h30m. Stays opt-in (default `s3`) until validated on a real
  group in production — pass `--transfer-mode nfs` explicitly to use it.

## Ingest flow (default since 2026-10-01)

Reading the multi-GB MCAPs back from the NAS over NFS is what made Stage 2/3 slow (Stage 2 dropped to
~36MB/s; the NFS-written files show `0% in Alluxio`, so Alluxio's cache is not in that read path).
The ingest flow therefore never reads raw data from the NAS: everything is read from the source SSDs
(USB 3, ~1GB/s) or from a local copy, and the raw data is uploaded to the NAS in the background.

1. **Prepare** — detect the SSDs, check link speed, create `<group>/{ecu0,ecu1,gps-route}`
   on the NAS (root-owned parents need `docker exec`, done by the script):
   ```bash
   python3 tools/drs-ingest/detect_and_upload.py --group <group> --prepare-only
   ```
   It prints `INGEST_ECU0=...`, `INGEST_ECU1=...`, `INGEST_GROUP_DIR=...` for the next steps. Same
   `[warn] portX is linked below USB 3 speed` handling as Stage 1 below.

2. **Convert + upload raw (one process)** — `run_batch_t4_pipeline.py` in ingest mode (needs the sibling
   `rosbag-data-pipeline` repo, located as in Step 3 below; this branch needs `feat/chunked-parallel-pipeline`
   or later):
   ```bash
   cd "$ROSBAG_PIPELINE_REPO"
   nohup python3 python/run_batch_t4_pipeline.py "$INGEST_GROUP_DIR" \
     --ingest-ecu0 "$INGEST_ECU0" --ingest-ecu1 "$INGEST_ECU1" [--ingest-exclude-date YYYY-MM-DD ...] \
     --convert-config config/convert_rosbag2_to_non_annotated_t4_fisheye_delay.yaml \
     --duration 30 --process-mode balanced --merge-mode smart --min-overlap-sec 5 \
     --driver auto --max-parallel 4 --docker-container rosbag-pipeline-dev \
     --output-format tar --alluxio-worker-container petadata-alluxio-worker-warm-1 \
     > <scratchpad>/<group>_ingest.log 2>&1 &
   ```
   Per chunk (default 10 ECU0 files, 2 chunks in parallel, 600GB local budget): copy from the SSD to the
   local workspace -> merge/decode/convert from the local copy while the chunk's raw files upload to
   `<group>/{ecu0,ecu1}/` in the background -> (after the raw upload is size-verified) copy the converted
   segments to the corpus `datasets/<group>/` (next to the vehicle dirs; `--datasets-root` overrides). The log says
   whether each chunk's raw upload was hidden behind processing. The tar shards go straight into
   `datasets/<group>/<session>/`; at the end the corpus index `_index/` and `_manifest.json` are written
   (format `t4dataset-v1-shard-ext`, see `docs/t4dataset-v1-shard-ext.md`).
   Failure/exit semantics are those of Step 3 (lock file, `Batch run complete: processed=X failed=Y`).
   `--output-format tar` is recommended: the converted segment copy of ~3800 small files per segment is
   otherwise the slow step; consumers of the dataset must read the tar shards.

   At the end it re-verifies **every** raw file of the SSD against the NAS by size (also MCAPs skipped as
   unreadable and recordings without an ecu0/ecu1 partner), repairs what is missing, fixes permissions
   (`chmod -R o+rwX` as root in the Alluxio worker container), syncs Alluxio metadata, and prints
   `SAFE TO WIPE: ...` or `DO NOT WIPE: ...`. **The SSDs must stay plugged in until that line appears.**
   Recordings excluded with `--ingest-exclude-date` are not uploaded (the script warns).

3. **GPS routes, from the SSD** — can run at the same time as step 2 (they share the SSD's read bandwidth):
   ```bash
   tools/drs-ingest/run_gps_route_ephemeral.sh "$INGEST_GROUP_DIR" --ecu0-dir "$INGEST_ECU0" \
     [--exclude-date YYYY-MM-DD ...]
   ```
   Same warnings as Step 2 below (basemap failures, GPS-quality flags).

4. **Final report** as in Step 4 below, taking `SAFE TO WIPE` from step 2.

Token collisions: the converter also occasionally gives two records the same token (sample_data.json: broken
prev/next links; ego_pose.json: two frames resolve to the same pose) or writes a number-like token without
quotes (invalid JSON). The pipeline's pre-copy gate catches duplicates and re-converts; for data that is
already on the NAS use `python/scan_t4_links.py <root> --check-files --out report.json` (read-only) and
`python/repair_t4_tokens.py` (dry-run by default; `--apply --backup-dir ... --root-container
petadata-alluxio-worker-warm-1` repairs in place with backups, atomically, keeping owner/mode) in the
rosbag-data-pipeline repo. 2026-10-04: all of jt10 was repaired this way (269 segments).

Known issues worth reporting: an MCAP without a summary section (a recording that was never finalized — the
pipeline skips it with a warning, and Stage 2's gps-route fails on that recording); the converter
occasionally emits segments with broken prev/next links (the pipeline validates and re-runs convert up to
`--convert-retries` times, and fails the chunk rather than copying a bad segment).

The sections below describe the previous flow (upload to the NAS first, then process from the NAS). Use it
only for an upload-only run or when the ingest flow cannot be used.

## Step 0 — Prerequisites and group naming

```bash
mount | grep /mnt/ssd
```

Both `/mnt/ssd/portA` and `/mnt/ssd/portB` should be listed as mounted
(the existing `/etc/udev/rules.d/99-usb-ssd.rules` auto-mounts them on USB
insertion). If a port isn't mounted, ask the operator to check the cable/
insertion before proceeding — don't guess at partial data.

Ask the operator for the recording date and a place/keyword, and form the
group name following the existing convention (`<date>_<vehicle>_<place>`, e.g.
`20260916_jt10_example`; the vehicle name stays in the name so corpus names never collide). Confirm it doesn't already exist:

```bash
AWS_ACCESS_KEY_ID=x AWS_SECRET_ACCESS_KEY=x \
aws --endpoint-url http://127.0.0.1:39999/api/v1/s3 s3 ls s3://raw/jt10/ --region us-east-1 \
  2>/dev/null | grep <group> || echo "new group, OK to proceed"
```

(If it *does* exist, this is a resume — `detect_and_upload.py` skips
recordings already uploaded+persisted, so it's safe to re-run.)

## Step 1 (legacy flow) — Detect and upload

`cd` to this repo's root first — it's two levels above this skill file
(`.claude/skills/drs-ingest/SKILL.md` → repo root), which the invocation
already told you ("Base directory for this skill: .../my-drs-util/.claude/skills/drs-ingest"),
so don't hardcode a path here; this repo can be cloned anywhere.

```bash
python3 tools/drs-ingest/detect_and_upload.py --group <group>
```

Add `--dry-run` first if you want to see what it *would* do without
uploading anything. If only one of `ecu0`/`ecu1` is found, the script
refuses by default — confirm with the operator whether to proceed with
just that ECU (`--allow-partial`) or wait until both drives are correctly
inserted.

The script always prints each port's USB link speed first and warns if
either is below USB 3 — if you see `[warn] portX is linked below USB 3
speed`, mention it to the operator (reseating the cable/port before a
large upload avoids a much slower transfer) but proceed; it's a warning,
not a blocker. For a large group, `--transfer-mode nfs` (opt-in, default
is `s3`) uploads ~3x faster via direct rsync instead of the S3 API — offer
it for large groups once it's been validated in production, but don't
switch the default without being asked.

Report the per-ECU persisted/verified counts. Only tell the operator the
SSD is safe to remove/wipe when the script's own final line says
`SAFE TO WIPE: ...` — a `DO NOT WIPE: ...` line means some recordings
didn't verify and the source data must be kept until resolved (re-run the
script; it resumes from where it left off).

This step can take a while for large groups (hundreds of GB to low TBs) —
prefer running it in the background (`nohup ... > <scratchpad>/<group>_upload.log 2>&1 &`)
and monitoring the log rather than blocking, same as the batch steps below.

## Step 2 — GPS route generation (legacy flow reads the NAS; the ingest flow reads the SSD)

```bash
tools/drs-ingest/run_gps_route_ephemeral.sh /mnt/DataPool/SharedData/raw/jt10/<group>
```

This spins up its own ephemeral ROS2 container (installs `mcap`/
`mcap-ros2-support`/`contextily`/etc. into it, smoke-tests one recording to
confirm the OSM basemap actually renders before committing to the full
batch, then runs the batch with `--tag-maneuvers`), consolidates into a
sibling `gps-route/` folder, and tears the container down when done. See
[gps-route-batch-s3](../gps-route-batch-s3/SKILL.md) and
[gps-route-batch](../gps-route-batch/SKILL.md) for background on the
underlying tool and known failure patterns worth watching for in the
script's output — in particular:

- A `[warn] N basemap fetch failure(s)` line means some maps may have a
  blank background; investigate before trusting the batch as complete.
- The "GPS-quality flags" section lists any recording where nearly every
  fix was excluded as a speed jump — this is a known data-quality pattern
  (INS reporting a physically implausible fix, or a GPS cold-start at the
  very start of a session) seen on several prior jt10 groups. Report these
  to the operator individually; they are not a tool bug.

Run this in the background for large groups (`nohup ... &`) and monitor
with the `Monitor` tool, filtered to milestones/errors, the same way this
session's manual gps-route runs were handled.

## Step 3 — T4/NuScenes conversion (legacy flow: whole-dataset or `--chunk-files N` from the NAS)

This step needs the sibling `rosbag-data-pipeline` repo, which — unlike
`my-drs-util` itself — has no equivalent "base directory" the invocation
tells you. Locate it rather than assuming a fixed path: check
`$ROSBAG_PIPELINE_REPO` first if set, then look for a `rosbag-data-pipeline`
directory next to (a sibling of) this repo's own root, e.g.:

```bash
# REPO_ROOT = this repo's root, e.g. the skill invocation's own "Base
# directory for this skill" two levels up (.claude/skills/drs-ingest/..)
ROSBAG_PIPELINE_REPO="${ROSBAG_PIPELINE_REPO:-$(dirname "$REPO_ROOT")/rosbag-data-pipeline}"
[ -d "$ROSBAG_PIPELINE_REPO/python" ] || { echo "rosbag-data-pipeline not found — ask the operator for its path" >&2; exit 1; }
```

If it isn't a sibling of `my-drs-util` on this host, ask the operator for
its path rather than guessing further.

```bash
cd "$ROSBAG_PIPELINE_REPO"
nohup python3 python/run_batch_t4_pipeline.py \
  /mnt/DataPool/SharedData/raw/jt10/<group> \
  --convert-config config/convert_rosbag2_to_non_annotated_t4_fisheye_delay.yaml \
  --duration 30 --process-mode balanced --merge-mode smart --min-overlap-sec 5 \
  --driver auto --max-parallel 4 --docker-container rosbag-pipeline-dev \
  > <scratchpad>/<group>_t4convert.log 2>&1 &
disown
```

`root_dir` is pointed **directly at the new group's own path**, not its
`raw/jt10/` parent — `find_group_dirs()` starts scanning at `root_dir` itself,
so this scopes the run to exactly this one group instead of rescanning all
existing jt10 groups (confirmed against the script's source,
`run_batch_t4_pipeline.py:77-96`). `--min-overlap-sec 5` must be passed
explicitly — the script's own default (28) does not match this team's
standard config (confirmed against
`rosbag-data-pipeline/.claude/skills/t4-batch-convert/SKILL.md`).

If this exits immediately with `sys.exit(1)` and a lock-related message,
another batch (possibly from a different session) is already running —
report this to the operator as "skipped, another conversion is already in
progress," not as a failure; do not retry in a loop.

Monitor with the `Monitor` tool filtered to
`FAILED dataset|OK dataset PROCESSED|Batch run complete|Traceback|Error:|CRITICAL`
(same filter `t4-batch-convert` uses). On completion, parse the
`Batch run complete: processed=X failed=Y total=Z` line — the process exit
code is 0 even when some datasets failed, so this line (or
`pipeline_log/batch_t4_pipeline_status.json`) is the real signal.

## Step 4 — Final report

Summarize to the operator: recordings uploaded/persisted per ECU (and
whether the SSDs are safe to wipe), gps-route PNG/CSV counts plus any
basemap or GPS-quality flags, and the T4 conversion's processed/failed
counts. Call out anything that needs follow-up explicitly rather than
letting a partial failure blend into an otherwise-successful summary.
