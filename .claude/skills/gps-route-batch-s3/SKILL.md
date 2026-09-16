---
name: gps-route-batch-s3
description: Fetch a measurement group's ecu0/ recordings from an Alluxio S3 API (no local FUSE/NFS mount required for the transfer), run tools/gps-route over every recording (maneuver tagging + elapsed-time ticks), then upload the resulting *_gps-route.png / *_maneuvers.csv files back to a sibling "gps-route" folder at the same hierarchy level as ecu0, all via the S3 API. Use when asked to generate GPS route maps for an S3/Alluxio-hosted (jt10-style) group, or explicitly asked to do gps-route "via S3 API" / "S3 APIで".
---

# gps-route-batch-s3

S3 (Alluxio S3 API) version of [gps-route-batch](../gps-route-batch/SKILL.md):
downloads a measurement group's `ecu0/` recordings from S3, runs
`tools/gps-route` locally to produce route maps + maneuver CSVs, then
uploads the results to `gps-route/` — a folder at the **same hierarchy
level as `ecu0`** (a sibling, not a subfolder of it) — back through S3.

The S3 access pattern (endpoint, credentials, bucket/key mapping, the
post-upload permission fix) is the one verified end-to-end against the real
Alluxio deployment in `rosbag-data-pipeline`'s `t4-batch-convert` skill —
reuse those same conventions here rather than re-deriving them; see that
skill's "0b. Alternative environment: Docker + Alluxio S3 API" section for
the underlying investigation if anything below needs re-verifying on a new
host.

## 0. Prerequisites

- `aws` CLI installed and on `PATH` (`aws --version`). No real AWS account
  is needed — Alluxio's S3 API doesn't verify the request signature, but
  does require a well-formed SigV4 request, which the CLI provides
  automatically.
- The repo's `.venv` with `tools/gps-route`'s deps (`mcap`,
  `mcap-ros2-support`, `numpy`, `matplotlib`, `contextily`) plus a sourced
  ROS 2 environment for `rclpy`/`sensor_msgs` — same requirement as plain
  `gps-route-batch` (see `tools/gps-route/README.md`). If this host has no
  native ROS 2 and only reaches the data through Alluxio (the "no native
  ROS2" case), the batch step can instead run inside an existing
  ROS2-enabled Docker container the way `t4-batch-convert` §0b sets one
  up — not detailed again here, since it's identical to that skill's setup.
- Connection details for the Alluxio S3 endpoint. Ask the user for
  whichever of these aren't already known from context (e.g. a prior
  `t4-batch-convert` session on the same host):

  | Param | Example | Notes |
  |---|---|---|
  | S3 endpoint URL | `http://127.0.0.1:39999/api/v1/s3` | Alluxio S3 API endpoint |
  | Access key | `root` | **Must be a real Unix username inside the Alluxio worker container** — Alluxio uses it directly as the UFS chown target for the written file; a made-up value (e.g. a project name) silently breaks every large-file (multipart) upload with `AccessDenied`, while small files tolerate it — so this only shows up once real recording data is involved. Check via `docker exec <alluxio-worker-container> cat /etc/passwd` if unsure; `root` reliably exists. |
  | Secret key | any non-empty string | Not verified by Alluxio |
  | Target group's `ecu0` S3 path | `s3://jt10/<group>/ecu0` | The path segment right after `s3://` is the bucket (Alluxio maps each top-level FUSE dir 1:1 to a bucket); the rest is the key. Ask the user for the full path if not given — don't guess the group name. |
  | FUSE mount root (optional) | `/home/shimaki/peta-data/.data/alluxio-mount` | Only needed for the permission fix in step 4 — a POSIX path onto the *same* Alluxio namespace as the S3 endpoint above. If unavailable on this host, skip step 4's `chmod` and tell the user the uploaded files may be unreadable to other users until someone with FUSE access runs it. |

**Bash tool calls in this session do not share shell state** (env vars,
aliases) between separate invocations — the working directory persists,
nothing else does. So prepend this block to the start of *every* step
below rather than exporting credentials once and assuming they carry over:

```bash
S3_ENDPOINT="<endpoint-url>"
export AWS_ACCESS_KEY_ID=<access-key>
export AWS_SECRET_ACCESS_KEY=<secret-key>
s3() { aws --endpoint-url "$S3_ENDPOINT" s3 --region us-east-1 "$@"; }
```

Wrap every transfer in the shell `timeout` command (not just aws-cli's own
`--cli-read-timeout`) — under cluster strain an S3 transfer can stall
(trickling a few bytes at a time) without ever hitting aws-cli's own
read-timeout; confirmed in `t4-batch-convert` production use: one file
stalled 1.5 hours before finally erroring on its own. Pick a longer
timeout for a whole-recording `--recursive` copy than for a single small
file.

## 1. Discover ecu0 recordings for the target group

```bash
timeout 60s s3 ls "s3://<bucket>/<group>/ecu0/"
```

Each `PRE <name>/` line is one recording (split rosbag2 directory —
`*.mcap` + `metadata.yaml` — or occasionally a single `.mcap` object).
Confirm the listing looks like what `gps-route-batch` expects before
proceeding; a non-standard layout (raw files with no per-recording
subfolder, mismatched naming, etc. — see `t4-batch-convert` step 1) should
be surfaced to the user rather than guessed past.

## 2. Download ecu0 into local staging, one recording at a time

Use the repo's git-ignored `output/` dir as scratch space, not `/tmp` —
this can be tens of GB and needs to survive across the several steps below:

```bash
local_root="<repo_root>/output/gps-route-s3/<group>"
mkdir -p "$local_root/ecu0"

timeout 60s s3 ls "s3://<bucket>/<group>/ecu0/" \
  | awk '/PRE/ {print $2}' | sed 's:/$::' > "$local_root/.recordings.txt"

: > "$local_root/.download_failures.txt"
while read -r rec; do
  echo "downloading $rec"
  mkdir -p "$local_root/ecu0/$rec"
  timeout 1800s s3 cp --recursive \
    "s3://<bucket>/<group>/ecu0/$rec/" "$local_root/ecu0/$rec/" \
    || echo "$rec" >> "$local_root/.download_failures.txt"
done < "$local_root/.recordings.txt"

cat "$local_root/.download_failures.txt"
```

Per-recording (not one giant recursive copy of all of `ecu0/`) so a stall
or failure on one recording doesn't force restarting the whole group's
transfer, and matches `batch_plot_gps_route.py`'s own philosophy of
"one bad subfolder doesn't stop the batch." Report any lines in
`.download_failures.txt` to the user before continuing — retry those
individually rather than silently dropping them from the run.

Sanity-check the counts:
```bash
wc -l < "$local_root/.recordings.txt"
find "$local_root/ecu0" -mindepth 1 -maxdepth 1 -type d | wc -l
```

## 3. Run gps-route locally (same as gps-route-batch)

Follow [gps-route-batch](../gps-route-batch/SKILL.md) steps 2-3 verbatim
against `$local_root/ecu0`, including its basemap-availability check
(confirm `contextily` is importable and do one single-folder run first —
see that skill for why a missing/failed basemap silently produces a blank
map without failing the batch):

```bash
cd <repo_root>/tools/gps-route
<repo_root>/.venv/bin/python3 batch_plot_gps_route.py "$local_root/ecu0" -- --tag-maneuvers
```

Then consolidate into a sibling `gps-route/` folder, same as that skill's
step 3:

```bash
dest="$local_root/gps-route"
mkdir -p "$dest"
find "$local_root/ecu0" -maxdepth 2 \( -name "*_gps-route.png" -o -name "*_maneuvers.csv" \) -exec mv -t "$dest" {} +
```

## 4. Upload gps-route/ back via S3, same hierarchy level as ecu0

```bash
timeout 600s s3 cp --recursive "$dest/" "s3://<bucket>/<group>/gps-route/"
```

Alluxio's S3 API always writes new objects `rwxrwx---` (owner/group only,
unreadable by anyone but the `--s3-access-key` user) regardless of access
key — neither `PutObjectAcl` nor an `--acl` upload flag work around it (the
former is rejected as `NotImplemented`, the latter hangs the request).
Fix it through the FUSE view instead, if available (Alluxio has its own
permission enforcement disabled cluster-wide, so a `chmod` through FUSE
reaches the real NAS files even from a non-owning host user):

```bash
chmod -R o+rX "<fuse_mount_root>/<bucket>/<group>/gps-route"
```

If no FUSE mount is reachable from this host, tell the user the uploaded
files may be unreadable to other users until someone with FUSE access runs
that `chmod`.

## 5. Verify and report

```bash
s3 ls --recursive "s3://<bucket>/<group>/gps-route/" | grep -c "_gps-route.png"
s3 ls --recursive "s3://<bucket>/<group>/gps-route/" | grep -c "_maneuvers.csv"
ls "$dest"/*_gps-route.png 2>/dev/null | wc -l
ls "$dest"/*_maneuvers.csv 2>/dev/null | wc -l
```

The S3-side and local counts should match. Writes through Alluxio can take
a moment to become visible even to a direct listing (NFS attribute-cache
lag observed elsewhere in this pipeline) — retry the `s3 ls` count once
after a few seconds if it looks short rather than assuming data loss.

Report to the user: recordings discovered vs. successfully downloaded
(from step 2's failure log), the batch tool's own success/fail summary
line from step 3, and the final S3 object counts here. Don't conflate a
download failure (step 2) with a plotting failure (step 3, e.g. no
NavSatFix topic) when reporting — they need different follow-up.

## 6. Clean up local staging

Once the S3 upload is verified, the local copy is disposable (it's just a
staging cache of what's now in S3):

```bash
rm -rf "$local_root"
```

Confirm with the user before deleting if anything in `$local_root` wasn't
verified as uploaded yet, or if they want to keep a local copy around.

## Long runs

A group with many recordings can take a while (download + ~15-20s/recording
plotting, per `gps-route-batch`'s own observation). For a large group,
assemble steps 1-6 into a single background script (`nohup ... &`,
`disown`) and monitor its log with the `Monitor` tool instead of blocking
the conversation on it — the same guidance `t4-batch-convert` gives for its
own long batch runs.
