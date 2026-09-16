#!/usr/bin/env bash
# Runs tools/gps-route over a jt10 group's ecu0/ data using a throwaway ROS2
# container (this host has no native ROS2; the shared rosbag-pipeline-dev
# container doesn't have mcap/contextily installed) - the exact recipe run
# by hand 4 times this session (kamakura, kujukuri, todenarakawa x2).
#
# Writes output directly to <group>/gps-route/ over the jt10 NFS mount (same
# underlying storage as the Alluxio S3 API - see .claude/skills/drs-ingest),
# then verifies the result is visible via the S3 API before exiting, since
# Alluxio's own UFS metadata sync is what other tools/hosts will see.
#
# Usage:
#   run_gps_route_ephemeral.sh <group_path> [-- <plot_gps_route.py extra args>]
#
# <group_path> is the jt10 group directory itself, e.g.
#   /mnt/DataPool/SharedData/jt10/20260916_jt10_example
# (must contain an ecu0/ subfolder; gps-route/ is created alongside it).
set -uo pipefail

IMAGE="data-converter-ros2-dev:latest"
S3_ENDPOINT="${S3_ENDPOINT:-http://127.0.0.1:39999/api/v1/s3}"
S3_ACCESS_KEY="${S3_ACCESS_KEY:-root}"
S3_SECRET_KEY="${S3_SECRET_KEY:-unused-not-verified}"
S3_BUCKET="${S3_BUCKET:-jt10}"

if [ $# -lt 1 ]; then
  echo "usage: $0 <group_path> [-- <plot_gps_route.py extra args>]" >&2
  exit 2
fi
GROUP_PATH="$(realpath "$1")"
shift
EXTRA_ARGS=()
if [ "${1:-}" = "--" ]; then
  shift
  EXTRA_ARGS=("$@")
fi

GROUP_NAME="$(basename "$GROUP_PATH")"
ECU0_DIR="$GROUP_PATH/ecu0"
DEST_DIR="$GROUP_PATH/gps-route"
# Derived from this script's own location (tools/drs-ingest/<this file>), not
# hardcoded - works regardless of where this repo happens to be cloned.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOLS_DIR="$REPO_ROOT/tools/gps-route"

if [ ! -d "$ECU0_DIR" ]; then
  echo "[error] no ecu0/ under $GROUP_PATH" >&2
  exit 1
fi

CONTAINER="gps-route-ingest-$$-$(date +%s)"
cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1; }
trap cleanup EXIT

echo "[1/6] starting ephemeral container $CONTAINER"
# Bind-mount the group's own path (not a hardcoded NFS mount root) and this
# repo - both at identical absolute paths, so paths used on the host and
# inside the container match regardless of where the jt10 mount or this repo
# actually live on this particular host.
docker run -dit --name "$CONTAINER" \
  -v "$GROUP_PATH:$GROUP_PATH" \
  -v "$REPO_ROOT:$REPO_ROOT" \
  "$IMAGE" bash >/dev/null

echo "[2/6] installing gps-route python deps"
docker exec "$CONTAINER" bash -c \
  "pip3 install -q mcap mcap-ros2-support 'matplotlib>=3.7' contextily numpy" \
  || { echo "[error] pip install failed" >&2; exit 1; }

echo "[3/6] smoke-testing one recording (confirms OSM basemap actually renders)"
FIRST_REC="$(ls "$ECU0_DIR" | head -1)"
if [ -z "$FIRST_REC" ]; then
  echo "[error] $ECU0_DIR has no recordings" >&2
  exit 1
fi
SMOKE_LOG="$(mktemp)"
docker exec -w "$TOOLS_DIR" "$CONTAINER" bash -c \
  "source /opt/ros/humble/setup.bash && python3 plot_gps_route.py \
   '$ECU0_DIR/$FIRST_REC' --out-dir /tmp/smoketest --tag-maneuvers" \
  > "$SMOKE_LOG" 2>&1
if grep -q "basemap fetch failed" "$SMOKE_LOG"; then
  echo "[error] basemap fetch failed on smoke test - aborting before the full batch:" >&2
  cat "$SMOKE_LOG" >&2
  exit 1
fi
if ! grep -q "wrote /tmp/smoketest/.*_gps-route.png" "$SMOKE_LOG"; then
  echo "[error] smoke test did not produce a PNG - aborting:" >&2
  cat "$SMOKE_LOG" >&2
  exit 1
fi
echo "  smoke test OK ($FIRST_REC)"
rm -f "$SMOKE_LOG"

echo "[4/6] running full batch over $ECU0_DIR"
# ecu0/ came from an S3 upload and is not writable by this (unprivileged)
# container, so plot_gps_route.py's default "write alongside the input"
# behavior would fail - write to gps-route/ directly via --out-dir instead
# (nested one level, <group>/gps-route/<recording>/<recording>_*.{png,csv});
# flattened to match the established convention in step 5 below.
mkdir -p "$DEST_DIR" 2>/dev/null || true
BATCH_LOG="$(mktemp)"
docker exec -w "$TOOLS_DIR" "$CONTAINER" bash -c \
  "source /opt/ros/humble/setup.bash && python3 batch_plot_gps_route.py \
   '$ECU0_DIR' --out-dir '$DEST_DIR' -- --tag-maneuvers ${EXTRA_ARGS[*]}" \
  > "$BATCH_LOG" 2>&1
tail -5 "$BATCH_LOG"

N_BASEMAP_FAIL="$(grep -c "basemap fetch failed" "$BATCH_LOG" || true)"
if [ "$N_BASEMAP_FAIL" -gt 0 ]; then
  echo "  [warn] $N_BASEMAP_FAIL basemap fetch failure(s) during the full batch - see $BATCH_LOG"
fi
echo "  GPS-quality flags (near-total point exclusion) found during this run:"
grep -B1 "\[note\].*excluded [0-9]* speed Jump" "$BATCH_LOG" | grep -E "^\[batch\]|\[note\]" || echo "    none"

echo "[5/6] flattening $DEST_DIR (--out-dir nests one level per recording)"
find "$DEST_DIR" -mindepth 2 -maxdepth 2 \( -name "*_gps-route.png" -o -name "*_maneuvers.csv" \) \
  -exec mv -t "$DEST_DIR" {} +
find "$DEST_DIR" -mindepth 1 -maxdepth 1 -type d -empty -delete
N_PNG="$(find "$DEST_DIR" -maxdepth 1 -name "*_gps-route.png" | wc -l)"
N_CSV="$(find "$DEST_DIR" -maxdepth 1 -name "*_maneuvers.csv" | wc -l)"
echo "  $N_PNG PNG(s), $N_CSV CSV(s) in $DEST_DIR"

echo "[6/6] verifying visibility via S3 API (Alluxio UFS sync of the direct-NFS write)"
export AWS_ACCESS_KEY_ID="$S3_ACCESS_KEY"
export AWS_SECRET_ACCESS_KEY="$S3_SECRET_KEY"
N_LOCAL=$((N_PNG + N_CSV))
N_REMOTE=0
for _ in 1 2 3 4 5; do
  # List the group root first - Alluxio only discovers a subdirectory that
  # was created out-of-band (docker exec mkdir / direct NFS write, as
  # gps-route/ was) once its PARENT has been (re-)listed; a listing of the
  # nested path alone can return empty even though the files are already on
  # disk (confirmed directly: listing gps-route/ before ever listing the
  # group root returned nothing, listing the group root first fixed it).
  timeout 30s aws --endpoint-url "$S3_ENDPOINT" s3 ls "s3://$S3_BUCKET/$GROUP_NAME/" \
    --region us-east-1 >/dev/null 2>&1
  N_REMOTE="$(timeout 30s aws --endpoint-url "$S3_ENDPOINT" s3 ls --recursive \
    "s3://$S3_BUCKET/$GROUP_NAME/gps-route/" --region us-east-1 2>/dev/null | wc -l)"
  [ "$N_REMOTE" -ge "$N_LOCAL" ] && break
  sleep 3
done
if [ "$N_REMOTE" -ge "$N_LOCAL" ]; then
  echo "  OK: $N_REMOTE object(s) visible via S3 API (local: $N_LOCAL)"
else
  echo "  [warn] only $N_REMOTE/$N_LOCAL object(s) visible via S3 API after retries - files are on" \
       "disk but may not be synced to Alluxio's view yet" >&2
fi

rm -f "$BATCH_LOG"
echo "done: $N_PNG PNG(s), $N_CSV CSV(s), $N_BASEMAP_FAIL basemap failure(s)"
[ "$N_PNG" -gt 0 ]
