#!/usr/bin/env bash
# Run an auto-calibration step (or any command) with ROS 2 Humble sourced and
# the project venv's python. All steps share this one venv, so run_pipeline.py
# runs end to end without switching interpreters.
#
# Usage:
#     tools/auto-calib/run.sh step0_validate.py <bag> [args...]
#     tools/auto-calib/run.sh run_pipeline.py <bag> --out-dir OUT
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="$REPO/.venv"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "venv not found at $VENV — run tools/auto-calib/setup.sh first" >&2
    exit 1
fi

# ROS setup.bash references unbound vars, so relax `-u` only while sourcing.
set +u
# shellcheck disable=SC1090
source "$ROS_SETUP"
set -u

# Run from the step directory so `import common` resolves.
cd "$HERE"
exec "$VENV/bin/python" "$@"
