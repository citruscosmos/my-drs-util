#!/usr/bin/env bash
# Create the project venv for the auto-calibration pipeline and install deps.
#
# Creates a CLEAN venv (no --system-site-packages) at <repo>/.venv and installs
# requirements.txt into it. rclpy / ROS message packages are provided by
# sourcing ROS 2 Humble at run time, not installed here.
#
# Usage:  tools/auto-calib/setup.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
VENV="$REPO/.venv"

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV"
    python3 -m venv "$VENV"
fi

"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

echo
echo "venv ready: $VENV"
echo "Run steps with:  $HERE/run.sh step0_validate.py <bag>"
