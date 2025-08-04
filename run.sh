#!/usr/bin/env bash
set -euo pipefail

# ensure systemd user manager is active
[ -n "$XDG_RUNTIME_DIR" ] || \
  { echo "ERROR: XDG_RUNTIME_DIR undefined – log in via system loginctl"; exit 1; }

logfile="out.log"
unitname="job-1"

# Build & run, capturing output
systemd-run --user \
  --unit="${unitname}" \
  --quiet \
  --no-block \
  bash -c '
    cd "$(dirname "$0")"
    python run.py > "'"$logfile"'" 2>&1
  '

