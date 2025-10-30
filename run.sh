#!/usr/bin/env bash
set -euo pipefail

# Ensure systemd user manager is active
[ -n "$XDG_RUNTIME_DIR" ] || { 
    echo "ERROR: XDG_RUNTIME_DIR undefined – log in via systemd" >&2
    exit 1
}

logfile="$(pwd)/job.log"
launch_log="$(pwd)/launch.log" 
repo_dir="$(pwd)"

# Clear previous logs 
> "$logfile"
> "$launch_log"

# First log the environment and setup
{
    echo "=== Starting job at $(date) ==="
    echo "Working directory: $repo_dir"
    echo "Systemd user status:"
    systemctl --user status --no-pager
    echo "=== SYSTEMD JOURNAL BEFORE ==="
    journalctl --user -n 20 --no-pager
    echo "=== ENVIRONMENT ==="
    printenv | sort
    echo "=== Python version ==="
    python3 --version
    echo "=== Docker info ==="
    docker --version
    
} >> "$launch_log"

# Run with full error capture
systemd-run --user \
  --unit="job-1" \
  --working-directory="$repo_dir" \
  --quiet \
  --no-block \
  --property="Delegate=yes" \
  --property="StandardOutput=file:$logfile" \
  --property="StandardError=inherit" \
  bash -c '
    echo "=== JOB STARTED AT $(date) ==="
    sg docker -c "python3 run.py"
    exit_code=$?
    echo "=== JOB EXITED WITH STATUS $exit_code ==="
    exit $exit_code
  '

# Capture service status after launch
{
    sleep 2  # Give service time to start
    echo ""
    echo "=== SERVICE STATUS AFTER LAUNCH ==="
    systemctl --user status job-1.service --no-pager
    echo "=== JOURNAL AFTER LAUNCH ==="
    journalctl --user -u job-1.service -n 20 --no-pager
    echo "=== LAUNCH COMPLETE AT $(date) ==="
} >> "$launch_log" 2>&1

# Immediate log check
echo "Job log: tail -f '$logfile'"
echo "Launch log: tail -f '$launch_log'"