#!/usr/bin/env bash
# LOC-LOC v2.2.0 launcher (macOS / Linux)
set -e
cd "$(dirname "$0")"
echo "Checking dependencies..."
python3 -m pip install -r requirements.txt --quiet

echo "Starting camera worker (supervised — auto-restarts if it ever exits)..."
(
  while true; do
    python3 camera_worker.py
    code=$?
    # Exit code 2 = refused to start (no credentials). Restarting won't help.
    if [ "$code" -eq 2 ]; then
      echo "Worker refused to start (missing credentials). Not restarting."
      break
    fi
    echo "Worker exited (code $code). Restarting in 5s..."
    sleep 5
  done
) &
SUPERVISOR_PID=$!
trap 'kill $SUPERVISOR_PID 2>/dev/null; pkill -P $SUPERVISOR_PID 2>/dev/null || true' EXIT

echo "Starting dashboard..."
python3 -m streamlit run app.py --server.address 0.0.0.0
