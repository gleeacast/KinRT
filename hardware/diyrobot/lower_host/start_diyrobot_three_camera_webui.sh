#!/usr/bin/env bash
set -euo pipefail

# Resolve every runtime path relative to this released DIYRobot package.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LEROBOT_SRC="${LEROBOT_SRC:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8090}"
CAMERA_RIGHT_GRIPPER="${CAMERA_RIGHT_GRIPPER:-right_gripper=/dev/diyrobot/camera-right-wrist:640x480:mjpeg}"
CAMERA_LEFT_GRIPPER="${CAMERA_LEFT_GRIPPER:-left_gripper=/dev/diyrobot/camera-left-wrist:640x480:mjpeg}"
CAMERA_OVERHEAD="${CAMERA_OVERHEAD:-overhead=/dev/diyrobot/camera-overhead:640x480:opencv}"

if ss -ltn "sport = :${PORT}" | grep -q ":${PORT}"; then
  old_pids="$(pgrep -f "diyrobot_three_camera_webui.py .*--port ${PORT}" || true)"
  if [[ -z "${old_pids}" ]]; then
    echo "Port ${PORT} is already in use by another process; refusing to start." >&2
    ss -ltnp "sport = :${PORT}" >&2 || true
    exit 1
  fi
  echo "Stopping existing DIYRobot camera WebUI on port ${PORT}: ${old_pids}"
  kill ${old_pids} 2>/dev/null || true
  sleep 1
  still_running="$(pgrep -f "diyrobot_three_camera_webui.py .*--port ${PORT}" || true)"
  if [[ -n "${still_running}" ]]; then
    echo "Force-stopping stuck DIYRobot camera WebUI: ${still_running}"
    kill -9 ${still_running} 2>/dev/null || true
    sleep 0.5
  fi
fi

PYTHONUNBUFFERED=1 \
PYTHONPATH="${LEROBOT_SRC}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" ./diyrobot_three_camera_webui.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --camera "${CAMERA_RIGHT_GRIPPER}" \
  --camera "${CAMERA_LEFT_GRIPPER}" \
  --camera "${CAMERA_OVERHEAD}"
