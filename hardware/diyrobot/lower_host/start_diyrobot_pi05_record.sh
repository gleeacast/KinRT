#!/usr/bin/env bash
set -euo pipefail

# The recorder owns all three cameras, so stop only the matching preview process.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LEROBOT_SRC="${LEROBOT_SRC:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/diyrobot/datasets}"
mkdir -p logs "${DATA_ROOT}"

TASK="${TASK:-pick up the object and place it at the target location}"
REPO_ID="${REPO_ID:-local/diyrobot_pi05}"
ROOT="${ROOT:-${DATA_ROOT}/diyrobot_pi05}"
EPISODES="${EPISODES:-1}"
EPISODE_TIME="${EPISODE_TIME:-60}"
RESET_TIME="${RESET_TIME:-0}"
FPS="${FPS:-20}"
DATASET_VERSION="${DATASET_VERSION:-v2.1}"
CAMERA_RIGHT_GRIPPER="${CAMERA_RIGHT_GRIPPER:-right_gripper=/dev/diyrobot/camera-right-wrist:640x480:30}"
CAMERA_LEFT_GRIPPER="${CAMERA_LEFT_GRIPPER:-left_gripper=/dev/diyrobot/camera-left-wrist:640x480:30}"
CAMERA_OVERHEAD="${CAMERA_OVERHEAD:-overhead=/dev/diyrobot/camera-overhead:640x480:30}"

webui_pids="$(pgrep -f "diyrobot_three_camera_webui.py" || true)"
if [[ -n "${webui_pids}" ]]; then
  echo "Stopping camera WebUI before recording: ${webui_pids}"
  kill ${webui_pids} 2>/dev/null || true
  sleep 1
  webui_pids="$(pgrep -f "diyrobot_three_camera_webui.py" || true)"
  if [[ -n "${webui_pids}" ]]; then
    echo "Force-stopping stuck camera WebUI before recording: ${webui_pids}"
    kill -9 ${webui_pids} 2>/dev/null || true
    sleep 0.5
  fi
fi

PYTHONUNBUFFERED=1 \
PYTHONPATH="${LEROBOT_SRC}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" ./diyrobot_pi05_record.py \
  --repo-id "${REPO_ID}" \
  --root "${ROOT}" \
  --dataset-version "${DATASET_VERSION}" \
  --task "${TASK}" \
  --num-episodes "${EPISODES}" \
  --episode-time-s "${EPISODE_TIME}" \
  --reset-time-s "${RESET_TIME}" \
  --fps "${FPS}" \
  --camera "${CAMERA_RIGHT_GRIPPER}" \
  --camera "${CAMERA_LEFT_GRIPPER}" \
  --camera "${CAMERA_OVERHEAD}" \
  --vcodec h264 \
  --streaming-encoding \
  --encoder-threads 2 \
  --follower-tx-min-gap-s 0.003 \
  --max-step-deg 0.75 \
  --max-catchup-step-deg 2.40 \
  --wrist-max-step-deg 1.20 \
  --wrist-max-catchup-step-deg 4.20 \
  --gripper-max-step-deg 4.5 \
  --gripper-max-catchup-step-deg 10.0 \
  --catchup-start-error-deg 0.15 \
  --catchup-full-error-deg 0.8 \
  --active-feedback-motors right_wrist_flex \
  --startup-threshold-deg 7.0 \
  --startup-range-slack-deg 0.5 \
  --startup-hold-guard-s 1.2 \
  --startup-hold-max-drift-deg 1.0 \
  --feedback-max-age-s 0.30 \
  --feedback-retry-count 3 \
  --feedback-retry-sleep-s 0.015 \
  --live-feedback-missing-grace-s 1.0 \
  --hold-settle-s 0.2 \
  --log-file ./logs/diyrobot_pi05_record_$(date +%Y%m%d_%H%M%S).jsonl \
  --log-decimate 10
