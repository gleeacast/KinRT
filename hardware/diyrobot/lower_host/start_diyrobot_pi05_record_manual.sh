#!/usr/bin/env bash
set -euo pipefail

# Manual episode control: SPACE=start, ENTER=stop
# Preflight passes -> teleop active -> you control episodes with keyboard
#
# Env vars:
#   TASK           task description. Required for TASK_MODE=new.
#   TASK_MODE      auto|same|new. auto keeps old behavior; same must match an existing task;
#                  new must not match an existing task and will append a new task_index.
#   SAME_TASK=1    shortcut for TASK_MODE=same.
#   NEW_TASK=1     shortcut for TASK_MODE=new.
#   REPO_ID        dataset repo id
#   ROOT           dataset output directory
#   RESUME=1       append to existing dataset instead of failing

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LEROBOT_SRC="${LEROBOT_SRC:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/data/diyrobot/datasets}"
mkdir -p logs "${DATA_ROOT}"

DEFAULT_TASK="pick up the object and place it at the target location"
TASK="${TASK:-}"
REPO_ID="${REPO_ID:-local/diyrobot_pi05_manual}"
ROOT="${ROOT:-${DATA_ROOT}/diyrobot_pi05_manual}"
FPS="${FPS:-20}"
RESUME="${RESUME:-0}"
TASK_MODE="${TASK_MODE:-auto}"

if [[ "${NEW_TASK:-0}" == "1" && "${SAME_TASK:-0}" == "1" ]]; then
  echo "ERROR: NEW_TASK=1 and SAME_TASK=1 cannot both be set." >&2
  exit 2
fi
if [[ "${NEW_TASK:-0}" == "1" ]]; then
  TASK_MODE="new"
fi
if [[ "${SAME_TASK:-0}" == "1" ]]; then
  TASK_MODE="same"
fi

case "${TASK_MODE}" in
  auto) ;;
  same|existing|continue|reuse) TASK_MODE="same" ;;
  new|different) TASK_MODE="new" ;;
  *)
    echo "ERROR: TASK_MODE must be one of: auto, same, new." >&2
    exit 2
    ;;
esac

case "${RESUME}" in
  0|1) ;;
  *)
    echo "ERROR: RESUME must be 0 or 1." >&2
    exit 2
    ;;
esac

CAMERA_RIGHT_GRIPPER="${CAMERA_RIGHT_GRIPPER:-right_gripper=/dev/diyrobot/camera-right-wrist:640x480:30}"
CAMERA_LEFT_GRIPPER="${CAMERA_LEFT_GRIPPER:-left_gripper=/dev/diyrobot/camera-left-wrist:640x480:30}"
CAMERA_OVERHEAD="${CAMERA_OVERHEAD:-overhead=/dev/diyrobot/camera-overhead:640x480:30}"

if [[ "${RESUME}" == "1" ]]; then
  TASK="$(
    TASK="${TASK}" TASK_MODE="${TASK_MODE}" ROOT="${ROOT}" DEFAULT_TASK="${DEFAULT_TASK}" \
      "${PYTHON_BIN}" - <<'PY_TASK_CHECK'
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["ROOT"]).expanduser()
tasks_path = root / "meta" / "tasks.jsonl"
mode = os.environ["TASK_MODE"]
task = os.environ.get("TASK", "")
default_task = os.environ["DEFAULT_TASK"]

tasks = []
if tasks_path.exists():
    with tasks_path.open("r", encoding="utf-8") as fh:
        tasks = [json.loads(line) for line in fh if line.strip()]
existing = {item["task"] for item in tasks}

def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    if tasks:
        print("Existing tasks:", file=sys.stderr)
        for item in tasks:
            print(f"  [{item['task_index']}] {item['task']}", file=sys.stderr)
    sys.exit(2)

if mode == "same":
    if not tasks:
        fail(f"TASK_MODE=same requires an existing dataset with {tasks_path}.")
    if not task:
        if len(tasks) == 1:
            task = tasks[0]["task"]
        else:
            fail("TASK is required for TASK_MODE=same when multiple tasks already exist.")
    if task not in existing:
        fail("TASK_MODE=same was requested, but TASK does not exactly match an existing task.")
elif mode == "new":
    if not task:
        fail("TASK is required for TASK_MODE=new.")
    if task in existing:
        fail("TASK_MODE=new was requested, but TASK already exists. Use TASK_MODE=same to continue it.")
elif mode == "auto":
    if not task:
        task = default_task
else:
    fail(f"unknown TASK_MODE: {mode}")

print(task, end="")
PY_TASK_CHECK
  )"
else
  if [[ "${TASK_MODE}" == "same" ]]; then
    echo "ERROR: TASK_MODE=same requires RESUME=1." >&2
    exit 2
  fi
  if [[ -z "${TASK}" ]]; then
    TASK="${DEFAULT_TASK}"
  fi
fi

# Stop any running WebUI to free cameras
webui_pids="$(pgrep -f "diyrobot_three_camera_webui.py" || true)"
if [[ -n "${webui_pids}" ]]; then
  echo "Stopping camera WebUI before recording: ${webui_pids}"
  kill ${webui_pids} 2>/dev/null || true
  sleep 1
  webui_pids="$(pgrep -f "diyrobot_three_camera_webui.py" || true)"
  if [[ -n "${webui_pids}" ]]; then
    echo "Force-stopping stuck camera WebUI: ${webui_pids}"
    kill -9 ${webui_pids} 2>/dev/null || true
    sleep 0.5
  fi
fi

# Build extra args
EXTRA_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  EXTRA_ARGS+=(--resume)
fi

echo "=== DIYRobot PI0.5 Manual Recording ==="
echo "  Task:    ${TASK}"
echo "  Repo:    ${REPO_ID}"
echo "  Root:    ${ROOT}"
echo "  FPS:     ${FPS}"
echo "  Resume:  ${RESUME}"
echo "  Mode:    ${TASK_MODE}"
echo ""
echo "  SPACE = start recording episode"
echo "  ENTER = stop & save current episode"
echo "  Ctrl-C = exit"
echo "======================================"

PYTHONUNBUFFERED=1 \
PYTHONPATH="${LEROBOT_SRC}${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_BIN}" ./diyrobot_pi05_record.py \
  --manual \
  --repo-id "${REPO_ID}" \
  --root "${ROOT}" \
  --task "${TASK}" \
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
  --log-file "./logs/diyrobot_pi05_manual_$(date +%Y%m%d_%H%M%S).jsonl" \
  --log-decimate 10 \
  "${EXTRA_ARGS[@]}"
