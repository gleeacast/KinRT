#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LEROBOT_SRC="${LEROBOT_SRC:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
mkdir -p logs

# Training-time prompts for the current 5-task promptv2 dataset:
#   0 = pill box to notebook center
#   1 = same-side black pen insertion, no handover
#   2 = opposite-side black pen insertion, right-to-left handover
#   3 = press black remote
#   4 = pull pill box onto the black pad
TASK_ID="${TASK_ID:-}"
PROMPT_VARIANT="${PROMPT_VARIANT:-canonical}"
TASKS=(
  "Put the pill box into the center of the notebook."
  "The black pen and the black pen holder are on the same side; pick up the pen and insert it directly into the holder without a handover."
  "Pick up the black pen, perform a right-to-left handover, and insert the pen into the black pen holder on the left side."
  "Press the black remote control until the green indicator light turns on."
  "Pull the pill box onto the black pad beside it."
)

load_prompt_variants() {
  case "$1" in
    0)
      PROMPT_VARIANTS=(
        "Put the pill box into the center of the notebook."
        "Place the pill box in the middle of the notebook."
        "Move the pill box to the center area of the notebook."
        "Pick up the pill box and set it at the notebook's center."
        "Put the medicine box onto the center of the notebook."
        "Position the pill box in the middle of the notebook."
        "Transfer the pill box onto the central part of the notebook."
        "Set the pill box down at the center of the notebook."
        "Place the small pill box on the notebook, centered."
        "Move the pill container to the middle of the notebook."
      )
      ;;
    1)
      PROMPT_VARIANTS=(
        "The black pen and the black pen holder are on the same side; pick up the pen and insert it directly into the holder without a handover."
        "When the pen holder is on the same side as the black pen, directly place the pen into the black holder; do not hand it over."
        "Pick up the black pen and put it straight into the same-side black pen holder, with no arm-to-arm handover."
        "For the same-side pen task, grasp the black pen and insert it directly into the nearby black pen holder."
        "Since the black pen and holder are on the same side, use the local arm to place the pen into the black holder without handover."
        "Directly move the black pen into the black pen holder on the same side; no cross-arm transfer is needed."
        "Insert the black pen into the same-side black pen holder without passing it to the other hand."
        "Grasp the black pen and place it directly in the black holder beside it, avoiding any handover."
        "Use the arm on the pen side to pick up the black pen and drop it into the same-side black pen holder."
        "Put the black pen into the black holder on its own side; keep the pen in the same hand and do not hand it over."
      )
      ;;
    2)
      PROMPT_VARIANTS=(
        "The black pen and the black pen holder are on opposite sides; pick up the pen, hand it over from the right arm to the left arm, then insert it into the left black holder."
        "For the opposite-side pen task, grasp the black pen with the right arm, transfer it to the left arm, and place it into the black holder on the left."
        "Pick up the black pen, perform a right-to-left handover, and insert the pen into the black pen holder on the left side."
        "Because the holder is on the opposite side, pass the black pen from the right hand to the left hand before putting it into the left holder."
        "Move the black pen across sides by handing it from the right arm to the left arm, then put it into the black holder on the left."
        "Use a handover for the black pen: right arm picks it up, left arm receives it, and the left arm inserts it into the black holder."
        "The pen must go to the opposite-side holder; hand it from right to left and then place it into the left black pen holder."
        "Grasp the black pen with the right arm, hand it to the left arm, and finish by inserting it into the left-side black holder."
        "Transfer the black pen from the right side to the left hand, then put it into the black pen holder on the left."
        "Complete the opposite-side insertion by doing a right-to-left handover of the black pen before placing it in the left holder."
      )
      ;;
    3)
      PROMPT_VARIANTS=(
        "Press the black remote control until the green indicator light turns on."
        "Push the button on the black remote until the green light is on."
        "Press down on the black remote control and keep going until the green indicator lights up."
        "Activate the black remote by pressing it until the green light turns on."
        "Use the robot to press the black remote control until its green indicator is illuminated."
        "Press the black remote's button until the green LED comes on."
        "Push the black remote control so that the green indicator light turns on."
        "Operate the black remote by pressing it until the green light appears."
        "Depress the black remote control button until the green indicator switches on."
        "Press the black remote and stop after the green indicator light is lit."
      )
      ;;
    4)
      PROMPT_VARIANTS=(
        "Pull the pill box onto the black pad beside it."
        "Move the medicine box onto the nearby black pad."
        "Drag the pill box into the black pad next to it."
        "Use the robot to pull the medicine box onto the adjacent black pad."
        "Slide the pill box onto the black mat beside it."
        "Move the pill box from its current position onto the black pad beside it."
        "Pull the small medicine box toward the nearby black pad and place it on the pad."
        "Bring the pill box onto the adjacent black pad."
        "Shift the medicine box onto the black pad beside it."
        "Pull the pill box until it is on the nearby black pad."
      )
      ;;
    *)
      echo "TASK_ID must be one of 0,1,2,3,4; got '$1'" >&2
      exit 2
      ;;
  esac
}

# Override legacy task IDs with the current five-task prompt-v3 evaluation table.
source "${SCRIPT_DIR}/diyrobot_pi05_policy_tasks_v3.sh"

select_prompt() {
  local task_id="$1"
  load_prompt_variants "${task_id}"
  case "${PROMPT_VARIANT}" in
    canonical|exact|"")
      SELECTED_TASK="${TASKS[${task_id}]}"
      SELECTED_PROMPT_VARIANT="canonical"
      ;;
    random)
      local n="${#PROMPT_VARIANTS[@]}"
      local idx="$((RANDOM % n))"
      SELECTED_TASK="${PROMPT_VARIANTS[${idx}]}"
      SELECTED_PROMPT_VARIANT="random:${idx}/${n}"
      ;;
    *[!0-9]*)
      echo "PROMPT_VARIANT must be canonical, random, or a numeric variant index; got '${PROMPT_VARIANT}'" >&2
      exit 2
      ;;
    *)
      local n="${#PROMPT_VARIANTS[@]}"
      if (( PROMPT_VARIANT < 0 || PROMPT_VARIANT >= n )); then
        echo "PROMPT_VARIANT for TASK_ID=${task_id} must be in [0,$((n - 1))]; got '${PROMPT_VARIANT}'" >&2
        exit 2
      fi
      SELECTED_TASK="${PROMPT_VARIANTS[${PROMPT_VARIANT}]}"
      SELECTED_PROMPT_VARIANT="${PROMPT_VARIANT}/${n}"
      ;;
  esac
}

if [[ -n "${TASK:-}" ]]; then
  SELECTED_TASK="${TASK}"
  SELECTED_PROMPT_VARIANT="manual"
else
  if [[ -z "${TASK_ID}" ]]; then
    TASK_ID=0
  fi
  if ! [[ "${TASK_ID}" =~ ^[0-4]$ ]]; then
    echo "TASK_ID must be one of 0,1,2,3,4; got '${TASK_ID}'" >&2
    exit 2
  fi
  select_prompt "${TASK_ID}"
fi

POLICY_MODE="${POLICY_MODE:-diyrobot_tcp}"
POLICY_SERVER="${POLICY_SERVER:-}"
FPS="${FPS:-20}"
DURATION="${DURATION:-3000}"
ALLOW_MOTION="${ALLOW_MOTION:-0}"
ALLOW_INFINITE_LIVE="${ALLOW_INFINITE_LIVE:-0}"
HOLD_LEFT_ARM="${HOLD_LEFT_ARM:-0}"
OFFLINE_SELF_TEST="${OFFLINE_SELF_TEST:-0}"
STATE_KEY="${STATE_KEY:-state}"
PROMPT_KEY="${PROMPT_KEY:-prompt}"
POLICY_TIMEOUT_S="${POLICY_TIMEOUT_S:-60.0}"
ACTION_CHUNK_STEPS="${ACTION_CHUNK_STEPS:-20}"
STARTUP_THRESHOLD_DEG="${STARTUP_THRESHOLD_DEG:-15.0}"

CAMERA_CAM_HIGH="${CAMERA_CAM_HIGH:-cam_high=/dev/diyrobot/camera-overhead:640x480:30}"
CAMERA_CAM_LEFT_WRIST="${CAMERA_CAM_LEFT_WRIST:-cam_left_wrist=/dev/diyrobot/camera-left-wrist:640x480:30}"
CAMERA_CAM_RIGHT_WRIST="${CAMERA_CAM_RIGHT_WRIST:-cam_right_wrist=/dev/diyrobot/camera-right-wrist:640x480:30}"

args=(
  ./diyrobot_pi05_policy_client.py
  --task "${SELECTED_TASK}"
  --fps "${FPS}"
  --camera "${CAMERA_CAM_HIGH}"
  --camera "${CAMERA_CAM_LEFT_WRIST}"
  --camera "${CAMERA_CAM_RIGHT_WRIST}"
  --state-key "${STATE_KEY}"
  --prompt-key "${PROMPT_KEY}"
  --image-key "cam_high=images.cam_high"
  --image-key "cam_left_wrist=images.cam_left_wrist"
  --image-key "cam_right_wrist=images.cam_right_wrist"
  --allow-startup-low-disturbance-sample
  --follower-tx-min-gap-s 0.003
  --max-step-deg 0.75
  --max-catchup-step-deg 2.40
  --wrist-max-step-deg 1.20
  --wrist-max-catchup-step-deg 4.20
  --gripper-max-step-deg 4.5
  --gripper-max-catchup-step-deg 10.0
  --catchup-start-error-deg 0.15
  --catchup-full-error-deg 0.8
  --active-feedback-motors right_wrist_flex
  --startup-threshold-deg "${STARTUP_THRESHOLD_DEG}"
  --startup-range-slack-deg 0.5
  --startup-hold-guard-s 1.2
  --startup-hold-max-drift-deg 1.0
  --feedback-max-age-s 0.30
  --feedback-retry-count 3
  --feedback-retry-sleep-s 0.015
  --live-feedback-missing-grace-s 1.0
  --policy-timeout-s "${POLICY_TIMEOUT_S}"
  --action-chunk-steps "${ACTION_CHUNK_STEPS}"
  --hold-settle-s 0.2
  --log-file "./logs/diyrobot_pi05_policy_$(date +%Y%m%d_%H%M%S).jsonl"
  --log-decimate 10
)

if [[ -n "${POLICY_SERVER}" ]]; then
  args+=(--policy-server "${POLICY_SERVER}")
fi

if [[ "${OFFLINE_SELF_TEST}" == "1" ]]; then
  args+=(--policy-mode noop --skip-cameras --offline-self-test)
else
  args+=(--policy-mode "${POLICY_MODE}")
fi

if [[ "${DURATION}" != "0" ]]; then
  args+=(--duration "${DURATION}")
fi

if [[ "${ALLOW_MOTION}" == "1" ]]; then
  args+=(--allow-motion)
fi

if [[ "${HOLD_LEFT_ARM}" == "1" ]]; then
  args+=(--hold-left-arm)
fi

if [[ "${ALLOW_INFINITE_LIVE}" == "1" ]]; then
  args+=(--allow-infinite-live)
fi

echo "policy_server=${POLICY_SERVER:-'(none)'}"
echo "task_id=${TASK_ID:-'(manual)'}"
echo "prompt_variant=${SELECTED_PROMPT_VARIANT}"
echo "task=${SELECTED_TASK}"
echo "allow_motion=${ALLOW_MOTION} hold_left_arm=${HOLD_LEFT_ARM} duration=${DURATION}"

PYTHONUNBUFFERED=1 PYTHONPATH="${LEROBOT_SRC}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON_BIN}" "${args[@]}"
