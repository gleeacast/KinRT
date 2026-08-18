#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/private/yth/projects/RoboTwin/policy/pi05
CONFIG=kinrt_adamoe_diyrobot
EXP=hiarm_500_new_button_adamoe_kinrt_k4_8000_bs32_20260724
GPUS=0,1,2,3
VAR_ROOT=/var/ckpt
FINAL_ROOT=${ROOT}/checkpoints
LOG_DIR=${ROOT}/logs
STATUS=${LOG_DIR}/queue_adamoe_kinrt_after_himoe_20260724.status.log
TRAIN_LOG=${LOG_DIR}/train_${EXP}.log
HIMOE_OUTPUT=/private/yth/projects/zyz26/HiMoE-VLA/checkpoints/hiarm_new_button_finetune_chunk50/hiarm_new_button_native_chunk50_pd4_ga8_4gpu_8000_20260722/checkpoint-8000/pytorch_model.pth

mkdir -p "${LOG_DIR}" "${VAR_ROOT}" "${FINAL_ROOT}"
log() { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "${STATUS}"; }
trap 'rc=$?; log "ERROR at line ${LINENO}, exit=${rc}"; exit "${rc}"' ERR

log "queued: wait for HiMoE-50 checkpoint-8000"
while pgrep -f 'hiarm_new_button_native_chunk50_pd4_ga8_4gpu_8000_20260722' >/dev/null; do
  sleep 60
done
[[ -f "${HIMOE_OUTPUT}" ]]
log "HiMoE-50 completed; starting AdaMoE+KinRT K4 validation"

"${ROOT}/.venv/bin/python" -m py_compile "${ROOT}/src/openpi/training/config.py"
LABELS=/private/yth/projects/cache/huggingface/lerobot/hiarm_pi05_manual_500_promptv3_new_button_augmented/meta/router_labels_action_vel_k4_pi05_800_full_logic_20260714/router_labels.npy
"${ROOT}/.venv/bin/python" -c 'import numpy as np,sys; a=np.load(sys.argv[1]); assert a.shape==(159104,); assert np.array_equal(np.unique(a),np.arange(4)); print("labels_ok",a.shape)' "${LABELS}"

SMOKE_ROOT=${VAR_ROOT}/_smoke_adamoe_kinrt_20260724
SMOKE_EXP=${EXP}_smoke
if [[ ! -d "${SMOKE_ROOT}/${CONFIG}/${SMOKE_EXP}/2/params" ]]; then
  log "starting 2-step smoke: 4 GPUs, global batch 32"
  (
    cd "${ROOT}"
    timeout 90m bash finetune.sh "${CONFIG}" "${SMOKE_EXP}" "${GPUS}" "${SMOKE_ROOT}" \
      --num-train-steps=2 --save-interval=1 --keep-period=4000
  ) >"${LOG_DIR}/train_${SMOKE_EXP}.log" 2>&1
fi
[[ -d "${SMOKE_ROOT}/${CONFIG}/${SMOKE_EXP}/2/params" ]]
log "smoke passed"

if [[ ! -d "${VAR_ROOT}/${CONFIG}/${EXP}/8000/params" ]]; then
  log "starting/resuming AdaMoE+KinRT K4: 8000 steps, 4 GPUs, global batch 32, EMA none"
  (
    cd "${ROOT}"
    bash finetune.sh "${CONFIG}" "${EXP}" "${GPUS}" "${VAR_ROOT}" \
      --num-train-steps=8000 --save-interval=1000 --keep-period=4000
  ) >>"${TRAIN_LOG}" 2>&1
fi
[[ -d "${VAR_ROOT}/${CONFIG}/${EXP}/8000/params" ]]

mkdir -p "${FINAL_ROOT}/${CONFIG}"
rsync -a "${VAR_ROOT}/${CONFIG}/${EXP}/" "${FINAL_ROOT}/${CONFIG}/${EXP}/"
[[ -d "${FINAL_ROOT}/${CONFIG}/${EXP}/8000/params" ]]
log "completed and copied to ${FINAL_ROOT}/${CONFIG}/${EXP}/8000"
