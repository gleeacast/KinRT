#!/usr/bin/env bash
set -Eeuo pipefail

HIMOE_ROOT=/private/yth/projects/zyz26/HiMoE-VLA
HIMOE_PYTHON=/var/tmp/himoe_vla_venv_py311/bin/python
GPUS=0,1,2,3
STATUS=${HIMOE_ROOT}/logs/chain_himoe50_only_20260723.status.log
OLD_STATUS=/private/yth/projects/RoboTwin/policy/pi05/logs/chain_pi0_variants_then_himoe50_20260722.status.log

mkdir -p "${HIMOE_ROOT}/logs"
: >"${STATUS}"
log() {
  local message
  message="$(date '+%F %T') $*"
  printf '%s\n' "${message}" | tee -a "${STATUS}"
}
trap 'rc=$?; log "ERROR: HiMoE recovery stopped at line ${LINENO} with exit code ${rc}"; exit "${rc}"' ERR

run_launch() {
  local config=$1
  local exp=$2
  local logfile=$3
  shift 3
  (
    cd "${HIMOE_ROOT}"
    export USE_FLAX=0 USE_TF=0 CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1
    "${HIMOE_PYTHON}" -m accelerate.commands.accelerate_cli launch --num_processes=4 --mixed_precision=bf16 scripts/train.py \
      --deepspeed=src/moevla/training/zero2.json \
      --config="${config}" --exp-name="${exp}" "$@"
  ) >>"${logfile}" 2>&1
}

run_smoke() {
  local config=$1
  local exp=$2
  local logfile="${HIMOE_ROOT}/logs/train_${exp}.log"
  : >"${logfile}"
  log "HiMoE-50 smoke starting: ${config}"
  if ! run_launch "${config}" "${exp}" "${logfile}"; then
    log "HiMoE-50 smoke failed: ${config}"
    return 1
  fi
  if grep -q -E 'Traceback|ChildFailedError|CUDA out of memory|OutOfMemoryError' "${logfile}"; then
    log "HiMoE-50 smoke log contains a fatal error: ${config}"
    return 1
  fi
  log "HiMoE-50 smoke passed: ${config}"
}

log "Recovery started after valid PI0-Full-MoE checkpoint-8000"
printf '%s\n' "$(date '+%F %T') RECOVERY: Full-MoE 8000 valid; continuing with HiMoE-50" >>"${OLD_STATUS}"

HIMOE_CONFIG=hiarm_new_button_finetune_chunk50
HIMOE_SMOKE_CONFIG=hiarm_new_button_finetune_chunk50_smoke
HIMOE_SMOKE_EXP=hiarm_new_button_chunk50_pd4_ga8_smoke_20260722
HIMOE_EXP=hiarm_new_button_native_chunk50_pd4_ga8_4gpu_8000_20260722
if ! run_smoke "${HIMOE_SMOKE_CONFIG}" "${HIMOE_SMOKE_EXP}"; then
  log "Per-device batch 4 smoke failed; falling back to batch 2 and accumulation 16"
  HIMOE_CONFIG=hiarm_new_button_finetune_chunk50_bs2
  HIMOE_SMOKE_CONFIG=hiarm_new_button_finetune_chunk50_bs2_smoke
  HIMOE_SMOKE_EXP=hiarm_new_button_chunk50_pd2_ga16_smoke_20260722
  HIMOE_EXP=hiarm_new_button_native_chunk50_pd2_ga16_4gpu_8000_20260722
  run_smoke "${HIMOE_SMOKE_CONFIG}" "${HIMOE_SMOKE_EXP}"
fi

OUTPUT="${HIMOE_ROOT}/checkpoints/${HIMOE_CONFIG}/${HIMOE_EXP}/checkpoint-8000/pytorch_model.pth"
LOGFILE="${HIMOE_ROOT}/logs/train_${HIMOE_EXP}.log"
if [[ -f "${OUTPUT}" ]]; then
  log "HiMoE-50 checkpoint-8000 already exists; nothing to do"
  exit 0
fi
RESUME=()
if compgen -G "${HIMOE_ROOT}/checkpoints/${HIMOE_CONFIG}/${HIMOE_EXP}/checkpoint-*" >/dev/null; then
  RESUME+=(--resume=True)
fi
log "HiMoE-50 formal training starting: config=${HIMOE_CONFIG}, 4 GPUs, effective batch 128, native chunk 50"
run_launch "${HIMOE_CONFIG}" "${HIMOE_EXP}" "${LOGFILE}" "${RESUME[@]}"
[[ -f "${OUTPUT}" ]]
log "HiMoE-50 completed at checkpoint-8000"
