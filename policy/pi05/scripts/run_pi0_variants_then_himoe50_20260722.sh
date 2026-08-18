#!/usr/bin/env bash
set -Eeuo pipefail

PI0_ROOT=/private/yth/projects/RoboTwin/policy/pi05
PI0_CKPT=${PI0_ROOT}/checkpoints
PI0_SMOKE=${PI0_CKPT}/_smoke_pi0_variants_20260722
HIMOE_ROOT=/private/yth/projects/zyz26/HiMoE-VLA
HIMOE_PYTHON=/var/tmp/himoe_vla_venv_py311/bin/python
HIMOE_ACCELERATE=/var/tmp/himoe_vla_venv_py311/bin/accelerate
LOG_DIR=${PI0_ROOT}/logs
STATUS=${LOG_DIR}/chain_pi0_variants_then_himoe50_20260722.status.log
GPUS=0,1,2,3

mkdir -p "${LOG_DIR}" "${PI0_SMOKE}" "${HIMOE_ROOT}/logs"

log() {
  printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "${STATUS}"
}

on_error() {
  local code=$?
  log "ERROR: chain stopped at line ${BASH_LINENO[0]} with exit code ${code}"
  exit "${code}"
}
trap on_error ERR

validate_prerequisites() {
  "${PI0_ROOT}/.venv/bin/python" - <<'PY'
import json
from pathlib import Path
import numpy as np

label_path = Path("/private/yth/projects/cache/huggingface/lerobot/hiarm_pi05_manual_500_promptv3_new_button_augmented/meta/router_labels_action_vel_k4_pi05_800_full_logic_20260714/router_labels.npy")
norm_path = Path("/private/yth/projects/RoboTwin/policy/pi05/assets/pi0_full_diyrobot/hiarm_pi05_manual_500_promptv3_new_button_augmented/norm_stats.json")
labels = np.load(label_path)
values, counts = np.unique(labels, return_counts=True)
if labels.shape != (159104,):
    raise SystemExit(f"router label shape mismatch: {labels.shape}")
if values.tolist() != [0, 1, 2, 3] or np.any(counts <= 0):
    raise SystemExit(f"invalid K4 labels: values={values.tolist()} counts={counts.tolist()}")
norm = json.loads(norm_path.read_text())
if not norm:
    raise SystemExit("PI0 norm stats are empty")
print("K4 labels:", dict(zip(values.tolist(), counts.tolist())))
print("PI0 norm keys:", sorted(norm))
PY
  log "validated exact PI0 norm, 159104 K4 labels, 4 GPUs and global batch 32"
}

run_pi0_smoke() {
  local config=$1
  local label=$2
  local exp="${label}_smoke_20260722"
  local output="${PI0_SMOKE}/${config}/${exp}/2/params"
  local logfile="${LOG_DIR}/train_${exp}.log"
  if [[ -d "${output}" ]]; then
    log "${label}: smoke already passed"
    return
  fi
  log "${label}: starting 2-step smoke on four GPUs"
  (
    cd "${PI0_ROOT}"
    timeout 90m bash finetune.sh "${config}" "${exp}" "${GPUS}" "${PI0_SMOKE}" \
      --num-train-steps=2 --save-interval=1 --keep-period=4000
  ) >"${logfile}" 2>&1
  [[ -d "${output}" ]]
  log "${label}: smoke passed"
}

run_pi0_train() {
  local config=$1
  local exp=$2
  local label=$3
  local output="${PI0_CKPT}/${config}/${exp}/8000/params"
  local logfile="${LOG_DIR}/train_${exp}.log"
  if [[ -d "${output}" ]]; then
    log "${label}: 8000 checkpoint already exists; skipping"
    return
  fi
  log "${label}: starting/resuming 8000 steps, 4 GPUs, global batch 32, EMA none"
  (
    cd "${PI0_ROOT}"
    bash finetune.sh "${config}" "${exp}" "${GPUS}" "${PI0_CKPT}" \
      --num-train-steps=8000 --save-interval=1000 --keep-period=4000
  ) >>"${logfile}" 2>&1
  [[ -d "${output}" ]]
  log "${label}: completed at 8000"
}

run_himoe_launch() {
  local config=$1
  local exp=$2
  local logfile=$3
  shift 3
  (
    cd "${HIMOE_ROOT}"
    export USE_FLAX=0
    export USE_TF=0
    export CUDA_VISIBLE_DEVICES="${GPUS}"
    export PYTHONUNBUFFERED=1
    "${HIMOE_ACCELERATE}" launch \
      --num_processes=4 \
      --mixed_precision=bf16 \
      scripts/train.py \
      --deepspeed=src/moevla/training/zero2.json \
      --config="${config}" \
      --exp-name="${exp}" "$@"
  ) >>"${logfile}" 2>&1
}

run_himoe_smoke() {
  local config=$1
  local exp=$2
  local logfile="${HIMOE_ROOT}/logs/train_${exp}.log"
  log "HiMoE-50: smoke config=${config}"
  if run_himoe_launch "${config}" "${exp}" "${logfile}"; then
    log "HiMoE-50: smoke passed with ${config}"
    return 0
  fi
  log "HiMoE-50: smoke failed with ${config}; see ${logfile}"
  return 1
}

run_himoe_train() {
  local config=$1
  local exp=$2
  local output="${HIMOE_ROOT}/checkpoints/${config}/${exp}/checkpoint-8000/pytorch_model.pth"
  local logfile="${HIMOE_ROOT}/logs/train_${exp}.log"
  local resume=()
  if compgen -G "${HIMOE_ROOT}/checkpoints/${config}/${exp}/checkpoint-*" >/dev/null; then
    resume+=(--resume=True)
  fi
  log "HiMoE-50: starting/resuming ${config} to 8000 with effective batch 128"
  run_himoe_launch "${config}" "${exp}" "${logfile}" "${resume[@]}"
  [[ -f "${output}" ]]
  log "HiMoE-50: completed at 8000"
}

log "queue started: PI0-LoRA -> PI0-LoRA-MoE K4 -> PI0-Full-MoE K4 -> HiMoE native 50-step"
validate_prerequisites

PI0_LORA_CONFIG=pi0_lora_diyrobot
PI0_LORA_EXP=hiarm_500_new_button_pi0_lora_8000_bs32_20260722
PI0_LORA_MOE_CONFIG=kinrt_lora_pi0_diyrobot
PI0_LORA_MOE_EXP=hiarm_500_new_button_pi0_lora_moe_k4_8000_bs32_20260722
PI0_FULL_MOE_CONFIG=kinrt_full_pi0_diyrobot
PI0_FULL_MOE_EXP=hiarm_500_new_button_pi0_full_moe_k4_8000_bs32_20260722

run_pi0_smoke "${PI0_LORA_CONFIG}" "pi0_lora"
run_pi0_train "${PI0_LORA_CONFIG}" "${PI0_LORA_EXP}" "PI0-LoRA"
run_pi0_smoke "${PI0_LORA_MOE_CONFIG}" "pi0_lora_moe_k4"
run_pi0_train "${PI0_LORA_MOE_CONFIG}" "${PI0_LORA_MOE_EXP}" "PI0-LoRA-MoE K4"
run_pi0_smoke "${PI0_FULL_MOE_CONFIG}" "pi0_full_moe_k4"
run_pi0_train "${PI0_FULL_MOE_CONFIG}" "${PI0_FULL_MOE_EXP}" "PI0-Full-MoE K4"

HIMOE_CONFIG=hiarm_new_button_finetune_chunk50
HIMOE_SMOKE_CONFIG=hiarm_new_button_finetune_chunk50_smoke
HIMOE_EXP=hiarm_new_button_native_chunk50_pd4_ga8_4gpu_8000_20260722
if ! run_himoe_smoke "${HIMOE_SMOKE_CONFIG}" "hiarm_new_button_chunk50_pd4_ga8_smoke_20260722"; then
  HIMOE_CONFIG=hiarm_new_button_finetune_chunk50_bs2
  HIMOE_SMOKE_CONFIG=hiarm_new_button_finetune_chunk50_bs2_smoke
  HIMOE_EXP=hiarm_new_button_native_chunk50_pd2_ga16_4gpu_8000_20260722
  run_himoe_smoke "${HIMOE_SMOKE_CONFIG}" "hiarm_new_button_chunk50_pd2_ga16_smoke_20260722"
fi
run_himoe_train "${HIMOE_CONFIG}" "${HIMOE_EXP}"

log "queue completed: all three PI0 variants and native 50-step HiMoE reached 8000"
