#!/bin/bash

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 # ensure GPU < 24G

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." # move to root

config=${1:-policy/pi05/deploy_policy.yml}

get_yaml() {
    python -c 'import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")); value = cfg.get(sys.argv[2]); print("" if value is None else value)' "$config" "$1"
}

gpu_id=${SERVER_GPU_ID:-$(get_yaml server_gpu_id)}
host=${SERVER_BIND_HOST:-$(get_yaml server_bind_host)}
port=${SERVER_PORT:-$(get_yaml port)}
gpu_id=${gpu_id:-0}
host=${host:-0.0.0.0}
port=${port:-9000}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mserver gpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mserver address: ${host}:${port}\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python script/policy_model_server.py \
    --host ${host} \
    --port ${port} \
    --config ${config}
