#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.." # move to root

config=${CONFIG:-policy/pi05/deploy_policy.yml}
task_name=${1:-}
gpu_arg=${2:-}

get_yaml() {
    python -c 'import sys, yaml; cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8")); value = cfg.get(sys.argv[2]); print("" if value is None else value)' "$config" "$1"
}

server_host=${MODEL_SERVER_HOST:-$(get_yaml model_server_host)}
server_port=${MODEL_SERVER_PORT:-$(get_yaml port)}
gpu_id=${gpu_arg:-${EVAL_GPU_ID:-$(get_yaml eval_gpu_id)}}
task_name=${task_name:-${TASK_NAME:-$(get_yaml task_name)}}
server_port=${server_port:-9000}
gpu_id=${gpu_id:-0}

if [ -z "${server_host}" ]; then
    echo -e "\033[31mmodel_server_host is empty. Set it in ${config} or export MODEL_SERVER_HOST.\033[0m"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33meval gpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33meval task name: ${task_name}\033[0m"
echo -e "\033[33mremote model server: ${server_host}:${server_port}\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy_client.py \
    --host ${server_host} \
    --port ${server_port} \
    --config ${config} \
    --overrides \
    --task_name ${task_name}
