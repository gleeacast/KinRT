train_config_name=$1
model_name=$2
gpu_use=$3
checkpoint_base_dir=${4:-/private/yth/projects/RoboTwin/policy/pi05/checkpoints}
shift $(( $# < 4 ? $# : 4 ))

export CUDA_VISIBLE_DEVICES=$gpu_use
export PATH=/opt/conda/envs/pi05/bin:$PATH
export XDG_CACHE_HOME=/private/yth/projects/cache
export HF_HOME=/private/yth/projects/cache/huggingface
export HF_LEROBOT_HOME=/private/yth/projects/cache/huggingface/lerobot
export HF_DATASETS_CACHE=/private/yth/projects/cache/huggingface/datasets
export OPENPI_DATA_HOME=/private/yth/models/pi05_base
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/opt/conda/envs/pi05/lib:$LD_LIBRARY_PATH  # Link libraries from the pi05 Conda environment.
export LD_PRELOAD=/private/yth/projects/RoboTwin/policy/pi05/nolock_preload2.so  # Intercept flock/fcntl to avoid ENOLCK on EPC NFS.
echo $CUDA_VISIBLE_DEVICES

XLA_PYTHON_CLIENT_ALLOCATOR=platform uv run python scripts/train.py $train_config_name --exp-name=$model_name --resume --checkpoint-base-dir $checkpoint_base_dir "$@"
