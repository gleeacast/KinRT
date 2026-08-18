train_config_name=$1

export XDG_CACHE_HOME=/private/yth/projects/cache           # LeRobot dataset cache.
export OPENPI_DATA_HOME=/private/yth/projects/cache/openpi  # PI0.5 model cache.
export LD_LIBRARY_PATH=/opt/conda/envs/pi05/lib:$LD_LIBRARY_PATH  # Link libraries from the pi05 Conda environment.
export LD_PRELOAD=/private/yth/projects/nolock_preload.so   # Work around unsupported file locking on this filesystem.

uv run scripts/compute_norm_stats.py --config-name $train_config_name
