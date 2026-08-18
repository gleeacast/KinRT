export XDG_CACHE_HOME=/private/yth/projects/cache
export LD_LIBRARY_PATH=/opt/conda/envs/pi05/lib:$LD_LIBRARY_PATH
export HDF5_USE_FILE_LOCKING=FALSE

data_dir=${1}
repo_id=${2}
uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py --raw_dir $data_dir --repo_id $repo_id
