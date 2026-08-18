task_name=${1}
setting=${2}
expert_data_num=${3}

export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

python scripts/process_data.py $task_name $setting $expert_data_num