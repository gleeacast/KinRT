#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${REPO_ROOT}/data"
PROCESSED_ROOT="${SCRIPT_DIR}/processed_data"
LOG_ROOT="${SCRIPT_DIR}/process_logs"
TASK_FILTER="${1:-${TASK_FILTER:-}}"
SETTING_FILTER="${2:-${SETTING_FILTER:-}}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-10}"
TIME_BIN=""

export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

for candidate in gtime /usr/bin/time /bin/time; do
  if command -v "${candidate}" >/dev/null 2>&1 && "${candidate}" -v true >/dev/null 2>&1; then
    TIME_BIN="$(command -v "${candidate}")"
    break
  fi
done

if [[ "${TASK_FILTER}" == "-h" || "${TASK_FILTER}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  process_all_data_pi05.sh [task_name [setting]]

Examples:
  DRY_RUN=1 process_all_data_pi05.sh click_alarmclock
  process_all_data_pi05.sh click_alarmclock
  MONITOR_INTERVAL=5 process_all_data_pi05.sh click_alarmclock demo_randomized

Environment:
  DRY_RUN=1              Preview selected data without processing.
  MONITOR_INTERVAL=10    Seconds between CPU/memory samples.
EOF
  exit 0
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Data root not found: ${DATA_ROOT}" >&2
  exit 1
fi

mkdir -p "${PROCESSED_ROOT}"
mkdir -p "${LOG_ROOT}"
cd "${SCRIPT_DIR}"

collect_descendants() {
  local parent_pid="$1"
  local child_pid
  local child_pids

  child_pids="$(pgrep -P "${parent_pid}" 2>/dev/null || true)"
  for child_pid in ${child_pids}; do
    printf '%s\n' "${child_pid}"
    collect_descendants "${child_pid}"
  done
}

monitor_process_tree() {
  local root_pid="$1"
  local log_file="$2"
  local interval="$3"
  local pids
  local timestamp

  printf 'timestamp\tpid\tppid\telapsed_sec\tcpu_pct\tmem_pct\trss_kb\tvsz_kb\tcommand\n' > "${log_file}"

  while kill -0 "${root_pid}" 2>/dev/null; do
    pids="$(printf '%s\n%s\n' "${root_pid}" "$(collect_descendants "${root_pid}")" | awk 'NF' | sort -n | uniq | tr '\n' ',' | sed 's/,$//')"
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    if [[ -n "${pids}" ]]; then
      ps -o pid=,ppid=,etimes=,%cpu=,%mem=,rss=,vsz=,comm= -p "${pids}" 2>/dev/null \
        | awk -v ts="${timestamp}" '{print ts "\t" $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6 "\t" $7 "\t" $8}' \
        >> "${log_file}" || true
    fi
    sleep "${interval}"
  done
}

run_with_monitoring() {
  local task="$1"
  local setting="$2"
  local episode_num="$3"
  local target_dir="$4"
  local timestamp
  local log_dir
  local command_pid
  local monitor_pid
  local status
  local start_epoch
  local end_epoch
  local time_summary

  timestamp="$(date '+%Y%m%d_%H%M%S')"
  log_dir="${LOG_ROOT}/${timestamp}_${task}_${setting}_${episode_num}"
  mkdir -p "${log_dir}"

  echo "[process] ${task}/${setting}: ${episode_num} episodes"
  echo "[log] ${log_dir}"

  if [[ -n "${TIME_BIN}" ]]; then
    "${TIME_BIN}" -v -o "${log_dir}/time.txt" \
      bash process_data_pi05.sh "${task}" "${setting}" "${episode_num}" \
      > "${log_dir}/process.log" 2>&1 &
  else
    start_epoch="$(date +%s)"
    {
      echo "GNU time not found; recording wall-clock duration only."
      echo "Start: $(date '+%Y-%m-%d %H:%M:%S')"
    } > "${log_dir}/time.txt"
    bash process_data_pi05.sh "${task}" "${setting}" "${episode_num}" \
      > "${log_dir}/process.log" 2>&1 &
  fi
  command_pid="$!"

  monitor_process_tree "${command_pid}" "${log_dir}/resource_samples.tsv" "${MONITOR_INTERVAL}" &
  monitor_pid="$!"

  set +e
  wait "${command_pid}"
  status="$?"
  wait "${monitor_pid}" 2>/dev/null
  set -e

  if [[ -z "${TIME_BIN}" ]]; then
    end_epoch="$(date +%s)"
    {
      echo "End: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "Elapsed seconds: $((end_epoch - start_epoch))"
    } >> "${log_dir}/time.txt"
  fi

  if [[ "${status}" -ne 0 ]]; then
    echo "[error] ${task}/${setting}: process failed with exit code ${status}" >&2
    echo "        See ${log_dir}/process.log and ${log_dir}/time.txt" >&2
    return "${status}"
  fi

  touch "${target_dir}/.complete"
  echo "[done] ${task}/${setting}: ${target_dir}"
  time_summary="$(grep -E 'Elapsed \(wall clock\) time|Maximum resident set size|Elapsed seconds' "${log_dir}/time.txt" | paste -sd ';' - || true)"
  if [[ -n "${time_summary}" ]]; then
    echo "[time] ${time_summary}"
  fi
}

while IFS=$'\t' read -r task setting episode_num; do
  if [[ -n "${TASK_FILTER}" && "${task}" != "${TASK_FILTER}" ]]; then
    continue
  fi

  if [[ -n "${SETTING_FILTER}" && "${setting}" != "${SETTING_FILTER}" ]]; then
    continue
  fi

  target_dir="${PROCESSED_ROOT}/${task}-${setting}-${episode_num}"

  if [[ -e "${target_dir}" ]]; then
    if [[ -f "${target_dir}/.complete" ]]; then
      echo "[skip] ${task}/${setting}: target already completed: ${target_dir}"
      continue
    fi

    echo "[error] ${task}/${setting}: target exists without .complete marker: ${target_dir}" >&2
    echo "        Inspect that directory before moving or deleting it, then rerun this script." >&2
    exit 1
  fi

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry-run] ${task}/${setting}: ${episode_num} episodes -> ${target_dir}"
  else
    run_with_monitoring "${task}" "${setting}" "${episode_num}" "${target_dir}"
  fi
done < <(
  python3 - <<'PY'
from pathlib import Path
import re
import sys

repo_root = Path(__file__).resolve().parents[2]
data_root = repo_root / "data"

for task_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
    for setting_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
        data_dir = setting_dir / "data"
        instr_dir = setting_dir / "instructions"
        if not data_dir.is_dir() or not instr_dir.is_dir():
            continue

        hdf5_ids = {
            int(match.group(1))
            for path in data_dir.glob("episode*.hdf5")
            if (match := re.fullmatch(r"episode(\d+)\.hdf5", path.name))
        }
        json_ids = {
            int(match.group(1))
            for path in instr_dir.glob("episode*.json")
            if (match := re.fullmatch(r"episode(\d+)\.json", path.name))
        }
        paired_ids = hdf5_ids & json_ids

        episode_num = 0
        while episode_num in paired_ids:
            episode_num += 1

        if episode_num == 0:
            print(f"warning: no paired consecutive episodes in {setting_dir}", file=sys.stderr)
            continue

        missing_hdf5 = sorted(json_ids - hdf5_ids)
        missing_json = sorted(hdf5_ids - json_ids)
        if missing_hdf5 or missing_json:
            print(
                f"warning: unmatched files in {setting_dir}; "
                f"missing hdf5={missing_hdf5[:5]}, missing json={missing_json[:5]}",
                file=sys.stderr,
            )

        print(f"{task_dir.name}\t{setting_dir.name}\t{episode_num}")
PY
)
