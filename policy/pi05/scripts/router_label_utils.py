"""Shared utilities for router-label generation scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


JOINT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
LEFT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
RIGHT_DIMS = np.asarray([7, 8, 9, 10, 11, 12, 13], dtype=np.int64)
LEFT_GRIPPER = 6
RIGHT_GRIPPER = 13


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_path(repo_root: Path, data_path_pattern: str, episode_index: int, chunks_size: int) -> Path:
    return repo_root / data_path_pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def fixed_list_column_to_numpy(table, name: str, dtype=np.float32) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=dtype)


def int_column_to_numpy(table, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=np.int64)


def read_parquet_episode(
    repo_root: Path,
    data_path_pattern: str,
    episode_index: int,
    chunks_size: int,
    *,
    extra_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Read actions, states, and optional extra columns from one episode parquet file.

    Returns
    -------
    actions : (T, action_dim)
    states  : (T, state_dim)
    extras  : dict of column_name -> np.ndarray, only for requested extra_columns
    """
    columns = ["action", "observation.state"] + (extra_columns or [])
    table = pq.read_table(
        episode_path(repo_root, data_path_pattern, episode_index, chunks_size),
        columns=columns,
    )
    actions = fixed_list_column_to_numpy(table, "action")
    states = fixed_list_column_to_numpy(table, "observation.state")
    extras = {col: int_column_to_numpy(table, col) for col in (extra_columns or [])}
    return actions, states, extras
