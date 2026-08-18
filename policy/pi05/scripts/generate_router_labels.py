"""Generate offline MoE router labels from LeRobot action chunks.

This script is intentionally independent from the training/model code. It reads
local LeRobot parquet episodes, builds future-action chunk features for each
frame, clusters them, and writes labels aligned by the dataset's global
``index`` field.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


JOINT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)


def _parse_episodes(value: str | None, total_episodes: int) -> list[int]:
    if value is None:
        return list(range(total_episodes))
    value = value.strip()
    if ":" in value:
        start_s, end_s = value.split(":", maxsplit=1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else total_episodes
        return list(range(start, end))
    return [int(x) for x in value.split(",") if x.strip()]


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _episode_path(repo_root: Path, data_path_pattern: str, episode_index: int, chunks_size: int) -> Path:
    return repo_root / data_path_pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def _fixed_list_column_to_numpy(table, name: str, dtype=np.float32) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=dtype)


def _int_column_to_numpy(table, name: str) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=np.int64)


def _build_episode_features(
    actions: np.ndarray,
    states: np.ndarray | None,
    *,
    action_horizon: int,
    feature_mode: str,
    relative_to_state: bool,
) -> np.ndarray:
    num_frames, action_dim = actions.shape
    if num_frames == 0:
        return np.empty((0, 0), dtype=np.float32)

    tail = np.repeat(actions[-1:], max(action_horizon - 1, 0), axis=0)
    padded_actions = np.concatenate([actions, tail], axis=0)
    indices = np.arange(num_frames)[:, None] + np.arange(action_horizon)[None, :]
    chunks = padded_actions[indices].astype(np.float32, copy=True)

    if relative_to_state:
        if states is None:
            raise ValueError("states must be provided when relative_to_state=True")
        chunks[:, :, JOINT_DIMS] -= states[:, None, JOINT_DIMS].astype(np.float32)

    if feature_mode == "chunk":
        parts = [chunks.reshape(num_frames, action_horizon * action_dim)]
    elif feature_mode == "chunk_velocity":
        velocity = np.diff(chunks, axis=1)
        parts = [
            chunks.reshape(num_frames, action_horizon * action_dim),
            velocity.reshape(num_frames, (action_horizon - 1) * action_dim),
        ]
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _iter_episode_batches(
    repo_root: Path,
    episodes: list[int],
    *,
    data_path_pattern: str,
    chunks_size: int,
    action_column: str,
    state_column: str,
    action_horizon: int,
    feature_mode: str,
    relative_to_state: bool,
    batch_size: int,
    desc: str,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    feature_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    episode_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    task_parts: list[np.ndarray] = []
    pending = 0

    for episode_index in tqdm(episodes, desc=desc):
        path = _episode_path(repo_root, data_path_pattern, episode_index, chunks_size)
        columns = [action_column, "index", "episode_index", "frame_index", "task_index"]
        if relative_to_state:
            columns.append(state_column)
        table = pq.read_table(
            path,
            columns=columns,
        )
        actions = _fixed_list_column_to_numpy(table, action_column)
        states = _fixed_list_column_to_numpy(table, state_column) if relative_to_state else None
        features = _build_episode_features(
            actions,
            states,
            action_horizon=action_horizon,
            feature_mode=feature_mode,
            relative_to_state=relative_to_state,
        )

        feature_parts.append(features)
        index_parts.append(_int_column_to_numpy(table, "index"))
        episode_parts.append(_int_column_to_numpy(table, "episode_index"))
        frame_parts.append(_int_column_to_numpy(table, "frame_index"))
        task_parts.append(_int_column_to_numpy(table, "task_index"))
        pending += features.shape[0]

        if pending >= batch_size:
            yield (
                np.concatenate(feature_parts, axis=0),
                np.concatenate(index_parts, axis=0),
                np.concatenate(episode_parts, axis=0),
                np.concatenate(frame_parts, axis=0),
                np.concatenate(task_parts, axis=0),
            )
            feature_parts.clear()
            index_parts.clear()
            episode_parts.clear()
            frame_parts.clear()
            task_parts.clear()
            pending = 0

    if pending:
        yield (
            np.concatenate(feature_parts, axis=0),
            np.concatenate(index_parts, axis=0),
            np.concatenate(episode_parts, axis=0),
            np.concatenate(frame_parts, axis=0),
            np.concatenate(task_parts, axis=0),
        )


def _fit_pca(
    repo_root: Path,
    episodes: list[int],
    *,
    data_path_pattern: str,
    chunks_size: int,
    action_column: str,
    state_column: str,
    action_horizon: int,
    feature_mode: str,
    relative_to_state: bool,
    batch_size: int,
    scaler: StandardScaler,
    pca_components: int,
) -> IncrementalPCA:
    pca = IncrementalPCA(n_components=pca_components, batch_size=batch_size)
    carry: np.ndarray | None = None

    for features, *_ in _iter_episode_batches(
        repo_root,
        episodes,
        data_path_pattern=data_path_pattern,
        chunks_size=chunks_size,
        action_column=action_column,
        state_column=state_column,
        action_horizon=action_horizon,
        feature_mode=feature_mode,
        relative_to_state=relative_to_state,
        batch_size=batch_size,
        desc="Fitting PCA",
    ):
        scaled = scaler.transform(features)
        if carry is not None:
            scaled = np.concatenate([carry, scaled], axis=0)
            carry = None
        if scaled.shape[0] < pca_components:
            carry = scaled
            continue
        usable = (scaled.shape[0] // pca_components) * pca_components
        pca.partial_fit(scaled[:usable])
        if usable < scaled.shape[0]:
            carry = scaled[usable:]

    if carry is not None and carry.shape[0] >= pca_components:
        pca.partial_fit(carry)

    return pca


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True, help="Path to a local LeRobot dataset repository.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes", type=str, default=None, help="Episode list like '0,1,2' or range like '0:800'.")
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--action-column", type=str, default="action")
    parser.add_argument("--state-column", type=str, default="observation.state")
    parser.add_argument("--feature-mode", choices=["chunk", "chunk_velocity"], default="chunk_velocity")
    parser.add_argument(
        "--relative-to-state",
        action="store_true",
        help="Use future action minus current state for joint dims. Default is pure action chunk clustering.",
    )
    parser.add_argument(
        "--kmeans-fit-stride",
        type=int,
        default=1,
        help="Fit KMeans on every Nth sample, then predict labels for every frame. 1 means full fit.",
    )
    parser.add_argument(
        "--max-kmeans-fit-samples",
        type=int,
        default=None,
        help="Optional cap for KMeans fitting samples after stride selection.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    info = _read_json(repo_root / "meta" / "info.json")
    episodes = _parse_episodes(args.episodes, int(info["total_episodes"]))
    output_dir = args.output_dir or (repo_root / "meta" / f"router_labels_k{args.num_clusters}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path_pattern = info["data_path"]
    chunks_size = int(info["chunks_size"])
    total_frames = int(info["total_frames"])

    first_path = _episode_path(repo_root, data_path_pattern, episodes[0], chunks_size)
    first_columns = [args.action_column, "index"]
    if args.relative_to_state:
        first_columns.append(args.state_column)
    first_table = pq.read_table(first_path, columns=first_columns)
    first_features = _build_episode_features(
        _fixed_list_column_to_numpy(first_table, args.action_column),
        _fixed_list_column_to_numpy(first_table, args.state_column) if args.relative_to_state else None,
        action_horizon=args.action_horizon,
        feature_mode=args.feature_mode,
        relative_to_state=args.relative_to_state,
    )
    feature_dim = first_features.shape[1]
    pca_components = min(args.pca_components, feature_dim)

    scaler = StandardScaler()
    num_samples = 0
    max_index = -1
    for features, indices, *_ in _iter_episode_batches(
        repo_root,
        episodes,
        data_path_pattern=data_path_pattern,
        chunks_size=chunks_size,
        action_column=args.action_column,
        state_column=args.state_column,
        action_horizon=args.action_horizon,
        feature_mode=args.feature_mode,
        relative_to_state=args.relative_to_state,
        batch_size=args.batch_size,
        desc="Fitting scaler",
    ):
        scaler.partial_fit(features)
        num_samples += features.shape[0]
        max_index = max(max_index, int(indices.max()))

    pca = _fit_pca(
        repo_root,
        episodes,
        data_path_pattern=data_path_pattern,
        chunks_size=chunks_size,
        action_column=args.action_column,
        state_column=args.state_column,
        action_horizon=args.action_horizon,
        feature_mode=args.feature_mode,
        relative_to_state=args.relative_to_state,
        batch_size=args.batch_size,
        scaler=scaler,
        pca_components=pca_components,
    )

    reduced = np.empty((num_samples, pca_components), dtype=np.float32)
    sample_indices = np.empty((num_samples,), dtype=np.int64)
    sample_episodes = np.empty((num_samples,), dtype=np.int64)
    sample_frames = np.empty((num_samples,), dtype=np.int64)
    sample_tasks = np.empty((num_samples,), dtype=np.int64)
    offset = 0

    for features, indices, episode_indices, frame_indices, task_indices in _iter_episode_batches(
        repo_root,
        episodes,
        data_path_pattern=data_path_pattern,
        chunks_size=chunks_size,
        action_column=args.action_column,
        state_column=args.state_column,
        action_horizon=args.action_horizon,
        feature_mode=args.feature_mode,
        relative_to_state=args.relative_to_state,
        batch_size=args.batch_size,
        desc="Transforming features",
    ):
        batch_reduced = pca.transform(scaler.transform(features)).astype(np.float32, copy=False)
        end = offset + features.shape[0]
        reduced[offset:end] = batch_reduced
        sample_indices[offset:end] = indices
        sample_episodes[offset:end] = episode_indices
        sample_frames[offset:end] = frame_indices
        sample_tasks[offset:end] = task_indices
        offset = end

    if args.kmeans_fit_stride < 1:
        raise ValueError("--kmeans-fit-stride must be >= 1")
    fit_mask = np.arange(num_samples) % args.kmeans_fit_stride == 0
    fit_indices = np.nonzero(fit_mask)[0]
    if args.max_kmeans_fit_samples is not None and fit_indices.shape[0] > args.max_kmeans_fit_samples:
        rng = np.random.default_rng(args.seed)
        fit_indices = np.sort(rng.choice(fit_indices, size=args.max_kmeans_fit_samples, replace=False))
    if fit_indices.shape[0] < args.num_clusters:
        raise ValueError(
            f"KMeans needs at least {args.num_clusters} fit samples, got {fit_indices.shape[0]}. "
            "Use a smaller --kmeans-fit-stride or larger --max-kmeans-fit-samples."
        )

    kmeans = KMeans(n_clusters=args.num_clusters, n_init=20, random_state=args.seed)
    kmeans.fit(reduced[fit_indices])
    labels = kmeans.predict(reduced).astype(np.int32)

    labels_by_index = np.full((max(total_frames, max_index + 1),), -1, dtype=np.int32)
    labels_by_index[sample_indices] = labels

    np.save(output_dir / "router_labels.npy", labels_by_index)
    np.save(output_dir / "sample_indices.npy", sample_indices)
    np.save(output_dir / "sample_labels.npy", labels)
    np.save(output_dir / "cluster_centers_pca.npy", kmeans.cluster_centers_.astype(np.float32))
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "kmeans": kmeans,
            "joint_dims": JOINT_DIMS,
            "feature_mode": args.feature_mode,
            "relative_to_state": args.relative_to_state,
            "action_column": args.action_column,
            "state_column": args.state_column,
            "action_horizon": args.action_horizon,
        },
        output_dir / "router_label_model.joblib",
    )

    counts = np.bincount(labels, minlength=args.num_clusters)
    episode_summary_path = output_dir / "episode_router_summary.jsonl"
    with episode_summary_path.open("w", encoding="utf-8") as f:
        for episode_index in sorted(set(sample_episodes.tolist())):
            mask = sample_episodes == episode_index
            ep_counts = np.bincount(labels[mask], minlength=args.num_clusters)
            record = {
                "episode_index": int(episode_index),
                "num_frames": int(mask.sum()),
                "dominant_cluster": int(ep_counts.argmax()),
                "cluster_counts": ep_counts.astype(int).tolist(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "episodes": args.episodes or f"0:{info['total_episodes']}",
        "num_samples": int(num_samples),
        "global_label_array_length": int(labels_by_index.shape[0]),
        "num_clusters": int(args.num_clusters),
        "cluster_counts": counts.astype(int).tolist(),
        "cluster_fractions": (counts / max(counts.sum(), 1)).astype(float).tolist(),
        "kmeans_inertia": float(kmeans.inertia_),
        "kmeans_fit_samples": int(fit_indices.shape[0]),
        "kmeans_fit_stride": int(args.kmeans_fit_stride),
        "max_kmeans_fit_samples": args.max_kmeans_fit_samples,
        "action_horizon": int(args.action_horizon),
        "feature_mode": args.feature_mode,
        "relative_to_state": bool(args.relative_to_state),
        "action_column": args.action_column,
        "state_column": args.state_column,
        "feature_dim": int(feature_dim),
        "pca_components": int(pca_components),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "files": {
            "labels_by_global_index": "router_labels.npy",
            "sample_indices": "sample_indices.npy",
            "sample_labels": "sample_labels.npy",
            "model": "router_label_model.joblib",
            "episode_summary": "episode_router_summary.jsonl",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote router labels to: {output_dir}")
    print(f"Cluster counts: {counts.tolist()}")
    print(f"PCA explained variance ratio sum: {summary['pca_explained_variance_ratio_sum']:.4f}")


if __name__ == "__main__":
    main()

