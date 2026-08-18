"""Generate frame-aligned MoE router labels from phase-level action chunks.

The output format matches ``DataConfig.router_labels_path``: an int32 numpy
array indexed by LeRobot global ``index``.  Each frame is labeled by a
MiniBatchKMeans cluster over its local future-action chunk.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import joblib

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from router_label_utils import (
    JOINT_DIMS, LEFT_DIMS, RIGHT_DIMS, LEFT_GRIPPER, RIGHT_GRIPPER,
    read_json, episode_path, fixed_list_column_to_numpy, int_column_to_numpy,
)


def build_frame_features(actions: np.ndarray, states: np.ndarray, *, horizon: int) -> np.ndarray:
    num_frames = actions.shape[0]
    tail = np.repeat(actions[-1:], max(horizon - 1, 0), axis=0)
    padded = np.concatenate([actions, tail], axis=0)
    indices = np.arange(num_frames)[:, None] + np.arange(horizon)[None, :]
    chunks = padded[indices].astype(np.float32)
    state0 = states.astype(np.float32)

    rel = chunks.copy()
    rel[:, :, JOINT_DIMS] -= state0[:, None, JOINT_DIMS]
    velocity = np.diff(chunks, axis=1, prepend=chunks[:, :1, :])

    left_motion = np.mean(np.abs(velocity[:, :, LEFT_DIMS]), axis=(1, 2))
    right_motion = np.mean(np.abs(velocity[:, :, RIGHT_DIMS]), axis=(1, 2))
    both_motion = np.minimum(left_motion, right_motion) / np.clip(np.maximum(left_motion, right_motion), 1e-6, None)
    grip = np.stack(
        [
            chunks[:, 0, LEFT_GRIPPER],
            chunks[:, -1, LEFT_GRIPPER],
            chunks[:, -1, LEFT_GRIPPER] - chunks[:, 0, LEFT_GRIPPER],
            chunks[:, 0, RIGHT_GRIPPER],
            chunks[:, -1, RIGHT_GRIPPER],
            chunks[:, -1, RIGHT_GRIPPER] - chunks[:, 0, RIGHT_GRIPPER],
        ],
        axis=1,
    )
    stats = np.concatenate(
        [
            chunks.mean(axis=1),
            chunks.std(axis=1),
            np.mean(np.abs(velocity), axis=1),
            np.stack([left_motion, right_motion, both_motion], axis=1),
            grip,
        ],
        axis=1,
    )
    return np.concatenate(
        [
            rel[:, :, JOINT_DIMS].reshape(num_frames, -1),
            velocity[:, :, JOINT_DIMS].reshape(num_frames, -1),
            stats,
        ],
        axis=1,
    ).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo"),
    )
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-clusters", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=48)
    parser.add_argument("--fit-stride", type=int, default=4)
    parser.add_argument("--max-fit-samples", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    info = read_json(repo_root / "meta" / "info.json")
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    data_path_pattern = info["data_path"]
    chunks_size = int(info["chunks_size"])
    output_dir = args.output_dir or (repo_root / "meta" / f"router_labels_k{args.num_clusters}_phase_h{args.horizon}")
    output_dir.mkdir(parents=True, exist_ok=True)

    features_all = []
    indices_all = []
    episodes_all = []
    frames_all = []
    for episode_index in range(total_episodes):
        table = pq.read_table(
            episode_path(repo_root, data_path_pattern, episode_index, chunks_size),
            columns=["action", "observation.state", "index", "frame_index", "episode_index"],
        )
        actions = fixed_list_column_to_numpy(table, "action")
        states = fixed_list_column_to_numpy(table, "observation.state")
        features = build_frame_features(actions, states, horizon=args.horizon)
        features_all.append(features)
        indices_all.append(int_column_to_numpy(table, "index"))
        frames_all.append(int_column_to_numpy(table, "frame_index"))
        episodes_all.append(int_column_to_numpy(table, "episode_index"))

    x = np.concatenate(features_all, axis=0)
    sample_indices = np.concatenate(indices_all, axis=0)
    sample_frames = np.concatenate(frames_all, axis=0)
    sample_episodes = np.concatenate(episodes_all, axis=0)

    scaler = StandardScaler()
    for start in range(0, x.shape[0], args.batch_size):
        scaler.partial_fit(x[start : start + args.batch_size])
    x_scaled = scaler.transform(x)
    pca_components = min(args.pca_components, x_scaled.shape[1], x_scaled.shape[0] - 1)
    pca = PCA(n_components=pca_components, random_state=args.seed)
    x_reduced = pca.fit_transform(x_scaled).astype(np.float32)

    fit_idx = np.arange(0, x_reduced.shape[0], max(args.fit_stride, 1), dtype=np.int64)
    if args.max_fit_samples is not None and fit_idx.shape[0] > args.max_fit_samples:
        rng = np.random.default_rng(args.seed)
        fit_idx = np.sort(rng.choice(fit_idx, size=args.max_fit_samples, replace=False))

    kmeans = MiniBatchKMeans(
        n_clusters=args.num_clusters,
        batch_size=args.batch_size,
        n_init=20,
        random_state=args.seed,
        reassignment_ratio=0.01,
    )
    kmeans.fit(x_reduced[fit_idx])
    labels = kmeans.predict(x_reduced).astype(np.int32)

    labels_by_index = np.full((total_frames,), -1, dtype=np.int32)
    labels_by_index[sample_indices] = labels
    np.save(output_dir / "router_labels.npy", labels_by_index)
    np.save(output_dir / "sample_indices.npy", sample_indices)
    np.save(output_dir / "sample_labels.npy", labels)
    joblib.dump(
        {
            "scaler": scaler,
            "pca": pca,
            "kmeans": kmeans,
            "horizon": int(args.horizon),
            "joint_dims": JOINT_DIMS,
            "left_dims": LEFT_DIMS,
            "right_dims": RIGHT_DIMS,
        },
        output_dir / "router_label_model.joblib",
    )

    counts = np.bincount(labels, minlength=args.num_clusters)
    with (output_dir / "episode_router_summary.jsonl").open("w", encoding="utf-8") as f:
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
        "num_samples": int(labels.shape[0]),
        "global_label_array_length": int(labels_by_index.shape[0]),
        "num_clusters": int(args.num_clusters),
        "cluster_counts": counts.astype(int).tolist(),
        "cluster_fractions": (counts / max(counts.sum(), 1)).astype(float).tolist(),
        "horizon": int(args.horizon),
        "feature_dim": int(x.shape[1]),
        "pca_components": int(pca_components),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "fit_stride": int(args.fit_stride),
        "fit_samples": int(fit_idx.shape[0]),
        "files": {
            "labels_by_global_index": "router_labels.npy",
            "model": "router_label_model.joblib",
            "episode_summary": "episode_router_summary.jsonl",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote phase router labels to: {output_dir}")
    print(f"Cluster counts: {counts.tolist()}")
    print(f"PCA explained variance ratio sum: {summary['pca_explained_variance_ratio_sum']:.4f}")


if __name__ == "__main__":
    main()
