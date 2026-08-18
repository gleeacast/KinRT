"""Generate frame-aligned MoE router labels with improved clustering quality.

Improvements over v1 (generate_phase_router_labels.py):

1. **Feature rebalancing**: After StandardScaler normalization, multiply the
   discriminative stats dimensions (gripper deltas, motion magnitudes, arm
   asymmetry) by ``--stats-weight`` (default 4.0).  In v1 these 51 dims
   competed against 576 raw-joint dims in PCA, so they had ~11× less
   influence.  Scaling them up to equal footing prevents the two dominant
   "generic" clusters (k18 clusters 1 and 8, together ≈70% of frames) that
   made max_fraction=0.43 ineffective.

2. **Episode-progress feature**: A single normalized frame-position value
   [0, 1] is appended to the stats block.  This temporal context helps
   k-means discover phase sequences (early→grasp, mid→transfer,
   late→release) without manually encoding them.

3. **Cross-repo fitting**: Optionally include ``--extra-episodes`` episodes
   sampled uniformly from an ``--extra-repo`` (e.g. demo_randomized_repo).
   These extra frames are used only for scaler / PCA / k-means *fitting*;
   router labels are output only for the primary repo.  The result is a
   model that generalises to action styles beyond the 8 training tasks.
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
    read_json, episode_path, fixed_list_column_to_numpy, int_column_to_numpy,)

# Number of raw-trajectory dims produced by build_frame_features_v2:
#   rel:  horizon * len(JOINT_DIMS) = 24 * 12 = 288
#   vel:  horizon * len(JOINT_DIMS) = 24 * 12 = 288
# Stats dimensions start after these.
_TRAJ_DIMS_PER_PART = None  # computed lazily from first feature vector


def build_frame_features_v2(
    actions: np.ndarray,
    states: np.ndarray,
    *,
    horizon: int,
) -> tuple[np.ndarray, int]:
    """Build per-frame feature matrix with episode-progress appended to stats.

    Returns
    -------
    features : np.ndarray, shape (num_frames, feature_dim)
    traj_dim : int
        Number of leading trajectory dimensions (rel + vel).
        Stats occupy ``features[:, traj_dim:]``.
    """
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
    both_motion = np.minimum(left_motion, right_motion) / np.clip(
        np.maximum(left_motion, right_motion), 1e-6, None
    )

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
    )  # (num_frames, 6)

    # Episode-progress: normalised position [0, 1] within this episode.
    progress = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)  # (num_frames,)

    stats = np.concatenate(
        [
            chunks.mean(axis=1),                            # 14: mean joint positions
            chunks.std(axis=1),                             # 14: joint position variance
            np.mean(np.abs(velocity), axis=1),              # 14: mean abs velocity
            np.stack([left_motion, right_motion, both_motion], axis=1),  # 3: hand activity
            grip,                                           # 6: gripper states & deltas
            progress[:, None],                              # 1: episode progress  ← NEW
        ],
        axis=1,
    )  # (num_frames, 52)

    rel_flat = rel[:, :, JOINT_DIMS].reshape(num_frames, -1)       # (N, 288)
    vel_flat = velocity[:, :, JOINT_DIMS].reshape(num_frames, -1)  # (N, 288)
    traj_dim = rel_flat.shape[1] + vel_flat.shape[1]

    features = np.concatenate([rel_flat, vel_flat, stats], axis=1).astype(np.float32)
    return features, traj_dim


def _sample_extra_episodes(
    extra_repo: Path,
    *,
    num_episodes: int,
    seed: int,
) -> list[int]:
    """Return a list of episode indices sampled uniformly from extra_repo."""
    info = read_json(extra_repo / "meta" / "info.json")
    total = int(info["total_episodes"])
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=min(num_episodes, total), replace=False)
    return sorted(indices.tolist())


def _load_repo_features(
    repo_root: Path,
    episode_indices: list[int],
    *,
    horizon: int,
    primary: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, int]:
    """Load frame features for a list of episodes from *repo_root*.

    Returns
    -------
    x             : (total_frames, feature_dim)  feature matrix
    global_idx    : (total_frames,) global LeRobot index  -- primary only
    frame_idx     : (total_frames,) frame_index            -- primary only
    episode_idx   : (total_frames,) episode_index          -- primary only
    traj_dim      : scalar, offset where stats block starts
    """
    info = read_json(repo_root / "meta" / "info.json")
    data_path_pattern = info["data_path"]
    chunks_size = int(info["chunks_size"])

    features_all: list[np.ndarray] = []
    global_idx_all: list[np.ndarray] = []
    frame_idx_all: list[np.ndarray] = []
    episode_idx_all: list[np.ndarray] = []
    traj_dim: int = 0

    for episode_index in episode_indices:
        columns = ["action", "observation.state"]
        if primary:
            columns += ["index", "frame_index", "episode_index"]
        table = pq.read_table(
            episode_path(repo_root, data_path_pattern, episode_index, chunks_size),
            columns=columns,
        )
        actions = fixed_list_column_to_numpy(table, "action")
        states = fixed_list_column_to_numpy(table, "observation.state")
        feats, traj_dim = build_frame_features_v2(actions, states, horizon=horizon)
        features_all.append(feats)

        if primary:
            global_idx_all.append(int_column_to_numpy(table, "index"))
            frame_idx_all.append(int_column_to_numpy(table, "frame_index"))
            episode_idx_all.append(int_column_to_numpy(table, "episode_index"))

    x = np.concatenate(features_all, axis=0)
    if primary:
        return (
            x,
            np.concatenate(global_idx_all),
            np.concatenate(frame_idx_all),
            np.concatenate(episode_idx_all),
            traj_dim,
        )
    return x, None, None, None, traj_dim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo"),
        help="Primary repo. Router labels are produced for this repo only.",
    )
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--num-clusters", type=int, default=20)
    parser.add_argument("--pca-components", type=int, default=48)
    parser.add_argument("--fit-stride", type=int, default=4)
    parser.add_argument("--max-fit-samples", type=int, default=60000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)

    # Feature rebalancing ────────────────────────────────────────────────
    parser.add_argument(
        "--stats-weight",
        type=float,
        default=4.0,
        help=(
            "After StandardScaler, multiply stats dimensions by this factor so "
            "that the 52-dim discriminative block (gripper, motion, asymmetry, "
            "progress) has equal total influence as the 576-dim raw-joint block "
            "in PCA.  Formula: weight ≈ sqrt(576/52) ≈ 3.3; default 4.0 gives "
            "a slight over-weighting for sharper phase separation."
        ),
    )

    # Cross-repo generalisation ──────────────────────────────────────────
    parser.add_argument(
        "--extra-repo",
        type=Path,
        default=None,
        help=(
            "Optional secondary repo whose episodes are included in scaler / "
            "PCA / k-means *fitting* but NOT labelled.  Use demo_randomized_repo "
            "for better generalisation beyond the 8 training tasks."
        ),
    )
    parser.add_argument(
        "--extra-episodes",
        type=int,
        default=400,
        help="Number of episodes to sample uniformly from --extra-repo.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    info = read_json(repo_root / "meta" / "info.json")
    total_episodes = int(info["total_episodes"])
    total_frames = int(info["total_frames"])
    data_path_pattern = info["data_path"]
    chunks_size = int(info["chunks_size"])

    tag = f"k{args.num_clusters}_phase_h{args.horizon}_v2"
    if args.extra_repo is not None:
        tag += "_crossrepo"
    output_dir = args.output_dir or (repo_root / "meta" / f"router_labels_{tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading primary repo: {repo_root}  ({total_episodes} episodes)")
    primary_x, sample_indices, sample_frames, sample_episodes, traj_dim = (
        _load_repo_features(
            repo_root,
            list(range(total_episodes)),
            horizon=args.horizon,
            primary=True,
        )
    )

    # Optionally load extra-repo frames for model fitting only.
    extra_x: np.ndarray | None = None
    if args.extra_repo is not None:
        extra_repo = args.extra_repo.resolve()
        extra_ep_indices = _sample_extra_episodes(
            extra_repo, num_episodes=args.extra_episodes, seed=args.seed
        )
        print(
            f"Loading extra repo: {extra_repo}  "
            f"(sampling {len(extra_ep_indices)} / "
            f"{read_json(extra_repo / 'meta' / 'info.json')['total_episodes']} episodes)"
        )
        extra_x, *_ = _load_repo_features(
            extra_repo,
            extra_ep_indices,
            horizon=args.horizon,
            primary=False,
        )
        print(f"  Extra frames: {extra_x.shape[0]}")

    # ── Standardise ──────────────────────────────────────────────────────
    # Fit scaler on combined data (primary + extra) for better normalisation.
    fit_x = np.concatenate([primary_x, extra_x], axis=0) if extra_x is not None else primary_x
    scaler = StandardScaler()
    for start in range(0, fit_x.shape[0], args.batch_size):
        scaler.partial_fit(fit_x[start : start + args.batch_size])

    # Scale primary and extra separately, keeping them aligned.
    primary_scaled = scaler.transform(primary_x)
    extra_scaled = scaler.transform(extra_x) if extra_x is not None else None

    # ── Feature rebalancing ───────────────────────────────────────────────
    # stats_weight amplifies the discriminative block (gripper / motion /
    # progress) so PCA picks up these directions instead of being dominated
    # by the 576 raw-joint dims.
    if args.stats_weight != 1.0:
        primary_scaled[:, traj_dim:] *= args.stats_weight
        if extra_scaled is not None:
            extra_scaled[:, traj_dim:] *= args.stats_weight

    fit_scaled = (
        np.concatenate([primary_scaled, extra_scaled], axis=0)
        if extra_scaled is not None
        else primary_scaled
    )

    # ── PCA ───────────────────────────────────────────────────────────────
    pca_components = min(args.pca_components, fit_scaled.shape[1], fit_scaled.shape[0] - 1)
    pca = PCA(n_components=pca_components, random_state=args.seed)
    pca.fit(fit_scaled)

    primary_reduced = pca.transform(primary_scaled).astype(np.float32)
    extra_reduced = pca.transform(extra_scaled).astype(np.float32) if extra_scaled is not None else None

    fit_reduced = (
        np.concatenate([primary_reduced, extra_reduced], axis=0)
        if extra_reduced is not None
        else primary_reduced
    )

    # ── k-means fit ──────────────────────────────────────────────────────
    # Subsample for fitting, balancing primary / extra if both present.
    rng = np.random.default_rng(args.seed)

    def _subsample(arr: np.ndarray, target: int) -> np.ndarray:
        step = max(arr.shape[0] // target, 1)
        idx = np.arange(0, arr.shape[0], step, dtype=np.int64)
        if idx.shape[0] > target:
            idx = rng.choice(idx, size=target, replace=False)
        return np.sort(idx)

    if extra_reduced is not None:
        # Equal budget split between primary and extra so extra doesn't dominate.
        half = args.max_fit_samples // 2
        p_idx = _subsample(primary_reduced, half)
        e_offset = primary_reduced.shape[0]
        e_idx = _subsample(extra_reduced, half) + e_offset
        fit_idx = np.concatenate([p_idx, e_idx])
    else:
        fit_idx = _subsample(fit_reduced, args.max_fit_samples)

    kmeans = MiniBatchKMeans(
        n_clusters=args.num_clusters,
        batch_size=args.batch_size,
        n_init=20,
        random_state=args.seed,
        reassignment_ratio=0.01,
    )
    kmeans.fit(fit_reduced[fit_idx])

    # ── Predict labels for primary repo only ─────────────────────────────
    labels = kmeans.predict(primary_reduced).astype(np.int32)

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
            "stats_weight": float(args.stats_weight),
            "traj_dim": int(traj_dim),
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
        "extra_repo": str(args.extra_repo) if args.extra_repo else None,
        "extra_episodes_used": int(len(extra_ep_indices)) if args.extra_repo else 0,
        "num_samples": int(labels.shape[0]),
        "global_label_array_length": int(labels_by_index.shape[0]),
        "num_clusters": int(args.num_clusters),
        "cluster_counts": counts.astype(int).tolist(),
        "cluster_fractions": (counts / max(counts.sum(), 1)).astype(float).tolist(),
        "max_fraction": float(counts.max() / max(counts.sum(), 1)),
        "horizon": int(args.horizon),
        "stats_weight": float(args.stats_weight),
        "traj_dim": int(traj_dim),
        "feature_dim": int(primary_x.shape[1]),
        "pca_components": int(pca_components),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "fit_samples": int(fit_idx.shape[0]),
        "files": {
            "labels_by_global_index": "router_labels.npy",
            "model": "router_label_model.joblib",
            "episode_summary": "episode_router_summary.jsonl",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\nWrote phase router labels to: {output_dir}")
    print(f"Cluster counts:   {counts.tolist()}")
    print(f"Cluster fractions: {[f'{v:.3f}' for v in summary['cluster_fractions']]}")
    print(f"Max fraction:      {summary['max_fraction']:.3f}  (target < 0.25 after merge)")
    print(f"PCA explained var: {summary['pca_explained_variance_ratio_sum']:.4f}")


if __name__ == "__main__":
    main()
