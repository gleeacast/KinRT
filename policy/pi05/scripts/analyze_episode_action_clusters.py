"""Analyze episode-level action similarity for candidate MoE expert groups.

This is an analysis script, not a training entrypoint. It samples episodes from
the mixed LeRobot repo, builds compact trajectory descriptors, clusters episodes
for several values of K, and reports whether clusters are coherent by task and
clean/random split.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


JOINT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _episode_path(repo_root: Path, data_path_pattern: str, episode_index: int, chunks_size: int) -> Path:
    return repo_root / data_path_pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def _fixed_list_column_to_numpy(table, name: str, dtype=np.float32) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=dtype)


def _resample_sequence(values: np.ndarray, steps: int) -> np.ndarray:
    if values.shape[0] == 0:
        return np.zeros((steps, values.shape[1]), dtype=np.float32)
    if values.shape[0] == 1:
        return np.repeat(values, steps, axis=0).astype(np.float32)
    source = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    target = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    columns = [np.interp(target, source, values[:, dim]) for dim in range(values.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def _episode_feature(actions: np.ndarray, states: np.ndarray, *, resample_steps: int) -> np.ndarray:
    action = actions.astype(np.float32)
    state = states.astype(np.float32)
    delta_to_state = action - state
    velocity = np.diff(action, axis=0, prepend=action[:1])

    # Use joint dims for trajectory shape; keep grippers in aggregate stats.
    action_j = action[:, JOINT_DIMS]
    delta_j = delta_to_state[:, JOINT_DIMS]
    velocity_j = velocity[:, JOINT_DIMS]

    resampled = np.concatenate(
        [
            _resample_sequence(action_j, resample_steps),
            _resample_sequence(delta_j, resample_steps),
            _resample_sequence(velocity_j, resample_steps),
        ],
        axis=1,
    ).reshape(-1)

    stats = np.concatenate(
        [
            action.mean(axis=0),
            action.std(axis=0),
            delta_to_state.mean(axis=0),
            delta_to_state.std(axis=0),
            np.mean(np.abs(velocity), axis=0),
            np.asarray([float(action.shape[0])], dtype=np.float32),
        ],
        axis=0,
    )
    return np.concatenate([resampled, stats], axis=0).astype(np.float32)


def _infer_task_and_variant(episode_index: int, episodes_per_task: int) -> tuple[int, str]:
    task_id = episode_index // episodes_per_task
    offset = episode_index % episodes_per_task
    variant = "clean" if offset < episodes_per_task // 2 else "random"
    return int(task_id), variant


def _sample_episodes(total_episodes: int, episodes_per_task: int, sample_per_variant: int) -> list[int]:
    if total_episodes % episodes_per_task != 0:
        raise ValueError(
            f"Expected total_episodes to be divisible by episodes_per_task, got "
            f"{total_episodes} and {episodes_per_task}"
        )
    sampled = []
    for task_start in range(0, total_episodes, episodes_per_task):
        clean_start = task_start
        random_start = task_start + episodes_per_task // 2
        sampled.extend(range(clean_start, min(clean_start + sample_per_variant, random_start)))
        sampled.extend(range(random_start, min(random_start + sample_per_variant, task_start + episodes_per_task)))
    return sampled


def _counts(values: np.ndarray, minlength: int | None = None) -> list[int]:
    if minlength is None:
        minlength = int(values.max()) + 1 if values.size else 0
    return np.bincount(values.astype(np.int64), minlength=minlength).astype(int).tolist()


def _distribution(values: np.ndarray, *, minlength: int) -> np.ndarray:
    counts = np.bincount(values.astype(np.int64), minlength=minlength).astype(np.float64)
    return counts / max(float(counts.sum()), 1.0)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _clean_fit_random_predict(
    x: np.ndarray,
    task_ids: np.ndarray,
    variant_ids: np.ndarray,
    records: list[dict],
    *,
    total_tasks: int,
    k_values: list[int],
    seed: int,
) -> list[dict]:
    clean_mask = variant_ids == 0
    random_mask = variant_ids == 1
    scaler = StandardScaler().fit(x[clean_mask])
    x_clean = scaler.transform(x[clean_mask])
    x_random = scaler.transform(x[random_mask])
    task_clean = task_ids[clean_mask]
    task_random = task_ids[random_mask]
    records_clean = [records[i] for i in np.flatnonzero(clean_mask)]
    records_random = [records[i] for i in np.flatnonzero(random_mask)]

    results = []
    for k in k_values:
        if k < 2 or k >= x_clean.shape[0]:
            continue
        kmeans = KMeans(n_clusters=k, n_init=50, random_state=seed)
        clean_labels = kmeans.fit_predict(x_clean)
        random_labels = kmeans.predict(x_random)

        clean_task_cluster = np.zeros((total_tasks, k), dtype=np.float64)
        random_task_cluster = np.zeros((total_tasks, k), dtype=np.float64)
        for task_id in range(total_tasks):
            clean_task_cluster[task_id] = _distribution(clean_labels[task_clean == task_id], minlength=k)
            random_task_cluster[task_id] = _distribution(random_labels[task_random == task_id], minlength=k)

        task_cosines = np.asarray(
            [
                _cosine_similarity(clean_task_cluster[task_id], random_task_cluster[task_id])
                for task_id in range(total_tasks)
            ],
            dtype=np.float64,
        )
        task_top1_match = np.argmax(clean_task_cluster, axis=1) == np.argmax(random_task_cluster, axis=1)

        cluster_records = []
        for cluster_id in range(k):
            clean_cluster_mask = clean_labels == cluster_id
            random_cluster_mask = random_labels == cluster_id
            clean_task_counts = _counts(task_clean[clean_cluster_mask], minlength=total_tasks)
            random_task_counts = _counts(task_random[random_cluster_mask], minlength=total_tasks)
            clean_examples = [
                {
                    "episode_index": records_clean[i]["episode_index"],
                    "task_id": records_clean[i]["task_id"],
                    "prompt": records_clean[i]["prompt"],
                }
                for i in np.flatnonzero(clean_cluster_mask)[:3]
            ]
            random_examples = [
                {
                    "episode_index": records_random[i]["episode_index"],
                    "task_id": records_random[i]["task_id"],
                    "prompt": records_random[i]["prompt"],
                }
                for i in np.flatnonzero(random_cluster_mask)[:3]
            ]
            cluster_records.append(
                {
                    "cluster": int(cluster_id),
                    "clean_count": int(clean_cluster_mask.sum()),
                    "random_count": int(random_cluster_mask.sum()),
                    "clean_task_counts": clean_task_counts,
                    "random_task_counts": random_task_counts,
                    "clean_examples": clean_examples,
                    "random_examples": random_examples,
                }
            )

        results.append(
            {
                "k": int(k),
                "clean_silhouette": float(silhouette_score(x_clean, clean_labels)),
                "clean_nmi_task": float(normalized_mutual_info_score(task_clean, clean_labels)),
                "clean_ari_task": float(adjusted_rand_score(task_clean, clean_labels)),
                "random_nmi_task": float(normalized_mutual_info_score(task_random, random_labels)),
                "random_ari_task": float(adjusted_rand_score(task_random, random_labels)),
                "clean_cluster_counts": _counts(clean_labels, minlength=k),
                "random_cluster_counts": _counts(random_labels, minlength=k),
                "task_clean_random_cluster_cosine": task_cosines.astype(float).tolist(),
                "mean_task_cluster_cosine": float(task_cosines.mean()),
                "min_task_cluster_cosine": float(task_cosines.min()),
                "task_top1_cluster_match": task_top1_match.astype(bool).tolist(),
                "top1_match_rate": float(task_top1_match.mean()),
                "clusters": cluster_records,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo"),
    )
    parser.add_argument("--episodes-per-task", type=int, default=100)
    parser.add_argument("--sample-per-variant", type=int, default=8)
    parser.add_argument("--resample-steps", type=int, default=20)
    parser.add_argument("--k-values", type=str, default="2,3,4,5,6,8")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--clean-fit-random-predict",
        action="store_true",
        help="Fit scaler/KMeans on clean episodes only, then assign random episodes to nearest centers.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    info = _read_json(repo_root / "meta" / "info.json")
    episodes_meta = _read_jsonl(repo_root / "meta" / "episodes.jsonl")
    data_path_pattern = info["data_path"]
    chunks_size = int(info["chunks_size"])
    total_episodes = int(info["total_episodes"])
    total_tasks = total_episodes // args.episodes_per_task
    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]
    episode_indices = _sample_episodes(total_episodes, args.episodes_per_task, args.sample_per_variant)

    features = []
    task_ids = []
    variant_ids = []
    records = []
    for episode_index in episode_indices:
        path = _episode_path(repo_root, data_path_pattern, episode_index, chunks_size)
        table = pq.read_table(path, columns=["action", "observation.state"])
        actions = _fixed_list_column_to_numpy(table, "action")
        states = _fixed_list_column_to_numpy(table, "observation.state")
        feature = _episode_feature(actions, states, resample_steps=args.resample_steps)
        task_id, variant = _infer_task_and_variant(episode_index, args.episodes_per_task)
        features.append(feature)
        task_ids.append(task_id)
        variant_ids.append(0 if variant == "clean" else 1)
        records.append(
            {
                "episode_index": int(episode_index),
                "task_id": int(task_id),
                "variant": variant,
                "num_frames": int(actions.shape[0]),
                "prompt": episodes_meta[episode_index].get("tasks", [""])[0],
            }
        )

    x = np.stack(features, axis=0)
    x_scaled = StandardScaler().fit_transform(x)
    task_ids_arr = np.asarray(task_ids, dtype=np.int64)
    variant_ids_arr = np.asarray(variant_ids, dtype=np.int64)

    output_dir = args.output_dir or repo_root / "meta" / "episode_action_cluster_probe"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for k in k_values:
        if k < 2 or k >= x_scaled.shape[0]:
            continue
        clusterer = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = clusterer.fit_predict(x_scaled)
        silhouette = silhouette_score(x_scaled, labels) if len(set(labels.tolist())) > 1 else math.nan
        result = {
            "k": int(k),
            "silhouette": float(silhouette),
            "nmi_task": float(normalized_mutual_info_score(task_ids_arr, labels)),
            "ari_task": float(adjusted_rand_score(task_ids_arr, labels)),
            "nmi_variant": float(normalized_mutual_info_score(variant_ids_arr, labels)),
            "ari_variant": float(adjusted_rand_score(variant_ids_arr, labels)),
            "cluster_counts": _counts(labels, minlength=k),
            "task_by_cluster": [],
        }
        for cluster_id in range(k):
            mask = labels == cluster_id
            task_counts = _counts(task_ids_arr[mask], minlength=total_tasks)
            variant_counts = _counts(variant_ids_arr[mask], minlength=2)
            examples = [
                {
                    "episode_index": records[i]["episode_index"],
                    "task_id": records[i]["task_id"],
                    "variant": records[i]["variant"],
                    "prompt": records[i]["prompt"],
                }
                for i in np.flatnonzero(mask)[:3]
            ]
            result["task_by_cluster"].append(
                {
                    "cluster": int(cluster_id),
                    "count": int(mask.sum()),
                    "task_counts": task_counts,
                    "variant_counts": {"clean": int(variant_counts[0]), "random": int(variant_counts[1])},
                    "examples": examples,
                }
            )
        results.append(result)

    report = {
        "repo_root": str(repo_root),
        "total_episodes": total_episodes,
        "episodes_per_task": int(args.episodes_per_task),
        "sample_per_variant": int(args.sample_per_variant),
        "sampled_episodes": episode_indices,
        "num_sampled_episodes": len(episode_indices),
        "resample_steps": int(args.resample_steps),
        "feature_dim": int(x.shape[1]),
        "results": results,
    }
    if args.clean_fit_random_predict:
        report["clean_fit_random_predict"] = _clean_fit_random_predict(
            x,
            task_ids_arr,
            variant_ids_arr,
            records,
            total_tasks=total_tasks,
            k_values=k_values,
            seed=args.seed,
        )
    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Wrote report to: {output_dir / 'report.json'}")
    print("k | silhouette | nmi_task | ari_task | nmi_variant | cluster_counts")
    for result in results:
        print(
            f"{result['k']:>2} | {result['silhouette']:.4f} | "
            f"{result['nmi_task']:.4f} | {result['ari_task']:.4f} | "
            f"{result['nmi_variant']:.4f} | {result['cluster_counts']}"
        )
    if args.clean_fit_random_predict:
        print("\nclean-fit -> random-predict")
        print("k | clean_sil | clean_nmi | random_nmi | mean_task_cos | top1_match | random_counts")
        for result in report["clean_fit_random_predict"]:
            print(
                f"{result['k']:>2} | {result['clean_silhouette']:.4f} | "
                f"{result['clean_nmi_task']:.4f} | {result['random_nmi_task']:.4f} | "
                f"{result['mean_task_cluster_cosine']:.4f} | {result['top1_match_rate']:.4f} | "
                f"{result['random_cluster_counts']}"
            )


if __name__ == "__main__":
    main()
