"""Explore reusable phase-level skills from mixed LeRobot episodes.

Unlike episode-level clustering, this script samples action chunks from within
episodes and clusters local motion phases.  The goal is to discover reusable
skills such as reach/pick, transfer, rotate, open, press, and settle that can
span multiple tasks.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


JOINT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
LEFT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
RIGHT_DIMS = np.asarray([7, 8, 9, 10, 11, 12, 13], dtype=np.int64)
LEFT_GRIPPER = 6
RIGHT_GRIPPER = 13
FAMILY_NAMES = {
    0: "block_handover",
    1: "mug_hang",
    2: "move_can_to_pot",
    3: "open_laptop",
    4: "shoe_into_box",
    5: "mouse_to_mat",
    6: "rotate_payment_sign",
    7: "click_switch",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_path(repo_root: Path, data_path_pattern: str, episode_index: int, chunks_size: int) -> Path:
    return repo_root / data_path_pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def fixed_list_column_to_numpy(table, name: str, dtype=np.float32) -> np.ndarray:
    return np.asarray(table[name].combine_chunks().to_pylist(), dtype=dtype)


def hand_activity(velocity: np.ndarray) -> tuple[float, float, float]:
    left = float(np.mean(np.abs(velocity[:, LEFT_DIMS])))
    right = float(np.mean(np.abs(velocity[:, RIGHT_DIMS])))
    both = float(min(left, right) / max(max(left, right), 1e-6))
    return left, right, both


def phase_name(chunk: np.ndarray, state0: np.ndarray) -> str:
    velocity = np.diff(chunk, axis=0, prepend=chunk[:1])
    left, right, both = hand_activity(velocity)
    left_grip_delta = float(chunk[-1, LEFT_GRIPPER] - chunk[0, LEFT_GRIPPER])
    right_grip_delta = float(chunk[-1, RIGHT_GRIPPER] - chunk[0, RIGHT_GRIPPER])
    left_wrist_rot = float(np.mean(np.abs(np.diff(chunk[:, [4, 5]], axis=0))))
    right_wrist_rot = float(np.mean(np.abs(np.diff(chunk[:, [11, 12]], axis=0))))
    total_motion = max(left, right)

    if total_motion < 0.004:
        return "settle/hold"
    if both > 0.65 and total_motion > 0.012:
        return "bimanual_motion"
    if min(left_grip_delta, right_grip_delta) < -0.18:
        return "gripper_close/pick"
    if max(left_grip_delta, right_grip_delta) > 0.18:
        return "gripper_open/release"
    if max(left_wrist_rot, right_wrist_rot) > 0.015:
        return "wrist_rotate/orient"
    if total_motion > 0.025:
        return "large_transfer"
    if left > right * 1.4:
        return "left_arm_adjust"
    if right > left * 1.4:
        return "right_arm_adjust"
    return "small_adjust"


def build_chunk_features(
    actions: np.ndarray,
    states: np.ndarray,
    *,
    horizon: int,
    stride: int,
    max_chunks_per_episode: int,
    episode_index: int,
    family_id: int,
    prompt: str,
) -> tuple[list[np.ndarray], list[dict]]:
    num_frames, action_dim = actions.shape
    if num_frames < 2:
        return [], []
    starts = list(range(0, max(num_frames - 1, 1), stride))
    if len(starts) > max_chunks_per_episode:
        starts = np.linspace(0, starts[-1], max_chunks_per_episode, dtype=np.int64).astype(int).tolist()

    features = []
    records = []
    tail = np.repeat(actions[-1:], max(horizon - 1, 0), axis=0)
    padded = np.concatenate([actions, tail], axis=0)
    for start in starts:
        indices = start + np.arange(horizon)
        chunk = padded[indices].astype(np.float32)
        state0 = states[min(start, states.shape[0] - 1)].astype(np.float32)
        rel = chunk.copy()
        rel[:, JOINT_DIMS] -= state0[JOINT_DIMS]
        velocity = np.diff(chunk, axis=0, prepend=chunk[:1])
        left, right, both = hand_activity(velocity)
        grip = np.asarray(
            [
                chunk[0, LEFT_GRIPPER],
                chunk[-1, LEFT_GRIPPER],
                chunk[-1, LEFT_GRIPPER] - chunk[0, LEFT_GRIPPER],
                chunk[0, RIGHT_GRIPPER],
                chunk[-1, RIGHT_GRIPPER],
                chunk[-1, RIGHT_GRIPPER] - chunk[0, RIGHT_GRIPPER],
            ],
            dtype=np.float32,
        )
        stats = np.concatenate(
            [
                chunk.mean(axis=0),
                chunk.std(axis=0),
                np.mean(np.abs(velocity), axis=0),
                np.asarray([left, right, both, start / max(num_frames - 1, 1)], dtype=np.float32),
                grip,
            ],
            axis=0,
        )
        feature = np.concatenate(
            [
                rel[:, JOINT_DIMS].reshape(-1),
                velocity[:, JOINT_DIMS].reshape(-1),
                stats,
            ],
            axis=0,
        ).astype(np.float32)
        features.append(feature)
        records.append(
            {
                "episode_index": int(episode_index),
                "family_id": int(family_id),
                "family": FAMILY_NAMES[int(family_id)],
                "start_frame": int(start),
                "phase_pos": float(start / max(num_frames - 1, 1)),
                "num_frames": int(num_frames),
                "heuristic_phase": phase_name(chunk, state0),
                "prompt": prompt,
            }
        )
    return features, records


def summarize_labels(labels: np.ndarray, records: list[dict], *, k: int) -> list[dict]:
    rows = []
    positions = np.asarray([record["phase_pos"] for record in records], dtype=np.float32)
    phase_names = [record["heuristic_phase"] for record in records]
    family_ids = np.asarray([record["family_id"] for record in records], dtype=np.int64)
    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        family_counts = Counter(FAMILY_NAMES[int(family_id)] for family_id in family_ids[mask])
        phase_counts = Counter(phase_names[i] for i in np.flatnonzero(mask))
        examples = []
        for i in np.flatnonzero(mask)[:8]:
            record = records[i]
            examples.append(
                {
                    "episode_index": record["episode_index"],
                    "family": record["family"],
                    "start_frame": record["start_frame"],
                    "phase_pos": record["phase_pos"],
                    "heuristic_phase": record["heuristic_phase"],
                    "prompt": record["prompt"],
                }
            )
        rows.append(
            {
                "cluster": int(cluster_id),
                "count": int(mask.sum()),
                "family_counts": dict(family_counts.most_common()),
                "phase_counts": dict(phase_counts.most_common()),
                "num_families": int(len(family_counts)),
                "dominant_family_fraction": float(family_counts.most_common(1)[0][1] / mask.sum()),
                "mean_phase_pos": float(np.mean(positions[mask])),
                "std_phase_pos": float(np.std(positions[mask])),
                "examples": examples,
            }
        )
    return rows


def cross_task_score(rows: list[dict]) -> float:
    scores = []
    weights = []
    for row in rows:
        n = row["num_families"]
        dom = row["dominant_family_fraction"]
        if n == 1:
            score = 0.0
        elif n <= 4:
            score = 1.0 - abs(dom - 0.45)
        else:
            score = 0.65 * (1.0 - dom)
        scores.append(max(score, 0.0))
        weights.append(row["count"])
    return float(np.average(np.asarray(scores), weights=np.asarray(weights)))


def write_summary(report: dict, path: Path) -> None:
    lines = ["# Mixed-Only Phase Cluster Exploration", ""]
    lines.append("Clusters are local action chunks, not whole episodes.")
    lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| k | silhouette | cross-task score | counts |")
    lines.append("|---:|---:|---:|---|")
    for run in report["runs"]:
        lines.append(
            f"| {run['k']} | {run['silhouette']:.3f} | {run['cross_task_score']:.3f} | "
            f"{run['cluster_counts']} |"
        )
    for run in report["runs"]:
        lines.append("")
        lines.append(f"## k={run['k']}")
        lines.append("")
        lines.append("| cluster | count | families | heuristic phases | mean pos |")
        lines.append("|---:|---:|---|---|---:|")
        for cluster in run["clusters"]:
            families = ", ".join(f"{key}:{value}" for key, value in cluster["family_counts"].items())
            phases = ", ".join(f"{key}:{value}" for key, value in cluster["phase_counts"].items())
            lines.append(
                f"| {cluster['cluster']} | {cluster['count']} | {families} | {phases} | "
                f"{cluster['mean_phase_pos']:.2f} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo"),
    )
    parser.add_argument("--episodes-per-family", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-chunks-per-episode", type=int, default=16)
    parser.add_argument("--pca-components", type=int, default=48)
    parser.add_argument("--k-values", type=str, default="8,12,16,20")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or (repo_root / "meta" / "mixed_phase_cluster_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    info = read_json(repo_root / "meta" / "info.json")
    episodes_meta = read_jsonl(repo_root / "meta" / "episodes.jsonl")
    total_episodes = int(info["total_episodes"])

    features = []
    records = []
    for episode_index in range(total_episodes):
        table = pq.read_table(
            episode_path(repo_root, info["data_path"], episode_index, int(info["chunks_size"])),
            columns=["action", "observation.state"],
        )
        actions = fixed_list_column_to_numpy(table, "action")
        states = fixed_list_column_to_numpy(table, "observation.state")
        family_id = episode_index // args.episodes_per_family
        prompt = episodes_meta[episode_index].get("tasks", [""])[0]
        episode_features, episode_records = build_chunk_features(
            actions,
            states,
            horizon=args.horizon,
            stride=args.stride,
            max_chunks_per_episode=args.max_chunks_per_episode,
            episode_index=episode_index,
            family_id=family_id,
            prompt=prompt,
        )
        features.extend(episode_features)
        records.extend(episode_records)

    x = np.stack(features, axis=0)
    x_scaled = StandardScaler().fit_transform(x)
    pca_components = min(args.pca_components, x_scaled.shape[1], x_scaled.shape[0] - 1)
    x_reduced = PCA(n_components=pca_components, random_state=args.seed).fit_transform(x_scaled)

    runs = []
    for k in [int(value) for value in args.k_values.split(",") if value.strip()]:
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            batch_size=2048,
            n_init=20,
            random_state=args.seed,
            reassignment_ratio=0.01,
        )
        labels = kmeans.fit_predict(x_reduced).astype(np.int64)
        clusters = summarize_labels(labels, records, k=k)
        sample_size = min(4000, x_reduced.shape[0])
        rng = np.random.default_rng(args.seed)
        sample_idx = np.sort(rng.choice(np.arange(x_reduced.shape[0]), size=sample_size, replace=False))
        runs.append(
            {
                "k": int(k),
                "cluster_counts": np.bincount(labels, minlength=k).astype(int).tolist(),
                "silhouette": float(silhouette_score(x_reduced[sample_idx], labels[sample_idx])),
                "cross_task_score": cross_task_score(clusters),
                "clusters": clusters,
            }
        )

    report = {
        "repo_root": str(repo_root),
        "total_episodes": total_episodes,
        "num_chunks": len(records),
        "args": {
            "horizon": int(args.horizon),
            "stride": int(args.stride),
            "max_chunks_per_episode": int(args.max_chunks_per_episode),
            "pca_components": int(pca_components),
        },
        "runs": runs,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(report, output_dir / "summary.md")

    print(f"Wrote report to: {output_dir / 'report.json'}")
    print(f"Wrote summary to: {output_dir / 'summary.md'}")
    print(f"num_chunks={len(records)}")
    for run in runs:
        print(
            f"k={run['k']} silhouette={run['silhouette']:.3f} "
            f"cross_task={run['cross_task_score']:.3f} counts={run['cluster_counts']}"
        )


if __name__ == "__main__":
    main()
