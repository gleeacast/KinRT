"""Probe mixed-repo episode clusters on randomized-repo episodes.

The mixed repo is treated as 8 task families with 100 episodes each.  We fit an
episode-level action trajectory clustering model on all mixed episodes, then
assign sampled randomized episodes to the nearest cluster center and report
whether the assignment is plausible from both action similarity and language.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


JOINT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)

MIXED_FAMILY_NAMES = {
    0: "block_handover",
    1: "mug_hang",
    2: "move_can_to_pot",
    3: "open_laptop",
    4: "shoe_into_box",
    5: "mouse_to_mat",
    6: "rotate_payment_sign",
    7: "click_switch",
}

RANDOMIZED_BLOCK_HINTS = {
    0: "hammer_block",
    1: "stack_blocks",
    2: "click_alarm_clock",
    3: "click_bell",
    4: "block_handover",
    5: "mug_hang",
    6: "lift_pot",
    7: "move_can_to_pot",
    8: "open_laptop",
    9: "open_microwave",
    10: "pick_bottles",
}


@dataclass(frozen=True)
class EpisodeRecord:
    repo: str
    episode_index: int
    prompt: str
    length: int
    feature: np.ndarray
    family_id: int | None = None
    family_name: str | None = None
    block_id: int | None = None
    block_hint: str | None = None


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


def resample_sequence(values: np.ndarray, steps: int) -> np.ndarray:
    if values.shape[0] == 0:
        return np.zeros((steps, values.shape[1]), dtype=np.float32)
    if values.shape[0] == 1:
        return np.repeat(values, steps, axis=0).astype(np.float32)
    source = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    target = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    columns = [np.interp(target, source, values[:, dim]) for dim in range(values.shape[1])]
    return np.stack(columns, axis=1).astype(np.float32)


def episode_feature(actions: np.ndarray, states: np.ndarray, *, resample_steps: int) -> np.ndarray:
    action = actions.astype(np.float32)
    state = states.astype(np.float32)
    delta_to_state = action - state
    velocity = np.diff(action, axis=0, prepend=action[:1])

    action_j = action[:, JOINT_DIMS]
    delta_j = delta_to_state[:, JOINT_DIMS]
    velocity_j = velocity[:, JOINT_DIMS]

    resampled = np.concatenate(
        [
            resample_sequence(action_j, resample_steps),
            resample_sequence(delta_j, resample_steps),
            resample_sequence(velocity_j, resample_steps),
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


def load_episode_feature(
    repo_root: Path,
    info: dict,
    episodes_meta: list[dict],
    episode_index: int,
    *,
    resample_steps: int,
) -> tuple[np.ndarray, str, int]:
    path = episode_path(repo_root, info["data_path"], episode_index, int(info["chunks_size"]))
    table = pq.read_table(path, columns=["action", "observation.state"])
    actions = fixed_list_column_to_numpy(table, "action")
    states = fixed_list_column_to_numpy(table, "observation.state")
    prompt = episodes_meta[episode_index].get("tasks", [""])[0]
    return episode_feature(actions, states, resample_steps=resample_steps), prompt, int(actions.shape[0])


def load_mixed_records(repo_root: Path, *, episodes_per_family: int, resample_steps: int) -> list[EpisodeRecord]:
    info = read_json(repo_root / "meta" / "info.json")
    episodes_meta = read_jsonl(repo_root / "meta" / "episodes.jsonl")
    total_episodes = int(info["total_episodes"])
    records = []
    for episode_index in range(total_episodes):
        feature, prompt, length = load_episode_feature(
            repo_root,
            info,
            episodes_meta,
            episode_index,
            resample_steps=resample_steps,
        )
        family_id = episode_index // episodes_per_family
        records.append(
            EpisodeRecord(
                repo="mixed",
                episode_index=episode_index,
                prompt=prompt,
                length=length,
                feature=feature,
                family_id=family_id,
                family_name=MIXED_FAMILY_NAMES.get(family_id, f"family_{family_id}"),
            )
        )
    return records


def uniform_indices(start: int, end: int, count: int) -> list[int]:
    if end <= start:
        return []
    if end - start <= count:
        return list(range(start, end))
    return np.linspace(start, end - 1, count, dtype=np.int64).astype(int).tolist()


def randomized_sample_indices(total_episodes: int, *, block_size: int, samples_per_block: int) -> list[int]:
    sampled = []
    for start in range(0, total_episodes, block_size):
        end = min(start + block_size, total_episodes)
        sampled.extend(uniform_indices(start, end, samples_per_block))
    return sampled


def load_randomized_records(
    repo_root: Path,
    *,
    block_size: int,
    samples_per_block: int,
    resample_steps: int,
) -> list[EpisodeRecord]:
    info = read_json(repo_root / "meta" / "info.json")
    episodes_meta = read_jsonl(repo_root / "meta" / "episodes.jsonl")
    total_episodes = int(info["total_episodes"])
    records = []
    for episode_index in randomized_sample_indices(
        total_episodes,
        block_size=block_size,
        samples_per_block=samples_per_block,
    ):
        feature, prompt, length = load_episode_feature(
            repo_root,
            info,
            episodes_meta,
            episode_index,
            resample_steps=resample_steps,
        )
        block_id = episode_index // block_size
        records.append(
            EpisodeRecord(
                repo="randomized",
                episode_index=episode_index,
                prompt=prompt,
                length=length,
                feature=feature,
                block_id=block_id,
                block_hint=RANDOMIZED_BLOCK_HINTS.get(block_id, f"block_{block_id}"),
            )
        )
    return records


def short_prompt(prompt: str, limit: int = 130) -> str:
    return prompt if len(prompt) <= limit else prompt[: limit - 3] + "..."


def summarize_training_clusters(
    labels: np.ndarray,
    records: list[EpisodeRecord],
    *,
    num_clusters: int,
    num_families: int,
) -> tuple[list[dict], dict[int, int]]:
    family_ids = np.asarray([record.family_id for record in records], dtype=np.int64)
    clusters = []
    cluster_to_family = {}
    for cluster_id in range(num_clusters):
        mask = labels == cluster_id
        family_counts = np.bincount(family_ids[mask], minlength=num_families)
        dominant_family = int(family_counts.argmax())
        cluster_to_family[cluster_id] = dominant_family
        examples = []
        for i in np.flatnonzero(mask)[:5]:
            examples.append(
                {
                    "episode_index": int(records[i].episode_index),
                    "family_id": int(records[i].family_id),
                    "family_name": records[i].family_name,
                    "prompt": records[i].prompt,
                }
            )
        clusters.append(
            {
                "cluster": int(cluster_id),
                "count": int(mask.sum()),
                "dominant_family_id": dominant_family,
                "dominant_family_name": MIXED_FAMILY_NAMES.get(dominant_family, f"family_{dominant_family}"),
                "purity": float(family_counts.max() / max(mask.sum(), 1)),
                "family_counts": family_counts.astype(int).tolist(),
                "examples": examples,
            }
        )
    return clusters, cluster_to_family


def nearest_examples(
    x_train: np.ndarray,
    train_records: list[EpisodeRecord],
    query: np.ndarray,
    *,
    count: int,
) -> list[dict]:
    distances = np.linalg.norm(x_train - query[None, :], axis=1)
    nearest = np.argsort(distances)[:count]
    return [
        {
            "episode_index": int(train_records[i].episode_index),
            "family_id": int(train_records[i].family_id),
            "family_name": train_records[i].family_name,
            "distance": float(distances[i]),
            "prompt": train_records[i].prompt,
        }
        for i in nearest
    ]


def assignment_confidence(center_distances: np.ndarray) -> float:
    ordered = np.sort(center_distances)
    if ordered.shape[0] < 2:
        return 0.0
    return float((ordered[1] - ordered[0]) / max(ordered[1], 1e-9))


def write_markdown_summary(report: dict, path: Path) -> None:
    lines = []
    lines.append("# Mixed to Randomized Cluster Probe")
    lines.append("")
    lines.append(
        f"Train: {report['train']['num_episodes']} mixed episodes, k={report['args']['num_clusters']}, "
        f"silhouette={report['train']['silhouette']:.4f}, "
        f"NMI(task_family)={report['train']['nmi_family']:.4f}, "
        f"ARI(task_family)={report['train']['ari_family']:.4f}."
    )
    lines.append("")
    lines.append("## Training Clusters")
    lines.append("")
    lines.append("| cluster | count | dominant mixed family | purity | family_counts |")
    lines.append("|---:|---:|---|---:|---|")
    for cluster in report["train"]["clusters"]:
        lines.append(
            f"| {cluster['cluster']} | {cluster['count']} | "
            f"{cluster['dominant_family_name']} | {cluster['purity']:.2f} | "
            f"{cluster['family_counts']} |"
        )
    lines.append("")
    lines.append("## Randomized Blocks")
    lines.append("")
    lines.append("| randomized block | samples | nearest mixed families | center-assigned families | mean confidence | OOD ratio | note |")
    lines.append("|---|---:|---|---|---:|---:|---|")
    for block in report["randomized"]["blocks"]:
        nearest = ", ".join(f"{name}:{count}" for name, count in block["nearest_family_counts"].items())
        assigned = ", ".join(f"{name}:{count}" for name, count in block["assigned_family_counts"].items())
        lines.append(
            f"| {block['block_hint']} ({block['episode_range']}) | {block['num_samples']} | "
            f"{nearest} | {assigned} | {block['mean_confidence']:.3f} | "
            f"{block['ood_fraction_vs_train_p95']:.2f} | {block['interpretation']} |"
        )
    lines.append("")
    lines.append("## Representative Assignments")
    lines.append("")
    for block in report["randomized"]["blocks"]:
        lines.append(f"### {block['block_hint']} ({block['episode_range']})")
        for example in block["examples"]:
            nearest = example["nearest_mixed_examples"][0]
            lines.append(
                f"- ep {example['episode_index']} -> cluster {example['cluster']} "
                f"({example['assigned_family_name']}), conf={example['confidence']:.3f}; "
                f"text: {short_prompt(example['prompt'])}"
            )
            lines.append(
                f"  nearest mixed: ep {nearest['episode_index']} "
                f"({nearest['family_name']}), dist={nearest['distance']:.2f}; "
                f"{short_prompt(nearest['prompt'])}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mixed-repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo"),
    )
    parser.add_argument(
        "--randomized-repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_randomized_repo"),
    )
    parser.add_argument("--episodes-per-family", type=int, default=100)
    parser.add_argument("--num-clusters", type=int, default=8)
    parser.add_argument("--resample-steps", type=int, default=20)
    parser.add_argument("--randomized-block-size", type=int, default=500)
    parser.add_argument("--samples-per-randomized-block", type=int, default=20)
    parser.add_argument("--nearest-examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    mixed_repo_root = args.mixed_repo_root.resolve()
    randomized_repo_root = args.randomized_repo_root.resolve()
    output_dir = args.output_dir or (mixed_repo_root / "meta" / "mixed_to_randomized_cluster_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    mixed_records = load_mixed_records(
        mixed_repo_root,
        episodes_per_family=args.episodes_per_family,
        resample_steps=args.resample_steps,
    )
    randomized_records = load_randomized_records(
        randomized_repo_root,
        block_size=args.randomized_block_size,
        samples_per_block=args.samples_per_randomized_block,
        resample_steps=args.resample_steps,
    )

    x_mixed = np.stack([record.feature for record in mixed_records], axis=0)
    family_ids = np.asarray([record.family_id for record in mixed_records], dtype=np.int64)
    num_families = int(math.ceil(len(mixed_records) / args.episodes_per_family))
    scaler = StandardScaler().fit(x_mixed)
    x_mixed_scaled = scaler.transform(x_mixed)

    kmeans = KMeans(n_clusters=args.num_clusters, n_init=100, random_state=args.seed)
    train_labels = kmeans.fit_predict(x_mixed_scaled)
    train_clusters, cluster_to_family = summarize_training_clusters(
        train_labels,
        mixed_records,
        num_clusters=args.num_clusters,
        num_families=num_families,
    )

    x_randomized = np.stack([record.feature for record in randomized_records], axis=0)
    x_randomized_scaled = scaler.transform(x_randomized)
    center_distances = pairwise_distances(x_randomized_scaled, kmeans.cluster_centers_)
    randomized_labels = center_distances.argmin(axis=1)
    train_center_distances = pairwise_distances(x_mixed_scaled, kmeans.cluster_centers_)
    train_own_center_distances = train_center_distances[np.arange(train_labels.shape[0]), train_labels]
    train_center_distance_p95 = float(np.percentile(train_own_center_distances, 95))

    randomized_assignments = []
    for i, record in enumerate(randomized_records):
        cluster_id = int(randomized_labels[i])
        assigned_family_id = cluster_to_family[cluster_id]
        nearest = nearest_examples(
            x_mixed_scaled,
            mixed_records,
            x_randomized_scaled[i],
            count=args.nearest_examples,
        )
        randomized_assignments.append(
            {
                "episode_index": int(record.episode_index),
                "block_id": int(record.block_id),
                "block_hint": record.block_hint,
                "length": int(record.length),
                "prompt": record.prompt,
                "cluster": cluster_id,
                "assigned_family_id": int(assigned_family_id),
                "assigned_family_name": MIXED_FAMILY_NAMES.get(assigned_family_id, f"family_{assigned_family_id}"),
                "distance_to_center": float(center_distances[i, cluster_id]),
                "confidence": assignment_confidence(center_distances[i]),
                "is_ood_vs_train_p95": bool(center_distances[i, cluster_id] > train_center_distance_p95),
                "nearest_family_name": nearest[0]["family_name"],
                "nearest_mixed_examples": nearest,
            }
        )

    block_reports = []
    for block_id in sorted({record.block_id for record in randomized_records if record.block_id is not None}):
        block_items = [item for item in randomized_assignments if item["block_id"] == block_id]
        assigned_counter = Counter(item["assigned_family_name"] for item in block_items)
        nearest_counter = Counter(item["nearest_family_name"] for item in block_items)
        cluster_counter = Counter(str(item["cluster"]) for item in block_items)
        start = block_id * args.randomized_block_size
        end = min(start + args.randomized_block_size - 1, int(read_json(randomized_repo_root / "meta" / "info.json")["total_episodes"]) - 1)
        dominant_family, dominant_count = assigned_counter.most_common(1)[0]
        block_hint = RANDOMIZED_BLOCK_HINTS.get(block_id, f"block_{block_id}")
        seen_match = block_hint == dominant_family
        interpretation = "seen-task text/action match" if seen_match else "unseen or action-similar assignment"
        block_reports.append(
            {
                "block_id": int(block_id),
                "block_hint": block_hint,
                "episode_range": f"{start}-{end}",
                "num_samples": len(block_items),
                "dominant_assigned_family": dominant_family,
                "dominant_assigned_count": int(dominant_count),
                "nearest_family_counts": dict(nearest_counter.most_common()),
                "assigned_family_counts": dict(assigned_counter.most_common()),
                "cluster_counts": dict(cluster_counter.most_common()),
                "mean_confidence": float(np.mean([item["confidence"] for item in block_items])),
                "mean_distance_to_center": float(np.mean([item["distance_to_center"] for item in block_items])),
                "ood_fraction_vs_train_p95": float(np.mean([item["is_ood_vs_train_p95"] for item in block_items])),
                "interpretation": interpretation,
                "examples": block_items[:3],
            }
        )

    report = {
        "args": {
            "mixed_repo_root": str(mixed_repo_root),
            "randomized_repo_root": str(randomized_repo_root),
            "episodes_per_family": int(args.episodes_per_family),
            "num_clusters": int(args.num_clusters),
            "resample_steps": int(args.resample_steps),
            "randomized_block_size": int(args.randomized_block_size),
            "samples_per_randomized_block": int(args.samples_per_randomized_block),
            "seed": int(args.seed),
        },
        "train": {
            "num_episodes": len(mixed_records),
            "num_families": num_families,
            "cluster_counts": np.bincount(train_labels, minlength=args.num_clusters).astype(int).tolist(),
            "silhouette": float(silhouette_score(x_mixed_scaled, train_labels)),
            "nmi_family": float(normalized_mutual_info_score(family_ids, train_labels)),
            "ari_family": float(adjusted_rand_score(family_ids, train_labels)),
            "mean_center_distance": float(np.mean(train_own_center_distances)),
            "p95_center_distance": train_center_distance_p95,
            "clusters": train_clusters,
        },
        "randomized": {
            "num_sampled_episodes": len(randomized_records),
            "blocks": block_reports,
            "assignments": randomized_assignments,
        },
    }

    report_path = output_dir / "report.json"
    summary_path = output_dir / "summary.md"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_summary(report, summary_path)

    print(f"Wrote report to: {report_path}")
    print(f"Wrote summary to: {summary_path}")
    print(
        "train: "
        f"silhouette={report['train']['silhouette']:.4f}, "
        f"NMI={report['train']['nmi_family']:.4f}, "
        f"ARI={report['train']['ari_family']:.4f}, "
        f"counts={report['train']['cluster_counts']}"
    )
    print("randomized blocks:")
    for block in block_reports:
        print(
            f"  {block['block_hint']:>20s} {block['episode_range']}: "
            f"nearest={block['nearest_family_counts']} "
            f"center={block['assigned_family_counts']} "
            f"conf={block['mean_confidence']:.3f} "
            f"ood={block['ood_fraction_vs_train_p95']:.2f}"
        )


if __name__ == "__main__":
    main()
