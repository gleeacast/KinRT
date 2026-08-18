"""Explore expert partition candidates using only demo_mixed_repo.

This script compares task-family labels, instruction slot routing, and
episode-action clustering on the mixed LeRobot repo.  It intentionally does not
read randomized data, so the resulting proposal is based only on the training
data distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


JOINT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
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


def slot_route(prompt: str) -> str:
    s = prompt.lower()
    if any(w in s for w in ["shoe", "sneaker", "footwear"]):
        return "box_pack/shoe"
    if any(w in s for w in ["mug", "cup"]):
        return "mug_rotate_hang"
    if any(w in s for w in ["payment", "qr", "qrcode"]):
        return "rotate_orient/payment_sign"
    if "laptop" in s:
        return "open_hinged/laptop"
    if "switch" in s:
        return "press_click/switch"
    if "mouse" in s or re.search(r"\bmat\b", s):
        return "single_pick_place/mouse_mat"
    if "sauce" in s or ("can" in s and ("pot" in s or "kitchenpot" in s)):
        return "pot_can/move_can_to_pot"
    if "block" in s:
        if any(w in s for w in ["hand over", "handover", "pass", "transfer", "right arm", "right hand", "blue pad"]):
            return "handover/block"
        return "block_generic"
    return "unknown"


def purity(labels: np.ndarray, targets: np.ndarray) -> float:
    total = 0
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        counts = np.bincount(targets[mask])
        total += int(counts.max()) if counts.size else 0
    return total / max(labels.shape[0], 1)


def distribution(labels: np.ndarray, minlength: int) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64), minlength=minlength).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def summarize_partition(name: str, labels: np.ndarray, family_ids: np.ndarray, variant_ids: np.ndarray, prompts: list[str]) -> dict:
    unique_labels = sorted(set(labels.tolist()))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    dense = np.asarray([label_to_id[label] for label in labels], dtype=np.int64)
    num_labels = len(unique_labels)
    clusters = []
    for label in unique_labels:
        mask = labels == label
        dense_id = label_to_id[label]
        family_counts = np.bincount(family_ids[mask], minlength=8)
        variant_counts = np.bincount(variant_ids[mask], minlength=2)
        examples = []
        for i in np.flatnonzero(mask)[:5]:
            examples.append({"episode_index": int(i), "family": FAMILY_NAMES[int(family_ids[i])], "prompt": prompts[i]})
        clusters.append(
            {
                "label": str(label),
                "dense_label": int(dense_id),
                "count": int(mask.sum()),
                "dominant_family": FAMILY_NAMES[int(family_counts.argmax())],
                "purity": float(family_counts.max() / max(mask.sum(), 1)),
                "family_counts": family_counts.astype(int).tolist(),
                "variant_counts": {"clean": int(variant_counts[0]), "random": int(variant_counts[1])},
                "examples": examples,
            }
        )

    clean_mask = variant_ids == 0
    random_mask = variant_ids == 1
    per_family_cosine = []
    for family_id in range(8):
        clean_dist = distribution(dense[clean_mask & (family_ids == family_id)], num_labels)
        random_dist = distribution(dense[random_mask & (family_ids == family_id)], num_labels)
        per_family_cosine.append(cosine(clean_dist, random_dist))

    return {
        "name": name,
        "num_labels": num_labels,
        "counts": np.bincount(dense, minlength=num_labels).astype(int).tolist(),
        "nmi_family": float(normalized_mutual_info_score(family_ids, dense)),
        "ari_family": float(adjusted_rand_score(family_ids, dense)),
        "purity_family": float(purity(dense, family_ids)),
        "nmi_variant": float(normalized_mutual_info_score(variant_ids, dense)),
        "mean_clean_random_family_cosine": float(np.mean(per_family_cosine)),
        "min_clean_random_family_cosine": float(np.min(per_family_cosine)),
        "clusters": clusters,
    }


def write_summary(report: dict, path: Path) -> None:
    lines = []
    lines.append("# Mixed-Only Expert Partition Exploration")
    lines.append("")
    lines.append("This report uses only `demo_mixed_repo` episodes.")
    lines.append("")
    lines.append("## Partition Metrics")
    lines.append("")
    lines.append("| partition | labels | NMI family | ARI family | purity | variant NMI | clean/random cosine |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for item in report["partitions"]:
        lines.append(
            f"| {item['name']} | {item['num_labels']} | {item['nmi_family']:.3f} | "
            f"{item['ari_family']:.3f} | {item['purity_family']:.3f} | "
            f"{item['nmi_variant']:.3f} | {item['mean_clean_random_family_cosine']:.3f} |"
        )

    for item in report["partitions"]:
        lines.append("")
        lines.append(f"## {item['name']}")
        lines.append("")
        lines.append("| label | count | dominant family | purity | clean/random | family counts |")
        lines.append("|---|---:|---|---:|---|---|")
        for cluster in item["clusters"]:
            lines.append(
                f"| {cluster['label']} | {cluster['count']} | {cluster['dominant_family']} | "
                f"{cluster['purity']:.2f} | {cluster['variant_counts']['clean']}/"
                f"{cluster['variant_counts']['random']} | {cluster['family_counts']} |"
            )

    lines.append("")
    lines.append("## Suggested Expert Sets")
    lines.append("")
    for suggestion in report["suggestions"]:
        lines.append(f"- `{suggestion['name']}`: {suggestion['description']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo"),
    )
    parser.add_argument("--episodes-per-family", type=int, default=100)
    parser.add_argument("--resample-steps", type=int, default=20)
    parser.add_argument("--k-values", type=str, default="8,10,12")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or (repo_root / "meta" / "mixed_expert_partition_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    info = read_json(repo_root / "meta" / "info.json")
    episodes_meta = read_jsonl(repo_root / "meta" / "episodes.jsonl")
    total_episodes = int(info["total_episodes"])
    prompts = [record.get("tasks", [""])[0] for record in episodes_meta]
    family_ids = np.asarray([i // args.episodes_per_family for i in range(total_episodes)], dtype=np.int64)
    variant_ids = np.asarray(
        [0 if (i % args.episodes_per_family) < args.episodes_per_family // 2 else 1 for i in range(total_episodes)],
        dtype=np.int64,
    )

    features = []
    lengths = []
    for episode_index in range(total_episodes):
        table = pq.read_table(
            episode_path(repo_root, info["data_path"], episode_index, int(info["chunks_size"])),
            columns=["action", "observation.state"],
        )
        actions = fixed_list_column_to_numpy(table, "action")
        states = fixed_list_column_to_numpy(table, "observation.state")
        features.append(episode_feature(actions, states, resample_steps=args.resample_steps))
        lengths.append(int(actions.shape[0]))
    x = np.stack(features, axis=0)
    x_scaled = StandardScaler().fit_transform(x)

    partitions = []
    partitions.append(
        summarize_partition(
            "task_family_oracle",
            np.asarray([FAMILY_NAMES[int(family_id)] for family_id in family_ids], dtype=object),
            family_ids,
            variant_ids,
            prompts,
        )
    )
    partitions.append(
        summarize_partition(
            "instruction_slot_rules",
            np.asarray([slot_route(prompt) for prompt in prompts], dtype=object),
            family_ids,
            variant_ids,
            prompts,
        )
    )

    for k in [int(value) for value in args.k_values.split(",") if value.strip()]:
        if k < 2 or k >= total_episodes:
            continue
        labels = KMeans(n_clusters=k, n_init=100, random_state=args.seed).fit_predict(x_scaled)
        summary = summarize_partition(f"action_kmeans_k{k}", labels.astype(object), family_ids, variant_ids, prompts)
        summary["silhouette"] = float(silhouette_score(x_scaled, labels))
        partitions.append(summary)

    slot_labels = np.asarray([slot_route(prompt) for prompt in prompts], dtype=object)
    slot_counts = Counter(slot_labels.tolist())
    suggestions = [
        {
            "name": "8_task_family",
            "description": "Use the known mixed episode ranges as experts. This is the cleanest training label and matches the data construction.",
        },
        {
            "name": "slot_rules",
            "description": (
                "Use instruction object/action slots as route labels. On mixed it recovers "
                f"{len(slot_counts)} labels: {dict(slot_counts.most_common())}."
            ),
        },
        {
            "name": "action_subexperts_k10_or_k12",
            "description": "Use action clusters as subexperts inside task families, not as global semantic experts.",
        },
        {
            "name": "hybrid_top2",
            "description": "Primary route by task family or slot; use action nearest cluster only for auxiliary top-2 routing or OOD checks.",
        },
    ]

    report = {
        "repo_root": str(repo_root),
        "total_episodes": total_episodes,
        "episodes_per_family": int(args.episodes_per_family),
        "family_names": FAMILY_NAMES,
        "length_stats": {
            "min": int(min(lengths)),
            "mean": float(np.mean(lengths)),
            "max": int(max(lengths)),
        },
        "partitions": partitions,
        "suggestions": suggestions,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(report, output_dir / "summary.md")

    print(f"Wrote report to: {output_dir / 'report.json'}")
    print(f"Wrote summary to: {output_dir / 'summary.md'}")
    print("partition | labels | nmi | ari | purity | variant_nmi | cr_cos")
    for item in partitions:
        print(
            f"{item['name']} | {item['num_labels']} | {item['nmi_family']:.3f} | "
            f"{item['ari_family']:.3f} | {item['purity_family']:.3f} | "
            f"{item['nmi_variant']:.3f} | {item['mean_clean_random_family_cosine']:.3f}"
        )


if __name__ == "__main__":
    main()
