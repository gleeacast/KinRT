"""Explore mixed-only expert clusters from action, language, vision, and fusion.

The goal is not to recover the 8 task ids.  Instead, this script searches for
cross-task cluster structures that may be useful as generalizing MoE experts.
It reads only demo_mixed_repo.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler


JOINT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12], dtype=np.int64)
LEFT_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
RIGHT_DIMS = np.asarray([7, 8, 9, 10, 11, 12, 13], dtype=np.int64)
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


def action_feature(actions: np.ndarray, states: np.ndarray, *, resample_steps: int) -> np.ndarray:
    action = actions.astype(np.float32)
    state = states.astype(np.float32)
    delta_to_state = action - state
    velocity = np.diff(action, axis=0, prepend=action[:1])

    action_j = action[:, JOINT_DIMS]
    delta_j = delta_to_state[:, JOINT_DIMS]
    velocity_j = velocity[:, JOINT_DIMS]

    traj = np.concatenate(
        [
            resample_sequence(action_j, resample_steps),
            resample_sequence(delta_j, resample_steps),
            resample_sequence(velocity_j, resample_steps),
        ],
        axis=1,
    ).reshape(-1)

    left_motion = float(np.mean(np.abs(velocity[:, LEFT_DIMS])))
    right_motion = float(np.mean(np.abs(velocity[:, RIGHT_DIMS])))
    handedness = np.asarray(
        [
            left_motion,
            right_motion,
            left_motion - right_motion,
            max(left_motion, right_motion),
            float(action.shape[0]),
        ],
        dtype=np.float32,
    )
    stats = np.concatenate(
        [
            action.mean(axis=0),
            action.std(axis=0),
            delta_to_state.mean(axis=0),
            delta_to_state.std(axis=0),
            np.mean(np.abs(velocity), axis=0),
            handedness,
        ],
        axis=0,
    )
    return np.concatenate([traj, stats], axis=0).astype(np.float32)


def image_to_array(value: dict, *, size: int) -> np.ndarray:
    image = Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    image = image.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 255.0


def image_stats(image: np.ndarray) -> np.ndarray:
    # Compact visual descriptor: color, coarse spatial layout, and gradient energy.
    mean = image.mean(axis=(0, 1))
    std = image.std(axis=(0, 1))
    grid = image.reshape(4, image.shape[0] // 4, 4, image.shape[1] // 4, 3).mean(axis=(1, 3)).reshape(-1)
    gray = image.mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    edge = np.asarray([np.mean(np.abs(gx)), np.mean(np.abs(gy)), np.std(gray)], dtype=np.float32)
    return np.concatenate([mean, std, grid, edge], axis=0).astype(np.float32)


def vision_feature(table, *, image_columns: list[str], image_size: int) -> np.ndarray:
    features = []
    for column in image_columns:
        values = table[column].combine_chunks().to_pylist()
        images = [image_to_array(value, size=image_size) for value in values]
        per_frame = [image_stats(image) for image in images]
        features.extend(per_frame)
        if len(images) >= 2:
            diffs = [np.mean(np.abs(images[i + 1] - images[i]), axis=(0, 1)) for i in range(len(images) - 1)]
            features.append(np.concatenate(diffs, axis=0).astype(np.float32))
    return np.concatenate(features, axis=0).astype(np.float32)


def reduce_dense(x: np.ndarray, *, max_dim: int, seed: int) -> np.ndarray:
    x_scaled = StandardScaler().fit_transform(x)
    if x_scaled.shape[1] <= max_dim:
        return x_scaled.astype(np.float32)
    pca = PCA(n_components=max_dim, random_state=seed)
    return pca.fit_transform(x_scaled).astype(np.float32)


def language_features(prompts: list[str], *, max_dim: int, seed: int) -> dict[str, np.ndarray]:
    word = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2), min_df=1, max_features=256)
    char = TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_features=384)
    x_word = word.fit_transform(prompts).astype(np.float32)
    x_char = char.fit_transform(prompts).astype(np.float32)
    x_word = Normalizer().fit_transform(x_word).toarray().astype(np.float32)
    x_char = Normalizer().fit_transform(x_char).toarray().astype(np.float32)
    return {
        "language_word_tfidf": reduce_dense(x_word, max_dim=max_dim, seed=seed),
        "language_char_tfidf": reduce_dense(x_char, max_dim=max_dim, seed=seed),
        "language_word_char": reduce_dense(np.concatenate([x_word, x_char], axis=1), max_dim=max_dim, seed=seed),
    }


def top_family_counts(labels: np.ndarray, family_ids: np.ndarray) -> list[dict]:
    rows = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        counts = Counter(FAMILY_NAMES[int(f)] for f in family_ids[mask])
        rows.append(
            {
                "label": int(label),
                "count": int(mask.sum()),
                "family_counts": dict(counts.most_common()),
                "dominant_family": counts.most_common(1)[0][0],
                "dominant_fraction": float(counts.most_common(1)[0][1] / max(mask.sum(), 1)),
                "num_families": len(counts),
            }
        )
    return rows


def family_mixing_score(rows: list[dict]) -> float:
    # Higher means clusters are not just single-task buckets.
    weights = np.asarray([row["count"] for row in rows], dtype=np.float64)
    mixed = np.asarray([1.0 - row["dominant_fraction"] for row in rows], dtype=np.float64)
    return float(np.average(mixed, weights=weights))


def useful_cluster_score(rows: list[dict]) -> float:
    # Favor cross-task clusters but penalize clusters that mix almost everything.
    scores = []
    weights = []
    for row in rows:
        n = row["num_families"]
        dom = row["dominant_fraction"]
        if n == 1:
            score = 0.0
        elif n <= 3:
            score = 1.0 - abs(dom - 0.55)
        else:
            score = 0.35 * (1.0 - dom)
        scores.append(max(score, 0.0))
        weights.append(row["count"])
    return float(np.average(np.asarray(scores), weights=np.asarray(weights)))


def clean_random_stability(labels: np.ndarray, family_ids: np.ndarray, variant_ids: np.ndarray, k: int) -> float:
    sims = []
    for family_id in range(8):
        clean = labels[(family_ids == family_id) & (variant_ids == 0)]
        random = labels[(family_ids == family_id) & (variant_ids == 1)]
        c = np.bincount(clean, minlength=k).astype(np.float64)
        r = np.bincount(random, minlength=k).astype(np.float64)
        c /= max(c.sum(), 1.0)
        r /= max(r.sum(), 1.0)
        denom = np.linalg.norm(c) * np.linalg.norm(r)
        sims.append(float(np.dot(c, r) / denom) if denom > 1e-12 else 0.0)
    return float(np.mean(sims))


def cluster_one(x: np.ndarray, *, method: str, k: int, seed: int) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=50, random_state=seed).fit_predict(x)
    if method == "gmm":
        return GaussianMixture(n_components=k, covariance_type="diag", n_init=10, random_state=seed).fit_predict(x)
    if method == "agglomerative":
        return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
    if method == "spectral":
        return SpectralClustering(
            n_clusters=k,
            assign_labels="kmeans",
            affinity="nearest_neighbors",
            n_neighbors=min(20, x.shape[0] - 1),
            random_state=seed,
        ).fit_predict(x)
    raise ValueError(f"Unsupported method: {method}")


def summarize_clustering(
    name: str,
    x: np.ndarray,
    labels: np.ndarray,
    family_ids: np.ndarray,
    variant_ids: np.ndarray,
) -> dict:
    k = len(set(labels.tolist()))
    rows = top_family_counts(labels, family_ids)
    return {
        "name": name,
        "k": k,
        "cluster_counts": np.bincount(labels, minlength=k).astype(int).tolist(),
        "family_mixing_score": family_mixing_score(rows),
        "useful_cross_task_score": useful_cluster_score(rows),
        "clean_random_stability": clean_random_stability(labels, family_ids, variant_ids, k),
        "clusters": rows,
    }


def write_summary(report: dict, path: Path) -> None:
    lines = ["# Mixed-Only Multimodal Cluster Exploration", ""]
    lines.append("The score favors cross-task clusters; it is not measuring recovery of the 8 task ids.")
    lines.append("")
    lines.append("## Top Clusterings")
    lines.append("")
    lines.append("| rank | clustering | k | useful cross-task | family mixing | clean/random stability | counts |")
    lines.append("|---:|---|---:|---:|---:|---:|---|")
    for i, item in enumerate(report["top_results"], start=1):
        lines.append(
            f"| {i} | {item['name']} | {item['k']} | {item['useful_cross_task_score']:.3f} | "
            f"{item['family_mixing_score']:.3f} | {item['clean_random_stability']:.3f} | "
            f"{item['cluster_counts']} |"
        )
    for item in report["top_results"][:12]:
        lines.append("")
        lines.append(f"## {item['name']}")
        lines.append("")
        lines.append("| cluster | count | families | dominant frac |")
        lines.append("|---:|---:|---|---:|")
        for cluster in item["clusters"]:
            families = ", ".join(f"{k}:{v}" for k, v in cluster["family_counts"].items())
            lines.append(
                f"| {cluster['label']} | {cluster['count']} | {families} | {cluster['dominant_fraction']:.2f} |"
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
    parser.add_argument("--resample-steps", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-columns", type=str, default="observation.images.cam_high")
    parser.add_argument("--max-dim", type=int, default=64)
    parser.add_argument("--k-values", type=str, default="4,6,8,10,12")
    parser.add_argument("--methods", type=str, default="kmeans,gmm,agglomerative,spectral")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or (repo_root / "meta" / "mixed_multimodal_cluster_probe")
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

    image_columns = [value.strip() for value in args.image_columns.split(",") if value.strip()]
    action_parts = []
    vision_parts = []
    for episode_index in range(total_episodes):
        table = pq.read_table(
            episode_path(repo_root, info["data_path"], episode_index, int(info["chunks_size"])),
            columns=["action", "observation.state"],
        )
        actions = fixed_list_column_to_numpy(table, "action")
        states = fixed_list_column_to_numpy(table, "observation.state")
        action_parts.append(action_feature(actions, states, resample_steps=args.resample_steps))
        target_frames = sorted(
            {
                0,
                max(int(actions.shape[0] // 2), 0),
                max(int(actions.shape[0] - 1), 0),
            }
        )
        image_table = pq.read_table(
            episode_path(repo_root, info["data_path"], episode_index, int(info["chunks_size"])),
            columns=["frame_index", *image_columns],
            filters=[("frame_index", "in", target_frames)],
        )
        vision_parts.append(
            vision_feature(
                image_table,
                image_columns=image_columns,
                image_size=args.image_size,
            )
        )

    feature_sets = {}
    feature_sets["action"] = reduce_dense(np.stack(action_parts, axis=0), max_dim=args.max_dim, seed=args.seed)
    feature_sets["vision_stats"] = reduce_dense(np.stack(vision_parts, axis=0), max_dim=args.max_dim, seed=args.seed)
    feature_sets.update(language_features(prompts, max_dim=args.max_dim, seed=args.seed))
    feature_sets["action_vision"] = reduce_dense(
        np.concatenate([feature_sets["action"], feature_sets["vision_stats"]], axis=1),
        max_dim=args.max_dim,
        seed=args.seed,
    )
    feature_sets["action_language"] = reduce_dense(
        np.concatenate([feature_sets["action"], feature_sets["language_word_char"]], axis=1),
        max_dim=args.max_dim,
        seed=args.seed,
    )
    feature_sets["vision_language"] = reduce_dense(
        np.concatenate([feature_sets["vision_stats"], feature_sets["language_word_char"]], axis=1),
        max_dim=args.max_dim,
        seed=args.seed,
    )
    feature_sets["action_vision_language"] = reduce_dense(
        np.concatenate(
            [feature_sets["action"], feature_sets["vision_stats"], feature_sets["language_word_char"]],
            axis=1,
        ),
        max_dim=args.max_dim,
        seed=args.seed,
    )

    k_values = [int(value) for value in args.k_values.split(",") if value.strip()]
    methods = [value.strip() for value in args.methods.split(",") if value.strip()]
    results = []
    for feature_name, x in feature_sets.items():
        for method in methods:
            for k in k_values:
                if method == "spectral" and k > 10:
                    continue
                labels = cluster_one(x, method=method, k=k, seed=args.seed).astype(np.int64)
                results.append(
                    summarize_clustering(
                        f"{feature_name}/{method}/k{k}",
                        x,
                        labels,
                        family_ids,
                        variant_ids,
                    )
                )

    top_results = sorted(
        results,
        key=lambda item: (
            item["useful_cross_task_score"],
            item["clean_random_stability"],
            item["family_mixing_score"],
        ),
        reverse=True,
    )
    report = {
        "repo_root": str(repo_root),
        "total_episodes": total_episodes,
        "family_names": FAMILY_NAMES,
        "args": {
            "resample_steps": int(args.resample_steps),
            "image_size": int(args.image_size),
            "image_columns": image_columns,
            "max_dim": int(args.max_dim),
            "k_values": k_values,
            "methods": methods,
        },
        "top_results": top_results[:30],
        "all_results": results,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(report, output_dir / "summary.md")

    print(f"Wrote report to: {output_dir / 'report.json'}")
    print(f"Wrote summary to: {output_dir / 'summary.md'}")
    print("top results:")
    for item in top_results[:12]:
        print(
            f"{item['name']} useful={item['useful_cross_task_score']:.3f} "
            f"mix={item['family_mixing_score']:.3f} stable={item['clean_random_stability']:.3f} "
            f"counts={item['cluster_counts']}"
        )


if __name__ == "__main__":
    main()
