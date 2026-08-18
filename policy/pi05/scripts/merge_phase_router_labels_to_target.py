"""Merge oversegmented phase-router labels into a smaller expert set.

This script is intended for cases where direct KMeans at a small ``k`` collapses
most frames into one default cluster.  It starts from a larger phase vocabulary
such as k18, profiles each old cluster, then greedily merges similar clusters
under a maximum-load constraint.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler


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
PHASE_NAMES = [
    "gripper_close/pick",
    "gripper_open/release",
    "large_transfer",
    "left_arm_adjust",
    "right_arm_adjust",
    "wrist_rotate/orient",
    "bimanual_motion",
    "settle/hold",
    "small_adjust",
]
STAT_NAMES = [
    "phase_pos",
    "left_motion",
    "right_motion",
    "both_motion",
    "total_motion",
    "left_grip_delta",
    "right_grip_delta",
    "left_wrist_rot",
    "right_wrist_rot",
]


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


def phase_indices_and_stats(actions: np.ndarray, *, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    num_frames = actions.shape[0]
    tail = np.repeat(actions[-1:], max(horizon - 1, 0), axis=0)
    padded = np.concatenate([actions, tail], axis=0)
    indices = np.arange(num_frames)[:, None] + np.arange(horizon)[None, :]
    chunks = padded[indices].astype(np.float32)
    velocity = np.diff(chunks, axis=1, prepend=chunks[:, :1, :])

    left_motion = np.mean(np.abs(velocity[:, :, LEFT_DIMS]), axis=(1, 2))
    right_motion = np.mean(np.abs(velocity[:, :, RIGHT_DIMS]), axis=(1, 2))
    both_motion = np.minimum(left_motion, right_motion) / np.clip(
        np.maximum(left_motion, right_motion), 1e-6, None
    )
    total_motion = np.maximum(left_motion, right_motion)
    left_grip_delta = chunks[:, -1, LEFT_GRIPPER] - chunks[:, 0, LEFT_GRIPPER]
    right_grip_delta = chunks[:, -1, RIGHT_GRIPPER] - chunks[:, 0, RIGHT_GRIPPER]
    left_wrist_rot = np.mean(np.abs(np.diff(chunks[:, :, [4, 5]], axis=1)), axis=(1, 2))
    right_wrist_rot = np.mean(np.abs(np.diff(chunks[:, :, [11, 12]], axis=1)), axis=(1, 2))
    phase_pos = np.arange(num_frames, dtype=np.float32) / max(num_frames - 1, 1)

    phase = np.full(num_frames, PHASE_NAMES.index("small_adjust"), dtype=np.int64)
    phase[total_motion < 0.004] = PHASE_NAMES.index("settle/hold")
    mask = (both_motion > 0.65) & (total_motion > 0.012) & (total_motion >= 0.004)
    phase[mask] = PHASE_NAMES.index("bimanual_motion")
    phase[np.minimum(left_grip_delta, right_grip_delta) < -0.18] = PHASE_NAMES.index("gripper_close/pick")
    phase[np.maximum(left_grip_delta, right_grip_delta) > 0.18] = PHASE_NAMES.index("gripper_open/release")
    phase[np.maximum(left_wrist_rot, right_wrist_rot) > 0.015] = PHASE_NAMES.index("wrist_rotate/orient")
    phase[total_motion > 0.025] = PHASE_NAMES.index("large_transfer")
    phase[left_motion > right_motion * 1.4] = PHASE_NAMES.index("left_arm_adjust")
    phase[right_motion > left_motion * 1.4] = PHASE_NAMES.index("right_arm_adjust")

    stats = np.stack(
        [
            phase_pos,
            left_motion,
            right_motion,
            both_motion,
            total_motion,
            left_grip_delta,
            right_grip_delta,
            left_wrist_rot,
            right_wrist_rot,
        ],
        axis=1,
    ).astype(np.float64)
    return phase, stats


def profile_old_clusters(
    repo_root: Path,
    input_dir: Path,
    *,
    horizon: int,
    examples_per_cluster: int,
) -> dict:
    info = read_json(repo_root / "meta" / "info.json")
    labels_by_index = np.load(input_dir / "router_labels.npy")
    input_summary = read_json(input_dir / "summary.json")
    num_old = int(input_summary["num_clusters"])
    total_frames = int(info["total_frames"])
    if labels_by_index.shape[0] != total_frames:
        raise ValueError(f"Label length {labels_by_index.shape[0]} != total_frames {total_frames}")

    counts = np.zeros(num_old, dtype=np.int64)
    family_counts = np.zeros((num_old, len(FAMILY_NAMES)), dtype=np.int64)
    phase_counts = np.zeros((num_old, len(PHASE_NAMES)), dtype=np.int64)
    stat_sums = np.zeros((num_old, len(STAT_NAMES)), dtype=np.float64)
    examples: list[list[dict]] = [[] for _ in range(num_old)]
    episodes_path = repo_root / "meta" / "episodes.jsonl"
    prompts = {}
    with episodes_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            prompts[int(record["episode_index"])] = record["tasks"][0]

    data_path_pattern = info["data_path"]
    chunks_size = int(info["chunks_size"])
    for episode_index in range(int(info["total_episodes"])):
        table = pq.read_table(
            episode_path(repo_root, data_path_pattern, episode_index, chunks_size),
            columns=["action", "index", "frame_index", "episode_index"],
        )
        actions = fixed_list_column_to_numpy(table, "action")
        indices = int_column_to_numpy(table, "index")
        frames = int_column_to_numpy(table, "frame_index")
        labels = labels_by_index[indices]
        valid = labels >= 0
        labels = labels[valid]
        if labels.size == 0:
            continue
        phases, stats = phase_indices_and_stats(actions, horizon=horizon)
        phases = phases[valid]
        stats = stats[valid]
        family_id = min(int(episode_index // 100), len(FAMILY_NAMES) - 1)

        counts += np.bincount(labels, minlength=num_old)
        for old_id in np.unique(labels):
            mask = labels == old_id
            old_i = int(old_id)
            family_counts[old_i, family_id] += int(mask.sum())
            phase_counts[old_i] += np.bincount(phases[mask], minlength=len(PHASE_NAMES))
            stat_sums[old_i] += stats[mask].sum(axis=0)
            if len(examples[old_i]) < examples_per_cluster:
                frame_values = frames[valid][mask]
                for frame in frame_values[: examples_per_cluster - len(examples[old_i])]:
                    examples[old_i].append(
                        {
                            "episode_index": int(episode_index),
                            "family": FAMILY_NAMES[family_id],
                            "frame_index": int(frame),
                            "phase_pos": float(frame / max(actions.shape[0] - 1, 1)),
                            "prompt": prompts.get(episode_index, ""),
                        }
                    )

    stat_means = stat_sums / np.clip(counts[:, None], 1, None)
    return {
        "counts": counts,
        "family_counts": family_counts,
        "phase_counts": phase_counts,
        "stat_means": stat_means,
        "examples": examples,
        "input_summary": input_summary,
    }


def build_cluster_vectors(input_dir: Path, profile: dict) -> np.ndarray:
    counts = profile["counts"].astype(np.float64)
    family_frac = profile["family_counts"] / np.clip(counts[:, None], 1, None)
    phase_frac = profile["phase_counts"] / np.clip(counts[:, None], 1, None)
    stat_means = profile["stat_means"]
    parts = [
        StandardScaler().fit_transform(stat_means) * 1.5,
        phase_frac * 2.0,
        family_frac * 0.5,
    ]
    model_path = input_dir / "router_label_model.joblib"
    if model_path.exists():
        model = joblib.load(model_path)
        centers = np.asarray(model["kmeans"].cluster_centers_, dtype=np.float64)
        parts.insert(0, StandardScaler().fit_transform(centers) * 1.0)
    return np.concatenate(parts, axis=1)


def greedy_merge(
    vectors: np.ndarray,
    counts: np.ndarray,
    *,
    target_clusters: int,
    max_fraction: float,
    protect_gripper_phases: bool = True,
    phase_counts: np.ndarray | None = None,
) -> list[list[int]]:
    GRIPPER_CLOSE_IDX = PHASE_NAMES.index("gripper_close/pick")
    GRIPPER_OPEN_IDX = PHASE_NAMES.index("gripper_open/release")

    def _gripper_dominant_phase(old_ids: list[int]) -> int | None:
        """Return GRIPPER_CLOSE_IDX or GRIPPER_OPEN_IDX if that phase strongly
        dominates a group, else None."""
        if phase_counts is None:
            return None
        pc = phase_counts[old_ids].sum(axis=0)
        total = pc.sum()
        if total == 0:
            return None
        close_frac = pc[GRIPPER_CLOSE_IDX] / total
        open_frac = pc[GRIPPER_OPEN_IDX] / total
        if close_frac > 0.40:
            return GRIPPER_CLOSE_IDX
        if open_frac > 0.40:
            return GRIPPER_OPEN_IDX
        return None

    total = int(counts.sum())
    groups = [
        {
            "ids": [old_id],
            "count": int(counts[old_id]),
            "vector": vectors[old_id].astype(np.float64),
        }
        for old_id in range(vectors.shape[0])
    ]
    cap = int(np.ceil(total * max_fraction))
    relaxed = False
    gripper_relaxed = False  # second relaxation: allow cross-gripper merge
    while len(groups) > target_clusters:
        best = None
        best_dist = float("inf")
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                merged_count = groups[i]["count"] + groups[j]["count"]
                if merged_count > cap and not relaxed:
                    continue
                # Semantic protection: never merge gripper_close with gripper_open
                # unless we have no other option.
                if protect_gripper_phases and not gripper_relaxed:
                    pi = _gripper_dominant_phase(groups[i]["ids"])
                    pj = _gripper_dominant_phase(groups[j]["ids"])
                    gripper_phases = {GRIPPER_CLOSE_IDX, GRIPPER_OPEN_IDX}
                    if (
                        pi in gripper_phases
                        and pj in gripper_phases
                        and pi != pj
                    ):
                        continue
                dist = float(np.linalg.norm(groups[i]["vector"] - groups[j]["vector"]))
                if dist < best_dist:
                    best = (i, j)
                    best_dist = dist
        if best is None:
            if not relaxed:
                relaxed = True
            elif not gripper_relaxed:
                gripper_relaxed = True
            else:
                # Should never reach here, but break to avoid infinite loop.
                break
            continue
        i, j = best
        a, b = groups[i], groups[j]
        merged_count = a["count"] + b["count"]
        merged_vector = (a["vector"] * a["count"] + b["vector"] * b["count"]) / max(merged_count, 1)
        merged = {
            "ids": sorted(a["ids"] + b["ids"]),
            "count": int(merged_count),
            "vector": merged_vector,
        }
        for index in sorted((i, j), reverse=True):
            del groups[index]
        groups.append(merged)
    groups = sorted(groups, key=lambda item: (-item["count"], item["ids"][0]))
    return [group["ids"] for group in groups]


def parse_manual_groups(value: str) -> list[list[int]]:
    groups = []
    seen = set()
    for group_s in value.split(";"):
        group = sorted(int(item) for item in group_s.split(",") if item.strip())
        if not group:
            continue
        overlap = seen.intersection(group)
        if overlap:
            raise ValueError(f"Manual groups contain duplicate old clusters: {sorted(overlap)}")
        seen.update(group)
        groups.append(group)
    return groups


def summarize_groups(groups: list[list[int]], profile: dict) -> tuple[list[dict], dict[int, int]]:
    old_to_new = {}
    rows = []
    counts = profile["counts"]
    for new_id, old_ids in enumerate(groups):
        for old_id in old_ids:
            old_to_new[int(old_id)] = int(new_id)
        group_counts = counts[old_ids]
        family_counts = profile["family_counts"][old_ids].sum(axis=0)
        phase_counts = profile["phase_counts"][old_ids].sum(axis=0)
        stat_means = np.average(profile["stat_means"][old_ids], axis=0, weights=np.clip(group_counts, 1, None))
        family_counter = Counter(
            {
                FAMILY_NAMES[i]: int(value)
                for i, value in enumerate(family_counts.tolist())
                if int(value) > 0
            }
        )
        phase_counter = Counter(
            {
                PHASE_NAMES[i]: int(value)
                for i, value in enumerate(phase_counts.tolist())
                if int(value) > 0
            }
        )
        examples = []
        for old_id in old_ids:
            for example in profile["examples"][old_id][:2]:
                examples.append({**example, "old_cluster": int(old_id)})
        rows.append(
            {
                "cluster": int(new_id),
                "old_clusters": [int(x) for x in old_ids],
                "count": int(group_counts.sum()),
                "family_counts": dict(family_counter.most_common()),
                "phase_counts": dict(phase_counter.most_common()),
                "num_families": int(np.sum(family_counts > 0)),
                "dominant_family_fraction": float(family_counts.max() / max(family_counts.sum(), 1)),
                "mean_phase_pos": float(stat_means[STAT_NAMES.index("phase_pos")]),
                "stat_means": {name: float(value) for name, value in zip(STAT_NAMES, stat_means, strict=True)},
                "examples": examples[:10],
            }
        )
    return rows, old_to_new


def write_merged_labels(input_dir: Path, output_dir: Path, old_to_new: dict[int, int], num_new: int) -> np.ndarray:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = np.load(input_dir / "router_labels.npy")
    vectorized = np.full((max(old_to_new) + 1,), -1, dtype=np.int32)
    for old_id, new_id in old_to_new.items():
        vectorized[int(old_id)] = int(new_id)
    merged = labels.copy()
    valid = merged >= 0
    merged[valid] = vectorized[merged[valid]]
    np.save(output_dir / "router_labels.npy", merged.astype(np.int32, copy=False))

    sample_labels_path = input_dir / "sample_labels.npy"
    if sample_labels_path.exists():
        sample_labels = np.load(sample_labels_path)
        sample_valid = sample_labels >= 0
        sample_merged = sample_labels.copy()
        sample_merged[sample_valid] = vectorized[sample_labels[sample_valid]]
        np.save(output_dir / "sample_labels.npy", sample_merged.astype(np.int32, copy=False))
    sample_indices_path = input_dir / "sample_indices.npy"
    if sample_indices_path.exists():
        np.save(output_dir / "sample_indices.npy", np.load(sample_indices_path))
    return np.bincount(merged[merged >= 0], minlength=num_new).astype(np.int64)


def plot_old_to_new(groups: list[list[int]], old_counts: np.ndarray, output_path: Path) -> None:
    matrix = np.zeros((len(groups), len(old_counts)), dtype=np.float32)
    for new_id, old_ids in enumerate(groups):
        for old_id in old_ids:
            matrix[new_id, old_id] = old_counts[old_id]
    row_sum = np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None)
    frac = matrix / row_sum
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=160)
    im = ax.imshow(frac, aspect="auto", cmap="magma", vmin=0.0, vmax=max(float(frac.max()), 1e-6))
    ax.set_xlabel("old k18 cluster")
    ax.set_ylabel("new merged expert")
    ax.set_xticks(np.arange(len(old_counts)))
    ax.set_xticklabels([str(i) for i in range(len(old_counts))])
    ax.set_yticks(np.arange(len(groups)))
    ax.set_yticklabels([str(i) for i in range(len(groups))])
    ax.set_title("k18 -> merged expert composition")
    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            if frac[i, j] > 0:
                ax.text(j, i, f"{frac[i, j]:.2f}", ha="center", va="center", color="white", fontsize=7)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("row fraction")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo"),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo/meta/router_labels_k18_phase_h24"
        ),
    )
    parser.add_argument("--target-clusters", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument(
        "--max-fraction",
        type=float,
        default=0.25,
        help=(
            "Maximum fraction of total frames allowed in any single merged expert. "
            "Lowered from the old default (0.43) to 0.25 to prevent one expert "
            "dominating.  The greedy merge relaxes this constraint only when no "
            "valid merge exists."
        ),
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=0.05,
        help=(
            "After greedy merge, if any resulting expert group has fewer than "
            "this fraction of total frames, force-merge it into its nearest "
            "neighbour regardless of max_fraction.  Prevents stranded tiny "
            "groups.  Set to 0.0 to disable."
        ),
    )
    parser.add_argument("--examples-per-cluster", type=int, default=6)
    parser.add_argument(
        "--manual-groups",
        type=str,
        default=None,
        help="Semicolon-separated old-cluster groups, e.g. '1;8,5;2,4'.",
    )
    parser.add_argument(
        "--protect-gripper-phases",
        action="store_true",
        default=True,
        help=(
            "Prevent merging a cluster dominated by gripper_close/pick with one "
            "dominated by gripper_open/release.  Keeps grasping and releasing "
            "in separate experts for better specialisation."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir or (
        input_dir.parent / f"router_labels_k18_to_k{args.target_clusters}_phase_h{args.horizon}_merged"
    )
    profile = profile_old_clusters(
        repo_root,
        input_dir,
        horizon=args.horizon,
        examples_per_cluster=args.examples_per_cluster,
    )
    vectors = build_cluster_vectors(input_dir, profile)
    if args.manual_groups:
        groups = parse_manual_groups(args.manual_groups)
        expected = set(range(int(profile["input_summary"]["num_clusters"])))
        actual = {old_id for group in groups for old_id in group}
        if actual != expected:
            raise ValueError(
                f"Manual groups must cover all old clusters exactly once. "
                f"missing={sorted(expected - actual)} extra={sorted(actual - expected)}"
            )
        if len(groups) != args.target_clusters:
            raise ValueError(f"Manual groups define {len(groups)} clusters, expected {args.target_clusters}.")
    else:
        groups = greedy_merge(
            vectors,
            profile["counts"],
            target_clusters=args.target_clusters,
            max_fraction=args.max_fraction,
            protect_gripper_phases=args.protect_gripper_phases,
            phase_counts=profile["phase_counts"],
        )

    # ── Post-process: absorb any group below min_fraction ──────────────────
    # After greedy merge we may end up with one or more stranded tiny groups
    # if their smallest-possible merge partner was already at the cap.  Force
    # merge them unconditionally (ignoring max_fraction) so no expert is
    # starved for training data.
    if args.min_fraction > 0.0:
        total_frames = int(profile["counts"].sum())
        min_count = int(total_frames * args.min_fraction)
        counts_arr = profile["counts"]

        def _group_count(g: list[int]) -> int:
            return int(counts_arr[g].sum())

        changed = True
        while changed:
            changed = False
            tiny_idx = next(
                (i for i, g in enumerate(groups) if _group_count(g) < min_count),
                None,
            )
            if tiny_idx is None:
                break
            tiny = groups[tiny_idx]
            # Build a temporary vector for the tiny group using its old-cluster mean.
            tiny_vec = np.average(
                vectors[tiny], axis=0, weights=np.clip(counts_arr[tiny], 1, None)
            )
            # Find the nearest other group (no cap restriction).
            best_j, best_d = None, float("inf")
            for j, g in enumerate(groups):
                if j == tiny_idx:
                    continue
                g_vec = np.average(
                    vectors[g], axis=0, weights=np.clip(counts_arr[g], 1, None)
                )
                d = float(np.linalg.norm(tiny_vec - g_vec))
                if d < best_d:
                    best_j, best_d = j, d
            if best_j is None:
                break
            merged = sorted(groups[tiny_idx] + groups[best_j])
            for idx in sorted([tiny_idx, best_j], reverse=True):
                del groups[idx]
            groups.append(merged)
            changed = True
        groups = sorted(groups, key=lambda g: (-_group_count(g), g[0]))

    clusters, old_to_new = summarize_groups(groups, profile)
    counts = write_merged_labels(input_dir, output_dir, old_to_new, args.target_clusters)

    report = {
        "repo_root": str(repo_root),
        "input_dir": str(input_dir),
        "runs": [
            {
                "k": int(args.target_clusters),
                "source_k": int(profile["input_summary"]["num_clusters"]),
                "cluster_counts": counts.astype(int).tolist(),
                "clusters": clusters,
            }
        ],
    }
    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "num_old_clusters": int(profile["input_summary"]["num_clusters"]),
        "num_clusters": int(args.target_clusters),
        "horizon": int(args.horizon),
        "max_fraction": float(args.max_fraction),
        "old_to_new": {str(k): int(v) for k, v in sorted(old_to_new.items())},
        "groups": [[int(x) for x in group] for group in groups],
        "manual_groups": args.manual_groups,
        "cluster_counts": counts.astype(int).tolist(),
        "cluster_fractions": (counts / max(counts.sum(), 1)).astype(float).tolist(),
        "old_cluster_counts": profile["counts"].astype(int).tolist(),
        "files": {
            "labels_by_global_index": "router_labels.npy",
            "sample_labels": "sample_labels.npy",
            "report": "report.json",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_old_to_new(groups, profile["counts"], figures_dir / "old_to_new_composition.png")
    print(f"Wrote merged labels to: {output_dir}")
    print(f"groups={groups}")
    print(f"counts={counts.tolist()}")


if __name__ == "__main__":
    main()
