"""Create figures for mixed phase cluster reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAMILY_ORDER = [
    "block_handover",
    "mug_hang",
    "move_can_to_pot",
    "open_laptop",
    "shoe_into_box",
    "mouse_to_mat",
    "rotate_payment_sign",
    "click_switch",
]

PHASE_ORDER = [
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


def load_report(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_run(report: dict, k: int) -> dict:
    for run in report["runs"]:
        if int(run["k"]) == k:
            return run
    raise ValueError(f"No run with k={k}")


def matrix_from_counts(clusters: list[dict], keys: list[str], count_key: str) -> np.ndarray:
    matrix = np.zeros((len(clusters), len(keys)), dtype=np.float32)
    for i, cluster in enumerate(clusters):
        counts = cluster[count_key]
        for j, key in enumerate(keys):
            matrix[i, j] = float(counts.get(key, 0))
    return matrix


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    denom = np.clip(matrix.sum(axis=1, keepdims=True), 1.0, None)
    return matrix / denom


def heatmap(
    matrix: np.ndarray,
    *,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cbar_label: str,
    output_path: Path,
) -> None:
    fig_w = max(10.0, len(xlabels) * 1.2)
    fig_h = max(6.0, len(ylabels) * 0.42)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=160)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(float(matrix.max()), 1e-6))
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    if matrix.shape[0] <= 24 and matrix.shape[1] <= 12:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if value >= 0.08:
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_cluster_overview(run: dict, output_path: Path) -> None:
    clusters = run["clusters"]
    labels = [str(cluster["cluster"]) for cluster in clusters]
    counts = np.asarray([cluster["count"] for cluster in clusters], dtype=np.float32)
    positions = np.asarray([cluster["mean_phase_pos"] for cluster in clusters], dtype=np.float32)
    dominance = np.asarray([cluster["dominant_family_fraction"] for cluster in clusters], dtype=np.float32)
    num_families = np.asarray([cluster["num_families"] for cluster in clusters], dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), dpi=160, sharex=True)
    x = np.arange(len(clusters))
    axes[0].bar(x, counts, color="#4c78a8")
    axes[0].set_ylabel("chunks")
    axes[0].set_title(f"Phase Cluster Overview, k={run['k']}")

    axes[1].bar(x, positions, color="#f58518")
    axes[1].set_ylabel("mean position")
    axes[1].set_ylim(0.0, 1.0)

    axes[2].bar(x - 0.2, dominance, width=0.4, color="#54a24b", label="dominant family fraction")
    axes[2].bar(x + 0.2, num_families / 8.0, width=0.4, color="#b279a2", label="families / 8")
    axes[2].set_ylabel("ratio")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(loc="upper right")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel("cluster")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_skill_map(run: dict, output_path: Path) -> None:
    family = row_normalize(matrix_from_counts(run["clusters"], FAMILY_ORDER, "family_counts"))
    phase = row_normalize(matrix_from_counts(run["clusters"], PHASE_ORDER, "phase_counts"))
    combined = np.concatenate([family, phase], axis=1)
    labels = [str(cluster["cluster"]) for cluster in run["clusters"]]
    xlabels = [f"fam:{value}" for value in FAMILY_ORDER] + [f"phase:{value}" for value in PHASE_ORDER]
    heatmap(
        combined,
        xlabels=xlabels,
        ylabels=labels,
        title=f"Cluster Skill Map, k={run['k']}",
        cbar_label="row fraction",
        output_path=output_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo/meta/mixed_phase_cluster_probe/report.json"),
    )
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    report = load_report(args.report)
    run = find_run(report, args.k)
    output_dir = args.output_dir or (args.report.parent / f"figures_k{args.k}")
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = run["clusters"]
    ylabels = [str(cluster["cluster"]) for cluster in clusters]
    family_matrix = row_normalize(matrix_from_counts(clusters, FAMILY_ORDER, "family_counts"))
    phase_matrix = row_normalize(matrix_from_counts(clusters, PHASE_ORDER, "phase_counts"))
    heatmap(
        family_matrix,
        xlabels=FAMILY_ORDER,
        ylabels=ylabels,
        title=f"Cluster x Task-Family Composition, k={args.k}",
        cbar_label="row fraction",
        output_path=output_dir / "cluster_family_heatmap.png",
    )
    heatmap(
        phase_matrix,
        xlabels=PHASE_ORDER,
        ylabels=ylabels,
        title=f"Cluster x Heuristic-Phase Composition, k={args.k}",
        cbar_label="row fraction",
        output_path=output_dir / "cluster_phase_heatmap.png",
    )
    plot_cluster_overview(run, output_dir / "cluster_overview.png")
    plot_skill_map(run, output_dir / "cluster_skill_map.png")

    print(f"Wrote figures to: {output_dir}")


if __name__ == "__main__":
    main()
