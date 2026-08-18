"""Merge redundant phase router clusters into fewer expert labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_MERGES = (
    (9, 16),   # mug late rotate/align, split mainly by time.
    (11, 17), # mug/shoe bimanual-orient late handling.
    (6, 15),  # right-arm adjust across block/open/click.
)


def parse_merges(value: str | None) -> list[tuple[int, ...]]:
    if not value:
        return [tuple(group) for group in DEFAULT_MERGES]
    groups = []
    for group_s in value.split(";"):
        group = tuple(int(item) for item in group_s.split(",") if item.strip())
        if len(group) >= 2:
            groups.append(group)
    return groups


def build_mapping(num_old: int, groups: list[tuple[int, ...]]) -> dict[int, int]:
    parent = list(range(num_old))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in groups:
        first = group[0]
        for item in group[1:]:
            union(first, item)

    roots = {}
    mapping = {}
    next_id = 0
    for old_id in range(num_old):
        root = find(old_id)
        if root not in roots:
            roots[root] = next_id
            next_id += 1
        mapping[old_id] = roots[root]
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo/meta/router_labels_k20_phase_h24"
        ),
    )
    parser.add_argument("--merges", type=str, default=None, help="Groups like '9,16;11,17;6,15'.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    labels = np.load(input_dir / "router_labels.npy")
    sample_labels = np.load(input_dir / "sample_labels.npy")
    input_summary = json.load((input_dir / "summary.json").open("r", encoding="utf-8"))
    num_old = int(input_summary["num_clusters"])
    groups = parse_merges(args.merges)
    mapping = build_mapping(num_old, groups)
    num_new = max(mapping.values()) + 1
    output_dir = args.output_dir or input_dir.parent / f"router_labels_k{num_new}_phase_h24_merged"
    output_dir.mkdir(parents=True, exist_ok=True)

    vectorized = np.full((max(mapping) + 1,), -1, dtype=np.int32)
    for old_id, new_id in mapping.items():
        vectorized[old_id] = new_id
    merged_labels = labels.copy()
    valid = labels >= 0
    merged_labels[valid] = vectorized[labels[valid]]
    merged_sample_labels = sample_labels.copy()
    sample_valid = sample_labels >= 0
    merged_sample_labels[sample_valid] = vectorized[sample_labels[sample_valid]]

    np.save(output_dir / "router_labels.npy", merged_labels.astype(np.int32, copy=False))
    np.save(output_dir / "sample_labels.npy", merged_sample_labels.astype(np.int32, copy=False))
    sample_indices_path = input_dir / "sample_indices.npy"
    if sample_indices_path.exists():
        np.save(output_dir / "sample_indices.npy", np.load(sample_indices_path))

    counts = np.bincount(merged_labels[merged_labels >= 0], minlength=num_new).astype(int)
    summary = {
        "input_dir": str(input_dir),
        "num_old_clusters": num_old,
        "num_clusters": int(num_new),
        "merge_groups": [list(group) for group in groups],
        "old_to_new": {str(k): int(v) for k, v in mapping.items()},
        "cluster_counts": counts.tolist(),
        "cluster_fractions": (counts / max(counts.sum(), 1)).astype(float).tolist(),
        "files": {
            "labels_by_global_index": "router_labels.npy",
            "sample_labels": "sample_labels.npy",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote merged labels to: {output_dir}")
    print(f"old_to_new={mapping}")
    print(f"counts={counts.tolist()}")


if __name__ == "__main__":
    main()
