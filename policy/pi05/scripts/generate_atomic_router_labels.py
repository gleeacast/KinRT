"""Generate atomic-action router labels from raw action trajectories.

Design
------
Unlike v1/v2 which cluster *features* with k-means, this script assigns
each frame a **semantically-grounded atomic action label** derived directly
from the kinematic signals in the trajectory:

  0  approach     – arm moving toward target, gripper open/neutral
  1  pre_grasp    – motion valley just before gripper closes
  2  grasp        – gripper actively closing
  3  carry        – arm moving with gripper closed (holding object)
  4  orient       – wrist rotation dominant (e.g. rotate mug, flip sign)
  5  pre_release  – motion valley after carry, before gripper opens
  6  release      – gripper actively opening

These 7 states cover every observed episode in the mixed dataset, span all
8 task families, and correspond directly to the "atomic skills" used in
AtomicVLA (pick≈grasp, place≈release, turn≈orient) but are **fully
automatic** — no per-episode human annotation required.

Soft labels
-----------
Instead of hard one-hot assignment, each frame gets a **7-dim soft weight
vector** that smoothly transitions at boundaries.  This is written to
``soft_router_labels.npy`` with shape ``(total_frames, 7)``.

The hard argmax is also saved to ``router_labels.npy`` (shape
``(total_frames,)``) for compatibility with the existing DataConfig API.

Validation
----------
The script prints a per-task distribution table and cross-task statistics
so you can verify the labels make semantic sense before training.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.signal import find_peaks

# ── Constants ────────────────────────────────────────────────────────────────
LEFT_DIMS   = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
RIGHT_DIMS  = np.asarray([7, 8, 9, 10, 11, 12], dtype=np.int64)
WRIST_DIMS  = np.asarray([4, 5, 11, 12], dtype=np.int64)
LEFT_GRIPPER  = 6
RIGHT_GRIPPER = 13

ATOMIC_NAMES = [
    "approach",     # 0
    "pre_grasp",    # 1
    "grasp",        # 2
    "carry",        # 3
    "orient",       # 4
    "pre_release",  # 5
    "release",      # 6
]
NUM_ATOMIC = len(ATOMIC_NAMES)

FAMILY_NAMES = [
    "block_handover", "mug_hang", "move_can_to_pot", "open_laptop",
    "shoe_into_box",  "mouse_to_mat", "rotate_payment_sign", "click_switch",
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_path(repo_root: Path, pattern: str, episode_index: int, chunks_size: int) -> Path:
    return repo_root / pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def _smooth(x: np.ndarray, w: int = 7) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same")


# ── Core label assignment ────────────────────────────────────────────────────

def assign_hard_labels(actions: np.ndarray) -> np.ndarray:
    """Assign each frame one of 7 atomic action labels.

    Rule priority (later rule overwrites earlier):
      default      → approach
      grip closed  → carry (if also moving)
      orient       → orient (wrist rotation dominant)
      low motion   → pre_grasp
      grip closing → grasp
      low motion after last grasp → pre_release
      grip opening → release
    """
    n = actions.shape[0]
    vel = np.diff(actions, axis=0, prepend=actions[:1])

    left_motion  = np.mean(np.abs(vel[:, LEFT_DIMS]),  axis=1)
    right_motion = np.mean(np.abs(vel[:, RIGHT_DIMS]), axis=1)
    total_motion = np.maximum(left_motion, right_motion)
    wrist_rot    = np.mean(np.abs(vel[:, WRIST_DIMS]), axis=1)

    lg     = actions[:, LEFT_GRIPPER]
    rg     = actions[:, RIGHT_GRIPPER]
    lg_vel = np.diff(lg, prepend=lg[:1])
    rg_vel = np.diff(rg, prepend=rg[:1])

    sm = _smooth(total_motion)
    low_motion   = sm < 0.004
    grip_closed  = (lg < 0.3) | (rg < 0.3)
    grasp_mask   = (lg_vel < -0.03) | (rg_vel < -0.03)
    release_mask = (lg_vel > 0.03)  | (rg_vel > 0.03)

    labels = np.zeros(n, dtype=np.int32)      # approach

    # carry: moving with closed gripper
    labels[(total_motion > 0.008) & grip_closed] = 3

    # orient: wrist rotation prominent
    labels[wrist_rot > 0.015] = 4

    # pre_grasp: low motion (default)
    labels[low_motion] = 1

    # grasp: gripper actively closing
    labels[grasp_mask] = 2

    # release: gripper actively opening
    labels[release_mask] = 6

    # pre_release: low motion BETWEEN last grasp and first following release
    grasp_frames   = np.where(grasp_mask)[0]
    release_frames = np.where(release_mask)[0]
    if len(grasp_frames) and len(release_frames):
        last_grasp = grasp_frames[-1]
        for rf in release_frames:
            if rf > last_grasp:
                idx = np.arange(n)
                pre_rel = (low_motion) & (idx > last_grasp) & (idx < rf)
                labels[pre_rel] = 5
                break

    return labels


def assign_soft_labels(
    hard_labels: np.ndarray,
    *,
    sigma: float = 4.0,
) -> np.ndarray:
    """Convert hard labels to soft Gaussian-blurred weights.

    For each frame t the soft weight for class c is:
        w_c(t) ∝ exp(-0.5 * min_dist(t,c)^2 / sigma^2)
    where min_dist(t,c) is the distance (in frames) to the nearest frame
    with hard label == c.

    This gives a smooth transition at segment boundaries instead of a
    hard step, which allows the router to express uncertainty during
    transition phases (e.g. the few frames at the end of a grasp motion
    that are simultaneously "carry" and "grasp").
    """
    n = len(hard_labels)
    soft = np.zeros((n, NUM_ATOMIC), dtype=np.float32)
    t_idx = np.arange(n, dtype=np.float32)

    for c in range(NUM_ATOMIC):
        frames_c = np.where(hard_labels == c)[0].astype(np.float32)
        if len(frames_c) == 0:
            continue
        # Distance of each frame to the nearest frame with label c
        dists = np.abs(t_idx[:, None] - frames_c[None, :]).min(axis=1)
        soft[:, c] = np.exp(-0.5 * (dists / sigma) ** 2)

    # Normalise to sum-1 per frame
    row_sum = soft.sum(axis=1, keepdims=True)
    soft = soft / np.clip(row_sum, 1e-8, None)
    return soft


# ── Main processing ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo"),
    )
    parser.add_argument(
        "--soft-sigma",
        type=float,
        default=4.0,
        help="Gaussian sigma (frames) for soft label smoothing.  "
             "4 ≈ 0.08s at 50fps; try 2-8.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--episodes-per-family", type=int, default=None,
                        help="Limit for debugging (None = all).")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    info = read_json(repo_root / "meta" / "info.json")
    total_episodes = int(info["total_episodes"])
    total_frames   = int(info["total_frames"])
    pattern        = info["data_path"]
    chunks_size    = int(info["chunks_size"])

    tag = f"atomic_sigma{int(args.soft_sigma)}"
    output_dir = args.output_dir or (repo_root / "meta" / f"router_labels_{tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output arrays (global index aligned)
    hard_by_index = np.full(total_frames, -1, dtype=np.int32)
    soft_by_index = np.zeros((total_frames, NUM_ATOMIC), dtype=np.float32)

    # Stats
    family_counts   = np.zeros((len(FAMILY_NAMES), NUM_ATOMIC), dtype=np.int64)
    global_counts   = np.zeros(NUM_ATOMIC, dtype=np.int64)

    eps_limit = args.episodes_per_family or 9999

    for episode_index in range(total_episodes):
        family_id = min(episode_index // 100, len(FAMILY_NAMES) - 1)
        ep_within = episode_index % 100
        if ep_within >= eps_limit:
            continue

        pq_path = episode_path(repo_root, pattern, episode_index, chunks_size)
        table = pq.read_table(
            pq_path, columns=["action", "index", "frame_index"]
        )
        actions = np.asarray(
            table["action"].combine_chunks().to_pylist(), dtype=np.float32
        )
        global_idx = np.asarray(
            table["index"].combine_chunks().to_pylist(), dtype=np.int64
        )

        hard = assign_hard_labels(actions)
        soft = assign_soft_labels(hard, sigma=args.soft_sigma)

        hard_by_index[global_idx] = hard
        soft_by_index[global_idx] = soft

        counts = np.bincount(hard, minlength=NUM_ATOMIC)
        global_counts  += counts
        family_counts[family_id] += counts

        if episode_index % 100 == 0:
            print(f"  ep {episode_index}/{total_episodes}")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(output_dir / "router_labels.npy",      hard_by_index)
    np.save(output_dir / "soft_router_labels.npy", soft_by_index)

    # ── Print validation table ────────────────────────────────────────────────
    total = int(global_counts.sum())
    print("\n=== Atomic label distribution (all episodes) ===")
    print(f"{'label':12s} {'count':>8s} {'frac':>6s}  bar")
    for i, name in enumerate(ATOMIC_NAMES):
        frac = global_counts[i] / max(total, 1)
        bar  = "#" * int(frac * 50)
        print(f"  {name:12s} {global_counts[i]:8d} {frac:.3f}  {bar}")

    print("\n=== Per-task distribution ===")
    header = "task" + "".join(f"  {n[:5]:>6s}" for n in ATOMIC_NAMES)
    print(header)
    for fi, fname in enumerate(FAMILY_NAMES):
        row_total = max(family_counts[fi].sum(), 1)
        vals = "".join(f"  {family_counts[fi, j]/row_total:6.3f}" for j in range(NUM_ATOMIC))
        print(f"  {fname:25s}{vals}")

    # ── Class weights for training ────────────────────────────────────────────
    # sqrt inverse frequency (same formula as existing code)
    class_weights = np.sqrt(total / (NUM_ATOMIC * np.clip(global_counts, 1, None)))
    class_weights = class_weights.astype(np.float32)

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "num_clusters": NUM_ATOMIC,
        "atomic_names": ATOMIC_NAMES,
        "soft_sigma": float(args.soft_sigma),
        "total_frames": total,
        "cluster_counts": global_counts.astype(int).tolist(),
        "cluster_fractions": (global_counts / max(total, 1)).astype(float).tolist(),
        "class_weights": class_weights.tolist(),
        "files": {
            "hard_labels_by_global_index": "router_labels.npy",
            "soft_labels_by_global_index": "soft_router_labels.npy",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved to: {output_dir}")
    print(f"class_weights: {[round(float(w),3) for w in class_weights]}")


if __name__ == "__main__":
    main()
