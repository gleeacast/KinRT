#!/usr/bin/env python3
"""
Convert DIYRobot data (JSONL + JPG) to LeRobot Parquet format
compatible with pi05_aloha_full_base fine-tuning.

Data layout expected:
  <input_dir>/
    meta.json
    episode_001/
      data.jsonl
      cam_gripper/frame_XXXXXX.jpg
      cam_overhead/frame_XXXXXX.jpg
    episode_002/
      ...

Output layout:
  <output_dir>/
    meta/
      info.json
      episodes.jsonl
      tasks.jsonl
      episodes_stats.jsonl
    data/
      chunk-000/
        episode_000000.parquet
        episode_000001.parquet
        ...

Usage:
    python scripts/convert_diyrobot_to_lerobot.py \\
        --input_dir /path/to/diyrobot_data_capture/data/task_pick_cube_20260514_220147 \\
        --output_dir /path/to/output_lerobot_dataset \\
        --task "Pick up the cube and place it in the box." \\
        --gripper_open_deg 100.0
"""

import argparse
import io
import json
import os
import re
from pathlib import Path

import numpy as np
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

# Constants

# Joint order in DIYRobot follower data (14 DOF across both arms).
DIYROBOT_JOINTS = [
    "left_shoulder_pan",
    "left_shoulder_lift",
    "left_elbow_flex",
    "left_wrist_pitch",
    "left_wrist_flex",
    "left_wrist_roll",
    "left_gripper",
    "right_shoulder_pan",
    "right_shoulder_lift",
    "right_elbow_flex",
    "right_wrist_pitch",
    "right_wrist_flex",
    "right_wrist_roll",
    "right_gripper",
]
N_JOINTS = len(DIYROBOT_JOINTS)

# Map policy image keys to source JSONL keys and camera directories.
CAM_MAPPING = {
    "cam_high":        "overhead",
    "cam_left_wrist":  "left_gripper",
    "cam_right_wrist": "right_gripper",
}

# Image helpers

def load_jpg_as_png_bytes(path: Path) -> bytes:
    """Load a JPG file and re-encode as PNG bytes (matching sim parquet format)."""
    img = PILImage.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def build_cam_index(episode_dir: Path, frames: list[dict], source_camera_key: str) -> list[Path | None]:
    """Align source camera frames to control frames using forward filling."""
    cam_dir = episode_dir / f"cam_{source_camera_key}"
    T = len(frames)
    resolved: list[Path | None] = [None] * T

    last_path: Path | None = None
    for t, frame in enumerate(frames):
        cam = frame.get("cam", {})
        if source_camera_key in cam:
            candidate = cam_dir / cam[source_camera_key]
            if candidate.exists():
                last_path = candidate
        resolved[t] = last_path

    # Backward-fill the leading None values (frames before first camera update)
    if resolved[0] is None:
        first_valid = next((p for p in resolved if p is not None), None)
        for t in range(T):
            if resolved[t] is None:
                resolved[t] = first_valid
            else:
                break

    return resolved


def build_image_column(cam_index: list[Path | None], col_name: str) -> list[dict]:
    """
    Load images along the cam_index list into parquet-compatible dicts.
    Caches bytes by path to avoid redundant disk reads for forward-filled frames.
    """
    cache: dict[Path, bytes] = {}
    result = []
    for i, path in enumerate(cam_index):
        if path is None:
            # Fallback: black image
            buf = io.BytesIO()
            PILImage.new("RGB", (640, 480), (0, 0, 0)).save(buf, format="PNG")
            b = buf.getvalue()
            fname = f"frame_{i:06d}.png"
        else:
            if path not in cache:
                cache[path] = load_jpg_as_png_bytes(path)
            b = cache[path]
            fname = path.stem + ".png"
        result.append({"bytes": b, "path": fname})
    return result


# State / action helpers

def extract_states(frames: list[dict], gripper_open_deg: float) -> np.ndarray:
    """
    Extract follower joint positions from frame list (both arms, 14 DOF).
    Arm joints (indices 0-5, 7-12): converted to radians.
    Gripper joints (indices 6, 13): normalized to [0, 1] via gripper_open_deg.
    Returns float32 array of shape [T, 14].
    """
    T = len(frames)
    states = np.zeros((T, N_JOINTS), dtype=np.float32)
    for t, frame in enumerate(frames):
        follower = frame["follower"]
        for j, joint in enumerate(DIYROBOT_JOINTS):
            states[t, j] = follower[joint]["pos_deg"]
    # Convert arm joints to radians (all except the two gripper indices 6 and 13)
    arm_indices = list(range(6)) + list(range(7, 13))
    states[:, arm_indices] = np.deg2rad(states[:, arm_indices])
    # Normalize both grippers to [0, 1]
    states[:, 6]  = np.clip(states[:, 6]  / gripper_open_deg, 0.0, 1.0)
    states[:, 13] = np.clip(states[:, 13] / gripper_open_deg, 0.0, 1.0)
    return states


def build_actions(states: np.ndarray) -> np.ndarray:
    """
    Action at time t = state at time t+1 (next-state as action).
    Final frame action repeats the last state.
    """
    actions = np.empty_like(states)
    actions[:-1] = states[1:]
    actions[-1] = states[-1]
    return actions


# Episode processing

def process_episode(
    episode_dir: Path,
    episode_index: int,
    global_frame_offset: int,
    task_index: int,
    fps: int,
    gripper_open_deg: float,
) -> tuple[Dataset, dict]:
    """
    Process one episode directory into a HF Dataset + episode stats dict.
    Returns (hf_dataset, stats_entry).
    """
    # Load frames
    frames = []
    with open(episode_dir / "data.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                frames.append(json.loads(line))
    T = len(frames)

    # Preserve one 14-D state/action convention across every episode.
    states_14 = extract_states(frames, gripper_open_deg)
    actions_14 = build_actions(states_14)

    cam_index = {
        source_camera_key: build_cam_index(episode_dir, frames, source_camera_key)
        for source_camera_key in ("left_gripper", "right_gripper", "overhead")
    }

    img_cols = {}
    for lerobot_key, source_camera_key in CAM_MAPPING.items():
        img_cols[f"observation.images.{lerobot_key}"] = build_image_column(
            cam_index[source_camera_key], lerobot_key
        )

    # LeRobot requires local, episode, and global frame indices.
    timestamps = (np.arange(T, dtype=np.float32) / fps).reshape(-1)
    frame_indices = np.arange(T, dtype=np.int64)
    episode_indices = np.full(T, episode_index, dtype=np.int64)
    global_indices = np.arange(global_frame_offset, global_frame_offset + T, dtype=np.int64)
    task_indices = np.full(T, task_index, dtype=np.int64)

    # Build through HF Dataset to retain the expected Parquet metadata.
    hf_features = Features({
        "observation.state": Sequence(Value("float32"), length=14),
        "action": Sequence(Value("float32"), length=14),
        "observation.images.cam_high": Image(),
        "observation.images.cam_left_wrist": Image(),
        "observation.images.cam_right_wrist": Image(),
        "timestamp": Value("float32"),
        "frame_index": Value("int64"),
        "episode_index": Value("int64"),
        "index": Value("int64"),
        "task_index": Value("int64"),
    })

    data_dict = {
        "observation.state": [states_14[t].tolist() for t in range(T)],
        "action": [actions_14[t].tolist() for t in range(T)],
        "observation.images.cam_high": img_cols["observation.images.cam_high"],
        "observation.images.cam_left_wrist": img_cols["observation.images.cam_left_wrist"],
        "observation.images.cam_right_wrist": img_cols["observation.images.cam_right_wrist"],
        "timestamp": timestamps.tolist(),
        "frame_index": frame_indices.tolist(),
        "episode_index": episode_indices.tolist(),
        "index": global_indices.tolist(),
        "task_index": task_indices.tolist(),
    }
    df = Dataset.from_dict(data_dict, features=hf_features)

    # Store per-episode statistics in the LeRobot v2.1 schema.
    def feat_stats(arr: np.ndarray, count: int) -> dict:
        return {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).clip(min=1e-8).tolist(),
            "count": [count],
        }

    def image_stats(count: int) -> dict:
        # lerobot expects per-channel stats with shape (3, 1, 1): [[[v]], [[v]], [[v]]]
        # Images are normalized to [0, 1] by the model pipeline; placeholder values suffice.
        return {
            "min": [[[0.0]], [[0.0]], [[0.0]]],
            "max": [[[1.0]], [[1.0]], [[1.0]]],
            "mean": [[[0.5]], [[0.5]], [[0.5]]],
            "std": [[[0.25]], [[0.25]], [[0.25]]],
            "count": [count],
        }

    stats = {
        "episode_index": episode_index,
        "stats": {
            "observation.state": feat_stats(states_14, T),
            "action": feat_stats(actions_14, T),
            "observation.images.cam_high": image_stats(T),
            "observation.images.cam_left_wrist": image_stats(T),
            "observation.images.cam_right_wrist": image_stats(T),
            "timestamp": feat_stats(timestamps.reshape(-1, 1), T),
            "frame_index": feat_stats(frame_indices.reshape(-1, 1).astype(np.float64), T),
            "episode_index": feat_stats(episode_indices.reshape(-1, 1).astype(np.float64), T),
            "index": feat_stats(global_indices.reshape(-1, 1).astype(np.float64), T),
            "task_index": feat_stats(task_indices.reshape(-1, 1).astype(np.float64), T),
        },
    }

    return df, stats


# Meta file generation

def write_meta(
    output_dir: Path,
    episode_lengths: list[int],
    episode_stats: list[dict],
    task_description: str,
    fps: int,
) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = len(episode_lengths)
    total_frames = sum(episode_lengths)

    # info.json defines the dataset-level schema and chunk layout.
    info = {
        "codebase_version": "v2.1",
        "robot_type": "aloha",
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [14],
                "names": [[
                    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                    "left_wrist_pitch", "left_wrist_flex", "left_wrist_roll", "left_gripper",
                    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
                    "right_wrist_pitch", "right_wrist_flex", "right_wrist_roll", "right_gripper",
                ]],
            },
            "action": {
                "dtype": "float32",
                "shape": [14],
                "names": [[
                    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
                    "left_wrist_pitch", "left_wrist_flex", "left_wrist_roll", "left_gripper",
                    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
                    "right_wrist_pitch", "right_wrist_flex", "right_wrist_roll", "right_gripper",
                ]],
            },
            "observation.images.cam_high": {
                "dtype": "image", "shape": [3, 480, 640],
                "names": ["channels", "height", "width"],
            },
            "observation.images.cam_left_wrist": {
                "dtype": "image", "shape": [3, 480, 640],
                "names": ["channels", "height", "width"],
            },
            "observation.images.cam_right_wrist": {
                "dtype": "image", "shape": [3, 480, 640],
                "names": ["channels", "height", "width"],
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Keep a single task index for this conversion invocation.
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")

    # Record one metadata row for each converted episode.
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for i, length in enumerate(episode_lengths):
            entry = {
                "episode_index": i,
                "tasks": [task_description],
                "length": length,
            }
            f.write(json.dumps(entry) + "\n")

    # Retain per-episode statistics for downstream normalization checks.
    with open(meta_dir / "episodes_stats.jsonl", "w") as f:
        for stats in episode_stats:
            f.write(json.dumps(stats) + "\n")


# Main

def discover_episodes(input_dir: Path) -> list[Path]:
    """Return sorted list of episode subdirectories (episode_XXX)."""
    dirs = sorted(
        [d for d in input_dir.iterdir() if d.is_dir() and re.match(r"episode_\d+", d.name)],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )
    if not dirs:
        raise FileNotFoundError(f"No episode_XXX directories found in {input_dir}")
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Convert DIYRobot JSONL+JPG data to LeRobot Parquet format"
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Task directory, e.g. diyrobot_data_capture/data/task_pick_cube_20260514_220147",
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Output LeRobot dataset directory",
    )
    parser.add_argument(
        "--task", type=str, default="Pick up the cube.",
        help="Task description string used as language prompt during training",
    )
    parser.add_argument(
        "--fps", type=int, default=40,
        help="Control loop FPS of the real robot (default: 40)",
    )
    parser.add_argument(
        "--gripper_open_deg", type=float, default=100.0,
        help="Gripper degrees at fully-open position, used to normalize to [0,1] (default: 100.0)",
    )
    parser.add_argument(
        "--episodes", type=str, default=None,
        help='Comma-separated episode numbers to include, e.g. "1,2,3". Default: all.',
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    # Discover episodes
    all_episode_dirs = discover_episodes(input_dir)
    if args.episodes is not None:
        selected = set(int(x) for x in args.episodes.split(","))
        episode_dirs = [
            d for d in all_episode_dirs
            if int(re.search(r"\d+", d.name).group()) in selected
        ]
    else:
        episode_dirs = all_episode_dirs

    print(f"Found {len(episode_dirs)} episode(s) in {input_dir}")
    print(f"Task: {args.task}")
    print(f"FPS: {args.fps}, gripper_open_deg: {args.gripper_open_deg}")
    print(f"Output: {output_dir}")
    print()

    # Prepare output directories
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    episode_lengths = []
    episode_stats_list = []
    global_frame_offset = 0

    for ep_idx, ep_dir in enumerate(episode_dirs):
        print(f"  Processing {ep_dir.name} -> episode_{ep_idx:06d}.parquet ...", end="", flush=True)

        df, stats = process_episode(
            episode_dir=ep_dir,
            episode_index=ep_idx,
            global_frame_offset=global_frame_offset,
            task_index=0,
            fps=args.fps,
            gripper_open_deg=args.gripper_open_deg,
        )

        T = len(df)
        parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
        df.to_parquet(str(parquet_path))

        episode_lengths.append(T)
        episode_stats_list.append(stats)
        global_frame_offset += T

        print(f" {T} frames, saved to {parquet_path.name}")

    # Write meta files
    print("\nWriting meta files...")
    write_meta(
        output_dir=output_dir,
        episode_lengths=episode_lengths,
        episode_stats=episode_stats_list,
        task_description=args.task,
        fps=args.fps,
    )

    total_frames = sum(episode_lengths)
    print(f"\nDone. {len(episode_dirs)} episodes, {total_frames} total frames.")
    print(f"Dataset written to: {output_dir}")
    print()
    print("Next step - update config.py:")
    print(f'  repo_id="{output_dir}"')


if __name__ == "__main__":
    main()
