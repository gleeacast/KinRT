#!/usr/bin/env python3
"""Append selected RoboTwin HDF5 episodes to an existing LeRobot v2.1 repo.

This intentionally updates only the parquet data files plus:
  - meta/info.json
  - meta/episodes.jsonl
  - meta/tasks.jsonl

It does not update episodes_stats.jsonl or stats.json.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

from datasets import Array2D
from datasets import Array3D
from datasets import Array4D
from datasets import Array5D
from datasets import Dataset
from datasets import Features
from datasets import Image
from datasets import Sequence
from datasets import Value
import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image as PILImage
from tqdm import tqdm


TASKS = [
    "beat_block_hammer",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "lift_pot",
    "open_microwave",
    "pick_dual_bottles",
    "stack_blocks_two",
]

VARIANTS = [
    ("clean", "demo_clean"),
    ("randomized", "demo_randomized"),
]

CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def episode_sort_key(path: Path) -> int:
    match = re.search(r"episode_?(\d+)$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse episode index from {path}")
    return int(match.group(1))


def find_variant_dir(processed_root: Path, task: str, variant_prefix: str) -> Path:
    matches = sorted(processed_root.glob(f"{task}-{variant_prefix}-*"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one directory for {task}-{variant_prefix}-*, found {matches}")
    return matches[0]


def selected_sources(processed_root: Path, episodes_per_variant: int) -> list[tuple[str, str, int, Path]]:
    sources: list[tuple[str, str, int, Path]] = []
    for task in TASKS:
        for variant_name, variant_prefix in VARIANTS:
            variant_dir = find_variant_dir(processed_root, task, variant_prefix)
            for ep_idx in range(episodes_per_variant):
                ep_dir = variant_dir / f"episode_{ep_idx}"
                hdf5_path = ep_dir / f"episode_{ep_idx}.hdf5"
                instr_path = ep_dir / "instructions.json"
                if not hdf5_path.is_file():
                    raise FileNotFoundError(hdf5_path)
                if not instr_path.is_file():
                    raise FileNotFoundError(instr_path)
                sources.append((task, variant_name, ep_idx, hdf5_path))
    return sources


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def choose_instruction(ep_dir: Path, source_episode_index: int) -> str:
    instructions = read_json(ep_dir / "instructions.json").get("instructions", [])
    instructions = unique_preserve_order([str(item) for item in instructions])
    if not instructions:
        raise ValueError(f"No instructions in {ep_dir / 'instructions.json'}")
    return instructions[source_episode_index % len(instructions)]


def normalize_features(features: dict) -> dict:
    normalized = {}
    for key, value in features.items():
        item = dict(value)
        if isinstance(item.get("shape"), list):
            item["shape"] = tuple(item["shape"])
        normalized[key] = item
    return normalized


def get_hf_features_from_info_features(features: dict) -> Features:
    hf_features = {}
    for key, ft in normalize_features(features).items():
        dtype = ft["dtype"]
        if dtype == "video":
            continue
        if dtype == "image":
            hf_features[key] = Image()
        elif ft["shape"] == (1,):
            hf_features[key] = Value(dtype=dtype)
        elif len(ft["shape"]) == 1:
            hf_features[key] = Sequence(length=ft["shape"][0], feature=Value(dtype=dtype))
        elif len(ft["shape"]) == 2:
            hf_features[key] = Array2D(shape=ft["shape"], dtype=dtype)
        elif len(ft["shape"]) == 3:
            hf_features[key] = Array3D(shape=ft["shape"], dtype=dtype)
        elif len(ft["shape"]) == 4:
            hf_features[key] = Array4D(shape=ft["shape"], dtype=dtype)
        elif len(ft["shape"]) == 5:
            hf_features[key] = Array5D(shape=ft["shape"], dtype=dtype)
        else:
            raise ValueError(f"Unsupported feature shape for {key}: {ft}")
    return Features(hf_features)


def image_array_to_pil_image(image_array: np.ndarray) -> PILImage.Image:
    if image_array.ndim != 3:
        raise ValueError(f"The array has {image_array.ndim} dimensions, but 3 are expected for an image.")
    if image_array.shape[0] == 3:
        image_array = image_array.transpose(1, 2, 0)
    elif image_array.shape[-1] != 3:
        raise ValueError(f"The image has {image_array.shape[-1]} channels, but 3 are expected.")
    if image_array.dtype != np.uint8:
        max_value = image_array.max().item()
        min_value = image_array.min().item()
        if max_value > 1.0 or min_value < 0.0:
            raise ValueError(f"Float image out of [0, 1] range: [{min_value}, {max_value}]")
        image_array = (image_array * 255).astype(np.uint8)
    return PILImage.fromarray(image_array)


def encoded_image_record(data: bytes | np.bytes_, frame_index: int) -> dict:
    image_bytes = bytes(data).rstrip(b"\0")
    if not image_bytes:
        raise ValueError(f"Empty encoded image at frame {frame_index}")
    return {"bytes": image_bytes, "path": f"frame_{frame_index:06d}.jpg"}


def load_image_records(ep: h5py.File, camera: str) -> list[dict]:
    dataset = ep[f"/observations/images/{camera}"]
    if dataset.ndim == 4:
        return [png_record(np.asarray(image), frame_idx) for frame_idx, image in enumerate(dataset[:])]

    return [encoded_image_record(data, frame_idx) for frame_idx, data in enumerate(dataset)]


def png_record(image: np.ndarray, frame_index: int) -> dict:
    pil_image = image_array_to_pil_image(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return {"bytes": buffer.getvalue(), "path": f"frame_{frame_index:06d}.png"}


def load_episode_dict(
    hdf5_path: Path,
    *,
    episode_index: int,
    global_start_index: int,
    task_index: int,
    fps: int,
) -> tuple[dict, int]:
    with h5py.File(hdf5_path, "r") as ep:
        state = np.asarray(ep["/observations/qpos"][:], dtype=np.float32)
        action = np.asarray(ep["/action"][:], dtype=np.float32)
        if state.shape != action.shape:
            raise ValueError(f"State/action shape mismatch in {hdf5_path}: {state.shape} vs {action.shape}")

        num_frames = int(state.shape[0])
        images = {camera: load_image_records(ep, camera) for camera in CAMERAS}

    for camera, values in images.items():
        if len(values) != num_frames:
            raise ValueError(f"{hdf5_path} {camera} has {len(values)} images, expected {num_frames}")

    episode = {
        "observation.state": state,
        "action": action,
        "timestamp": (np.arange(num_frames, dtype=np.float32) / float(fps)).astype(np.float32),
        "frame_index": np.arange(num_frames, dtype=np.int64),
        "episode_index": np.full((num_frames,), episode_index, dtype=np.int64),
        "index": np.arange(global_start_index, global_start_index + num_frames, dtype=np.int64),
        "task_index": np.full((num_frames,), task_index, dtype=np.int64),
    }
    for camera in CAMERAS:
        episode[f"observation.images.{camera}"] = images[camera]
    return episode, num_frames


def get_episode_length(hdf5_path: Path) -> int:
    with h5py.File(hdf5_path, "r") as ep:
        state_shape = ep["/observations/qpos"].shape
        action_shape = ep["/action"].shape
        if state_shape != action_shape:
            raise ValueError(f"State/action shape mismatch in {hdf5_path}: {state_shape} vs {action_shape}")
        return int(state_shape[0])


def write_episode_parquet(episode: dict, dest_path: Path, hf_features) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_episode = {key: episode[key] for key in hf_features.keys()}
    dataset = Dataset.from_dict(ordered_episode, features=hf_features, split="train")
    dataset.to_parquet(str(dest_path))


def get_data_file_path(info: dict, episode_index: int) -> Path:
    episode_chunk = episode_index // int(info["chunks_size"])
    return Path(
        info["data_path"].format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("/private/yth/projects/RoboTwin/policy/pi05/processed_data"),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo_1"),
    )
    parser.add_argument("--episodes-per-variant", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    meta_dir = args.repo_root / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"

    info = read_json(info_path)
    episodes = read_jsonl(episodes_path)
    tasks_rows = read_jsonl(tasks_path)
    task_to_index = {row["task"]: int(row["task_index"]) for row in tasks_rows}

    start_episode = int(info["total_episodes"])
    start_frame = int(info["total_frames"])
    if start_episode != len(episodes):
        raise RuntimeError(f"info.total_episodes={start_episode}, but episodes.jsonl has {len(episodes)} rows")
    existing_parquets = sorted((args.repo_root / "data").glob("chunk-*/episode_*.parquet"))
    if len(existing_parquets) != start_episode:
        raise RuntimeError(f"Expected {start_episode} parquet files, found {len(existing_parquets)}")

    sources = selected_sources(args.processed_root, args.episodes_per_variant)
    hf_features = get_hf_features_from_info_features(info["features"])

    new_task_rows: list[dict] = []
    new_episode_rows: list[dict] = []
    next_task_index = int(info["total_tasks"])
    next_frame_index = start_frame

    print(f"Appending {len(sources)} episodes to {args.repo_root}")
    print(f"Start episode={start_episode}, start frame index={start_frame}, start task count={next_task_index}")

    for offset, (task_name, variant_name, source_ep_idx, hdf5_path) in enumerate(tqdm(sources)):
        dest_episode_index = start_episode + offset
        dest_path = args.repo_root / get_data_file_path(info, dest_episode_index)
        if dest_path.exists():
            raise FileExistsError(dest_path)

        instruction = choose_instruction(hdf5_path.parent, source_ep_idx)
        task_index = task_to_index.get(instruction)
        if task_index is None:
            task_index = next_task_index
            next_task_index += 1
            task_to_index[instruction] = task_index
            new_task_rows.append({"task_index": task_index, "task": instruction})

        if args.dry_run:
            episode = None
            num_frames = get_episode_length(hdf5_path)
        else:
            episode, num_frames = load_episode_dict(
                hdf5_path,
                episode_index=dest_episode_index,
                global_start_index=next_frame_index,
                task_index=task_index,
                fps=int(info["fps"]),
            )

        if not args.dry_run and episode is not None:
            write_episode_parquet(episode, dest_path, hf_features)

        new_episode_rows.append(
            {
                "episode_index": dest_episode_index,
                "tasks": [instruction],
                "length": num_frames,
                "source": {
                    "task": task_name,
                    "variant": variant_name,
                    "episode_index": source_ep_idx,
                    "hdf5_path": str(hdf5_path),
                },
            }
        )
        next_frame_index += num_frames

    if args.dry_run:
        print(f"Dry run: would add {len(new_episode_rows)} episodes")
        print(f"Dry run: would add {len(new_task_rows)} new task strings")
        print(f"Dry run: resulting total_frames would be {next_frame_index}")
        return

    append_jsonl(tasks_path, new_task_rows)
    append_jsonl(
        episodes_path,
        [
            {
                "episode_index": row["episode_index"],
                "tasks": row["tasks"],
                "length": row["length"],
            }
            for row in new_episode_rows
        ],
    )

    info["total_episodes"] = start_episode + len(new_episode_rows)
    info["total_frames"] = next_frame_index
    info["total_tasks"] = next_task_index
    info["total_chunks"] = max(int(info.get("total_chunks", 1)), (info["total_episodes"] - 1) // int(info["chunks_size"]) + 1)
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    write_json(info_path, info)

    manifest_path = meta_dir / "append_selected_robotwin_manifest.json"
    write_json(
        manifest_path,
        {
            "processed_root": str(args.processed_root),
            "repo_root": str(args.repo_root),
            "start_episode": start_episode,
            "start_frame": start_frame,
            "added_episodes": len(new_episode_rows),
            "added_tasks": len(new_task_rows),
            "final_total_episodes": info["total_episodes"],
            "final_total_frames": info["total_frames"],
            "final_total_tasks": info["total_tasks"],
            "episodes": new_episode_rows,
            "stats_note": "episodes_stats.jsonl and stats.json were intentionally not updated.",
        },
    )

    print(f"Added {len(new_episode_rows)} episodes")
    print(f"Added {len(new_task_rows)} new task strings")
    print(f"Final total episodes: {info['total_episodes']}")
    print(f"Final total frames: {info['total_frames']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
