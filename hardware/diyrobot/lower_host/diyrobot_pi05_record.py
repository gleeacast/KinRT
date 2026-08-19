#!/usr/bin/env python3
"""Record DIYRobot teleoperation episodes as a PI0.5/OpenPI-ready dataset.

The recorder reuses the validated strict-v22 leader/follower safety path, but
writes training samples as:

- observation.state: follower absolute joint positions in degrees
- action: actual follower absolute joint targets sent to the motors
- observation.images.<camera>: RGB camera frames
- task: natural-language instruction
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import time
import select
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDataset

import dual_arm_teleop_strict_v22 as teleop


DEFAULT_MOTORS = (
    "left_shoulder_pan,left_shoulder_lift,left_elbow_flex,left_wrist_pitch,left_wrist_flex,"
    "left_wrist_roll,left_gripper,right_shoulder_pan,right_shoulder_lift,right_elbow_flex,"
    "right_wrist_pitch,right_wrist_flex,right_wrist_roll,right_gripper"
)


@dataclass(frozen=True)
class CameraSpec:
    name: str
    index_or_path: str
    width: int
    height: int
    fps: float


class OpenCVCameraSet:
    def __init__(self, specs: list[CameraSpec]) -> None:
        self.specs = specs
        self.captures: dict[str, cv2.VideoCapture] = {}

    def open(self) -> None:
        for spec in self.specs:
            source: int | str
            source = int(spec.index_or_path) if spec.index_or_path.isdigit() else spec.index_or_path
            cap = cv2.VideoCapture(source)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, spec.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, spec.height)
            cap.set(cv2.CAP_PROP_FPS, spec.fps)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {spec.name}={spec.index_or_path}")
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                raise RuntimeError(f"Failed to read first frame from camera {spec.name}")
            self.captures[spec.name] = cap

    def read(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for spec in self.specs:
            cap = self.captures[spec.name]
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                raise RuntimeError(f"Camera {spec.name} returned no frame")
            if frame_bgr.shape[1] != spec.width or frame_bgr.shape[0] != spec.height:
                frame_bgr = cv2.resize(frame_bgr, (spec.width, spec.height), interpolation=cv2.INTER_AREA)
            frames[spec.name] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frames

    def close(self) -> None:
        for cap in self.captures.values():
            cap.release()
        self.captures.clear()


def parse_camera_spec(value: str) -> CameraSpec:
    # NAME=INDEX_OR_PATH:WIDTHxHEIGHT[:FPS]
    if "=" not in value:
        raise ValueError(f"Bad --camera spec {value!r}; expected NAME=INDEX_OR_PATH:WIDTHxHEIGHT[:FPS]")
    name, rest = value.split("=", 1)
    parts = rest.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Bad --camera spec {value!r}; expected NAME=INDEX_OR_PATH:WIDTHxHEIGHT[:FPS]")
    index_or_path, size = parts[0], parts[1]
    if "x" not in size.lower():
        raise ValueError(f"Bad camera size {size!r}; expected WIDTHxHEIGHT")
    width_s, height_s = size.lower().split("x", 1)
    fps = float(parts[2]) if len(parts) == 3 else 30.0
    name = name.strip()
    if not name:
        raise ValueError("Camera name cannot be empty")
    return CameraSpec(
        name=name,
        index_or_path=index_or_path.strip(),
        width=int(width_s),
        height=int(height_s),
        fps=fps,
    )


def build_dataset_features(selected: list[str], cameras: list[CameraSpec], use_videos: bool) -> dict:
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(selected),),
            "names": list(selected),
        },
        "action": {
            "dtype": "float32",
            "shape": (len(selected),),
            "names": list(selected),
        },
    }
    image_dtype = "video" if use_videos else "image"
    for spec in cameras:
        features[f"observation.images.{spec.name}"] = {
            "dtype": image_dtype,
            "shape": (spec.height, spec.width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


def vector_from_positions(selected: list[str], positions: dict[str, float]) -> np.ndarray:
    return np.asarray([float(positions[name]) for name in selected], dtype=np.float32)


def add_dataset_frame(
    dataset: LeRobotDataset,
    *,
    selected: list[str],
    task: str,
    observation_positions: dict[str, float],
    action_positions: dict[str, float],
    camera_frames: dict[str, np.ndarray],
) -> None:
    frame = {
        "observation.state": vector_from_positions(selected, observation_positions),
        "action": vector_from_positions(selected, action_positions),
        "task": task,
    }
    for name, image in camera_frames.items():
        frame[f"observation.images.{name}"] = image
    dataset.add_frame(frame)


def write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_json(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(item, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def vector_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        raise RuntimeError("Cannot compute stats for an empty array.")
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def scalar_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        raise RuntimeError("Cannot compute stats for an empty array.")
    arr = values.astype(float)
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.shape[0])],
    }


def episode_stats_from_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    state = np.stack([row["observation.state"] for row in rows]).astype(np.float32)
    action = np.stack([row["action"] for row in rows]).astype(np.float32)
    timestamp = np.asarray([row["timestamp"] for row in rows], dtype=np.float32)
    frame_index = np.asarray([row["frame_index"] for row in rows], dtype=np.int64)
    episode_index = np.asarray([row["episode_index"] for row in rows], dtype=np.int64)
    index = np.asarray([row["index"] for row in rows], dtype=np.int64)
    task_index = np.asarray([row["task_index"] for row in rows], dtype=np.int64)
    return {
        "observation.state": vector_stats(state),
        "action": vector_stats(action),
        "timestamp": scalar_stats(timestamp),
        "frame_index": scalar_stats(frame_index),
        "episode_index": scalar_stats(episode_index),
        "index": scalar_stats(index),
        "task_index": scalar_stats(task_index),
    }


def aggregate_numeric_stats(episode_stats: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    if not episode_stats:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key in episode_stats[0]:
        counts = np.asarray([float(item[key]["count"][0]) for item in episode_stats], dtype=np.float64)
        mins = np.asarray([item[key]["min"] for item in episode_stats], dtype=np.float64)
        maxs = np.asarray([item[key]["max"] for item in episode_stats], dtype=np.float64)
        means = np.asarray([item[key]["mean"] for item in episode_stats], dtype=np.float64)
        stds = np.asarray([item[key]["std"] for item in episode_stats], dtype=np.float64)
        total = float(counts.sum())
        safe_total = np.maximum(total, 1.0)
        weights = counts.reshape((-1,) + (1,) * (means.ndim - 1))
        mean = (means * weights).sum(axis=0) / safe_total
        second_moment = ((stds**2 + means**2) * weights).sum(axis=0) / safe_total
        variance = np.maximum(second_moment - mean**2, 0.0)
        out[key] = {
            "min": mins.min(axis=0).astype(float).tolist(),
            "max": maxs.max(axis=0).astype(float).tolist(),
            "mean": mean.astype(float).tolist(),
            "std": np.sqrt(variance).astype(float).tolist(),
            "count": [int(total)],
        }
    return out


class V21EpisodeWriter:
    def __init__(
        self,
        *,
        root: Path,
        repo_id: str,
        fps: int,
        selected: list[str],
        cameras: list[CameraSpec],
        use_videos: bool,
        resume: bool,
    ) -> None:
        if not use_videos:
            raise RuntimeError("v2.1 writer currently requires video columns; remove --no-video.")
        self.root = root
        self.repo_id = repo_id
        self.fps = int(fps)
        self.selected = selected
        self.cameras = cameras
        self.use_videos = use_videos
        self.current_rows: list[dict[str, Any]] = []
        self.current_video_writers: dict[str, cv2.VideoWriter] = {}
        self.current_episode_index: int | None = None
        self.total_frames = 0
        self.episodes: list[dict[str, Any]] = []
        self.tasks: list[dict[str, Any]] = []
        self.task_to_index: dict[str, int] = {}
        self.episodes_stats: list[dict[str, Any]] = []
        self._closed = False

        if self.root.exists() and not resume:
            raise RuntimeError(f"Dataset root already exists: {self.root}. Use --resume or choose a new ROOT.")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "meta").mkdir(parents=True, exist_ok=True)
        if resume:
            self._load_existing_metadata()

    @property
    def next_episode_index(self) -> int:
        return len(self.episodes)

    def _load_existing_metadata(self) -> None:
        tasks_path = self.root / "meta/tasks.jsonl"
        episodes_path = self.root / "meta/episodes.jsonl"
        stats_path = self.root / "meta/episodes_stats.jsonl"
        if tasks_path.exists():
            self.tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines() if line]
            self.task_to_index = {item["task"]: int(item["task_index"]) for item in self.tasks}
        if episodes_path.exists():
            self.episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line]
            self.total_frames = sum(int(item["length"]) for item in self.episodes)
        if stats_path.exists():
            self.episodes_stats = [
                json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines() if line
            ]

    def task_index(self, task: str) -> int:
        if task not in self.task_to_index:
            idx = len(self.tasks)
            item = {"task_index": idx, "task": task}
            self.tasks.append(item)
            self.task_to_index[task] = idx
        return self.task_to_index[task]

    def start_episode(self) -> None:
        if self.current_episode_index is not None:
            raise RuntimeError("Episode already open.")
        ep_idx = self.next_episode_index
        self.current_episode_index = ep_idx
        self.current_rows = []
        if self.use_videos:
            self.current_video_writers = {}
            for spec in self.cameras:
                path = self.video_path(ep_idx, spec.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(self.fps),
                    (int(spec.width), int(spec.height)),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open video writer for {path}")
                self.current_video_writers[spec.name] = writer

    def add_frame(
        self,
        *,
        task: str,
        observation_positions: dict[str, float],
        action_positions: dict[str, float],
        camera_frames: dict[str, np.ndarray],
    ) -> None:
        if self.current_episode_index is None:
            self.start_episode()
        assert self.current_episode_index is not None
        ep_frame_idx = len(self.current_rows)
        row: dict[str, Any] = {
            "observation.state": vector_from_positions(self.selected, observation_positions),
            "action": vector_from_positions(self.selected, action_positions),
            "timestamp": np.float32(ep_frame_idx / float(self.fps)),
            "frame_index": np.int64(ep_frame_idx),
            "episode_index": np.int64(self.current_episode_index),
            "index": np.int64(self.total_frames + ep_frame_idx),
            "task_index": np.int64(self.task_index(task)),
        }
        self.current_rows.append(row)
        for name, frame_rgb in camera_frames.items():
            if name not in self.current_video_writers:
                continue
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            self.current_video_writers[name].write(frame_bgr)

    def save_episode(self, task: str | None = None) -> None:
        if self.current_episode_index is None:
            raise RuntimeError("No episode is open.")
        ep_idx = self.current_episode_index
        for writer in self.current_video_writers.values():
            writer.release()
        self.current_video_writers.clear()
        if not self.current_rows:
            self.current_episode_index = None
            return

        data_path = self.data_path(ep_idx)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.current_rows).to_parquet(data_path, index=False)

        stats = episode_stats_from_rows(self.current_rows)
        tasks = [task] if task is not None else []
        if not tasks and self.tasks:
            task_idx = int(self.current_rows[0]["task_index"])
            tasks = [self.tasks[task_idx]["task"]]
        episode_item = {"episode_index": ep_idx, "tasks": tasks, "length": len(self.current_rows)}
        self.episodes.append(episode_item)
        self.episodes_stats.append({"episode_index": ep_idx, "stats": stats})
        self.total_frames += len(self.current_rows)
        self.current_rows = []
        self.current_episode_index = None
        self.write_metadata()

    def data_path(self, episode_index: int) -> Path:
        return self.root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"

    def video_path(self, episode_index: int, video_key: str) -> Path:
        return (
            self.root
            / "videos"
            / f"chunk-{episode_index // 1000:03d}"
            / f"observation.images.{video_key}"
            / f"episode_{episode_index:06d}.mp4"
        )

    def build_info(self) -> dict[str, Any]:
        features = build_dataset_features(self.selected, self.cameras, use_videos=self.use_videos)
        info = {
            "codebase_version": "v2.1",
            "repo_id": self.repo_id,
            "robot_type": "diyrobot",
            "fps": int(self.fps),
            "total_episodes": int(len(self.episodes)),
            "total_frames": int(self.total_frames),
            "total_tasks": int(len(self.tasks)),
            "total_videos": int(len(self.episodes) * len(self.cameras)) if self.use_videos else 0,
            "total_chunks": int(max(1, (max(len(self.episodes), 1) + 999) // 1000)),
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
            if self.use_videos
            else None,
            "features": features,
        }
        for key, ft in info["features"].items():
            if ft["dtype"] == "video":
                ft["info"] = {
                    "video.height": int(ft["shape"][0]),
                    "video.width": int(ft["shape"][1]),
                    "video.codec": "mp4v",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": int(self.fps),
                    "video.channels": 3,
                    "has_audio": False,
                }
            else:
                ft["fps"] = int(self.fps)
        return info

    def write_metadata(self) -> None:
        write_jsonl(self.root / "meta/tasks.jsonl", self.tasks)
        write_jsonl(self.root / "meta/episodes.jsonl", self.episodes)
        write_jsonl(self.root / "meta/episodes_stats.jsonl", self.episodes_stats)
        stats = aggregate_numeric_stats([item["stats"] for item in self.episodes_stats])
        write_json(self.root / "meta/stats.json", stats)
        write_json(self.root / "meta/info.json", self.build_info())

    def finalize(self) -> None:
        if self._closed:
            return
        if self.current_episode_index is not None:
            for writer in self.current_video_writers.values():
                writer.release()
            self.current_video_writers.clear()
        self.write_metadata()
        self._closed = True


def add_writer_frame(
    dataset: LeRobotDataset | V21EpisodeWriter,
    *,
    selected: list[str],
    task: str,
    observation_positions: dict[str, float],
    action_positions: dict[str, float],
    camera_frames: dict[str, np.ndarray],
) -> None:
    if isinstance(dataset, V21EpisodeWriter):
        dataset.add_frame(
            task=task,
            observation_positions=observation_positions,
            action_positions=action_positions,
            camera_frames=camera_frames,
        )
        return
    add_dataset_frame(
        dataset,
        selected=selected,
        task=task,
        observation_positions=observation_positions,
        action_positions=action_positions,
        camera_frames=camera_frames,
    )


def save_writer_episode(dataset: LeRobotDataset | V21EpisodeWriter, task: str) -> None:
    if isinstance(dataset, V21EpisodeWriter):
        dataset.save_episode(task=task)
    else:
        dataset.save_episode()


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/diyrobot_pi05", help="LeRobot dataset repo_id.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/data/diyrobot/datasets/diyrobot_pi05"),
        help="Local dataset root directory.",
    )
    parser.add_argument("--resume", action="store_true", help="Append to an existing LeRobot dataset.")
    parser.add_argument(
        "--dataset-version",
        choices=("v2.1", "v3.0"),
        default="v2.1",
        help="On-disk LeRobot dataset format. Default is v2.1 for pi0.5 compatibility.",
    )
    parser.add_argument("--task", required=True, help="Natural-language task instruction saved per frame.")
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--episode-time-s", type=float, default=60.0)
    parser.add_argument("--reset-time-s", type=float, default=0.0)
    parser.add_argument("--manual", action="store_true", help="Keyboard control: SPACE starts episode, ENTER stops and saves.")
    parser.add_argument("--fps", type=int, default=20, help="Dataset/control recording FPS.")
    parser.add_argument(
        "--camera",
        action="append",
        default=[],
        help="Camera spec NAME=INDEX_OR_PATH:WIDTHxHEIGHT[:FPS], e.g. right_gripper=0:640x480:30.",
    )
    parser.add_argument("--no-video", action="store_true", help="Store image columns instead of encoded videos.")
    parser.add_argument("--vcodec", default="h264", help="Video codec passed to LeRobotDataset.")
    parser.add_argument("--streaming-encoding", action="store_true")
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--image-writer-processes", type=int, default=0)
    parser.add_argument("--image-writer-threads-per-camera", type=int, default=4)

    parser.add_argument("--kp", type=float, default=18.0)
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--max-step-deg", type=float, default=0.75)
    parser.add_argument("--max-catchup-step-deg", type=float, default=2.40)
    parser.add_argument("--wrist-max-step-deg", type=float, default=1.20)
    parser.add_argument("--wrist-max-catchup-step-deg", type=float, default=4.20)
    parser.add_argument("--gripper-max-step-deg", type=float, default=4.5)
    parser.add_argument("--gripper-max-catchup-step-deg", type=float, default=10.0)
    parser.add_argument("--catchup-start-error-deg", type=float, default=0.15)
    parser.add_argument("--catchup-full-error-deg", type=float, default=0.8)
    parser.add_argument("--min-range-width-deg", type=float, default=5.0)
    parser.add_argument("--startup-threshold-deg", type=float, default=7.0)
    parser.add_argument("--startup-range-slack-deg", type=float, default=0.5)
    parser.add_argument("--feedback-max-age-s", type=float, default=0.30)
    parser.add_argument("--feedback-retry-count", type=int, default=3)
    parser.add_argument("--feedback-retry-sleep-s", type=float, default=0.015)
    parser.add_argument("--active-feedback-motors", default="right_wrist_flex")
    parser.add_argument("--live-feedback-missing-grace-s", type=float, default=1.0)
    parser.add_argument("--follower-tx-min-gap-s", type=float, default=0.003)
    parser.add_argument("--hold-settle-s", type=float, default=0.2)
    parser.add_argument("--startup-hold-guard-s", type=float, default=1.2)
    parser.add_argument("--startup-hold-max-drift-deg", type=float, default=1.0)
    parser.add_argument("--allow-startup-low-disturbance-sample", action="store_true", default=True)
    parser.add_argument("--leader-port", default=teleop.LEADER_PORT)
    parser.add_argument("--left-follower-port", default=teleop.LEFT_FOLLOWER_PORT)
    parser.add_argument("--right-follower-port", default=teleop.RIGHT_FOLLOWER_PORT)
    parser.add_argument("--rest-range", type=Path, default=teleop.REST_RANGE_CALIB)
    parser.add_argument("--lerobot-calibration", type=Path, default=teleop.LEROBOT_CALIB)
    parser.add_argument("--motors", default=DEFAULT_MOTORS)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--log-decimate", type=int, default=10)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    skip_ep_checks = args.manual if hasattr(args, "manual") else False
    if not getattr(args, "manual", False) and args.num_episodes <= 0:
        raise RuntimeError("Refusing non-positive --num-episodes.")
    if not getattr(args, "manual", False) and args.episode_time_s <= 0:
        raise RuntimeError("Refusing non-positive --episode-time-s.")
    if args.reset_time_s < 0:
        raise RuntimeError("Refusing negative --reset-time-s.")
    if args.fps <= 0:
        raise RuntimeError("Refusing non-positive --fps.")
    if not args.camera:
        raise RuntimeError("Pi0.5 recording needs at least one --camera, e.g. --camera right_gripper=0:640x480:30.")
    if args.log_decimate <= 0:
        raise RuntimeError("Refusing non-positive --log-decimate.")
    if args.max_step_deg <= 0 or args.gripper_max_step_deg <= 0:
        raise RuntimeError("Refusing non-positive step limits.")
    if args.follower_tx_min_gap_s < 0:
        raise RuntimeError("Refusing negative --follower-tx-min-gap-s.")


def open_dataset(args: argparse.Namespace, selected: list[str], cameras: list[CameraSpec]) -> LeRobotDataset | V21EpisodeWriter:
    if args.dataset_version == "v2.1":
        return V21EpisodeWriter(
            root=args.root,
            repo_id=args.repo_id,
            fps=args.fps,
            selected=selected,
            cameras=cameras,
            use_videos=not args.no_video,
            resume=bool(args.resume),
        )

    if args.resume:
        dataset = LeRobotDataset(
            args.repo_id,
            root=args.root,
            batch_encoding_size=1,
            vcodec=args.vcodec,
            streaming_encoding=bool(args.streaming_encoding),
            encoder_threads=args.encoder_threads,
        )
        return dataset

    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        root=args.root,
        robot_type="diyrobot",
        features=build_dataset_features(selected, cameras, use_videos=not args.no_video),
        use_videos=not args.no_video,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads_per_camera * len(cameras),
        vcodec=args.vcodec,
        streaming_encoding=bool(args.streaming_encoding),
        encoder_threads=args.encoder_threads,
    )


def _set_cbreak():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    return old


def _reset_terminal(old):
    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old)


def _read_key(timeout_s=0.0):
    r, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if r:
        return sys.stdin.read(1)
    return None


def main() -> int:
    args = make_parser().parse_args()
    validate_args(args)
    cameras = [parse_camera_spec(item) for item in args.camera]

    leader_calibration = teleop.load_json_required(teleop.LEADER_CALIB)
    leader_rest_raw, follower_rest_deg, follower_ranges_abs = teleop.load_rest_range(args.rest_range)
    lerobot_calibration = teleop.load_json_optional(args.lerobot_calibration)

    left_motors, right_motors = teleop.build_dual_follower_motors()
    all_motors = {**left_motors, **right_motors}
    selected = teleop.parse_motor_selection(args.motors, all_motors)
    active_feedback_names = set(teleop.parse_active_feedback_motors(args.active_feedback_motors, all_motors))

    teleop.validate_rest_coverage(selected, leader_rest_raw, follower_rest_deg, follower_ranges_abs)
    teleop.validate_rest_ranges(selected, follower_rest_deg, follower_ranges_abs)
    teleop.validate_range_widths(selected, follower_ranges_abs, args.min_range_width_deg)
    calibration_audit = teleop.audit_calibration_consistency(
        selected,
        leader_calibration,
        follower_rest_deg,
        follower_ranges_abs,
        lerobot_calibration,
    )
    leader_mappings = teleop.build_leader_offset_mappings(
        selected,
        leader_calibration,
        leader_rest_raw,
        follower_rest_deg,
        follower_ranges_abs,
        lerobot_calibration,
    )

    teleop.print_header(selected, args, lerobot_calibration, calibration_audit)
    print(f"  dataset_root={args.root}")
    print(f"  dataset_repo_id={args.repo_id}")
    print(f"  dataset_version={args.dataset_version}")
    print("  cameras=" + ", ".join(f"{c.name}:{c.width}x{c.height}@{c.fps:g}" for c in cameras))

    leader_reader = teleop.LeaderReader(args.leader_port, leader_calibration)
    left_bus = None
    right_bus = None
    connected_buses = []
    log_handle = teleop.open_jsonl_log(args.log_file)
    camera_set = OpenCVCameraSet(cameras)
    dataset: LeRobotDataset | V21EpisodeWriter | None = None
    stop = {"value": False}

    def _stop(*_args):
        stop["value"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        print("Leader...")
        leader_reader.connect()
        leader_reader.assert_online(selected)
        print("  OK all online")

        left_names = [name for name in selected if name.startswith("left_")]
        right_names = [name for name in selected if name.startswith("right_")]
        if left_names:
            print(f"Follower LEFT ({args.left_follower_port})...")
            left_bus = teleop.RobStrideAtMotorsBus(
                port=args.left_follower_port,
                motors={name: left_motors[name] for name in left_names},
                min_tx_gap_s=args.follower_tx_min_gap_s,
            )
            left_bus.connect(handshake=False)
            left_bus.configure_motors()
            left_bus._zero_offsets_rad = {name: 0.0 for name in left_names}
            teleop.configure_active_feedback_reports(left_bus, left_names, left_motors, active_feedback_names, log_handle)
            connected_buses.append(left_bus)
            print(f"  OK {len(left_names)} motors")

        if right_names:
            print(f"Follower RIGHT ({args.right_follower_port})...")
            right_bus = teleop.RobStrideAtMotorsBus(
                port=args.right_follower_port,
                motors={name: right_motors[name] for name in right_names},
                min_tx_gap_s=args.follower_tx_min_gap_s,
            )
            right_bus.connect(handshake=False)
            right_bus.configure_motors()
            right_bus._zero_offsets_rad = {name: 0.0 for name in right_names}
            teleop.configure_active_feedback_reports(right_bus, right_names, right_motors, active_feedback_names, log_handle)
            connected_buses.append(right_bus)
            print(f"  OK {len(right_names)} motors")

        if not connected_buses:
            raise RuntimeError("No follower motors selected.")

        startup_bus_positions = teleop.sample_startup_positions(
            selected,
            left_bus,
            right_bus,
            left_motors,
            right_motors,
            allow_low_disturbance_sample=bool(args.allow_startup_low_disturbance_sample),
        )
        startup_leader_offsets, leader_unwrapped_raw, leader_zero_branches = teleop.read_leader_state(
            selected,
            leader_reader,
            leader_rest_raw,
        )
        startup_command_offsets = teleop.map_leader_offsets_to_follower_offsets(
            selected,
            startup_leader_offsets,
            leader_mappings,
            leader_unwrapped_raw=leader_unwrapped_raw,
        )
        startup_follower_offsets = teleop.compute_offsets_from_reference(
            selected,
            startup_bus_positions,
            follower_rest_deg,
            follower_ranges_abs,
            tolerance_deg=args.startup_range_slack_deg,
        )
        rows, bad = teleop.build_preflight_rows(
            selected,
            startup_leader_offsets,
            startup_command_offsets,
            startup_follower_offsets,
            args.startup_threshold_deg,
        )
        print("\nPRE-FLIGHT offsets from recorded zero\n")
        print("{:28s} {:>4s} {:>9s} {:>9s} {:>4s} {:>9s} {:>9s}".format(
            "Joint", "L-ID", "Leader", "CmdOfs", "F-ID", "Follower", "Err"))
        print("-" * 78)
        for row in rows:
            name = str(row["name"])
            l_id = leader_calibration[name]["id"]
            f_id = all_motors[name].id
            print("  {:26s} {:4d} {:+8.1f} {:+8.1f} {:4d} {:+8.1f} {:+8.1f}".format(
                name, l_id,
                float(row["leader_offset_deg"]),
                float(row["commanded_offset_deg"]),
                f_id,
                float(row["follower_offset_deg"]),
                float(row["error_deg"]),
            ))
        if bad:
            print("\nRefuse record: startup pose does not match recorded zero.\n")
            raise RuntimeError("Startup mismatch; stop before motion.")
        print("\nOK Pre-flight: startup pose matches recorded zero.\n")

        print("Cameras...")
        camera_set.open()
        print(f"  OK {len(cameras)} cameras")

        dataset = open_dataset(args, selected, cameras)
        teleop.warn_teleop_validation_gaps(lerobot_calibration, selected)
        teleop.enable_with_immediate_hold(
            selected,
            startup_bus_positions,
            left_bus,
            right_bus,
            lerobot_calibration,
            args.kp,
            args.kd,
            log_handle=log_handle,
        )

        live_anchor_positions = dict(startup_bus_positions)
        guard_start = time.monotonic()
        guard_frame_idx = 0
        while time.monotonic() - guard_start < args.startup_hold_guard_s:
            current_guard_positions = teleop.read_latest_frame_positions(
                selected,
                left_bus,
                right_bus,
                left_motors,
                right_motors,
                max_age_s=args.feedback_max_age_s,
                retry_count=args.feedback_retry_count,
                retry_sleep_s=args.feedback_retry_sleep_s,
            )
            if guard_frame_idx == 0:
                live_anchor_positions = dict(current_guard_positions)
                teleop.send_frame_targets(
                    selected,
                    live_anchor_positions,
                    left_bus,
                    right_bus,
                    lerobot_calibration,
                    args.kp,
                    args.kd,
                )
                guard_frame_idx += 1
                time.sleep(0.02)
                continue

            drifts = {
                name: current_guard_positions[name] - live_anchor_positions[name]
                for name in selected
            }
            if any(abs(value) > args.startup_hold_max_drift_deg for value in drifts.values()):
                raise RuntimeError("Startup hold drifted before recording; stop before motion.")
            teleop.send_frame_targets(
                selected,
                live_anchor_positions,
                left_bus,
                right_bus,
                lerobot_calibration,
                args.kp,
                args.kd,
            )
            guard_frame_idx += 1
            time.sleep(0.02)

        time.sleep(max(0.0, args.hold_settle_s))
        live_boot_positions = teleop.read_latest_frame_positions(
            selected,
            left_bus,
            right_bus,
            left_motors,
            right_motors,
            max_age_s=max(args.feedback_max_age_s, 0.50),
            retry_count=args.feedback_retry_count,
            retry_sleep_s=args.feedback_retry_sleep_s,
        )

        period = 1.0 / float(args.fps)
        commanded_bus_positions = dict(live_boot_positions)
        last_feedback_positions = dict(live_boot_positions)
        feedback_missing_since: dict[str, float] = {}
        if args.manual:
            print("\n" + "=" * 60)
            print("MANUAL EPISODE CONTROL")
            print("  SPACE = start recording episode")
            print("  ENTER = stop & save current episode")
            print("  Ctrl-C = exit")
            print("=" * 60)
            print("\nTeleop active.  Press SPACE to start recording...\n")

            old_term = _set_cbreak()
            recording = False
            episode_idx = 0
            frame_idx = 0
            just_ended = False

            while not stop["value"]:
                key = _read_key(timeout_s=0.0)

                if not recording:
                    if key == " ":
                        episode_idx += 1
                        recording = True
                        just_ended = False
                        frame_idx = 0
                        print(f"\nEpisode {episode_idx}: \u25cf RECORDING  (press ENTER to stop)")
                        if log_handle is not None:
                            teleop.write_log_event(log_handle, "episode_start", episode_index=int(episode_idx))
                else:
                    if key in ("\r", "\n"):
                        recording = False
                        just_ended = True

                # --- teleop control frame ---
                loop_start = time.monotonic()

                if episode_idx == 0 and not recording:
                    current_bus_positions = dict(live_boot_positions)
                    feedback_missing = []
                else:
                    fresh_positions, feedback_missing = teleop.try_read_latest_frame_positions(
                        selected, left_bus, right_bus, left_motors, right_motors,
                        max_age_s=max(args.feedback_max_age_s, 0.30),
                        retry_count=args.feedback_retry_count,
                        retry_sleep_s=args.feedback_retry_sleep_s,
                    )
                    now = time.monotonic()
                    last_feedback_positions.update(fresh_positions)
                    for name in selected:
                        if name in feedback_missing:
                            feedback_missing_since.setdefault(name, now)
                        else:
                            feedback_missing_since.pop(name, None)
                    stale_too_long = [
                        name for name in feedback_missing
                        if now - feedback_missing_since.get(name, now) > args.live_feedback_missing_grace_s
                    ]
                    if stale_too_long:
                        raise RuntimeError("Missing fresh follower feedback: " + ", ".join(stale_too_long))
                    current_bus_positions = dict(last_feedback_positions)

                leader_offsets, leader_unwrapped_raw, leader_zero_branches = teleop.read_leader_state(
                    selected, leader_reader, leader_rest_raw,
                    previous_unwrapped_raw=leader_unwrapped_raw,
                    zero_unwrapped_raw=leader_zero_branches,
                )
                current_command_offsets = teleop.map_leader_offsets_to_follower_offsets(
                    selected, leader_offsets, leader_mappings,
                    leader_unwrapped_raw=leader_unwrapped_raw,
                )

                next_bus_positions = {}
                target_bus_positions = {}
                for name in selected:
                    leader_delta = current_command_offsets[name] - startup_command_offsets[name]
                    prev_cmd_abs = commanded_bus_positions.get(name, current_bus_positions[name])
                    lo, hi = teleop.canonicalize_range_to_value(follower_ranges_abs[name], prev_cmd_abs)
                    lo, hi = teleop.expand_range_to_include_near_edge_value(
                        (lo, hi), prev_cmd_abs, slack_deg=args.startup_range_slack_deg,
                    )
                    target_abs = teleop.clamp(live_anchor_positions[name] + leader_delta, lo, hi)
                    step_limit = teleop.compute_live_step_limit_deg(
                        name=name,
                        tracking_error_deg=target_abs - prev_cmd_abs,
                        base_step_deg=args.max_step_deg,
                        catchup_step_deg=args.max_catchup_step_deg,
                        wrist_step_deg=args.wrist_max_step_deg,
                        wrist_catchup_step_deg=args.wrist_max_catchup_step_deg,
                        gripper_step_deg=args.gripper_max_step_deg,
                        gripper_catchup_step_deg=args.gripper_max_catchup_step_deg,
                        catchup_start_error_deg=args.catchup_start_error_deg,
                        catchup_full_error_deg=args.catchup_full_error_deg,
                    )
                    step = teleop.clamp(target_abs - prev_cmd_abs, -step_limit, step_limit)
                    next_abs = teleop.clamp(prev_cmd_abs + step, lo, hi)
                    if abs(next_abs - prev_cmd_abs) > step_limit + 1e-6:
                        raise RuntimeError(f"{name}: command step guard tripped.")
                    target_bus_positions[name] = target_abs
                    next_bus_positions[name] = next_abs

                camera_frames = camera_set.read()
                teleop.send_frame_targets(
                    selected, next_bus_positions, left_bus, right_bus,
                    lerobot_calibration, args.kp, args.kd,
                )

                if recording:
                    add_writer_frame(
                        dataset, selected=selected, task=args.task,
                        observation_positions=current_bus_positions,
                        action_positions=next_bus_positions,
                        camera_frames=camera_frames,
                    )
                    frame_idx += 1

                commanded_bus_positions = dict(next_bus_positions)

                if log_handle is not None and episode_idx > 0 and frame_idx % args.log_decimate == 0:
                    teleop.write_log_event(
                        log_handle, "record_frame",
                        episode_index=int(episode_idx),
                        frame_idx=int(frame_idx),
                        follower_current_abs_deg=teleop.float_dict(current_bus_positions),
                        follower_target_abs_deg=teleop.float_dict(target_bus_positions),
                        follower_command_abs_deg=teleop.float_dict(next_bus_positions),
                        feedback_missing=feedback_missing,
                        recording=recording,
                    )

                # Handle episode just ended
                if just_ended:
                    just_ended = False
                    if episode_idx > 0:
                        if frame_idx > 0:
                            print(f"\nSaving episode {episode_idx} ({frame_idx} frames)...")
                            save_writer_episode(dataset, args.task)
                            print("  saved")
                        else:
                            print(f"\nEpisode {episode_idx}: no frames, skip save")
                    if not stop["value"]:
                        print("\nPress SPACE for next episode, Ctrl-C to exit...\n")

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, period - elapsed))

            _reset_terminal(old_term)
        else:
            print("\nRECORDING - Ctrl-C to stop and save current episode\n")

            for episode_idx in range(args.num_episodes):
                if stop["value"]:
                    break
                print(f"Episode {episode_idx + 1}/{args.num_episodes}: recording {args.episode_time_s:.1f}s")
                episode_start = time.monotonic()
                frame_idx = 0
                while not stop["value"] and time.monotonic() - episode_start < args.episode_time_s:
                    loop_start = time.monotonic()
                    if frame_idx == 0 and episode_idx == 0:
                        current_bus_positions = dict(live_boot_positions)
                        feedback_missing = []
                    else:
                        fresh_positions, feedback_missing = teleop.try_read_latest_frame_positions(
                            selected,
                            left_bus,
                            right_bus,
                            left_motors,
                            right_motors,
                            max_age_s=max(args.feedback_max_age_s, 0.30),
                            retry_count=args.feedback_retry_count,
                            retry_sleep_s=args.feedback_retry_sleep_s,
                        )
                        now = time.monotonic()
                        last_feedback_positions.update(fresh_positions)
                        for name in selected:
                            if name in feedback_missing:
                                feedback_missing_since.setdefault(name, now)
                            else:
                                feedback_missing_since.pop(name, None)
                        stale_too_long = [
                            name
                            for name in feedback_missing
                            if now - feedback_missing_since.get(name, now) > args.live_feedback_missing_grace_s
                        ]
                        if stale_too_long:
                            raise RuntimeError("Missing fresh follower feedback: " + ", ".join(stale_too_long))
                        current_bus_positions = dict(last_feedback_positions)

                    leader_offsets, leader_unwrapped_raw, leader_zero_branches = teleop.read_leader_state(
                        selected,
                        leader_reader,
                        leader_rest_raw,
                        previous_unwrapped_raw=leader_unwrapped_raw,
                        zero_unwrapped_raw=leader_zero_branches,
                    )
                    current_command_offsets = teleop.map_leader_offsets_to_follower_offsets(
                        selected,
                        leader_offsets,
                        leader_mappings,
                        leader_unwrapped_raw=leader_unwrapped_raw,
                    )

                    next_bus_positions: dict[str, float] = {}
                    target_bus_positions: dict[str, float] = {}
                    for name in selected:
                        leader_delta = current_command_offsets[name] - startup_command_offsets[name]
                        prev_cmd_abs = commanded_bus_positions.get(name, current_bus_positions[name])
                        lo, hi = teleop.canonicalize_range_to_value(follower_ranges_abs[name], prev_cmd_abs)
                        lo, hi = teleop.expand_range_to_include_near_edge_value(
                            (lo, hi),
                            prev_cmd_abs,
                            slack_deg=args.startup_range_slack_deg,
                        )
                        target_abs = teleop.clamp(live_anchor_positions[name] + leader_delta, lo, hi)
                        step_limit = teleop.compute_live_step_limit_deg(
                            name=name,
                            tracking_error_deg=target_abs - prev_cmd_abs,
                            base_step_deg=args.max_step_deg,
                            catchup_step_deg=args.max_catchup_step_deg,
                            wrist_step_deg=args.wrist_max_step_deg,
                            wrist_catchup_step_deg=args.wrist_max_catchup_step_deg,
                            gripper_step_deg=args.gripper_max_step_deg,
                            gripper_catchup_step_deg=args.gripper_max_catchup_step_deg,
                            catchup_start_error_deg=args.catchup_start_error_deg,
                            catchup_full_error_deg=args.catchup_full_error_deg,
                        )
                        step = teleop.clamp(target_abs - prev_cmd_abs, -step_limit, step_limit)
                        next_abs = teleop.clamp(prev_cmd_abs + step, lo, hi)
                        if abs(next_abs - prev_cmd_abs) > step_limit + 1e-6:
                            raise RuntimeError(f"{name}: command step guard tripped.")
                        target_bus_positions[name] = target_abs
                        next_bus_positions[name] = next_abs

                    camera_frames = camera_set.read()
                    teleop.send_frame_targets(
                        selected,
                        next_bus_positions,
                        left_bus,
                        right_bus,
                        lerobot_calibration,
                        args.kp,
                        args.kd,
                    )
                    add_writer_frame(
                        dataset,
                        selected=selected,
                        task=args.task,
                        observation_positions=current_bus_positions,
                        action_positions=next_bus_positions,
                        camera_frames=camera_frames,
                    )
                    commanded_bus_positions = dict(next_bus_positions)
                    frame_idx += 1

                    if log_handle is not None and frame_idx % args.log_decimate == 0:
                        teleop.write_log_event(
                            log_handle,
                            "record_frame",
                            episode_index=int(episode_idx),
                            frame_idx=int(frame_idx),
                            follower_current_abs_deg=teleop.float_dict(current_bus_positions),
                            follower_target_abs_deg=teleop.float_dict(target_bus_positions),
                            follower_command_abs_deg=teleop.float_dict(next_bus_positions),
                            feedback_missing=feedback_missing,
                        )

                    elapsed = time.monotonic() - loop_start
                    time.sleep(max(0.0, period - elapsed))

                if frame_idx > 0:
                    print(f"Saving episode {episode_idx + 1} ({frame_idx} frames)...")
                    save_writer_episode(dataset, args.task)
                    print("  saved")
                if stop["value"]:
                    break

                if args.reset_time_s > 0 and episode_idx < args.num_episodes - 1:
                    print(f"Reset window {args.reset_time_s:.1f}s: teleop active, not recording")
                    reset_start = time.monotonic()
                    while not stop["value"] and time.monotonic() - reset_start < args.reset_time_s:
                        teleop.send_frame_targets(
                            selected,
                            commanded_bus_positions,
                            left_bus,
                            right_bus,
                            lerobot_calibration,
                            args.kp,
                            args.kd,
                        )
                        time.sleep(period)

    finally:
        print("\nStopping...")
        if dataset is not None:
            try:
                dataset.finalize()
            except Exception:
                pass
        for bus in connected_buses:
            try:
                bus.disable_torque()
            except Exception:
                pass
            try:
                bus.disconnect(disable_torque=False)
            except Exception:
                pass
        camera_set.close()
        leader_reader.disconnect()
        teleop.close_log_file(log_handle)
        print("Done.")

    if dataset is not None:
        print(f"Dataset ready: {dataset.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
