#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os
import threading
from pathlib import Path
import re


class PI0Session:

    def __init__(self, model):
        self.model = model
        self.instruction = None
        self.observation_window = None
        self.router_info_root_dir = None
        self.router_info_dir = None
        self.router_info_step = 0

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        result = self.model.infer(self.observation_window)
        router_info = result.pop("router_info", None)
        actions = result["actions"]
        if router_info is not None:
            self._save_router_info(
                router_info,
                actions=actions,
            )
        return actions

    def is_observation_window_empty(self):
        return self.observation_window is None

    def update_observation_window_rpc(self, obs):
        self.update_observation_window(obs["rgb"], obs["state"])

    def get_action_chunk(self):
        return self.get_action()[:self.model.pi0_step]

    def infer_action(self, obs):
        """Single-call inference for remote clients.
        obs: {"images": [cam_high, cam_right_wrist, cam_left_wrist],
              "state": state_vector,
              "instruction": language_string}
        """
        if self.observation_window is None:
            self.set_language(obs["instruction"])
        self.update_observation_window(obs["images"], obs["state"])
        return self.get_action_chunk()

    def get_router_info_dir(self):
        return None if self.router_info_root_dir is None else str(self.router_info_root_dir)

    def set_router_info_dir(self, router_info_dir):
        if router_info_dir is None:
            return self.get_router_info_dir()
        self.router_info_root_dir = Path(router_info_dir)
        self.router_info_root_dir.mkdir(parents=True, exist_ok=True)
        self.router_info_dir = self.router_info_root_dir
        self.router_info_step = 0
        print(f"recording router info to: {self.router_info_root_dir}")
        return str(self.router_info_root_dir)

    def start_router_info_episode(self, episode_info):
        if self.router_info_root_dir is None:
            return None

        episode_info = episode_info or {}
        episode = episode_info.get("episode")
        seed = episode_info.get("seed")
        prompt = episode_info.get("prompt", "")

        parts = []
        if episode is not None:
            parts.append(f"episode{episode}")
        else:
            parts.append("episode")
        if seed is not None:
            parts.append(f"seed{seed}")
        episode_dir_name = self._safe_path_name("_".join(parts))

        self.router_info_dir = self.router_info_root_dir / episode_dir_name
        self.router_info_dir.mkdir(parents=True, exist_ok=True)
        self.router_info_step = 0

        with open(self.router_info_dir / "prompt.txt", "w", encoding="utf-8") as f:
            f.write("" if prompt is None else str(prompt))
        with open(self.router_info_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(episode_info, f, ensure_ascii=False, indent=2)

        print(f"recording router info for episode to: {self.router_info_dir}")
        return str(self.router_info_dir)

    @staticmethod
    def _safe_path_name(name):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("._") or "episode"

    def _save_router_info(self, router_info, *, actions=None):
        if self.router_info_dir is None:
            return

        filename = f"infer_{self.router_info_step:08d}"
        self.router_info_step += 1
        output_path = self.router_info_dir / filename
        router_info = dict(router_info)
        if actions is not None:
            actions = np.asarray(actions)
            router_info["policy_action_chunk"] = actions[: self.model.pi0_step].astype(np.float32, copy=False)
        if self.model.router_info_compress:
            np.savez_compressed(output_path, **router_info)
        else:
            np.savez(output_path, **router_info)

    def reset_model(self):
        self.reset_obsrvationwindows()

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")


class PI0:

    def __init__(
        self,
        train_config_name,
        model_name,
        checkpoint_id,
        pi0_step,
        checkpoint_dir=None,
        record_router_info=False,
        router_info_dir=None,
        router_info_compress=True,
    ):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id
        self.record_router_info = self._as_bool(record_router_info)
        self.router_info_compress = self._as_bool(router_info_compress)

        checkpoint_dir = checkpoint_dir or "policy/pi05/checkpoints"
        if not os.path.isdir(os.path.join(checkpoint_dir, "assets")):
            checkpoint_dir = os.path.join(
                checkpoint_dir,
                self.train_config_name,
                self.model_name,
                str(self.checkpoint_id),
            )
        specified_path = os.path.join(checkpoint_dir, "assets")
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        config = _config.get_config(self.train_config_name)
        sample_kwargs = {"return_router_info": True} if self.record_router_info else None
        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            robotwin_repo_id=assets_id,
            sample_kwargs=sample_kwargs,
            )
        if self.record_router_info:
            print("router info recording enabled; waiting for eval session output dir")
        print("loading model success!")
        self.img_size = (224, 224)
        self.pi0_step = pi0_step
        self.infer_lock = threading.Lock()
        self._local_session = self.create_session()

    def set_router_info_dir(self, router_info_dir):
        return self._local_session.set_router_info_dir(router_info_dir)

    def start_router_info_episode(self, episode_info):
        return self._local_session.start_router_info_episode(episode_info)

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    def create_session(self):
        return PI0Session(self)

    def infer(self, observation_window):
        with self.infer_lock:
            return self.policy.infer(observation_window)

    @staticmethod
    def _as_bool(value):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "y", "on")
        return bool(value)

    # set language randomly
    def set_language(self, instruction):
        self._local_session.set_language(instruction)

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        self._local_session.update_observation_window(img_arr, state)

    def get_action(self):
        return self._local_session.get_action()

    def is_observation_window_empty(self):
        return self._local_session.is_observation_window_empty()

    def update_observation_window_rpc(self, obs):
        self._local_session.update_observation_window_rpc(obs)

    def get_action_chunk(self):
        return self._local_session.get_action_chunk()

    def infer_action(self, obs):
        return self._local_session.infer_action(obs)

    def get_router_info_dir(self):
        return self._local_session.get_router_info_dir()

    def reset_model(self):
        self.reset_obsrvationwindows()

    def reset_obsrvationwindows(self):
        self._local_session.reset_obsrvationwindows()

    @property
    def observation_window(self):
        return self._local_session.observation_window
