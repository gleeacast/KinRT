import sys
import os
import subprocess

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *


import sys
import os
import subprocess
import socket
import json
import csv
import threading
import time
import random
import traceback
import yaml
from datetime import datetime
import importlib
import argparse
from pathlib import Path
from collections import deque

import numpy as np
import json
from typing import Any

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

import numpy as np
import json
from typing import Any
import base64

class NumpyEncoder(json.JSONEncoder):
    """Enhanced json encoder for numpy types with array reconstruction info"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if str(obj.dtype) == "bfloat16":
                obj = obj.astype(np.float32)
            if obj.dtype == np.float32:
                dtype = 'float32'
            elif obj.dtype == np.float64:
                dtype = 'float64'
            elif obj.dtype == np.int32:
                dtype = 'int32'
            elif obj.dtype == np.int64:
                dtype = 'int64'
            else:
                dtype = str(obj.dtype)
            
            return {
                '__numpy_array__': True,
                'data': base64.b64encode(obj.tobytes()).decode('ascii'),
                'dtype': dtype,
                'shape': obj.shape
            }
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def numpy_to_json(data: Any) -> str:
    """Convert numpy-containing data to JSON string with reconstruction info"""
    return json.dumps(data, cls=NumpyEncoder)

def json_to_numpy(json_str: str) -> Any:
    """Convert JSON string back to Python objects with numpy arrays"""
    def object_hook(dct):
        if '__numpy_array__' in dct:
            data = base64.b64decode(dct['data'])
            dtype = dct['dtype']
            if dtype == 'bfloat16':
                import ml_dtypes

                return np.frombuffer(data, dtype=ml_dtypes.bfloat16).astype(np.float32).reshape(dct['shape'])
            return np.frombuffer(data, dtype=dtype).reshape(dct['shape'])
        return dct
    
    return json.loads(json_str, object_hook=object_hook)

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name, conda_env=None):
    # conda_env is abandoned
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e


def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

class ModelClient:
    def __init__(self, host='localhost', port=9999, timeout=30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock = None
        self._connect()

    def _connect(self):
        attempts = 0
        max_attempts = 1000
        retry_delay = 5
        
        while attempts < max_attempts:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.settimeout(self.timeout)
                self.sock.connect((self.host, self.port))
                print(f"🔗 Connected to model server at {self.host}:{self.port}")
                return
            except Exception as e:
                attempts += 1
                if self.sock:
                    self.sock.close()
                if attempts < max_attempts:
                    print(f"⚠️ Connection attempt {attempts} failed: {str(e)}")
                    print(f"🔄 Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    raise ConnectionError(
                        f"Failed to connect to server after {max_attempts} attempts: {str(e)}"
                    )

    def _send_recv(self, data):
        """Send request and receive response with numpy array support"""
        try:
            # Serialize with numpy support
            json_data = numpy_to_json(data).encode('utf-8')
            
            # Send data length and data
            self.sock.sendall(len(json_data).to_bytes(4, 'big'))
            self.sock.sendall(json_data)
            
            # Receive and deserialize response
            response = self._recv_response()
            return response
            
        except Exception as e:
            self.close()
            raise ConnectionError(f"Communication error: {str(e)}")

    def _recv_response(self):
        """Receive response with numpy array reconstruction"""
        # Read response length
        len_data = self.sock.recv(4)
        if not len_data:
            raise ConnectionError("Connection closed by server")
        
        size = int.from_bytes(len_data, 'big')
        
        # Read complete response
        chunks = []
        received = 0
        while received < size:
            chunk = self.sock.recv(min(size - received, 4096))
            if not chunk:
                raise ConnectionError("Incomplete response received")
            chunks.append(chunk)
            received += len(chunk)
        
        # Deserialize with numpy reconstruction
        return json_to_numpy(b''.join(chunks).decode('utf-8'))

    def call(self, func_name=None, obs=None):
        response = self._send_recv({"cmd": func_name, "obs": obs})
        if "error" in response:
            raise RuntimeError(response["error"] + "\n" + response.get("traceback", ""))
        return response['res']

    def close(self):
        """Close the connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
            finally:
                self.sock = None
                print("🔌 Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    host = usr_args.get("model_server_host") or usr_args.get("host", "localhost")
    port = usr_args["port"]
    save_dir = None
    video_save_dir = None
    video_size = None
    eval_result_root = Path(usr_args.get("eval_result_root", "eval_result"))

    policy_conda_env = usr_args.get("policy_conda_env", None)

    get_model = eval_function_decorator(policy_name, "get_model", conda_env=policy_conda_env)

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = eval_result_root / task_name / policy_name / task_config / ckpt_setting / current_time
    save_dir.mkdir(parents=True, exist_ok=True)
    router_info_root = usr_args.get("router_info_root")
    if router_info_root:
        router_info_save_dir = (
            Path(router_info_root)
            / task_name
            / policy_name
            / task_config
            / ckpt_setting
            / current_time
            / "router_info"
        )
    else:
        router_info_save_dir = save_dir / "router_info"

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    args["record_router_info"] = usr_args.get("record_router_info", False)
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = usr_args.get("test_num", 100)
    topk = 1

    file_path = os.path.join(save_dir, "_result.txt")
    episode_fieldnames = [
        "episode",
        "seed",
        "success",
        "success_count",
        "episode_count",
        "success_rate",
        "episode_eval_time_s",
        "cumulative_eval_time_s",
        "take_action_cnt",
        "step_lim",
        "router_info_episode_dir",
    ]
    model = ModelClient(host=host, port=port, timeout=300)
    router_info_dir = None
    try:
        if args["record_router_info"]:
            router_info_dir = model.call(func_name="set_router_info_dir", obs=str(router_info_save_dir))
        else:
            router_info_dir = model.call(func_name="get_router_info_dir")
    except Exception as e:
        print(f"Failed to query or set the router telemetry directory: {e}")

    with open(file_path, "w", newline="") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write(f"Eval Result Dir: {save_dir}\n")
        if router_info_dir is not None:
            file.write(f"Router Info Dir: {router_info_dir}\n")
        file.write("\nEpisode Metrics CSV\n")
        writer = csv.DictWriter(file, fieldnames=episode_fieldnames)
        writer.writeheader()

    st_seed, suc_num, episode_records = eval_policy(
        task_name,
        TASK_ENV,
        args,
        model,
        st_seed,
        test_num=test_num,
        video_size=video_size,
        instruction_type=instruction_type,
        policy_conda_env=policy_conda_env,
        result_file_path=file_path,
        episode_fieldnames=episode_fieldnames,
    )
    suc_nums.append(suc_num)

    # Emit eval_done only after every requested episode has produced a result record.
    # server_eval_loop.sh treats EVAL_DONE as the authoritative completion signal.
    if len(episode_records) == test_num:
        try:
            model.call(func_name="eval_done")
        except Exception:
            pass  # The server may close the connection immediately after acknowledging completion.
    model.close()

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    with open(file_path, "a") as file:
        file.write("\nFinal Summary\n")
        file.write(f"Final Success Rate: {suc_num}/{test_num} = {suc_num / test_num:.6f}\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("TopK Success Rates\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))
        file.write("\n")

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                policy_conda_env=None,
                result_file_path=None,
                episode_fieldnames=None):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval", conda_env=policy_conda_env)

    now_seed = st_seed
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]
    episode_records = []
    eval_start_time = time.perf_counter()

    args["eval_mode"] = True

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                print(" -------------")
                print("Error: ", e)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(" -------------")
                print("Error: ", stack_trace)
                print(" -------------")
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        # Isolate inference failures to the current episode. A transport failure terminates
        # evaluation because no further inference is possible; other failures count as a failed
        # episode and evaluation proceeds with the next seed.
        connection_lost = False
        succ = False
        video_started = False
        router_info_episode_dir = None
        episode_eval_start_time = time.perf_counter()
        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])
            TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

            episode_router_info = {
                "episode": TASK_ENV.test_num + 1,
                "seed": now_seed,
                "task_name": args["task_name"],
                "task_config": args["task_config"],
                "instruction_type": instruction_type,
                "prompt": str(instruction),
            }
            try:
                router_info_episode_dir = model.call(
                    func_name="start_router_info_episode", obs=episode_router_info
                )
            except Exception as e:
                if args.get("record_router_info", False):
                    print(f"Failed to start router telemetry for this episode: {e}")

            if TASK_ENV.eval_video_path is not None:
                ffmpeg = subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        video_size,
                        "-framerate",
                        "10",
                        "-i",
                        "-",
                        "-pix_fmt",
                        "yuv420p",
                        "-vcodec",
                        "libx264",
                        "-crf",
                        "23",
                        f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                    ],
                    stdin=subprocess.PIPE,
                )
                TASK_ENV._set_eval_video_ffmpeg(ffmpeg)
                video_started = True

            model.call(func_name='reset_model')
            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                observation = TASK_ENV.get_obs()
                eval_func(TASK_ENV, model, observation)
                if TASK_ENV.eval_success:
                    succ = True
                    break

        except ConnectionError as e:
            print(f"Model server connection lost during episode {succ_seed}: {e}")
            print("Stopping evaluation because inference is no longer available.")
            connection_lost = True

        except Exception as e:
            print(f"Evaluation failed during episode {succ_seed}; recording a failure and continuing:")
            print(traceback.format_exc())

        if video_started:
            try:
                TASK_ENV._del_eval_video_ffmpeg()
            except Exception:
                pass

        episode_eval_time_s = time.perf_counter() - episode_eval_start_time

        # Record metrics and release simulation resources even when the episode fails.
        # task_total_reward += TASK_ENV.episode_score
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        try:
            TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        except Exception:
            pass

        if TASK_ENV.render_freq:
            try:
                TASK_ENV.viewer.close()
            except Exception:
                pass

        TASK_ENV.test_num += 1
        success_rate = TASK_ENV.suc / TASK_ENV.test_num
        episode_record = {
            "episode": TASK_ENV.test_num,
            "seed": now_seed,
            "success": int(succ),
            "success_count": TASK_ENV.suc,
            "episode_count": TASK_ENV.test_num,
            "success_rate": round(success_rate, 6),
            "episode_eval_time_s": round(episode_eval_time_s, 3),
            "cumulative_eval_time_s": round(time.perf_counter() - eval_start_time, 3),
            "take_action_cnt": getattr(TASK_ENV, "take_action_cnt", 0),
            "step_lim": getattr(TASK_ENV, "step_lim", 0),
            "router_info_episode_dir": router_info_episode_dir or "",
        }
        episode_records.append(episode_record)
        if result_file_path is not None:
            with open(result_file_path, "a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=episode_fieldnames or episode_record.keys())
                writer.writerow(episode_record)

        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(success_rate*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m, episode eval time: \033[90m{episode_eval_time_s:.2f}s\033[0m\n"
        )
        now_seed += 1

        # Exit after recording the interrupted episode when the transport is unavailable.
        if connection_lost:
            break

    return now_seed, TASK_ENV.suc, episode_records


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.host is not None:
        config["host"] = args.host
    if args.port is not None:
        config["port"] = args.port

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
