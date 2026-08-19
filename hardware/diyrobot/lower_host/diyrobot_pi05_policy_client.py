#!/usr/bin/env python3
"""Run a trained PI0.5/OpenPI policy against the DIYRobot lower host.

This client mirrors the pi0.5 recorder's observation contract:

- observation.state: 14 follower absolute joint positions in degrees
- observation.images.right_gripper / left_gripper / overhead: RGB images
- prompt: natural-language task instruction

The policy is expected to return 14 absolute follower joint targets in degrees
with the same joint order used during recording.

Safety defaults are intentionally conservative: the script connects and
preflights the arm, but it never sends motion unless --allow-motion is passed.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import importlib
import json
import signal
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np

import dual_arm_teleop_strict_v22 as teleop
from diyrobot_pi05_record import CameraSpec, OpenCVCameraSet, parse_camera_spec


DEFAULT_MOTORS = (
    "left_shoulder_pan,left_shoulder_lift,left_elbow_flex,left_wrist_pitch,left_wrist_flex,"
    "left_wrist_roll,left_gripper,right_shoulder_pan,right_shoulder_lift,right_elbow_flex,"
    "right_wrist_pitch,right_wrist_flex,right_wrist_roll,right_gripper"
)

DEFAULT_CAMERA_RIGHT_GRIPPER = (
    "right_gripper=/dev/diyrobot/camera-right-wrist:640x480:30"
)
DEFAULT_CAMERA_LEFT_GRIPPER = (
    "left_gripper=/dev/diyrobot/camera-left-wrist:640x480:30"
)
DEFAULT_CAMERA_OVERHEAD = "overhead=/dev/diyrobot/camera-overhead:640x480:30"
EXPECTED_CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class PolicyClient(Protocol):
    def infer(self, obs: dict[str, Any]) -> Any:
        ...


@dataclass
class PolicyResult:
    actions: np.ndarray
    raw: Any


class DIYRobotTcpPolicyClient:
    """Adapter for the legacy length-prefixed TCP JSON policy protocol."""

    def __init__(self, host: str, port: int, timeout_s: float = 60.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout_s)
        self.sock.connect((host, port))

    @staticmethod
    def _numpy_json_hook(dct: dict[str, Any]) -> Any:
        if "__numpy_array__" in dct:
            return np.frombuffer(
                base64.b64decode(dct["data"]), dtype=np.dtype(dct["dtype"])
            ).reshape(dct["shape"])
        return dct

    def _send_recv(self, payload: dict[str, Any]) -> Any:
        data = json.dumps(payload).encode("utf-8")
        self.sock.sendall(len(data).to_bytes(4, "big"))
        self.sock.sendall(data)
        len_data = self._recv_exact(4)
        size = int.from_bytes(len_data, "big")
        response = json.loads(
            self._recv_exact(size).decode("utf-8"), object_hook=self._numpy_json_hook
        )
        if "error" in response:
            raise RuntimeError(f"DIYRobot TCP policy server error: {response['error']}")
        return response.get("res", response)

    def _recv_exact(self, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        while remaining > 0:
            chunk = self.sock.recv(min(remaining, 65536))
            if not chunk:
                raise ConnectionError("DIYRobot TCP policy server closed the connection")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _encode_image(self, value: Any) -> str:
        import cv2

        arr = np.asarray(value)
        if arr.ndim != 3:
            raise RuntimeError(f"Expected image array with 3 dims, got shape {arr.shape}")
        # The legacy TCP server expects base64 JPEG images rather than CHW arrays.
        if arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("Failed to encode image for DIYRobot TCP policy server")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        state = obs.get("state", obs.get("observation.state"))
        if state is None:
            raise RuntimeError("DIYRobot TCP policy observation missing state")
        raw_images = obs.get("images", {})
        images = {name: self._encode_image(value) for name, value in raw_images.items()}
        raw_previous_images = obs.get("images_prev", {})
        previous_images = {
            name: self._encode_image(value)
            for name, value in raw_previous_images.items()
        }
        server_obs = {
            "state": np.asarray(state, dtype=np.float32).tolist(),
            "images": images,
            "images_prev": previous_images,
            "prompt": str(obs.get("prompt", "")),
        }
        return self._send_recv({"cmd": "predict", "obs": server_obs})


class NoopPolicyClient:
    """Policy stub for dry-run plumbing checks.

    It returns the current state as the action, so the safety layer should hold
    position if someone explicitly combines it with --allow-motion.
    """

    def infer(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        state = None
        for value in obs.values():
            arr = np.asarray(value)
            if arr.ndim == 1 and arr.size == 14:
                state = arr.astype(np.float32)
                break
        if state is None:
            raise RuntimeError("Noop policy could not find a 14-D state vector in the observation.")
        return {"actions": state[None, :]}


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-mode",
        choices=("openpi", "diyrobot_tcp", "noop"),
        default="openpi",
        help="Use diyrobot_tcp for the release length-prefixed TCP protocol.",
    )
    parser.add_argument(
        "--policy-server",
        default="",
        help=(
            "OpenPI policy server host:port or ws:// URL. Required with --policy-mode openpi. "
            "Example: 192.168.1.50:8000"
        ),
    )
    parser.add_argument("--task", default="pick up the object and place it at the target location")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--kp", type=float, default=18.0)
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--max-step-deg", type=float, default=0.75)
    parser.add_argument("--max-catchup-step-deg", type=float, default=2.40)
    parser.add_argument("--gripper-max-step-deg", type=float, default=4.5)
    parser.add_argument("--wrist-max-step-deg", type=float, default=1.20)
    parser.add_argument("--wrist-max-catchup-step-deg", type=float, default=4.20)
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
    parser.add_argument("--policy-timeout-s", type=float, default=0.50)
    parser.add_argument(
        "--action-chunk-steps",
        type=int,
        default=1,
        help="How many returned policy actions to execute before querying the server again. 1 is safest.",
    )
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument(
        "--image-key",
        action="append",
        default=[],
        help=(
            "Map camera names to policy observation keys, e.g. "
            "right_gripper=observation.images.right_gripper. Repeat for each camera."
        ),
    )
    parser.add_argument("--follower-tx-min-gap-s", type=float, default=0.003)
    parser.add_argument("--hold-settle-s", type=float, default=0.20)
    parser.add_argument("--startup-hold-guard-s", type=float, default=1.2)
    parser.add_argument("--startup-hold-max-drift-deg", type=float, default=1.0)
    parser.add_argument(
        "--allow-startup-low-disturbance-sample",
        action="store_true",
        help="Same as strict v22: briefly enable/disable only to read missing startup feedback.",
    )
    parser.add_argument("--duration", type=float, default=0.0, help="0 means run until interrupted.")
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--log-decimate", type=int, default=10)
    parser.add_argument("--leader-port", default=teleop.LEADER_PORT)
    parser.add_argument("--left-follower-port", default=teleop.LEFT_FOLLOWER_PORT)
    parser.add_argument("--right-follower-port", default=teleop.RIGHT_FOLLOWER_PORT)
    parser.add_argument("--rest-range", type=Path, default=teleop.REST_RANGE_CALIB)
    parser.add_argument("--lerobot-calibration", type=Path, default=teleop.LEROBOT_CALIB)
    parser.add_argument("--motors", default=DEFAULT_MOTORS)
    parser.add_argument("--camera", action="append", default=[])
    parser.add_argument("--skip-cameras", action="store_true", help="Only for dry policy plumbing checks.")
    parser.add_argument(
        "--offline-self-test",
        action="store_true",
        help="Validate policy parsing/clamping in memory only; do not open cameras or serial/CAN devices.",
    )
    parser.add_argument("--allow-motion", action="store_true", help="Required before any policy action is sent.")
    parser.add_argument(
        "--hold-left-arm",
        action="store_true",
        help="Keep all selected left_* follower joints at their live startup pose while executing policy actions.",
    )
    parser.add_argument(
        "--allow-infinite-live",
        action="store_true",
        help="Allow --allow-motion without --duration. Use with a reachable E-stop.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise RuntimeError("Refusing non-positive --fps.")
    if args.max_step_deg <= 0:
        raise RuntimeError("Refusing non-positive --max-step-deg.")
    if args.max_catchup_step_deg < 0:
        raise RuntimeError("Refusing negative --max-catchup-step-deg.")
    if args.gripper_max_step_deg <= 0:
        raise RuntimeError("Refusing non-positive --gripper-max-step-deg.")
    if args.wrist_max_step_deg < 0:
        raise RuntimeError("Refusing negative --wrist-max-step-deg.")
    if args.wrist_max_catchup_step_deg < 0:
        raise RuntimeError("Refusing negative --wrist-max-catchup-step-deg.")
    if args.gripper_max_catchup_step_deg < 0:
        raise RuntimeError("Refusing negative --gripper-max-catchup-step-deg.")
    if args.catchup_start_error_deg < 0:
        raise RuntimeError("Refusing negative --catchup-start-error-deg.")
    if args.catchup_full_error_deg <= 0:
        raise RuntimeError("Refusing non-positive --catchup-full-error-deg.")
    if args.catchup_full_error_deg < args.catchup_start_error_deg:
        raise RuntimeError("Refusing catch-up full error smaller than catch-up start error.")
    if args.feedback_retry_count < 0:
        raise RuntimeError("Refusing negative --feedback-retry-count.")
    if args.feedback_retry_sleep_s < 0:
        raise RuntimeError("Refusing negative --feedback-retry-sleep-s.")
    if args.live_feedback_missing_grace_s < 0:
        raise RuntimeError("Refusing negative --live-feedback-missing-grace-s.")
    if args.policy_timeout_s <= 0:
        raise RuntimeError("Refusing non-positive --policy-timeout-s.")
    if args.action_chunk_steps <= 0:
        raise RuntimeError("Refusing non-positive --action-chunk-steps.")
    if args.follower_tx_min_gap_s < 0:
        raise RuntimeError("Refusing negative --follower-tx-min-gap-s.")
    if args.log_decimate <= 0:
        raise RuntimeError("Refusing non-positive --log-decimate.")
    if args.allow_motion and args.duration <= 0 and not args.allow_infinite_live:
        raise RuntimeError("Refusing live motion without --duration. Pass --allow-infinite-live to override.")
    if args.policy_mode in ("openpi", "diyrobot_tcp") and not args.policy_server and not args.offline_self_test:
        raise RuntimeError("--policy-server is required with the selected network policy mode.")
    if args.skip_cameras and args.policy_mode != "noop" and not args.offline_self_test:
        raise RuntimeError("--skip-cameras is only allowed with --policy-mode noop.")
    if args.offline_self_test and args.allow_motion:
        raise RuntimeError("--offline-self-test cannot be combined with --allow-motion.")


def default_cameras_if_needed(values: list[str]) -> list[str]:
    if values:
        return values
    return [DEFAULT_CAMERA_RIGHT_GRIPPER, DEFAULT_CAMERA_LEFT_GRIPPER, DEFAULT_CAMERA_OVERHEAD]


def validate_camera_names(cameras: list[CameraSpec]) -> None:
    names = [spec.name for spec in cameras]
    missing = [name for name in EXPECTED_CAMERAS if name not in names]
    if missing:
        raise RuntimeError("Missing required cameras for pi0.5 policy: " + ", ".join(missing))


def parse_image_key_map(values: list[str], cameras: list[CameraSpec]) -> dict[str, str]:
    out = {spec.name: f"observation.images.{spec.name}" for spec in cameras}
    for value in values:
        if "=" not in value:
            raise RuntimeError(f"Bad --image-key {value!r}; expected CAMERA=OBS_KEY")
        name, key = value.split("=", 1)
        name = name.strip()
        key = key.strip()
        if not name or not key:
            raise RuntimeError(f"Bad --image-key {value!r}; expected CAMERA=OBS_KEY")
        out[name] = key
    missing = [spec.name for spec in cameras if spec.name not in out]
    if missing:
        raise RuntimeError("Missing image key mapping for: " + ", ".join(missing))
    return out


def parse_policy_server(value: str) -> tuple[str, int]:
    if "://" in value:
        parsed = urlparse(value)
        if not parsed.hostname:
            raise RuntimeError(f"Bad --policy-server {value!r}")
        return parsed.hostname, int(parsed.port or 8000)
    if ":" in value:
        host, port_s = value.rsplit(":", 1)
        return host, int(port_s)
    return value, 8000


def make_policy_client(args: argparse.Namespace) -> PolicyClient:
    if args.policy_mode == "noop":
        return NoopPolicyClient()

    host, port = parse_policy_server(args.policy_server)
    if args.policy_mode == "diyrobot_tcp":
        return DIYRobotTcpPolicyClient(host=host, port=port, timeout_s=args.policy_timeout_s)

    try:
        module = importlib.import_module("openpi_client.websocket_client_policy")
        cls = getattr(module, "WebsocketClientPolicy")
    except Exception as exc:
        raise RuntimeError(
            "openpi-client is not installed on the lower host. Install the OpenPI client package "
            "in the active virtual environment, or use --policy-mode noop for local plumbing tests."
        ) from exc

    return cls(host=host, port=port)



def _encode_jpeg(frame):
    """Encode OpenCV BGR frame to JPEG bytes."""
    import cv2
    import io
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()
def build_observation(
    *,
    selected: list[str],
    positions: dict[str, float],
    camera_frames: dict[str, np.ndarray],
    task: str,
    state_key: str,
    prompt_key: str,
    image_key_map: dict[str, str],
    previous_camera_frames: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    state = np.asarray([float(positions[name]) for name in selected], dtype=np.float32)
    def map_images(frames: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        images = {}
        for name, frame in frames.items():
            mapped = image_key_map.get(name, name)
            sub_key = mapped.rsplit(".", 1)[-1]
            images[sub_key] = frame.transpose(2, 0, 1)  # HWC -> CHW
        return images

    observation = {
        state_key: state,
        prompt_key: task,
        "images": map_images(camera_frames),
    }
    if previous_camera_frames is not None:
        observation["images_prev"] = map_images(previous_camera_frames)
    return observation
def extract_action(result: Any, selected: list[str]) -> PolicyResult:
    raw = result
    candidates: list[Any] = []
    if isinstance(result, dict):
        for key in ("actions", "action", "action_pred", "predicted_actions"):
            if key in result:
                candidates.append(result[key])
    else:
        candidates.append(result)

    for candidate in candidates:
        arr = np.asarray(candidate, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == len(selected):
            return PolicyResult(actions=arr[None, :], raw=raw)
        if arr.ndim >= 2 and arr.shape[-1] == len(selected):
            return PolicyResult(actions=arr.reshape(-1, len(selected)), raw=raw)

    shapes = []
    for candidate in candidates:
        try:
            shapes.append(tuple(np.asarray(candidate).shape))
        except Exception:
            shapes.append(type(candidate).__name__)
    raise RuntimeError(f"Policy response did not contain a {len(selected)}-D action. candidate_shapes={shapes}")


def infer_with_timeout(
    executor: concurrent.futures.Executor,
    policy: PolicyClient,
    obs: dict[str, Any],
    selected: list[str],
    timeout_s: float,
) -> PolicyResult:
    start = time.monotonic()
    future = executor.submit(policy.infer, obs)
    try:
        result = future.result(timeout=timeout_s)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise RuntimeError(f"Policy response stale: > {timeout_s:.3f}s") from exc
    elapsed = time.monotonic() - start
    if elapsed > timeout_s:
        raise RuntimeError(f"Policy response stale: {elapsed:.3f}s > {timeout_s:.3f}s")
    return extract_action(result, selected)


def clamp_policy_action(
    *,
    selected: list[str],
    policy_action: np.ndarray,
    prev_commanded_positions: dict[str, float],
    follower_ranges_abs: dict[str, tuple[float, float]],
    args: argparse.Namespace,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    raw_targets: dict[str, float] = {}
    clamped_targets: dict[str, float] = {}
    next_positions: dict[str, float] = {}
    for idx, name in enumerate(selected):
        raw_abs = float(policy_action[idx])
        prev_cmd_abs = prev_commanded_positions[name]
        lo, hi = teleop.canonicalize_range_to_value(follower_ranges_abs[name], prev_cmd_abs)
        lo, hi = teleop.expand_range_to_include_near_edge_value(
            (lo, hi),
            prev_cmd_abs,
            slack_deg=args.startup_range_slack_deg,
        )
        target_abs = teleop.clamp(raw_abs, lo, hi)
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
            raise RuntimeError(
                f"{name}: policy command step guard tripped; "
                f"prev={prev_cmd_abs:+.2f} next={next_abs:+.2f} "
                f"limit={step_limit:.2f} range=[{lo:+.2f},{hi:+.2f}]"
            )
        raw_targets[name] = raw_abs
        clamped_targets[name] = target_abs
        next_positions[name] = next_abs
    return raw_targets, clamped_targets, next_positions


def apply_hold_left_arm(
    policy_action: np.ndarray,
    selected: list[str],
    hold_targets: dict[str, float],
) -> np.ndarray:
    if not hold_targets:
        return policy_action
    held_action = np.array(policy_action, dtype=np.float32, copy=True)
    for idx, name in enumerate(selected):
        if name in hold_targets:
            held_action[idx] = float(hold_targets[name])
    return held_action


def run_offline_self_test(args: argparse.Namespace, selected: list[str], follower_rest_deg: dict[str, float]) -> int:
    policy = NoopPolicyClient()
    positions = {name: float(follower_rest_deg[name]) for name in selected}
    image_key_map: dict[str, str] = {}
    obs = build_observation(
        selected=selected,
        positions=positions,
        camera_frames={},
        task=args.task,
        state_key=args.state_key,
        prompt_key=args.prompt_key,
        image_key_map=image_key_map,
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        result = infer_with_timeout(executor, policy, obs, selected, args.policy_timeout_s)
    if result.actions.shape[-1] != len(selected):
        raise RuntimeError(f"Offline self-test action shape mismatch: {result.actions.shape}")
    print("OFFLINE SELF-TEST OK")
    print(f"  motors={len(selected)}")
    print(f"  action_shape={tuple(result.actions.shape)}")
    print("  no cameras opened")
    print("  no serial/CAN devices opened")
    return 0


def main() -> int:
    args = make_parser().parse_args()
    validate_args(args)
    cameras = [] if args.skip_cameras else [parse_camera_spec(item) for item in default_cameras_if_needed(args.camera)]
    if cameras:
        validate_camera_names(cameras)

    leader_calibration = teleop.load_json_required(teleop.LEADER_CALIB)
    leader_rest_raw, follower_rest_deg, follower_ranges_abs = teleop.load_rest_range(args.rest_range)
    lerobot_calibration = teleop.load_json_optional(args.lerobot_calibration)

    left_motors, right_motors = teleop.build_dual_follower_motors()
    all_motors = {**left_motors, **right_motors}
    selected = teleop.parse_motor_selection(args.motors, all_motors)
    active_feedback_names = set(teleop.parse_active_feedback_motors(args.active_feedback_motors, all_motors))

    teleop.validate_rest_coverage(selected, leader_rest_raw, follower_rest_deg, follower_ranges_abs)
    if args.offline_self_test:
        return run_offline_self_test(args, selected, follower_rest_deg)

    teleop.validate_rest_ranges(selected, follower_rest_deg, follower_ranges_abs)
    teleop.validate_range_widths(selected, follower_ranges_abs, args.min_range_width_deg)
    calibration_audit = teleop.audit_calibration_consistency(
        selected,
        leader_calibration,
        follower_rest_deg,
        follower_ranges_abs,
        lerobot_calibration,
    )

    teleop.print_header(selected, args, lerobot_calibration, calibration_audit)
    print(f"  policy_mode={args.policy_mode}")
    print(f"  policy_server={args.policy_server or '(none)'}")
    print(f"  task={args.task!r}")
    if cameras:
        print("  cameras=" + ", ".join(f"{c.name}:{c.width}x{c.height}@{c.fps:g}" for c in cameras))
    else:
        print("  cameras=skipped")

    leader_reader = None
    left_bus = None
    right_bus = None
    connected_buses = []
    log_handle = teleop.open_jsonl_log(args.log_file)
    camera_set = OpenCVCameraSet(cameras)
    image_key_map = parse_image_key_map(args.image_key, cameras)
    stop = {"value": False}

    def _stop(*_args):
        stop["value"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        teleop.write_log_event(
            log_handle,
            "policy_session_start",
            selected=selected,
            motion_allowed=bool(args.allow_motion),
            policy_mode=args.policy_mode,
            policy_server=args.policy_server,
            state_key=args.state_key,
            prompt_key=args.prompt_key,
            image_key_map=image_key_map,
            action_chunk_steps=int(args.action_chunk_steps),
            hold_left_arm=bool(args.hold_left_arm),
            fps=float(args.fps),
            duration_s=float(args.duration),
            max_step_deg=float(args.max_step_deg),
            follower_tx_min_gap_s=float(args.follower_tx_min_gap_s),
            calibration_audit=calibration_audit,
        )
        print("Policy...")
        policy = make_policy_client(args)
        print("  OK policy client ready")
        policy_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        if cameras:
            print("Cameras...")
            camera_set.open()
            print(f"  OK {len(cameras)} cameras")

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
            teleop.configure_active_feedback_reports(
                right_bus, right_names, right_motors, active_feedback_names, log_handle
            )
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
        startup_follower_offsets = teleop.compute_offsets_from_reference(
            selected,
            startup_bus_positions,
            follower_rest_deg,
            follower_ranges_abs,
            tolerance_deg=args.startup_range_slack_deg,
        )
        zero_offsets = {name: 0.0 for name in selected}
        rows, bad = teleop.build_preflight_rows(
            selected,
            zero_offsets,
            zero_offsets,
            startup_follower_offsets,
            args.startup_threshold_deg,
        )
        print("\nPRE-FLIGHT follower offsets from recorded zero\n")
        teleop.print_preflight_rows(rows)
        if bad:
            print("\nRefuse policy live: startup pose does not match recorded zero.\n")
            raise RuntimeError("Startup mismatch; stop before motion.")
        print("\nOK Pre-flight: startup pose matches recorded zero.\n")

        teleop.warn_teleop_validation_gaps(lerobot_calibration, selected)

        if not args.allow_motion:
            frames = camera_set.read() if cameras else {}
            obs = build_observation(
                selected=selected,
                positions=startup_bus_positions,
                camera_frames=frames,
                task=args.task,
                state_key=args.state_key,
                prompt_key=args.prompt_key,
                image_key_map=image_key_map,
            )
            result = infer_with_timeout(policy_executor, policy, obs, selected, args.policy_timeout_s)
            dry_hold_targets = (
                {name: float(startup_bus_positions[name]) for name in selected if name.startswith("left_")}
                if args.hold_left_arm
                else {}
            )
            dry_policy_action = apply_hold_left_arm(result.actions[0], selected, dry_hold_targets)
            raw_targets, clamped_targets, next_positions = clamp_policy_action(
                selected=selected,
                policy_action=dry_policy_action,
                prev_commanded_positions=startup_bus_positions,
                follower_ranges_abs=follower_ranges_abs,
                args=args,
            )
            print("\nDRY POLICY CHECK - no motor commands sent")
            if dry_hold_targets:
                print("  hold_left_arm=ON: left_* policy targets are pinned to startup pose")
            for name in selected:
                print(
                    f"  {name:28s} state={startup_bus_positions[name]:+8.2f} "
                    f"policy={raw_targets[name]:+8.2f} clamp={clamped_targets[name]:+8.2f} "
                    f"next={next_positions[name]:+8.2f}"
                )
            print("\nPass --allow-motion to command motors.")
            return 0

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

            drifts = {name: current_guard_positions[name] - live_anchor_positions[name] for name in selected}
            if any(abs(value) > args.startup_hold_max_drift_deg for value in drifts.values()):
                raise RuntimeError("Startup hold drifted before policy live; stop before motion.")
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
        hold_left_targets = (
            {name: float(live_boot_positions[name]) for name in selected if name.startswith("left_")}
            if args.hold_left_arm
            else {}
        )
        if hold_left_targets:
            print("  hold_left_arm=ON: selected left_* joints will be held at live startup pose")
            for name, value in hold_left_targets.items():
                print(f"    {name:28s} hold={value:+8.2f}")

        print("\nPOLICY LIVE - Ctrl-C to stop\n")
        start = time.monotonic()
        period = 1.0 / float(args.fps)
        frame_idx = 0
        commanded_positions = dict(live_boot_positions)
        last_feedback_positions = dict(live_boot_positions)
        feedback_missing_since: dict[str, float] = {}
        pending_actions: list[np.ndarray] = []
        previous_camera_frames: dict[str, np.ndarray] | None = None
        while not stop["value"]:
            loop_start = time.monotonic()
            feedback_missing: list[str] = []

            if frame_idx == 0:
                current_positions = dict(live_boot_positions)
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
                current_positions = dict(last_feedback_positions)

            frames = camera_set.read() if cameras else {}
            prior_frames = previous_camera_frames
            previous_camera_frames = {
                name: frame.copy() for name, frame in frames.items()
            }
            if not pending_actions:
                obs = build_observation(
                    selected=selected,
                    positions=current_positions,
                    camera_frames=frames,
                    task=args.task,
                    state_key=args.state_key,
                    prompt_key=args.prompt_key,
                    image_key_map=image_key_map,
                    previous_camera_frames=(
                        prior_frames if args.policy_mode == "diyrobot_tcp" else None
                    ),
                )
                policy_result = infer_with_timeout(policy_executor, policy, obs, selected, args.policy_timeout_s)
                pending_actions = [
                    np.asarray(action, dtype=np.float32)
                    for action in policy_result.actions[: args.action_chunk_steps]
                ]
                if not pending_actions:
                    raise RuntimeError("Policy returned an empty action chunk.")
            policy_action = pending_actions.pop(0)
            model_raw_targets: dict[str, float] | None = None
            if hold_left_targets:
                model_raw_targets, _, _ = clamp_policy_action(
                    selected=selected,
                    policy_action=policy_action,
                    prev_commanded_positions=commanded_positions,
                    follower_ranges_abs=follower_ranges_abs,
                    args=args,
                )
                policy_action = apply_hold_left_arm(policy_action, selected, hold_left_targets)
            raw_targets, clamped_targets, next_positions = clamp_policy_action(
                selected=selected,
                policy_action=policy_action,
                prev_commanded_positions=commanded_positions,
                follower_ranges_abs=follower_ranges_abs,
                args=args,
            )
            teleop.send_frame_targets(
                selected,
                next_positions,
                left_bus,
                right_bus,
                lerobot_calibration,
                args.kp,
                args.kd,
            )
            commanded_positions = dict(next_positions)
            frame_idx += 1

            if log_handle is not None and frame_idx % args.log_decimate == 0:
                teleop.write_log_event(
                    log_handle,
                    "policy_frame",
                    frame_idx=int(frame_idx),
                    t_since_live_start_s=float(time.monotonic() - start),
                    follower_current_abs_deg=teleop.float_dict(current_positions),
                    hold_left_arm=bool(args.hold_left_arm),
                    hold_left_targets_abs_deg=teleop.float_dict(hold_left_targets),
                    policy_model_raw_abs_deg=teleop.float_dict(model_raw_targets or raw_targets),
                    policy_raw_abs_deg=teleop.float_dict(raw_targets),
                    policy_clamped_abs_deg=teleop.float_dict(clamped_targets),
                    follower_command_abs_deg=teleop.float_dict(next_positions),
                    feedback_missing=feedback_missing,
                    pending_action_count=int(len(pending_actions)),
                )

            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, period - elapsed))

    except Exception as exc:
        teleop.write_log_event(log_handle, "policy_error", message=str(exc), stop_requested=bool(stop["value"]))
        raise
    finally:
        print("\nStopping...")
        teleop.write_log_event(log_handle, "policy_session_stop", stop_requested=bool(stop["value"]))
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
        if "policy_executor" in locals():
            policy_executor.shutdown(wait=False, cancel_futures=True)
        if leader_reader is not None:
            leader_reader.disconnect()
        teleop.close_log_file(log_handle)
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
