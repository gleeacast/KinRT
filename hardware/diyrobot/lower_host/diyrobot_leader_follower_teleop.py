#!/usr/bin/env python3
"""Leader-to-follower teleoperation for DIYRobot arms only.

"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode

from robstride_at_bus import RobStrideAtMotorsBus

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler


ROOT = Path(__file__).resolve().parent
LEADER_CALIB = ROOT / "leader_calibration.json"
FOLLOWER_CALIB = ROOT / "follower_calibration.json"
SESSION_ZERO = ROOT / "teleop_session_zero.json"
LEROBOT_CALIB = ROOT / "diyrobot_lerobot_calibration.json"

LEADER_PORT = "/dev/diyrobot/leader"
FOLLOWER_PORT = "/dev/diyrobot/follower"

ARM = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_pitch",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--kp", type=float, default=18.0)
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--max-step-deg", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=0.0, help="0 means run until interrupted.")
    parser.add_argument("--leader-port", default=LEADER_PORT)
    parser.add_argument("--follower-port", default=FOLLOWER_PORT)
    parser.add_argument("--session-zero", type=Path, default=SESSION_ZERO)
    parser.add_argument("--lerobot-calibration", type=Path, default=LEROBOT_CALIB)
    parser.add_argument(
        "--use-session-zero",
        action="store_true",
        help="Use relative leader/follower session zero instead of absolute LeRobot range mapping.",
    )
    parser.add_argument("--motors", default="", help="Comma separated motor names for cautious single/joint testing.")
    parser.add_argument("--dry-run", action="store_true", help="Read leader and print targets without enabling follower.")
    parser.add_argument(
        "--allow-motion",
        action="store_true",
        help="Required for real follower motion. Keep omitted for dry-run and calibration work.",
    )
    args = parser.parse_args()

    follower_motors = build_follower_motors()
    leader_reader = LeaderReader(args.leader_port, load_leader_calibration_raw())
    follower_ranges = load_follower_ranges_deg()
    session_zero = load_session_zero(args.session_zero)
    selected_motors = parse_motor_selection(args.motors, follower_motors)
    selected_follower_motors = {name: follower_motors[name] for name in selected_motors}
    lerobot_calibration = load_lerobot_calibration(args.lerobot_calibration, selected_motors)

    follower = RobStrideAtMotorsBus(port=args.follower_port, motors=selected_follower_motors)

    stop = {"value": False}
    follower_connected = {"value": False}

    def _stop(*_):
        stop["value"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        print("Connecting leader...")
        leader_reader.connect()
        leader_reader.assert_online(selected_motors)
        if args.dry_run:
            run_dry(args, leader_reader, follower_ranges, session_zero, selected_motors, stop, lerobot_calibration)
            return 0
        if not args.allow_motion:
            raise RuntimeError(
                "Real teleop is locked after the collision. Use dry-run or the joint mapping calibration first. "
                "Pass --allow-motion only after each selected joint has been validated."
            )
        print("Connecting follower...")
        follower.connect(handshake=True)
        follower_connected["value"] = True
        follower.configure_motors()
        follower.enable_torque()

        if not args.use_session_zero:
            assert_lerobot_teleop_validated(lerobot_calibration, selected_motors)

        current = canonicalize_positions(
            initial_follower_positions(follower, follower_motors, selected_motors),
            follower_ranges,
        )
        validate_start_pose(current, session_zero["follower_zero_deg"], follower_ranges, selected_motors)
        start = time.monotonic()
        period = 1.0 / args.fps
        print(
            f"Teleop started: fps={args.fps:.1f}, kp={args.kp:.1f}, kd={args.kd:.2f}, "
            f"max_step={args.max_step_deg:.2f} deg"
        )
        print("Press Ctrl-C to stop.")
        while not stop["value"]:
            loop_start = time.monotonic()
            leader_pos = leader_reader.read_positions(selected_motors)
            commands = {}
            targets = compute_targets(
                leader_pos,
                follower_ranges,
                session_zero,
                selected_motors,
                lerobot_calibration,
                use_lerobot_mapping=not args.use_session_zero,
            )
            for name in selected_motors:
                target = float(targets[name])
                lo, hi = follower_ranges.get(name, (-180.0, 180.0))
                target = min(max(target, lo), hi)
                now = float(current.get(name, target))
                delta = min(max(target - now, -args.max_step_deg), args.max_step_deg)
                next_pos = now + delta
                current[name] = next_pos
                kp, kd, torque = joint_gains(name, args.kp, args.kd, lerobot_calibration)
                commands[name] = (kp, kd, next_pos, 0.0, torque)
            follower._mit_control_batch(commands)

            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break
            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, period - elapsed))
    finally:
        print("Stopping teleop; disabling follower torque...")
        if follower_connected["value"]:
            try:
                follower.disable_torque()
            finally:
                follower.disconnect(disable_torque=False)
        leader_reader.disconnect()
    return 0


def initial_follower_positions(
    follower: RobStrideAtMotorsBus,
    motors: dict[str, Motor],
    selected_motors: list[str],
) -> dict[str, float]:
    current: dict[str, float] = {}
    missing = []
    for name in selected_motors:
        motor = motors[name]
        state = None
        for _ in range(12):
            state = follower.bus.wait_feedback(motor.id, timeout=0.35)
            if state is not None:
                break
            time.sleep(0.04)
        if state is None:
            missing.append(name)
        else:
            offset = follower._zero_offsets_rad.get(name, 0.0)
            current[name] = math.degrees(float(state["position_rad"]) - offset)
    if missing:
        raise RuntimeError(
            "Refuse to start teleop: no startup feedback for "
            + ", ".join(missing)
            + ". Do not guess follower pose."
        )
    return current


def build_follower_motors() -> dict[str, Motor]:
    motors = {}
    for i, name in enumerate(ARM, 1):
        motors[f"left_{name}"] = Motor(i, "O3", MotorNormMode.DEGREES)
    for i, name in enumerate(ARM, 11):
        motors[f"right_{name}"] = Motor(i, "O3", MotorNormMode.DEGREES)
    return motors


def load_leader_calibration_raw() -> dict:
    return json.loads(LEADER_CALIB.read_text())


def load_follower_ranges_deg() -> dict[str, tuple[float, float]]:
    raw = json.loads(FOLLOWER_CALIB.read_text())
    ranges = {}
    for name, item in raw.items():
        if isinstance(item.get("range_min"), (int, float)) and isinstance(item.get("range_max"), (int, float)):
            ranges[name] = (math.degrees(item["range_min"]), math.degrees(item["range_max"]))
    return ranges


def load_session_zero(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run record_teleop_session_zero.py with both arms in the teleop start pose first."
        )
    data = json.loads(path.read_text())
    if "leader_zero_raw" not in data or "follower_zero_deg" not in data:
        raise ValueError(f"Invalid session zero file: {path}")
    return data


def load_lerobot_calibration(path: Path, selected_motors: list[str]) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    missing = []
    for section in ("leader", "follower"):
        for name in selected_motors:
            entry = data.get(section, {}).get(name)
            if not entry or "middle_recorded_at_unix" not in entry or "range_recorded_at_unix" not in entry:
                missing.append(f"{section}.{name}")
    if missing:
        raise RuntimeError(
            f"Incomplete LeRobot calibration file {path}: " + ", ".join(missing)
        )
    return data


def assert_lerobot_teleop_validated(calibration: dict | None, selected_motors: list[str]) -> None:
    if calibration is None:
        raise RuntimeError("Missing diyrobot_lerobot_calibration.json; run LeRobot-style calibration before motion.")
    missing = []
    for section in ("leader", "follower"):
        for name in selected_motors:
            if not calibration.get(section, {}).get(name, {}).get("teleop_validated"):
                missing.append(f"{section}.{name}")
    if missing:
        raise RuntimeError(
            "Refuse real teleop: selected joints are calibrated but not teleop-validated: "
            + ", ".join(missing)
        )


def parse_motor_selection(value: str, motors: dict[str, Motor]) -> list[str]:
    if not value.strip():
        return list(motors)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in selected if name not in motors]
    if unknown:
        raise ValueError(f"Unknown motor selection: {unknown}")
    return selected


def validate_start_pose(
    current: dict[str, float],
    follower_zero_deg: dict[str, float],
    follower_ranges: dict[str, tuple[float, float]],
    selected_motors: list[str],
    threshold_deg: float = 360.0,
) -> None:
    bad = []
    for name in selected_motors:
        zero = follower_zero_deg[name]
        if name not in current:
            bad.append(f"{name}=missing")
            continue
        lo_hi = follower_ranges.get(name)
        zero_value = float(zero) if lo_hi is None else canonicalize_deg(float(zero), *lo_hi)
        error = abs(float(current[name]) - zero_value)
        if error > threshold_deg:
            bad.append(f"{name} err={error:.1f}deg")
    if bad:
        raise RuntimeError(
            "Refuse to start teleop: follower pose no longer matches recorded session zero: " + ", ".join(bad)
        )


def compute_targets(
    leader_raw: dict[str, int],
    follower_ranges: dict[str, tuple[float, float]],
    session_zero: dict,
    selected_motors: list[str],
    lerobot_calibration: dict | None = None,
    use_lerobot_mapping: bool = True,
) -> dict[str, float]:
    if use_lerobot_mapping and lerobot_calibration is not None:
        return compute_targets_lerobot(leader_raw, selected_motors, lerobot_calibration)
    leader_zero = session_zero["leader_zero_raw"]
    follower_zero = session_zero["follower_zero_deg"]
    targets = {}
    for name in selected_motors:
        lo, hi = follower_ranges.get(name, (-180.0, 180.0))
        delta_raw = unwrap_12bit_delta(int(leader_raw[name]) - int(leader_zero[name]))
        delta_deg = delta_raw * 360.0 / 4095.0
        delta_deg *= session_zero_direction_scale(name, lerobot_calibration)
        base = canonicalize_deg(float(follower_zero[name]), lo, hi)
        target = canonicalize_deg(base + delta_deg, lo, hi)
        targets[name] = min(max(target, lo), hi)
    return targets


def session_zero_direction_scale(name: str, calibration: dict | None) -> float:
    if calibration is None:
        return 1.0
    follower = calibration.get("follower", {}).get(name, {})
    scale = float(follower.get("teleop_scale", 1.0))
    if int(follower.get("drive_mode", 0)):
        scale *= -1.0
    return scale


def joint_gains(name: str, default_kp: float, default_kd: float, calibration: dict | None) -> tuple[float, float, float]:
    kp = float(default_kp)
    kd = float(default_kd)
    torque_nm = 2.0  # base gravity compensation for all follower joints
    follower = (calibration or {}).get("follower", {}).get(name, {})
    if name == "left_shoulder_lift":
        kp = max(kp, 120.0)
        kd = max(kd, 3.0)
        torque_nm = 6.0
    elif follower.get("teleop_needs_higher_stiffness"):
        kp = max(kp, 120.0)
        kd = max(kd, 2.0)
    if name == "left_elbow_flex":
        kp = max(kp, 160.0)
        kd = max(kd, 2.4)
    if name == "left_gripper":
        kp = min(kp, 15.0)
        kd = min(kd, 0.8)
        torque_nm = 0.0
    return kp, kd, torque_nm


def compute_targets_lerobot(
    leader_raw: dict[str, int],
    selected_motors: list[str],
    calibration: dict,
) -> dict[str, float]:
    targets = {}
    for name in selected_motors:
        leader = calibration["leader"][name]
        follower = calibration["follower"][name]
        l_min = float(leader["range_min"])
        l_max = float(leader["range_max"])
        if abs(l_max - l_min) < 1:
            raise RuntimeError(f"Invalid leader range for {name}: {l_min}..{l_max}")
        value = float(leader_raw[name])
        if int(leader.get("drive_mode", 0)):
            value = l_max - (value - l_min)
        ratio = (value - l_min) / (l_max - l_min)
        ratio = min(max(ratio, 0.0), 1.0)

        f_min = float(follower.get("range_min", -math.pi))
        f_max = float(follower.get("range_max", math.pi))
        if int(follower.get("drive_mode", 0)):
            ratio = 1.0 - ratio
        target_rad = f_min + ratio * (f_max - f_min)
        targets[name] = math.degrees(target_rad)
    return targets


def canonicalize_positions(values: dict[str, float], ranges: dict[str, tuple[float, float]]) -> dict[str, float]:
    out = {}
    for name, value in values.items():
        lo_hi = ranges.get(name)
        out[name] = float(value) if lo_hi is None else canonicalize_deg(float(value), *lo_hi)
    return out


def canonicalize_deg(value: float, lo: float, hi: float) -> float:
    candidates = [value + 360.0 * k for k in range(-4, 5)]
    inside = [candidate for candidate in candidates if lo <= candidate <= hi]
    if inside:
        return min(inside, key=lambda candidate: abs(candidate - value))
    mid = (lo + hi) / 2.0
    return min(candidates, key=lambda candidate: abs(candidate - mid))


def unwrap_12bit_delta(delta: int) -> int:
    if delta > 2048:
        return delta - 4096
    if delta < -2048:
        return delta + 4096
    return delta


def run_dry(
    args: argparse.Namespace,
    leader_reader: "LeaderReader",
    follower_ranges: dict[str, tuple[float, float]],
    session_zero: dict,
    selected_motors: list[str],
    stop: dict,
    lerobot_calibration: dict | None = None,
) -> None:
    period = 1.0 / args.fps
    start = time.monotonic()
    print(f"Dry run started: fps={args.fps:.1f}. Follower is not connected or enabled.")
    while not stop["value"]:
        loop_start = time.monotonic()
        leader_raw = leader_reader.read_raw_positions(selected_motors)
        targets = compute_targets(
            leader_raw,
            follower_ranges,
            session_zero,
            selected_motors,
            lerobot_calibration,
            use_lerobot_mapping=not args.use_session_zero,
        )
        print(json.dumps({k: round(v, 2) for k, v in targets.items()}, sort_keys=True))
        if args.duration > 0 and time.monotonic() - start >= args.duration:
            return
        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))


class LeaderReader:
    def __init__(self, port: str, calibration: dict) -> None:
        self.port_name = port
        self.calibration = calibration
        self.port = PortHandler(port)
        self.packet = PacketHandler(0)

    def connect(self) -> None:
        if not self.port.openPort():
            raise RuntimeError(f"Failed to open leader port {self.port_name}")
        if not self.port.setBaudRate(1_000_000):
            raise RuntimeError("Failed to set leader baudrate")

    def disconnect(self) -> None:
        self.port.closePort()

    def assert_online(self, selected_motors: list[str] | None = None) -> None:
        missing = []
        names = selected_motors or list(self.calibration)
        for name in names:
            item = self.calibration[name]
            mid = int(item["id"])
            _, result, _ = self.packet.ping(self.port, mid)
            if result != COMM_SUCCESS:
                missing.append(mid)
        if missing:
            raise RuntimeError(f"Leader motors missing: {missing}")

    def read_raw_positions(self, selected_motors: list[str] | None = None) -> dict[str, int]:
        out = {}
        names = selected_motors or list(self.calibration)
        for name in names:
            item = self.calibration[name]
            out[name] = self._read_raw_position(int(item["id"]))
        return out

    def read_positions(self, selected_motors: list[str] | None = None) -> dict[str, int]:
        return self.read_raw_positions(selected_motors)

    def _read_raw_position(self, motor_id: int) -> int:
        last = None
        for _ in range(6):
            value, result, _ = self.packet.read2ByteTxRx(self.port, motor_id, 56)
            if result == COMM_SUCCESS:
                return int(value & 0xFFF)
            last = result
            time.sleep(0.006)
        raise ConnectionError(f"Failed to read leader id={motor_id}: {self.packet.getTxRxResult(last)}")


if __name__ == "__main__":
    raise SystemExit(main())
