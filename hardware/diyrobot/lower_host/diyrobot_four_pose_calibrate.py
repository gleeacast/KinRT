#!/usr/bin/env python3
"""Record the four DIYRobot calibration poses.

Follower sampling may briefly enable the selected motor and send a zero-gain
probe before disabling it. Physical supervision is required even though the
tool does not issue a pose target.

The four poses are intentionally stored separately from the motor calibration
files:

- middle: mechanical middle/reference pose
- zero: robot zero/base pose for teleoperation
- rotation: direction/range sanity-check pose
- rest: safe parked pose
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode

from robstride_at_bus import RobStrideAtMotorsBus

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "diyrobot_calibration_poses.json"
SESSION_ZERO = ROOT / "teleop_session_zero.json"
LEADER_PORT = "/dev/diyrobot/leader"
FOLLOWER_PORT = "/dev/diyrobot/follower"

POSES = ("middle", "zero", "rotation", "rest")
ARMS = ("leader", "follower")
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_pitch",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
LEADER_IDS = {
    **{f"left_{name}": i + 1 for i, name in enumerate(ARM_JOINTS)},
    **{f"right_{name}": i + 8 for i, name in enumerate(ARM_JOINTS)},
}
FOLLOWER_IDS = {
    **{f"left_{name}": i + 1 for i, name in enumerate(ARM_JOINTS)},
    **{f"right_{name}": i + 11 for i, name in enumerate(ARM_JOINTS)},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Record one pose for leader, follower, or both.")
    record.add_argument("--arm", choices=(*ARMS, "both"), required=True)
    record.add_argument("--pose", choices=POSES, required=True)
    record.add_argument("--side", choices=("both", "left", "right"), default="both")
    record.add_argument("--out", type=Path, default=DEFAULT_OUT)
    record.add_argument("--leader-port", default=LEADER_PORT)
    record.add_argument("--follower-port", default=FOLLOWER_PORT)

    export_zero = sub.add_parser("export-session-zero", help="Export recorded zero pose for teleop startup.")
    export_zero.add_argument("--out", type=Path, default=DEFAULT_OUT)
    export_zero.add_argument("--session-zero", type=Path, default=SESSION_ZERO)

    show = sub.add_parser("show", help="Print recorded pose coverage.")
    show.add_argument("--out", type=Path, default=DEFAULT_OUT)

    args = parser.parse_args()
    if args.command == "record":
        return record_pose(args)
    if args.command == "export-session-zero":
        return export_session_zero(args.out, args.session_zero)
    if args.command == "show":
        return show_coverage(args.out)
    raise AssertionError(args.command)


def record_pose(args: argparse.Namespace) -> int:
    data = load_pose_file(args.out)
    backup(args.out)

    arms = ARMS if args.arm == "both" else (args.arm,)
    for arm in arms:
        print(f"Recording {arm} {args.pose} pose...", flush=True)
        if arm == "leader":
            pose = {
                "port": args.leader_port,
                "units": "raw_ticks_0_4095",
                "positions": read_leader_raw(args.leader_port, args.side),
            }
        else:
            pose = {
                "port": args.follower_port,
                "units": "deg_relative_to_follower_software_zero",
                "positions": read_follower_deg(args.follower_port, args.side),
            }
        pose["recorded_at_unix"] = time.time()
        pose["description"] = pose_description(args.pose)
        poses = data.setdefault("arms", {}).setdefault(arm, {})
        if args.side == "both" or args.pose not in poses:
            poses[args.pose] = pose
        else:
            existing = poses[args.pose]
            existing.setdefault("positions", {}).update(pose["positions"])
            existing["port"] = pose["port"]
            existing["units"] = pose["units"]
            existing["recorded_at_unix"] = pose["recorded_at_unix"]
            existing["description"] = pose["description"]
            poses[args.pose] = existing

    data["schema"] = "diyrobot_four_pose_calibration_v1"
    data["updated_at_unix"] = time.time()
    save_pose_file(args.out, data)
    print(f"Saved: {args.out}")
    return show_coverage(args.out)


def export_session_zero(pose_path: Path, session_zero_path: Path) -> int:
    data = load_pose_file(pose_path)
    try:
        leader = data["arms"]["leader"]["zero"]
        follower = data["arms"]["follower"]["zero"]
    except KeyError as exc:
        raise SystemExit(f"Cannot export session zero; missing zero pose field: {exc}") from exc

    session = {
        "saved_at_unix": time.time(),
        "source": str(pose_path),
        "leader_port": leader["port"],
        "follower_port": follower["port"],
        "leader_zero_raw": leader["positions"],
        "follower_zero_deg": follower["positions"],
    }
    backup(session_zero_path)
    session_zero_path.write_text(json.dumps(session, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved: {session_zero_path}")
    for name in LEADER_IDS:
        print(
            f"{name:24s} leader_raw={int(session['leader_zero_raw'][name]):4d} "
            f"follower_zero_deg={float(session['follower_zero_deg'][name]):+.3f}"
        )
    return 0


def show_coverage(path: Path) -> int:
    data = load_pose_file(path)
    print(f"Pose file: {path}")
    ok = True
    for arm in ARMS:
        recorded = data.get("arms", {}).get(arm, {})
        for pose in POSES:
            item = recorded.get(pose)
            count = len(item.get("positions", {})) if isinstance(item, dict) else 0
            ready = count == 14
            ok = ok and ready
            if ready:
                status = "OK"
            elif count > 0:
                status = f"PARTIAL {count}/14"
            else:
                status = "MISSING"
            print(f"{arm:8s} {pose:8s} {status}")
    return 0 if ok else 2


def read_leader_raw(port_name: str, side: str = "both") -> dict[str, int]:
    port = PortHandler(port_name)
    pkt = PacketHandler(0)
    if not port.openPort():
        raise RuntimeError(f"Failed to open {port_name}")
    if not port.setBaudRate(1_000_000):
        raise RuntimeError("Failed to set leader baudrate")
    try:
        out: dict[str, int] = {}
        for name, mid in selected_ids(LEADER_IDS, side).items():
            out[name] = read_leader_position(pkt, port, mid)
        return out
    finally:
        port.closePort()


def read_leader_position(pkt: PacketHandler, port: PortHandler, mid: int) -> int:
    last = None
    for _ in range(8):
        value, result, _ = pkt.read2ByteTxRx(port, mid, 56)
        if result == COMM_SUCCESS:
            return int(value & 0xFFF)
        last = result
        time.sleep(0.02)
    raise RuntimeError(f"Failed to read leader ID={mid}: {pkt.getTxRxResult(last)}")


def read_follower_deg(port: str, side: str = "both") -> dict[str, float]:
    ids = selected_ids(FOLLOWER_IDS, side)
    motors = {name: Motor(mid, "O3", MotorNormMode.DEGREES) for name, mid in ids.items()}
    bus = RobStrideAtMotorsBus(port=port, motors=motors)
    bus.connect(handshake=False)
    try:
        out: dict[str, float] = {}
        missing = []
        for name, motor in motors.items():
            print(f"Sampling follower {motor.id:2d} {name}...", flush=True)
            state = sample_follower_low_disturbance(bus, motor.id)
            if state is None:
                missing.append(name)
                continue
            offset = bus._zero_offsets_rad.get(name, 0.0)
            out[name] = math.degrees(float(state["position_rad"]) - offset)
        if missing:
            raise RuntimeError("Missing follower feedback: " + ", ".join(missing))
        return out
    finally:
        for motor in motors.values():
            bus.bus.disable(motor.id)
            time.sleep(0.01)
        bus.disconnect(disable_torque=False)


def sample_follower_low_disturbance(bus: RobStrideAtMotorsBus, motor_id: int):
    """Sample one motor while minimizing persistent RobStride responses.

    Calibration only needs a fresh feedback frame.  The old implementation used
    repeated enable + MIT commands and disabled all motors at the end; on this
    hardware that can leave one motor flooding the bus.  This path touches one
    motor at a time, prefers zero-gain MIT probes, and disables immediately.
    """
    for attempt in range(4):
        since = time.monotonic()
        state = bus.bus.enable(motor_id, timeout=0.30)
        if state is None:
            state = wait_fresh_feedback(bus, motor_id, since=since, timeout=0.20)
        if state is not None:
            bus.bus.disable(motor_id)
            time.sleep(0.03)
            return state

        for _ in range(8):
            since = time.monotonic()
            bus.bus.motion_control(
                motor_id,
                position_rad=0.0,
                velocity_rad_s=0.0,
                kp=0.0,
                kd=0.2,
                torque_nm=0.0,
            )
            state = wait_fresh_feedback(bus, motor_id, since=since, timeout=0.12)
            if state is not None:
                break
            time.sleep(0.02)
        bus.bus.disable(motor_id)
        time.sleep(0.06 + attempt * 0.04)
        if state is not None:
            return state
    return None


def wait_fresh_feedback(bus: RobStrideAtMotorsBus, motor_id: int, since: float, timeout: float):
    frame = bus.bus.wait_for(
        lambda f: f.comm_type == 2 and (f.area2 & 0xFF) == motor_id,
        timeout=timeout,
        since=since,
    )
    if frame is None:
        return None
    return bus.bus.latest_feedback(motor_id, max_age_s=1.0)


def selected_ids(ids: dict[str, int], side: str) -> dict[str, int]:
    if side == "both":
        return dict(ids)
    prefix = f"{side}_"
    return {name: motor_id for name, motor_id in ids.items() if name.startswith(prefix)}


def pose_description(pose: str) -> str:
    return {
        "middle": "Mechanical middle/reference pose; not a teleop target.",
        "zero": "Robot zero/base pose used as teleop start reference.",
        "rotation": "Direction and encoder sanity-check pose.",
        "rest": "Safe parked pose for shutdown or non-working state.",
    }[pose]


def load_pose_file(path: Path) -> dict:
    if not path.exists():
        return {"schema": "diyrobot_four_pose_calibration_v1", "arms": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_pose_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def backup(path: Path) -> None:
    if path.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        shutil.copy2(path, path.with_suffix(path.suffix + f".bak_{stamp}"))


if __name__ == "__main__":
    raise SystemExit(main())
