#!/usr/bin/env python3
"""LeRobot-style calibration flow for DIYRobot leader and follower arms.

This follows the LeRobot pattern:

1. Put the selected joints at the middle of their safe travel and record homing.
2. Move the joints through their safe travel while recording min/max range.
3. Export calibration files used by the DIYRobot driver.

The script does not issue position targets. RobStride feedback is sampled with
a brief enable/read/disable cycle because the current bus requires it for fresh
feedback. Physical supervision and a reachable power disconnect are required.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode

from robstride_at_bus import RobStrideAtMotorsBus

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "diyrobot_lerobot_calibration.json"
SPLIT_DIR = ROOT / "diyrobot_lerobot_calibrations"
LEADER_EXPORT = ROOT / "leader_calibration.json"
FOLLOWER_EXPORT = ROOT / "follower_calibration.json"

LEADER_PORT = "/dev/diyrobot/leader"
FOLLOWER_PORT = "/dev/diyrobot/follower"

JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_pitch",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
LEADER_IDS = {
    **{f"left_{joint}": i + 1 for i, joint in enumerate(JOINTS)},
    **{f"right_{joint}": i + 8 for i, joint in enumerate(JOINTS)},
}
FOLLOWER_IDS = {
    **{f"left_{joint}": i + 1 for i, joint in enumerate(JOINTS)},
    **{f"right_{joint}": i + 11 for i, joint in enumerate(JOINTS)},
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    middle = sub.add_parser("middle", help="Record middle pose and homing offsets.")
    add_common_args(middle)

    ranges = sub.add_parser("range", help="Record min/max while joints are moved by hand.")
    add_common_args(ranges)
    ranges.add_argument("--duration", type=float, default=20.0)
    ranges.add_argument("--period", type=float, default=0.12)

    show = sub.add_parser("show", help="Show LeRobot-style calibration coverage.")
    show.add_argument("--out", type=Path, default=OUT)

    export = sub.add_parser("export", help="Export DIYRobot leader/follower calibration files.")
    export.add_argument("--out", type=Path, default=OUT)
    export.add_argument("--leader-export", type=Path, default=LEADER_EXPORT)
    export.add_argument("--follower-export", type=Path, default=FOLLOWER_EXPORT)
    export.add_argument("--allow-partial", action="store_true", help="Export incomplete calibration entries.")

    split = sub.add_parser("split-paths", help="Print the four separated calibration file paths.")
    split.add_argument("--dir", type=Path, default=SPLIT_DIR)

    validate = sub.add_parser("validate", help="Mark selected joints as teleop validated after physical testing.")
    validate.add_argument("--out", type=Path, default=OUT)
    validate.add_argument("--device", choices=("leader", "follower", "both"), required=True)
    validate.add_argument("--side", choices=("left", "right", "both"), default="both")
    validate.add_argument("--motors", default="", help="Comma-separated full motor names. Overrides --side.")

    drive = sub.add_parser("drive-mode", help="Set drive_mode/inversion flag for selected joints.")
    drive.add_argument("--out", type=Path, default=OUT)
    drive.add_argument("--device", choices=("leader", "follower", "both"), required=True)
    drive.add_argument("--side", choices=("left", "right", "both"), default="both")
    drive.add_argument("--motors", default="", help="Comma-separated full motor names. Overrides --side.")
    drive.add_argument("--drive-mode", type=int, choices=(0, 1), required=True)

    args = parser.parse_args()
    if args.command == "middle":
        return record_middle(args)
    if args.command == "range":
        return record_range(args)
    if args.command == "show":
        return show_calibration(args.out)
    if args.command == "export":
        return export_calibration(args)
    if args.command == "split-paths":
        return print_split_paths(args.dir)
    if args.command == "validate":
        return validate_joints(args)
    if args.command == "drive-mode":
        return set_drive_mode(args)
    raise AssertionError(args.command)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("leader", "follower", "both"), required=True)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--motors", default="", help="Comma-separated full motor names. Overrides --side.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument(
        "--split-out-dir",
        type=Path,
        default=None,
        help="Store each device/side in a separate file under this directory.",
    )
    parser.add_argument("--leader-port", default=LEADER_PORT)
    parser.add_argument("--follower-port", default=FOLLOWER_PORT)


def record_middle(args: argparse.Namespace) -> int:
    if args.split_out_dir is not None:
        return record_middle_split(args)
    data = load(args.out)
    backup(args.out)
    for device in selected_devices(args.device):
        names = selected_names(device, args.side, args.motors)
        positions = read_positions(device, names, args.leader_port, args.follower_port)
        dev = data.setdefault(device, {})
        for name in names:
            entry = dev.setdefault(name, calibration_entry(device, name))
            if device == "leader":
                entry["homing_offset"] = int(positions[name]) - 2047
                entry["middle_raw"] = int(positions[name])
            else:
                entry["software_zero_rad"] = float(positions[name])
                entry["homing_offset_rad"] = float(positions[name])
                entry["middle_raw_rad"] = float(positions[name])
            entry["middle_recorded_at_unix"] = time.time()
    data["schema"] = "diyrobot_lerobot_calibration_v1"
    data["updated_at_unix"] = time.time()
    save(args.out, data)
    print(f"Saved middle homing: {args.out}")
    return show_calibration(args.out)


def record_range(args: argparse.Namespace) -> int:
    if args.split_out_dir is not None:
        return record_range_split(args)
    data = load(args.out)
    backup(args.out)
    deadline = time.monotonic() + float(args.duration)
    for device in selected_devices(args.device):
        names = selected_names(device, args.side, args.motors)
        print(f"Recording {device} ranges for {args.duration:.1f}s: {', '.join(names)}", flush=True)
        first = read_positions(device, names, args.leader_port, args.follower_port)
        mins = dict(first)
        maxes = dict(first)
        while time.monotonic() < deadline:
            positions = read_positions(device, names, args.leader_port, args.follower_port)
            for name, value in positions.items():
                mins[name] = min(float(mins[name]), float(value))
                maxes[name] = max(float(maxes[name]), float(value))
            print_range_table(names, positions, mins, maxes)
            time.sleep(max(0.02, float(args.period)))
        dev = data.setdefault(device, {})
        for name in names:
            entry = dev.setdefault(name, calibration_entry(device, name))
            if device == "leader":
                entry["range_min"] = int(round(mins[name]))
                entry["range_max"] = int(round(maxes[name]))
            else:
                zero = float(entry.get("software_zero_rad", 0.0))
                entry["range_min_raw_rad"] = float(mins[name])
                entry["range_max_raw_rad"] = float(maxes[name])
                entry["range_min"] = float(mins[name]) - zero
                entry["range_max"] = float(maxes[name]) - zero
            entry["range_recorded_at_unix"] = time.time()
    data["schema"] = "diyrobot_lerobot_calibration_v1"
    data["updated_at_unix"] = time.time()
    save(args.out, data)
    print(f"Saved ranges: {args.out}")
    return show_calibration(args.out)


def record_middle_split(args: argparse.Namespace) -> int:
    for device, side in split_targets(args.device, args.side, args.motors):
        out = split_path(args.split_out_dir, device, side)
        split_args = clone_args(args, device=device, side=side, out=out, split_out_dir=None)
        record_middle(split_args)
    return print_split_paths(args.split_out_dir)


def record_range_split(args: argparse.Namespace) -> int:
    for device, side in split_targets(args.device, args.side, args.motors):
        out = split_path(args.split_out_dir, device, side)
        split_args = clone_args(args, device=device, side=side, out=out, split_out_dir=None)
        record_range(split_args)
    return print_split_paths(args.split_out_dir)


def read_positions(device: str, names: list[str], leader_port: str, follower_port: str) -> dict[str, float | int]:
    if device == "leader":
        return read_leader_raw(leader_port, names)
    return read_follower_rad(follower_port, names)


def read_leader_raw(port_name: str, names: list[str]) -> dict[str, int]:
    port = PortHandler(port_name)
    pkt = PacketHandler(0)
    if not port.openPort():
        raise RuntimeError(f"Failed to open leader port {port_name}")
    if not port.setBaudRate(1_000_000):
        raise RuntimeError("Failed to set leader baudrate")
    try:
        out: dict[str, int] = {}
        for name in names:
            out[name] = read_leader_one(pkt, port, LEADER_IDS[name])
        return out
    finally:
        port.closePort()


def read_leader_one(pkt: PacketHandler, port: PortHandler, motor_id: int) -> int:
    last = None
    for _ in range(8):
        value, result, _ = pkt.read2ByteTxRx(port, motor_id, 56)
        if result == COMM_SUCCESS:
            return int(value & 0xFFF)
        last = result
        time.sleep(0.01)
    raise RuntimeError(f"Failed to read leader id={motor_id}: {pkt.getTxRxResult(last)}")


def read_follower_rad(port: str, names: list[str]) -> dict[str, float]:
    motors = {name: Motor(FOLLOWER_IDS[name], "O3", MotorNormMode.DEGREES) for name in names}
    bus = RobStrideAtMotorsBus(port=port, motors=motors)
    bus.bus.connect()
    try:
        out: dict[str, float] = {}
        for name, motor in motors.items():
            state = None
            for _ in range(5):
                state = bus.bus.enable(motor.id, timeout=0.18)
                bus.bus.disable(motor.id)
                if state is not None:
                    break
                time.sleep(0.03)
            if state is None:
                raise RuntimeError(f"No RobStride feedback for {name} id={motor.id}")
            out[name] = float(state["position_rad"])
        return out
    finally:
        for motor in motors.values():
            bus.bus.disable(motor.id)
            time.sleep(0.01)
        bus.bus.disconnect()


def calibration_entry(device: str, name: str) -> dict:
    ids = LEADER_IDS if device == "leader" else FOLLOWER_IDS
    if device == "leader":
        return {"id": ids[name], "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
    return {
        "id": ids[name],
        "drive_mode": 0,
        "software_zero_rad": 0.0,
        "range_min": -math.pi,
        "range_max": math.pi,
    }


def export_calibration(args: argparse.Namespace) -> int:
    data = load(args.out)
    if "leader" in data:
        require_complete(data, "leader", allow_partial=args.allow_partial)
        leader = {}
        for name, entry in data["leader"].items():
            leader[name] = {
                "id": int(entry["id"]),
                "drive_mode": int(entry.get("drive_mode", 0)),
                "homing_offset": int(entry.get("homing_offset", 0)),
                "range_min": int(entry.get("range_min", 0)),
                "range_max": int(entry.get("range_max", 4095)),
            }
        backup(args.leader_export)
        save(args.leader_export, leader)
        print(f"Exported leader: {args.leader_export}")
    if "follower" in data:
        require_complete(data, "follower", allow_partial=args.allow_partial)
        follower = {}
        for name, entry in data["follower"].items():
            follower[name] = {
                "id": int(entry["id"]),
                "online": True,
                "software_zero_rad": float(entry.get("software_zero_rad", 0.0)),
                "range_min": float(entry.get("range_min", -math.pi)),
                "range_max": float(entry.get("range_max", math.pi)),
                "range_min_raw_rad": float(entry.get("range_min_raw_rad", 0.0)),
                "range_max_raw_rad": float(entry.get("range_max_raw_rad", 0.0)),
                "lerobot_calibrated": True,
            }
        backup(args.follower_export)
        save(args.follower_export, follower)
        print(f"Exported follower: {args.follower_export}")
    return 0


def print_split_paths(directory: Path) -> int:
    for device in ("leader", "follower"):
        for side in ("left", "right"):
            print(f"{device}_{side}: {split_path(directory, device, side)}")
    return 0


def split_path(directory: Path, device: str, side: str) -> Path:
    return directory / f"{device}_{side}.json"


def split_targets(device_value: str, side_value: str, motors: str) -> list[tuple[str, str]]:
    if motors.strip():
        targets = []
        for device in selected_devices(device_value):
            sides = {name.split("_", 1)[0] for name in selected_names(device, side_value, motors)}
            targets.extend((device, side) for side in sorted(sides))
        return targets
    return [(device, side) for device in selected_devices(device_value) for side in selected_sides(side_value)]


def selected_sides(value: str) -> tuple[str, ...]:
    return ("left", "right") if value == "both" else (value,)


def clone_args(args: argparse.Namespace, **updates) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def validate_joints(args: argparse.Namespace) -> int:
    data = load(args.out)
    backup(args.out)
    for device in selected_devices(args.device):
        dev = data.setdefault(device, {})
        for name in selected_names(device, args.side, args.motors):
            entry = dev.setdefault(name, calibration_entry(device, name))
            require_entry_complete(device, name, entry)
            entry["teleop_validated"] = True
            entry["teleop_validated_at_unix"] = time.time()
    data["updated_at_unix"] = time.time()
    save(args.out, data)
    print(f"Validated selected joints in {args.out}")
    return show_calibration(args.out)


def set_drive_mode(args: argparse.Namespace) -> int:
    data = load(args.out)
    backup(args.out)
    for device in selected_devices(args.device):
        dev = data.setdefault(device, {})
        for name in selected_names(device, args.side, args.motors):
            entry = dev.setdefault(name, calibration_entry(device, name))
            entry["drive_mode"] = int(args.drive_mode)
            entry["drive_mode_updated_at_unix"] = time.time()
    data["updated_at_unix"] = time.time()
    save(args.out, data)
    print(f"Updated drive_mode={args.drive_mode} in {args.out}")
    return show_calibration(args.out)


def show_calibration(path: Path) -> int:
    data = load(path)
    print(f"Calibration file: {path}")
    ok = True
    for device in ("leader", "follower"):
        dev = data.get(device, {})
        for name in sorted(dev):
            entry = dev[name]
            has_middle = "middle_recorded_at_unix" in entry
            has_range = "range_recorded_at_unix" in entry
            validated = bool(entry.get("teleop_validated"))
            ok = ok and has_middle and has_range
            print(
                f"{device:8s} {name:24s} middle={'OK' if has_middle else '--'} "
                f"range={'OK' if has_range else '--'} drive={int(entry.get('drive_mode', 0))} "
                f"teleop={'OK' if validated else '--'}"
            )
    return 0 if ok else 2


def require_complete(data: dict, device: str, allow_partial: bool = False) -> None:
    if allow_partial:
        return
    missing = []
    for name, entry in sorted(data.get(device, {}).items()):
        try:
            require_entry_complete(device, name, entry)
        except RuntimeError as exc:
            missing.append(str(exc))
    if missing:
        raise RuntimeError("Refuse to export incomplete calibration:\n" + "\n".join(missing))


def require_entry_complete(device: str, name: str, entry: dict) -> None:
    if "middle_recorded_at_unix" not in entry:
        raise RuntimeError(f"{device} {name}: missing middle")
    if "range_recorded_at_unix" not in entry:
        raise RuntimeError(f"{device} {name}: missing range")


def print_range_table(names: list[str], positions: dict, mins: dict, maxes: dict) -> None:
    print("-------------------------------------------")
    print(f"{'NAME':<24} | {'MIN':>9} | {'POS':>9} | {'MAX':>9}")
    for name in names:
        print(f"{name:<24} | {float(mins[name]):>+9.3f} | {float(positions[name]):>+9.3f} | {float(maxes[name]):>+9.3f}")


def selected_devices(value: str) -> tuple[str, ...]:
    return ("leader", "follower") if value == "both" else (value,)


def selected_names(device: str, side: str, motors: str) -> list[str]:
    ids = LEADER_IDS if device == "leader" else FOLLOWER_IDS
    if motors.strip():
        names = [item.strip() for item in motors.split(",") if item.strip()]
    elif side == "both":
        names = list(ids)
    else:
        names = [name for name in ids if name.startswith(f"{side}_")]
    unknown = [name for name in names if name not in ids]
    if unknown:
        raise ValueError(f"Unknown {device} motors: {unknown}")
    return names


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def backup(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + f".bak_{int(time.time())}"))


if __name__ == "__main__":
    raise SystemExit(main())
