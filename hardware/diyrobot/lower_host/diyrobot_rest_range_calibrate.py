#!/usr/bin/env python3
"""Rest-pose and range calibration for DIYRobot.

This intentionally avoids "middle" and "zero" terminology:

- rest: the safe startup/teleop alignment pose.
- range: min/max positions observed while the joint is moved by hand.

The script does not issue position targets. Follower sampling may briefly
enable and immediately disable one motor to obtain fresh feedback, so physical
supervision and a reachable power disconnect are still required.
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
OUT = ROOT / "diyrobot_rest_range_calibration.json"
SPLIT_DIR = ROOT / "diyrobot_rest_range_calibrations"
TELEOP_ZERO = ROOT / "teleop_session_zero.json"
FOLLOWER_EXPORT = ROOT / "follower_calibration.json"
LEROBOT_EXPORT = ROOT / "diyrobot_lerobot_calibration.json"

LEADER_PORT = "/dev/diyrobot/leader"
LEFT_FOLLOWER_PORT = "/dev/diyrobot/follower-left"
RIGHT_FOLLOWER_PORT = "/dev/diyrobot/follower-right"

ARM = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_pitch",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SIDES = ("left", "right")


def joint_names(side: str) -> list[str]:
    if side == "both":
        return [f"{s}_{j}" for s in SIDES for j in ARM]
    return [f"{side}_{j}" for j in ARM]


def leader_id(name: str) -> int:
    side, joint = name.split("_", 1)
    base = ARM.index(joint) + 1
    return base if side == "left" else base + 7


def follower_id(name: str) -> int:
    side, joint = name.split("_", 1)
    base = ARM.index(joint) + 1
    return base if side == "left" else base + 10


def load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema": "diyrobot_rest_range_calibration_v1",
        "created_at_unix": time.time(),
        "leader": {},
        "follower": {},
        "notes": {
            "rest": "Safe startup and teleop alignment pose.",
            "range": "Observed safe manual min/max around rest.",
            "wrist_mapping": "ID4=wrist_pitch, ID5=wrist_flex, ID6=wrist_roll.",
        },
    }


def save_json(path: Path, data: dict) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + f".bak_{int(time.time())}"))
    data["updated_at_unix"] = time.time()
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_save(data: dict, split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for device in ("leader", "follower"):
        for side in SIDES:
            names = joint_names(side)
            payload = {
                "schema": data["schema"],
                "updated_at_unix": data.get("updated_at_unix", time.time()),
                device: {name: data.get(device, {}).get(name, {}) for name in names if name in data.get(device, {})},
            }
            path = split_dir / f"{device}_{side}.json"
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_leader(names: list[str], port_name: str) -> dict[str, int]:
    port = PortHandler(port_name)
    pkt = PacketHandler(0)
    if not port.openPort():
        raise RuntimeError(f"Failed to open leader port {port_name}")
    if not port.setBaudRate(1_000_000):
        raise RuntimeError("Failed to set leader baudrate")
    try:
        out = {}
        for name in names:
            mid = leader_id(name)
            last = None
            for _ in range(8):
                value, result, _ = pkt.read2ByteTxRx(port, mid, 56)
                if result == COMM_SUCCESS:
                    out[name] = int(value & 0xFFF)
                    break
                last = result
                time.sleep(0.01)
            else:
                raise RuntimeError(f"Failed to read leader {name} id={mid}: {pkt.getTxRxResult(last)}")
        return out
    finally:
        port.closePort()


def follower_port_for(name: str) -> str:
    return LEFT_FOLLOWER_PORT if name.startswith("left_") else RIGHT_FOLLOWER_PORT


def read_follower(names: list[str]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(follower_port_for(name), []).append(name)
    out: dict[str, dict[str, float]] = {}
    for port, group in grouped.items():
        motors = {name: Motor(follower_id(name), "O3", MotorNormMode.DEGREES) for name in group}
        bus = RobStrideAtMotorsBus(port=port, motors=motors)
        bus.connect(handshake=True)
        try:
            # Rest/range calibration must be independent of any older software
            # zero offsets, so sample the follower in its raw absolute angle frame.
            bus._zero_offsets_rad = {name: 0.0 for name in group}
            states = bus.sync_read_all_states(group)
            for name, state in states.items():
                out[name] = {
                    "position_deg": float(state["position"]),
                    "velocity_deg_s": float(state["velocity"]),
                    "torque_nm": float(state["torque"]),
                    "temperature_c": float(state["temperature"]),
                    "fault": int(state["fault"]),
                    "mode_status": int(state["mode_status"]),
                }
        finally:
            bus.disconnect(disable_torque=False)
    return out


def record_rest(args: argparse.Namespace) -> int:
    names = parse_names(args)
    data = load_json(args.out)
    now = time.time()
    if args.device in ("leader", "both"):
        raw = read_leader(names, args.leader_port)
        for name, value in raw.items():
            data["leader"].setdefault(name, {})
            data["leader"][name].update({"id": leader_id(name), "rest_raw": value, "rest_recorded_at_unix": now})
    if args.device in ("follower", "both"):
        states = read_follower(names)
        for name, state in states.items():
            data["follower"].setdefault(name, {})
            data["follower"][name].update(
                {
                    "id": follower_id(name),
                    "rest_deg": state["position_deg"],
                    "rest_recorded_at_unix": now,
                    "last_state": state,
                }
            )
    save_json(args.out, data)
    split_save(data, args.split_out_dir)
    print_summary(data, names)
    return 0


def record_range(args: argparse.Namespace) -> int:
    names = parse_names(args)
    data = load_json(args.out)
    if args.device == "leader":
        raise RuntimeError("Range recording is intended for follower safety limits. Use --device follower or both.")
    end = time.monotonic() + args.duration
    mins = {name: math.inf for name in names}
    maxs = {name: -math.inf for name in names}
    samples = 0
    print(f"Recording follower ranges for {args.duration:.1f}s. Move joints by hand through safe limits.")
    while time.monotonic() < end:
        states = read_follower(names)
        for name, state in states.items():
            value = state["position_deg"]
            mins[name] = min(mins[name], value)
            maxs[name] = max(maxs[name], value)
        samples += 1
        time.sleep(args.period)
    now = time.time()
    for name in names:
        if math.isfinite(mins[name]) and math.isfinite(maxs[name]):
            data["follower"].setdefault(name, {})
            data["follower"][name].update(
                {
                    "id": follower_id(name),
                    "range_min_deg": mins[name],
                    "range_max_deg": maxs[name],
                    "range_recorded_at_unix": now,
                    "range_samples": samples,
                }
            )
    save_json(args.out, data)
    split_save(data, args.split_out_dir)
    print_summary(data, names)
    return 0


def export(args: argparse.Namespace) -> int:
    data = load_json(args.out)
    names = parse_names(args)
    missing = []
    for name in names:
        if "rest_raw" not in data.get("leader", {}).get(name, {}):
            missing.append(f"leader.{name}.rest")
        f = data.get("follower", {}).get(name, {})
        if "rest_deg" not in f:
            missing.append(f"follower.{name}.rest")
        if "range_min_deg" not in f or "range_max_deg" not in f:
            missing.append(f"follower.{name}.range")
    if missing:
        raise RuntimeError("Cannot export; missing: " + ", ".join(missing))

    teleop = {
        "saved_at_unix": time.time(),
        "scope": "rest_range_calibration",
        "leader_port": args.leader_port,
        "left_follower_port": args.left_follower_port,
        "right_follower_port": args.right_follower_port,
        "leader_zero_raw": {name: data["leader"][name]["rest_raw"] for name in names},
        "follower_zero_deg": {name: data["follower"][name]["rest_deg"] for name in names},
    }
    save_json(args.teleop_zero, teleop)

    if args.update_follower_calibration:
        follower = load_json(args.follower_export) if args.follower_export.exists() else {}
        for name in names:
            rest_deg = float(data["follower"][name]["rest_deg"])
            lo_deg = float(data["follower"][name]["range_min_deg"])
            hi_deg = float(data["follower"][name]["range_max_deg"])
            follower.setdefault(name, {})
            follower[name].update(
                {
                    "id": follower_id(name),
                    "software_zero_rad": math.radians(rest_deg),
                    "range_min": math.radians(lo_deg - rest_deg),
                    "range_max": math.radians(hi_deg - rest_deg),
                    "rest_based_calibration": True,
                    "rest_deg": rest_deg,
                    "range_min_abs_deg": lo_deg,
                    "range_max_abs_deg": hi_deg,
                }
            )
        save_json(args.follower_export, follower)
    print(f"Exported {args.teleop_zero}")
    if args.update_follower_calibration:
        print(f"Updated {args.follower_export}")
    return 0


def show(args: argparse.Namespace) -> int:
    data = load_json(args.out)
    print_summary(data, parse_names(args))
    return 0


def parse_names(args: argparse.Namespace) -> list[str]:
    if args.motors:
        return [item.strip() for item in args.motors.split(",") if item.strip()]
    return joint_names(args.side)


def print_summary(data: dict, names: list[str]) -> None:
    for name in names:
        l = data.get("leader", {}).get(name, {})
        f = data.get("follower", {}).get(name, {})
        bits = [
            f"{name:20s}",
            f"L.id={l.get('id', '-')}",
            f"L.rest={l.get('rest_raw', '-')}",
            f"F.id={f.get('id', '-')}",
            f"F.rest={fmt(f.get('rest_deg'))}",
            f"F.range=[{fmt(f.get('range_min_deg'))}, {fmt(f.get('range_max_deg'))}]",
        ]
        print("  ".join(bits))


def fmt(value) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):+.2f}"
    return "-"


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("leader", "follower", "both"), default="both")
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--motors", default="", help="Comma-separated full motor names; overrides --side.")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--split-out-dir", type=Path, default=SPLIT_DIR)
    parser.add_argument("--leader-port", default=LEADER_PORT)
    parser.add_argument("--left-follower-port", default=LEFT_FOLLOWER_PORT)
    parser.add_argument("--right-follower-port", default=RIGHT_FOLLOWER_PORT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rest = sub.add_parser("rest", help="Record rest pose.")
    add_common(rest)
    ranges = sub.add_parser("range", help="Record follower safe ranges by hand.")
    add_common(ranges)
    ranges.add_argument("--duration", type=float, default=20.0)
    ranges.add_argument("--period", type=float, default=0.15)
    exp = sub.add_parser("export", help="Export rest as teleop zero and optionally follower ranges.")
    add_common(exp)
    exp.add_argument("--teleop-zero", type=Path, default=TELEOP_ZERO)
    exp.add_argument("--follower-export", type=Path, default=FOLLOWER_EXPORT)
    exp.add_argument("--update-follower-calibration", action="store_true")
    show_parser = sub.add_parser("show", help="Show current rest/range coverage.")
    add_common(show_parser)
    args = parser.parse_args()

    if args.command == "rest":
        return record_rest(args)
    if args.command == "range":
        return record_range(args)
    if args.command == "export":
        return export(args)
    if args.command == "show":
        return show(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
