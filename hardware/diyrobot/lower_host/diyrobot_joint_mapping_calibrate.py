#!/usr/bin/env python3
"""Record per-joint DIYRobot mapping samples without position targets.

Follower sampling briefly enables and immediately disables the selected motor.
Physical supervision and a reachable power disconnect are required.

This replaces the unsafe practice of inferring teleop mapping from a few named
whole-arm poses.  Each joint is sampled independently at human-placed poses so
direction, scale, and wrap handling can be validated before real motion.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode

from diyrobot_four_pose_calibrate import (
    ARM_JOINTS,
    FOLLOWER_PORT,
    LEADER_PORT,
    read_leader_raw,
)
from robstride_at_bus import RobStrideAtMotorsBus


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "diyrobot_joint_mapping.json"
SIDES = ("left", "right")
SAMPLES = ("base", "positive", "negative")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Record one hand-placed joint sample.")
    record.add_argument("--side", choices=SIDES, required=True)
    record.add_argument("--joint", choices=ARM_JOINTS, required=True)
    record.add_argument("--sample", choices=SAMPLES, required=True)
    record.add_argument("--out", type=Path, default=DEFAULT_OUT)
    record.add_argument("--leader-port", default=LEADER_PORT)
    record.add_argument("--follower-port", default=FOLLOWER_PORT)

    show_parser = sub.add_parser("show", help="Show recorded coverage and inferred signs.")
    show_parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    args = parser.parse_args()
    if args.command == "record":
        return record_sample(args)
    if args.command == "show":
        return show(args.out)
    raise AssertionError(args.command)


def record_sample(args: argparse.Namespace) -> int:
    name = f"{args.side}_{args.joint}"
    data = load(args.out)
    backup(args.out)

    leader = read_leader_raw(args.leader_port, args.side)
    follower = read_one_follower_deg(args.follower_port, name)

    joint = data.setdefault("joints", {}).setdefault(name, {})
    joint.setdefault("samples", {})[args.sample] = {
        "recorded_at_unix": time.time(),
        "leader_raw": int(leader[name]),
        "follower_deg": float(follower),
    }
    joint["leader_port"] = args.leader_port
    joint["follower_port"] = args.follower_port
    data["schema"] = "diyrobot_joint_mapping_v1"
    data["updated_at_unix"] = time.time()
    save(args.out, data)

    print(f"Saved {args.sample}: {name} leader_raw={leader[name]} follower_deg={follower:+.3f}")
    return show(args.out, only=name)


def read_one_follower_deg(port: str, name: str) -> float:
    motor_id = follower_id(name)
    bus = RobStrideAtMotorsBus(port=port, motors={name: Motor(motor_id, "O3", MotorNormMode.DEGREES)})
    bus.bus.connect()
    try:
        state = None
        for _ in range(6):
            state = bus.bus.enable(motor_id, timeout=0.25)
            if state is not None:
                break
            time.sleep(0.03)
        if state is None:
            raise RuntimeError(f"No follower feedback for {name} id={motor_id}")
        offset = bus._zero_offsets_rad.get(name, 0.0)
        return math.degrees(float(state["position_rad"]) - offset)
    finally:
        bus.bus.disable(motor_id)
        bus.bus.disconnect()


def follower_id(name: str) -> int:
    side, joint = name.split("_", 1)
    index = ARM_JOINTS.index(joint)
    if side == "left":
        return index + 1
    if side == "right":
        return index + 11
    raise ValueError(name)


def show(path: Path, only: str | None = None) -> int:
    data = load(path)
    joints = data.get("joints", {})
    names = [only] if only else sorted(joints)
    print(f"Mapping file: {path}")
    for name in names:
        samples = joints.get(name, {}).get("samples", {})
        marks = " ".join(f"{sample}={'OK' if sample in samples else '--'}" for sample in SAMPLES)
        print(f"{name:24s} {marks}")
        if "base" in samples:
            base = samples["base"]
            for sample in ("positive", "negative"):
                if sample not in samples:
                    continue
                item = samples[sample]
                leader_delta = unwrap_12bit(int(item["leader_raw"]) - int(base["leader_raw"]))
                follower_delta = float(item["follower_deg"]) - float(base["follower_deg"])
                sign = 1 if leader_delta * follower_delta >= 0 else -1
                print(
                    f"  {sample:8s} leader_delta={leader_delta:+5d} "
                    f"follower_delta={follower_delta:+8.3f} sign={sign:+d}"
                )
    return 0


def unwrap_12bit(delta: int) -> int:
    if delta > 2048:
        return delta - 4096
    if delta < -2048:
        return delta + 4096
    return delta


def load(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def backup(path: Path) -> None:
    if path.exists():
        shutil.copy2(path, path.with_name(path.name + f".bak_{int(time.time())}"))


if __name__ == "__main__":
    raise SystemExit(main())
