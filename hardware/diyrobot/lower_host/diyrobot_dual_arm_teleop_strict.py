#!/usr/bin/env python3
"""Strict dual-arm leader-to-follower teleoperation for DIYRobot.

This rebuild fails closed:

- preflight is mandatory before motion
- startup mismatch is a hard stop, not a warning
- angle canonicalization is strict and never guesses another circle
- the first live command after torque enable is a hold-at-current command
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
import time
from pathlib import Path
from typing import TextIO

from lerobot.motors import Motor, MotorNormMode

from robstride_at_bus import RobStrideAtMotorsBus

from scservo_sdk import COMM_SUCCESS, PacketHandler, PortHandler


ROOT = Path(__file__).resolve().parent
LEADER_CALIB = ROOT / "leader_calibration.json"
REST_RANGE_CALIB = ROOT / "diyrobot_rest_range_calibration.json"
LEROBOT_CALIB = ROOT / "diyrobot_lerobot_calibration.json"

LEADER_PORT = "/dev/diyrobot/leader"
LEFT_FOLLOWER_PORT = "/dev/diyrobot/follower-left"
RIGHT_FOLLOWER_PORT = "/dev/diyrobot/follower-right"

ARM = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_pitch",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

EXTRA_SIGN_FLIPS = {
    "right_shoulder_lift",
    "right_wrist_flex",
    "right_gripper",
}
EXTRA_SCALES = {
    "right_gripper": 2.0,
}
# Near-full-turn leader ranges are ambiguous around encoder wrap, so only
# use range-ratio mapping when the recorded span is clearly bounded away from
# the wrap seam. Otherwise fall back to the older rest-relative offset map.
LEADER_RANGE_MAX_SAFE_WIDTH_DEG = 320.0
LEADER_RANGE_MIN_EDGE_MARGIN_DEG = 20.0
LEADER_RANGE_MAX_EDGE_ASYMMETRY = 3.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--kp", type=float, default=18.0)
    parser.add_argument("--kd", type=float, default=0.8)
    parser.add_argument("--max-step-deg", type=float, default=0.3)
    parser.add_argument(
        "--max-catchup-step-deg",
        type=float,
        default=0.0,
        help=(
            "Optional larger per-frame step cap used only when the follower is "
            "far behind the target. 0 disables catch-up scaling."
        ),
    )
    parser.add_argument(
        "--gripper-max-step-deg",
        type=float,
        default=1.2,
        help="Per-frame step limit for grippers. Defaults higher than arm joints so gripper motion is visible.",
    )
    parser.add_argument(
        "--wrist-max-step-deg",
        type=float,
        default=0.0,
        help="Per-frame step limit for wrist joints. 0 reuses --max-step-deg.",
    )
    parser.add_argument(
        "--wrist-max-catchup-step-deg",
        type=float,
        default=0.0,
        help="Optional larger catch-up step cap for wrist joints. 0 reuses --max-catchup-step-deg.",
    )
    parser.add_argument(
        "--gripper-max-catchup-step-deg",
        type=float,
        default=0.0,
        help="Optional larger catch-up step cap for grippers. 0 reuses --gripper-max-step-deg.",
    )
    parser.add_argument(
        "--catchup-start-error-deg",
        type=float,
        default=1.5,
        help="Tracking error where arm-joint catch-up scaling starts ramping up.",
    )
    parser.add_argument(
        "--catchup-full-error-deg",
        type=float,
        default=6.0,
        help="Tracking error where arm-joint catch-up scaling reaches its max catch-up step.",
    )
    parser.add_argument("--min-range-width-deg", type=float, default=5.0)
    parser.add_argument("--startup-threshold-deg", type=float, default=5.0)
    parser.add_argument("--startup-range-slack-deg", type=float, default=0.5)
    parser.add_argument("--feedback-max-age-s", type=float, default=0.40)
    parser.add_argument(
        "--feedback-retry-count",
        type=int,
        default=3,
        help="Retry missing fresh follower feedback this many times before stopping live motion.",
    )
    parser.add_argument(
        "--feedback-retry-sleep-s",
        type=float,
        default=0.015,
        help="Sleep between fresh-feedback retries when a follower frame is temporarily missing.",
    )
    parser.add_argument(
        "--active-feedback-motors",
        default="",
        help=(
            "Comma-separated motor names whose RobStride active reports should stay enabled. "
            "Selected motors not listed here are explicitly disabled to reduce bus load."
        ),
    )
    parser.add_argument(
        "--live-feedback-missing-grace-s",
        type=float,
        default=1.0,
        help="During LIVE, allow a motor's fresh feedback to be missing this long before stopping.",
    )
    parser.add_argument(
        "--follower-tx-min-gap-s",
        type=float,
        default=0.012,
        help="Minimum delay between follower USB-CAN transmit frames. Lower values improve multi-axis smoothness.",
    )
    parser.add_argument("--hold-settle-s", type=float, default=0.10)
    parser.add_argument("--startup-hold-guard-s", type=float, default=0.8)
    parser.add_argument("--startup-hold-max-drift-deg", type=float, default=2.0)
    parser.add_argument(
        "--allow-startup-low-disturbance-sample",
        action="store_true",
        help=(
            "If passive follower feedback is missing at startup, briefly enable each missing motor "
            "only to read feedback, then immediately disable it before preflight."
        ),
    )
    parser.add_argument("--duration", type=float, default=0.0, help="0 means run until interrupted.")
    parser.add_argument("--log-file", type=Path, default=None, help="Optional JSONL path for startup/live motor logs.")
    parser.add_argument("--log-decimate", type=int, default=1, help="Write every Nth live frame to the JSONL log.")
    parser.add_argument("--leader-port", default=LEADER_PORT)
    parser.add_argument("--left-follower-port", default=LEFT_FOLLOWER_PORT)
    parser.add_argument("--right-follower-port", default=RIGHT_FOLLOWER_PORT)
    parser.add_argument("--rest-range", type=Path, default=REST_RANGE_CALIB)
    parser.add_argument("--lerobot-calibration", type=Path, default=LEROBOT_CALIB)
    parser.add_argument("--motors", default="", help="Comma separated full motor names.")
    parser.add_argument("--dry-run", action="store_true", help="Read leader only; follower is not connected.")
    parser.add_argument("--allow-motion", action="store_true", help="Required for real follower motion.")
    parser.add_argument("--live", action="store_true", help="Alias for --allow-motion.")
    parser.add_argument(
        "--allow-infinite-live",
        action="store_true",
        help="Allow real motion without --duration. Use with care.",
    )
    args = parser.parse_args()

    motion_allowed = args.allow_motion or args.live
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
    if args.startup_range_slack_deg < 0:
        raise RuntimeError("Refusing negative --startup-range-slack-deg.")
    if args.startup_hold_guard_s < 0:
        raise RuntimeError("Refusing negative --startup-hold-guard-s.")
    if args.startup_hold_max_drift_deg <= 0:
        raise RuntimeError("Refusing non-positive --startup-hold-max-drift-deg.")
    if args.feedback_retry_count < 0:
        raise RuntimeError("Refusing negative --feedback-retry-count.")
    if args.feedback_retry_sleep_s < 0:
        raise RuntimeError("Refusing negative --feedback-retry-sleep-s.")
    if args.live_feedback_missing_grace_s < 0:
        raise RuntimeError("Refusing negative --live-feedback-missing-grace-s.")
    if args.follower_tx_min_gap_s < 0:
        raise RuntimeError("Refusing negative --follower-tx-min-gap-s.")
    if args.log_decimate <= 0:
        raise RuntimeError("Refusing non-positive --log-decimate.")
    if motion_allowed and args.duration <= 0 and not args.allow_infinite_live:
        raise RuntimeError("Refusing live mode without --duration. Pass --allow-infinite-live to override.")

    leader_calibration = load_json_required(LEADER_CALIB)
    leader_rest_raw, follower_rest_deg, follower_ranges_abs = load_rest_range(args.rest_range)
    lerobot_calibration = load_json_optional(args.lerobot_calibration)

    left_motors, right_motors = build_dual_follower_motors()
    all_motors = {**left_motors, **right_motors}
    selected = parse_motor_selection(args.motors, all_motors)
    active_feedback_names = set(parse_active_feedback_motors(args.active_feedback_motors, all_motors))

    validate_rest_coverage(selected, leader_rest_raw, follower_rest_deg, follower_ranges_abs)
    validate_rest_ranges(selected, follower_rest_deg, follower_ranges_abs)
    validate_range_widths(selected, follower_ranges_abs, args.min_range_width_deg)
    calibration_audit = audit_calibration_consistency(
        selected,
        leader_calibration,
        follower_rest_deg,
        follower_ranges_abs,
        lerobot_calibration,
    )
    leader_mappings = build_leader_offset_mappings(
        selected,
        leader_calibration,
        leader_rest_raw,
        follower_rest_deg,
        follower_ranges_abs,
        lerobot_calibration,
    )

    print_header(selected, args, lerobot_calibration, calibration_audit)

    leader_reader = LeaderReader(args.leader_port, leader_calibration)
    left_bus: RobStrideAtMotorsBus | None = None
    right_bus: RobStrideAtMotorsBus | None = None
    connected_buses: list[RobStrideAtMotorsBus] = []
    log_handle = open_jsonl_log(args.log_file)
    stop = {"value": False}

    def _stop(*_):
        stop["value"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        write_log_event(
            log_handle,
            "session_start",
            selected=selected,
            motion_allowed=motion_allowed,
            dry_run=bool(args.dry_run),
            fps=float(args.fps),
            duration_s=float(args.duration),
            max_step_deg=float(args.max_step_deg),
            follower_tx_min_gap_s=float(args.follower_tx_min_gap_s),
            startup_threshold_deg=float(args.startup_threshold_deg),
            startup_range_slack_deg=float(args.startup_range_slack_deg),
            startup_hold_guard_s=float(args.startup_hold_guard_s),
            startup_hold_max_drift_deg=float(args.startup_hold_max_drift_deg),
            feedback_max_age_s=float(args.feedback_max_age_s),
            hold_settle_s=float(args.hold_settle_s),
            calibration_audit=calibration_audit,
        )
        print("Leader...")
        leader_reader.connect()
        leader_reader.assert_online(selected)
        print("  OK all online")

        if args.dry_run:
            run_dry(
                args=args,
                leader_reader=leader_reader,
                selected=selected,
                leader_rest_raw=leader_rest_raw,
                follower_rest_deg=follower_rest_deg,
                follower_ranges_abs=follower_ranges_abs,
                leader_mappings=leader_mappings,
                stop=stop,
            )
            return 0

        left_names = [name for name in selected if name.startswith("left_")]
        right_names = [name for name in selected if name.startswith("right_")]

        if left_names:
            print(f"Follower LEFT ({args.left_follower_port})...")
            left_bus = RobStrideAtMotorsBus(
                port=args.left_follower_port,
                motors={name: left_motors[name] for name in left_names},
                min_tx_gap_s=args.follower_tx_min_gap_s,
            )
            left_bus.connect(handshake=False)
            left_bus.configure_motors()
            # Strict teleop treats rest/range calibration as the only follower
            # reference frame, so feedback/commands must stay in raw absolute
            # motor angles instead of any older software-zero frame.
            left_bus._zero_offsets_rad = {name: 0.0 for name in left_names}
            configure_active_feedback_reports(left_bus, left_names, left_motors, active_feedback_names, log_handle)
            connected_buses.append(left_bus)
            print(f"  OK {len(left_names)} motors")

        if right_names:
            print(f"Follower RIGHT ({args.right_follower_port})...")
            right_bus = RobStrideAtMotorsBus(
                port=args.right_follower_port,
                motors={name: right_motors[name] for name in right_names},
                min_tx_gap_s=args.follower_tx_min_gap_s,
            )
            right_bus.connect(handshake=False)
            right_bus.configure_motors()
            right_bus._zero_offsets_rad = {name: 0.0 for name in right_names}
            configure_active_feedback_reports(right_bus, right_names, right_motors, active_feedback_names, log_handle)
            connected_buses.append(right_bus)
            print(f"  OK {len(right_names)} motors")

        if not connected_buses:
            raise RuntimeError("No follower motors selected.")

        startup_bus_positions = sample_startup_positions(
            selected,
            left_bus,
            right_bus,
            left_motors,
            right_motors,
            allow_low_disturbance_sample=bool(args.allow_startup_low_disturbance_sample),
        )
        startup_leader_offsets, leader_unwrapped_raw, leader_zero_branches = read_leader_state(
            selected,
            leader_reader,
            leader_rest_raw,
        )
        startup_command_offsets = map_leader_offsets_to_follower_offsets(
            selected,
            startup_leader_offsets,
            leader_mappings,
            leader_unwrapped_raw=leader_unwrapped_raw,
        )
        startup_follower_offsets = compute_offsets_from_reference(
            selected,
            startup_bus_positions,
            follower_rest_deg,
            follower_ranges_abs,
            tolerance_deg=args.startup_range_slack_deg,
        )
        rows, bad = build_preflight_rows(
            selected,
            startup_leader_offsets,
            startup_command_offsets,
            startup_follower_offsets,
            args.startup_threshold_deg,
        )
        write_log_event(
            log_handle,
            "preflight",
            startup_threshold_deg=float(args.startup_threshold_deg),
            leader_offsets_deg=float_dict(startup_leader_offsets),
            commanded_offsets_deg=float_dict(startup_command_offsets),
            follower_offsets_deg=float_dict(startup_follower_offsets),
            follower_startup_abs_deg=float_dict(startup_bus_positions),
            rows=rows,
            bad_rows=bad,
        )

        print("\nPRE-FLIGHT offsets from recorded zero\n")
        print_preflight_rows(rows)
        if bad:
            print("\nRefuse live: startup pose does not match recorded zero.\n")
            print(f"{'Joint':28s} {'Leader':>9s} {'CmdOfs':>9s} {'Follower':>9s} {'Err':>9s}")
            print("-" * 70)
            for row in bad:
                print(
                    f"  {row['name']:26s} "
                    f"{row['leader_offset_deg']:+8.1f} "
                    f"{row['commanded_offset_deg']:+8.1f} "
                    f"{row['follower_offset_deg']:+8.1f} "
                    f"{row['error_deg']:+8.1f}"
                )
            raise RuntimeError("Startup mismatch; stop before motion.")

        print("\nOK Pre-flight: startup pose matches recorded zero.\n")
        if not motion_allowed:
            print("LIVE not requested; exiting after preflight. Pass --allow-motion to command motors.")
            return 0

        warn_teleop_validation_gaps(lerobot_calibration, selected)

        enable_with_immediate_hold(
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
            guard_elapsed = time.monotonic() - guard_start
            current_guard_positions = read_latest_frame_positions(
                selected,
                left_bus,
                right_bus,
                left_motors,
                right_motors,
                max_age_s=args.feedback_max_age_s,
                retry_count=args.feedback_retry_count,
                retry_sleep_s=args.feedback_retry_sleep_s,
            )
            drifts_deg = {
                name: current_guard_positions[name] - live_anchor_positions[name]
                for name in selected
            }
            bad_drifts = {
                name: drift
                for name, drift in drifts_deg.items()
                if abs(drift) > args.startup_hold_max_drift_deg
            }
            write_log_event(
                log_handle,
                "startup_hold_frame",
                frame_idx=int(guard_frame_idx),
                elapsed_s=float(guard_elapsed),
                drifts_deg=float_dict(drifts_deg),
                positions_deg=float_dict(current_guard_positions),
                anchor_positions_deg=float_dict(live_anchor_positions),
                max_abs_drift_deg=float(max((abs(drift) for drift in drifts_deg.values()), default=0.0)),
            )
            if guard_frame_idx == 0:
                live_anchor_positions = dict(current_guard_positions)
                write_log_event(
                    log_handle,
                    "startup_live_anchor_reset",
                    frame_idx=int(guard_frame_idx),
                    anchor_positions_deg=float_dict(live_anchor_positions),
                )
                send_frame_targets(
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
            if bad_drifts:
                write_log_event(
                    log_handle,
                    "startup_hold_drift_abort",
                    drifts_deg=float_dict(drifts_deg),
                    startup_positions_deg=float_dict(startup_bus_positions),
                    anchor_positions_deg=float_dict(live_anchor_positions),
                    current_positions_deg=float_dict(current_guard_positions),
                    limit_deg=float(args.startup_hold_max_drift_deg),
                )
                raise RuntimeError("Startup hold drifted before live; stop before motion.")
            send_frame_targets(
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
        # Prime one fresh follower frame after settle so LIVE does not start on
        # a feedback-age boundary and falsely report every motor as stale.
        live_boot_positions = read_latest_frame_positions(
            selected,
            left_bus,
            right_bus,
            left_motors,
            right_motors,
            max_age_s=max(args.feedback_max_age_s, 0.50),
            retry_count=args.feedback_retry_count,
            retry_sleep_s=args.feedback_retry_sleep_s,
        )
        write_log_event(
            log_handle,
            "live_start",
            startup_leader_offsets_deg=float_dict(startup_leader_offsets),
            startup_command_offsets_deg=float_dict(startup_command_offsets),
            startup_follower_abs_deg=float_dict(live_anchor_positions),
            pre_enable_follower_abs_deg=float_dict(startup_bus_positions),
            startup_follower_offsets_deg=float_dict(startup_follower_offsets),
            live_boot_positions_deg=float_dict(live_boot_positions),
        )

        start = time.monotonic()
        period = 1.0 / args.fps
        frame_idx = 0
        commanded_bus_positions = dict(live_boot_positions)
        last_feedback_positions = dict(live_boot_positions)
        feedback_missing_since: dict[str, float] = {}
        print("\nLIVE - Ctrl-C to stop\n")
        while not stop["value"]:
            loop_start = time.monotonic()
            feedback_missing: list[str] = []

            if frame_idx == 0:
                current_bus_positions = dict(live_boot_positions)
            else:
                fresh_positions, feedback_missing = try_read_latest_frame_positions(
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
            leader_offsets, leader_unwrapped_raw, leader_zero_branches = read_leader_state(
                selected,
                leader_reader,
                leader_rest_raw,
                previous_unwrapped_raw=leader_unwrapped_raw,
                zero_unwrapped_raw=leader_zero_branches,
            )
            current_command_offsets = map_leader_offsets_to_follower_offsets(
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
                lo, hi = canonicalize_range_to_value(follower_ranges_abs[name], prev_cmd_abs)
                lo, hi = expand_range_to_include_near_edge_value(
                    (lo, hi),
                    prev_cmd_abs,
                    slack_deg=args.startup_range_slack_deg,
                )
                # Keep startup anchoring so a small leader/follower mismatch at
                # launch does not create a step jump, but clamp and step in the
                # follower's absolute-angle frame used by the bus commands.
                target_abs = clamp(live_anchor_positions[name] + leader_delta, lo, hi)
                now_abs = current_bus_positions[name]
                step_limit = compute_live_step_limit_deg(
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
                step = clamp(target_abs - prev_cmd_abs, -step_limit, step_limit)
                target_bus_positions[name] = target_abs
                next_abs = clamp(prev_cmd_abs + step, lo, hi)
                actual_step = next_abs - prev_cmd_abs
                if abs(actual_step) > step_limit + 1e-6:
                    raise RuntimeError(
                        f"{name}: command step guard tripped; "
                        f"prev={prev_cmd_abs:+.2f} next={next_abs:+.2f} "
                        f"limit={step_limit:.2f} range=[{lo:+.2f},{hi:+.2f}]"
                    )
                next_bus_positions[name] = next_abs

            send_frame_targets(
                selected,
                next_bus_positions,
                left_bus,
                right_bus,
                lerobot_calibration,
                args.kp,
                args.kd,
            )
            commanded_bus_positions = dict(next_bus_positions)
            frame_idx += 1
            if log_handle is not None and frame_idx % args.log_decimate == 0:
                write_log_event(
                    log_handle,
                    "live_frame",
                    frame_idx=frame_idx,
                    t_since_live_start_s=time.monotonic() - start,
                    leader_offsets_deg=float_dict(leader_offsets),
                    leader_unwrapped_raw=float_dict(leader_unwrapped_raw),
                    commanded_offsets_deg=float_dict(current_command_offsets),
                    follower_current_abs_deg=float_dict(current_bus_positions),
                    follower_target_abs_deg=float_dict(target_bus_positions),
                    follower_command_abs_deg=float_dict(next_bus_positions),
                    follower_prev_command_abs_deg=float_dict(commanded_bus_positions),
                    feedback_missing=feedback_missing,
                )

            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break

            elapsed = time.monotonic() - loop_start
            time.sleep(max(0.0, period - elapsed))
    except Exception as exc:
        write_log_event(log_handle, "error", message=str(exc), stop_requested=bool(stop["value"]))
        raise
    finally:
        print("\nStopping...")
        write_log_event(log_handle, "session_stop", stop_requested=bool(stop["value"]))
        for bus in connected_buses:
            try:
                bus.disable_torque()
            except Exception:
                pass
            try:
                bus.disconnect(disable_torque=False)
            except Exception:
                pass
        leader_reader.disconnect()
        close_log_file(log_handle)
        print("Done.")
    return 0


def build_dual_follower_motors() -> tuple[dict[str, Motor], dict[str, Motor]]:
    left = {f"left_{name}": Motor(i, "O3", MotorNormMode.DEGREES) for i, name in enumerate(ARM, 1)}
    right = {f"right_{name}": Motor(i, "O3", MotorNormMode.DEGREES) for i, name in enumerate(ARM, 11)}
    return left, right


def parse_motor_selection(value: str, motors: dict[str, Motor]) -> list[str]:
    if not value.strip():
        return list(motors)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in selected if name not in motors]
    if unknown:
        raise ValueError(f"Unknown motor selection: {unknown}")
    return selected


def parse_active_feedback_motors(value: str, motors: dict[str, Motor]) -> list[str]:
    if not value.strip():
        return []
    selected = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in selected if name not in motors]
    if unknown:
        raise ValueError(f"Unknown active feedback motor selection: {unknown}")
    return selected


def load_json_required(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_json_optional(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def open_jsonl_log(path: Path | None) -> TextIO | None:
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", encoding="utf-8")


def close_log_file(handle: TextIO | None) -> None:
    if handle is None:
        return
    handle.close()


def write_log_event(handle: TextIO | None, event: str, **payload) -> None:
    if handle is None:
        return
    record = {
        "event": event,
        "wall_time_unix": time.time(),
        **payload,
    }
    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    handle.flush()


def float_dict(values: dict[str, float]) -> dict[str, float]:
    return {name: float(value) for name, value in values.items()}


def robstride_can_id(comm_type: int, data_area2: int, motor_id: int) -> int:
    return ((comm_type & 0x1F) << 24) | ((data_area2 & 0xFFFF) << 8) | (motor_id & 0xFF)


def configure_active_feedback_reports(
    bus: RobStrideAtMotorsBus,
    names: list[str],
    motors: dict[str, Motor],
    active_names: set[str],
    log_handle: TextIO | None,
) -> None:
    enabled: list[str] = []
    disabled: list[str] = []
    failed: list[str] = []
    for name in names:
        motor_id = motors[name].id
        enable = name in active_names
        try:
            bus.bus.set_active_report(motor_id, enable=enable)
            if enable:
                enabled.append(name)
            else:
                disabled.append(name)
            time.sleep(0.01)
        except Exception:
            failed.append(name)
    write_log_event(
        log_handle,
        "active_feedback_reports",
        enabled=enabled,
        disabled=disabled,
        failed=failed,
    )


def audit_calibration_consistency(
    selected: list[str],
    leader_calibration: dict,
    follower_rest_deg: dict[str, float],
    follower_ranges_abs: dict[str, tuple[float, float]],
    lerobot_calibration: dict | None,
) -> dict[str, object]:
    warnings: list[str] = []
    lerobot_follower = {} if not lerobot_calibration else lerobot_calibration.get("follower", {})
    lerobot_leader = {} if not lerobot_calibration else lerobot_calibration.get("leader", {})

    missing_lerobot_fields: list[str] = []
    stale_lerobot_joints: list[str] = []
    rest_updated_at = float(load_optional_number({"value": lerobot_calibration.get("updated_at_unix") if lerobot_calibration else None}, "value", default=0.0))
    leader_ids = []
    for name in selected:
        item = leader_calibration.get(name, {})
        if isinstance(item.get("id"), int):
            leader_ids.append(int(item["id"]))

    for name in selected:
        follower_entry = lerobot_follower.get(name, {})
        leader_entry = lerobot_leader.get(name, {})
        if not isinstance(follower_entry.get("drive_mode"), (int, float)):
            missing_lerobot_fields.append(f"follower.drive_mode.{name}")
        if not isinstance(leader_entry.get("drive_mode"), (int, float)):
            missing_lerobot_fields.append(f"leader.drive_mode.{name}")
        if not isinstance(follower_entry.get("teleop_scale"), (int, float)):
            missing_lerobot_fields.append(f"follower.teleop_scale.{name}")
        middle_ts = follower_entry.get("middle_recorded_at_unix")
        range_ts = follower_entry.get("range_recorded_at_unix")
        if isinstance(middle_ts, (int, float)) and isinstance(range_ts, (int, float)):
            if float(range_ts) < float(middle_ts):
                stale_lerobot_joints.append(name)
        elif follower_entry:
            stale_lerobot_joints.append(name)

    if not lerobot_calibration:
        warnings.append("missing_lerobot_calibration")
    if missing_lerobot_fields:
        warnings.append("incomplete_lerobot_fields")
    if stale_lerobot_joints:
        warnings.append("partial_or_stale_lerobot_joint_entries")
    if rest_updated_at and lerobot_calibration and isinstance(lerobot_calibration.get("updated_at_unix"), (int, float)):
        if float(lerobot_calibration["updated_at_unix"]) + 86400.0 < max(
            load_optional_number({"value": leader_calibration.get("_right_arm_calibrated_at_unix")}, "value", default=0.0),
            max((abs(float(follower_rest_deg[name])) for name in selected), default=0.0) * 0.0 + rest_updated_at,
        ):
            warnings.append("lerobot_calibration_older_than_rest_range")

    return {
        "warnings": warnings,
        "missing_lerobot_fields": missing_lerobot_fields,
        "stale_lerobot_joints": stale_lerobot_joints,
        "leader_ids": leader_ids,
        "rest_joint_count": len(follower_rest_deg),
        "range_joint_count": len(follower_ranges_abs),
        "lerobot_joint_count": len(lerobot_follower),
        "lerobot_updated_at_unix": None if not lerobot_calibration else lerobot_calibration.get("updated_at_unix"),
    }


def load_optional_number(container: dict, key: str, default: float = 0.0) -> float:
    value = container.get(key)
    return float(value) if isinstance(value, (int, float)) else float(default)


def load_rest_range(path: Path) -> tuple[dict[str, int], dict[str, float], dict[str, tuple[float, float]]]:
    data = load_json_required(path)
    leader_rest_raw: dict[str, int] = {}
    follower_rest_deg: dict[str, float] = {}
    follower_ranges_abs: dict[str, tuple[float, float]] = {}
    for name, item in data.get("leader", {}).items():
        if isinstance(item.get("rest_raw"), int):
            leader_rest_raw[name] = int(item["rest_raw"])
    for name, item in data.get("follower", {}).items():
        if isinstance(item.get("rest_deg"), (int, float)):
            follower_rest_deg[name] = float(item["rest_deg"])
        if isinstance(item.get("range_min_deg"), (int, float)) and isinstance(item.get("range_max_deg"), (int, float)):
            follower_ranges_abs[name] = (float(item["range_min_deg"]), float(item["range_max_deg"]))
    for name, rest_deg in follower_rest_deg.items():
        if name in follower_ranges_abs:
            follower_ranges_abs[name] = canonicalize_range_to_rest(
                follower_ranges_abs[name],
                rest_deg,
            )
    return leader_rest_raw, follower_rest_deg, follower_ranges_abs


def validate_rest_coverage(
    selected: list[str],
    leader_rest_raw: dict[str, int],
    follower_rest_deg: dict[str, float],
    follower_ranges_abs: dict[str, tuple[float, float]],
) -> None:
    missing = []
    for name in selected:
        if name not in leader_rest_raw:
            missing.append(f"leader.rest_raw.{name}")
        if name not in follower_rest_deg:
            missing.append(f"follower.rest_deg.{name}")
        if name not in follower_ranges_abs:
            missing.append(f"follower_ranges.{name}")
    if missing:
        raise RuntimeError("Missing calibration coverage: " + ", ".join(missing))


def validate_rest_ranges(
    selected: list[str],
    follower_rest_deg: dict[str, float],
    follower_ranges_abs: dict[str, tuple[float, float]],
) -> None:
    bad = []
    for name in selected:
        lo, hi = follower_ranges_abs[name]
        rest = float(follower_rest_deg[name])
        if not (lo <= rest <= hi):
            bad.append(f"{name}: rest={rest:+.2f} not in [{lo:+.2f}, {hi:+.2f}]")
    if bad:
        raise RuntimeError("Recorded rest is outside the follower range:\n" + "\n".join(bad))


def validate_range_widths(
    selected: list[str],
    follower_ranges_abs: dict[str, tuple[float, float]],
    min_range_width_deg: float,
) -> None:
    bad = []
    threshold = float(min_range_width_deg)
    for name in selected:
        lo, hi = follower_ranges_abs[name]
        width = float(hi) - float(lo)
        if width < threshold:
            bad.append(f"{name}: width={width:+.2f} deg < {threshold:.2f} deg")
    if bad:
        raise RuntimeError("Recorded follower range is too small:\n" + "\n".join(bad))


def canonicalize_range_to_rest(range_abs_deg: tuple[float, float], rest_deg: float) -> tuple[float, float]:
    lo, hi = map(float, range_abs_deg)
    rest = float(rest_deg)
    if lo > hi:
        lo, hi = hi, lo
    if lo <= rest <= hi:
        return lo, hi

    candidates: list[tuple[float, float, int]] = []
    for shift in range(-4, 5):
        shifted = (lo + 360.0 * shift, hi + 360.0 * shift)
        if shifted[0] <= rest <= shifted[1]:
            candidates.append((shifted[0], shifted[1], shift))
    if not candidates:
        return lo, hi
    best_lo, best_hi, _ = min(
        candidates,
        key=lambda item: (abs(item[2]), abs(((item[0] + item[1]) * 0.5) - rest)),
    )
    return best_lo, best_hi


def canonicalize_range_to_value(range_abs_deg: tuple[float, float], value_deg: float) -> tuple[float, float]:
    lo, hi = map(float, range_abs_deg)
    value = float(value_deg)
    if lo > hi:
        lo, hi = hi, lo

    best: tuple[float, float] | None = None
    best_distance = float("inf")
    for shift in range(-4, 5):
        shifted_lo = lo + 360.0 * shift
        shifted_hi = hi + 360.0 * shift
        if shifted_lo <= value <= shifted_hi:
            return shifted_lo, shifted_hi
        distance = shifted_lo - value if value < shifted_lo else value - shifted_hi
        if distance < best_distance:
            best_distance = distance
            best = (shifted_lo, shifted_hi)
    return (lo, hi) if best is None else best


def expand_range_to_include_near_edge_value(
    range_abs_deg: tuple[float, float],
    value_deg: float,
    *,
    slack_deg: float,
) -> tuple[float, float]:
    lo, hi = map(float, range_abs_deg)
    value = float(value_deg)
    slack = max(0.0, float(slack_deg))
    if lo > hi:
        lo, hi = hi, lo
    if value < lo and lo - value <= slack:
        lo = value
    elif value > hi and value - hi <= slack:
        hi = value
    return lo, hi


def warn_teleop_validation_gaps(calibration: dict | None, selected_motors: list[str]) -> None:
    if calibration is None:
        print("WARN missing diyrobot_lerobot_calibration.json; continuing with rest/range-only mapping.")
        return
    missing = []
    for section in ("leader", "follower"):
        for name in selected_motors:
            if not calibration.get(section, {}).get(name, {}).get("teleop_validated"):
                missing.append(f"{section}.{name}")
    if missing:
        print(
            "WARN legacy teleop_validated missing for: "
            + ", ".join(missing)
            + ". Continuing because strict rest/range preflight passed."
        )


def print_header(
    selected: list[str],
    args: argparse.Namespace,
    lerobot_calibration: dict,
    calibration_audit: dict[str, object] | None = None,
) -> None:
    print("=" * 60)
    print("  DIYRobot DUAL-ARM TELEOP (strict startup)")
    print("=" * 60)
    for name in selected:
        kp, kd, torque_ff = joint_gains(name, args.kp, args.kd, lerobot_calibration)
        print(f"  {name:28s} kp={kp:3.0f} kd={kd:3.1f} ff={torque_ff:3.0f}")
    print(f"  fps={args.fps:.0f}")
    print(f"  max_step={args.max_step_deg:.2f} deg/frame")
    print(f"  follower_tx_min_gap={args.follower_tx_min_gap_s * 1000.0:.1f} ms")
    if args.max_catchup_step_deg > args.max_step_deg:
        print(
            "  arm_catchup_step="
            f"{args.max_catchup_step_deg:.2f} deg/frame "
            f"(error {args.catchup_start_error_deg:.1f}->{args.catchup_full_error_deg:.1f} deg)"
        )
    if args.wrist_max_step_deg > 0 or args.wrist_max_catchup_step_deg > 0:
        wrist_step_deg = args.wrist_max_step_deg if args.wrist_max_step_deg > 0 else args.max_step_deg
        wrist_catchup_deg = (
            args.wrist_max_catchup_step_deg
            if args.wrist_max_catchup_step_deg > 0
            else max(args.max_catchup_step_deg, wrist_step_deg)
        )
        print(
            "  wrist_step="
            f"{wrist_step_deg:.2f} deg/frame "
            f"catchup={wrist_catchup_deg:.2f}"
        )
    if args.gripper_max_catchup_step_deg > 0:
        print(
            "  gripper_step="
            f"{args.gripper_max_step_deg:.2f} deg/frame "
            f"catchup={args.gripper_max_catchup_step_deg:.2f}"
        )
    print(f"  startup_threshold={args.startup_threshold_deg:.1f} deg")
    if calibration_audit and calibration_audit.get("warnings"):
        print("  calibration_warnings=" + ", ".join(str(item) for item in calibration_audit["warnings"]))
    print()


def run_dry(
    args: argparse.Namespace,
    leader_reader: "LeaderReader",
    selected: list[str],
    leader_rest_raw: dict[str, int],
    follower_rest_deg: dict[str, float],
    follower_ranges_abs: dict[str, tuple[float, float]],
    leader_mappings: dict[str, dict[str, float | bool | str]],
    stop: dict[str, bool],
) -> None:
    period = 1.0 / args.fps
    start = time.monotonic()
    previous_unwrapped_raw: dict[str, float] | None = None
    zero_unwrapped_raw: dict[str, float] | None = None
    print("\nDRY RUN - leader only; follower is not connected.\n")
    while not stop["value"]:
        loop_start = time.monotonic()
        leader_offsets, previous_unwrapped_raw, zero_unwrapped_raw = read_leader_state(
            selected,
            leader_reader,
            leader_rest_raw,
            previous_unwrapped_raw=previous_unwrapped_raw,
            zero_unwrapped_raw=zero_unwrapped_raw,
        )
        command_offsets = map_leader_offsets_to_follower_offsets(
            selected,
            leader_offsets,
            leader_mappings,
            leader_unwrapped_raw=previous_unwrapped_raw,
        )
        for name in selected:
            target_offset = clamp(
                command_offsets[name],
                follower_ranges_abs[name][0] - follower_rest_deg[name],
                follower_ranges_abs[name][1] - follower_rest_deg[name],
            )
            target_abs = float(follower_rest_deg[name]) + target_offset
            print(
                f"  {name:28s} "
                f"leader={leader_offsets[name]:+7.1f} "
                f"cmd_ofs={target_offset:+7.1f} "
                f"target={target_abs:+8.1f} "
                f"range=[{follower_ranges_abs[name][0]:+7.1f},{follower_ranges_abs[name][1]:+7.1f}]"
            )
        if args.duration > 0 and time.monotonic() - start >= args.duration:
            return
        time.sleep(max(0.0, period - (time.monotonic() - loop_start)))


def joint_gains(name: str, default_kp: float, default_kd: float, lerobot_calibration: dict) -> tuple[float, float, float]:
    kp = float(default_kp)
    kd = float(default_kd)
    # MIT command torque is feedforward torque, not a safety limit. Default to
    # zero so static hold does not bias joints toward one direction.
    torque_ff_nm = 0.0
    follower = lerobot_calibration.get("follower", {}).get(name, {})
    if name.endswith("shoulder_pan"):
        kp = max(kp, 120.0)
        kd = max(kd, 3.0)
    if name.endswith("shoulder_lift"):
        kp = max(kp, 120.0)
        kd = max(kd, 3.0)
    if name.endswith("elbow_flex"):
        kp = max(kp, 160.0)
        kd = max(kd, 2.4)
    if follower.get("teleop_needs_higher_stiffness"):
        kp = max(kp, 120.0)
        kd = max(kd, 2.0)
    if name.endswith("gripper"):
        kp = min(kp, 15.0)
        kd = min(kd, 0.8)
    explicit_torque_ff = follower.get("teleop_torque_ff_nm")
    if isinstance(explicit_torque_ff, (int, float)):
        torque_ff_nm = float(explicit_torque_ff)
    return kp, kd, torque_ff_nm


def map_leader_offset_to_follower_offset(name: str, leader_offset_deg: float, lerobot_calibration: dict) -> float:
    follower = lerobot_calibration.get("follower", {}).get(name, {})
    scale = float(follower.get("teleop_scale", 1.0))
    if int(follower.get("drive_mode", 0)):
        scale *= -1.0
    if name in EXTRA_SIGN_FLIPS:
        scale *= -1.0
    if name in EXTRA_SCALES:
        scale *= EXTRA_SCALES[name]
    return leader_offset_deg * scale


def build_leader_offset_mappings(
    selected: list[str],
    leader_calibration: dict,
    leader_zero_raw: dict[str, int],
    follower_rest_deg: dict[str, float],
    follower_ranges_abs: dict[str, tuple[float, float]],
    lerobot_calibration: dict,
) -> dict[str, dict[str, float | bool | str]]:
    mappings: dict[str, dict[str, float | bool | str]] = {}
    for name in selected:
        mappings[name] = build_single_leader_mapping(
            name,
            leader_calibration,
            leader_zero_raw,
            follower_rest_deg,
            follower_ranges_abs,
            lerobot_calibration,
        )
    return mappings


def build_single_leader_mapping(
    name: str,
    leader_calibration: dict,
    leader_zero_raw: dict[str, int],
    follower_rest_deg: dict[str, float],
    follower_ranges_abs: dict[str, tuple[float, float]],
    lerobot_calibration: dict,
) -> dict[str, float | bool | str]:
    fallback_scale = map_leader_offset_to_follower_offset(name, 1.0, lerobot_calibration)
    follower_lo, follower_hi = follower_ranges_abs[name]
    follower_rest = float(follower_rest_deg[name])
    follower_rel_lo = follower_lo - follower_rest
    follower_rel_hi = follower_hi - follower_rest

    mapping: dict[str, float | bool | str] = {
        "mode": "scale",
        "fallback_scale": fallback_scale,
        "leader_zero_raw": float(leader_zero_raw[name]),
        "follower_rel_lo": follower_rel_lo,
        "follower_rel_hi": follower_rel_hi,
    }

    leader_item = leader_calibration.get(name, {})
    try:
        leader_min = float(leader_item["range_min"])
        leader_max = float(leader_item["range_max"])
    except Exception:
        return mapping

    if abs(leader_max - leader_min) < 1.0:
        return mapping

    leader_rest = float(leader_zero_raw[name])
    leader_span = shortest_span_containing_point(leader_min, leader_max, leader_rest)
    if leader_span is None:
        return mapping

    leader_lo, leader_hi = leader_span
    if abs(leader_hi - leader_lo) < 1.0:
        return mapping
    if not leader_range_mapping_is_safe(leader_lo, leader_hi, leader_rest):
        return mapping

    drive_sign = -1.0 if int(leader_item.get("drive_mode", 0)) else 1.0
    follower_sign = -1.0 if fallback_scale < 0 else 1.0
    follower_abs_scale = abs(fallback_scale) if abs(fallback_scale) > 1e-6 else 1.0

    mapping.update(
        {
            "mode": "range",
            "leader_lo": leader_lo,
            "leader_hi": leader_hi,
            "leader_rest": leader_rest,
            "leader_drive_sign": drive_sign,
            "follower_sign": follower_sign,
            "follower_abs_scale": follower_abs_scale,
        }
    )
    return mapping


def shortest_span_containing_point(lo: float, hi: float, point: float) -> tuple[float, float] | None:
    base_lo = float(lo)
    base_hi = float(hi)
    best: tuple[float, float] | None = None
    for lo_shift in range(-2, 3):
        for hi_shift in range(-2, 3):
            cur_lo = base_lo + 4096.0 * lo_shift
            cur_hi = base_hi + 4096.0 * hi_shift
            if cur_hi <= cur_lo:
                continue
            if not (cur_lo <= point <= cur_hi):
                continue
            candidate = (cur_lo, cur_hi)
            if best is None or (candidate[1] - candidate[0]) < (best[1] - best[0]):
                best = candidate
    return best


def raw_12bit_to_deg(delta_raw: float) -> float:
    return float(delta_raw) * 360.0 / 4095.0


def leader_range_mapping_is_safe(lo: float, hi: float, rest: float) -> bool:
    total_span_deg = raw_12bit_to_deg(hi - lo)
    negative_span_deg = raw_12bit_to_deg(rest - lo)
    positive_span_deg = raw_12bit_to_deg(hi - rest)
    smaller_side_deg = min(negative_span_deg, positive_span_deg)
    larger_side_deg = max(negative_span_deg, positive_span_deg)
    if total_span_deg > LEADER_RANGE_MAX_SAFE_WIDTH_DEG:
        return False
    if smaller_side_deg < LEADER_RANGE_MIN_EDGE_MARGIN_DEG:
        return False
    if smaller_side_deg <= 1e-6:
        return False
    if larger_side_deg / smaller_side_deg > LEADER_RANGE_MAX_EDGE_ASYMMETRY:
        return False
    return True


def map_single_leader_offset_to_follower_offset(
    leader_offset_deg: float,
    mapping: dict[str, float | bool | str],
    leader_unwrapped_raw: float | None = None,
) -> float:
    fallback = float(mapping["fallback_scale"]) * float(leader_offset_deg)
    if mapping.get("mode") != "range" or leader_unwrapped_raw is None:
        return fallback

    leader_rest = float(mapping["leader_rest"])
    leader_lo = float(mapping["leader_lo"])
    leader_hi = float(mapping["leader_hi"])
    signed_ratio = normalized_signed_ratio(
        float(leader_unwrapped_raw),
        leader_lo,
        leader_hi,
        leader_rest,
        drive_sign=float(mapping["leader_drive_sign"]),
    )
    follower_sign = float(mapping["follower_sign"])
    follower_abs_scale = float(mapping["follower_abs_scale"])
    follower_span = (
        float(mapping["follower_rel_hi"]) if signed_ratio >= 0.0 else -float(mapping["follower_rel_lo"])
    )
    target = signed_ratio * follower_span * follower_sign

    # Keep the new mapping continuous, but do not let a questionable range
    # estimate explode more than the old offset-scale mapping.
    if abs(target - fallback) > max(20.0, abs(fallback) * 0.8 + 5.0):
        return fallback
    return target


def normalized_signed_ratio(value: float, lo: float, hi: float, center: float, drive_sign: float) -> float:
    if not (lo < center < hi):
        return 0.0
    if drive_sign < 0:
        value = hi - (value - lo)
        center = hi - (center - lo)
    if value >= center:
        span = hi - center
        return 0.0 if span <= 1e-9 else min(max((value - center) / span, 0.0), 1.0)
    span = center - lo
    return 0.0 if span <= 1e-9 else -min(max((center - value) / span, 0.0), 1.0)


def map_leader_offsets_to_follower_offsets(
    selected: list[str],
    leader_offsets: dict[str, float],
    leader_mappings: dict[str, dict[str, float | bool | str]],
    leader_unwrapped_raw: dict[str, float] | None = None,
) -> dict[str, float]:
    return {
        name: map_single_leader_offset_to_follower_offset(
            leader_offsets[name],
            leader_mappings[name],
            None if leader_unwrapped_raw is None else leader_unwrapped_raw.get(name),
        )
        for name in selected
    }


def read_leader_state(
    selected: list[str],
    leader_reader: "LeaderReader",
    leader_zero_raw: dict[str, int],
    previous_unwrapped_raw: dict[str, float] | None = None,
    zero_unwrapped_raw: dict[str, float] | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    raw = leader_reader.read_raw_positions(selected)
    offsets: dict[str, float] = {}
    unwrapped: dict[str, float] = {}
    zero_branches: dict[str, float] = {}
    for name in selected:
        value = float(raw[name])
        prev = None if previous_unwrapped_raw is None else previous_unwrapped_raw.get(name)
        current_unwrapped = unwrap_12bit_near_reference(value, prev)
        if zero_unwrapped_raw is not None and name in zero_unwrapped_raw:
            zero_unwrapped = float(zero_unwrapped_raw[name])
        else:
            zero_unwrapped = unwrap_12bit_near_reference(float(leader_zero_raw[name]), current_unwrapped)
        unwrapped[name] = current_unwrapped
        zero_branches[name] = zero_unwrapped
        offsets[name] = (current_unwrapped - zero_unwrapped) * 360.0 / 4095.0
    return offsets, unwrapped, zero_branches


def unwrap_12bit_near_reference(value: float, reference: float | None) -> float:
    base = float(value)
    if reference is None:
        return base
    return min((base + 4096.0 * k for k in range(-4, 5)), key=lambda candidate: abs(candidate - reference))


def build_preflight_rows(
    selected: list[str],
    leader_offsets: dict[str, float],
    commanded_offsets: dict[str, float],
    follower_offsets: dict[str, float],
    startup_threshold_deg: float,
) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    rows: list[dict[str, float | str]] = []
    bad: list[dict[str, float | str]] = []
    for name in selected:
        commanded_offset = commanded_offsets[name]
        error_deg = follower_offsets[name] - commanded_offset
        row = {
            "name": name,
            "leader_offset_deg": leader_offsets[name],
            "commanded_offset_deg": commanded_offset,
            "follower_offset_deg": follower_offsets[name],
            "error_deg": error_deg,
        }
        rows.append(row)
        if (
            abs(float(leader_offsets[name])) > startup_threshold_deg
            or abs(float(follower_offsets[name])) > startup_threshold_deg
            or abs(float(error_deg)) > startup_threshold_deg
        ):
            bad.append(row)
    bad.sort(key=lambda item: abs(float(item["error_deg"])), reverse=True)
    return rows, bad


def print_preflight_rows(rows: list[dict[str, float | str]]) -> None:
    print(f"{'Joint':28s} {'Leader':>9s} {'CmdOfs':>9s} {'Follower':>9s} {'Err':>9s}")
    print("-" * 70)
    for row in rows:
        print(
            f"  {str(row['name']):26s} "
            f"{float(row['leader_offset_deg']):+8.1f} "
            f"{float(row['commanded_offset_deg']):+8.1f} "
            f"{float(row['follower_offset_deg']):+8.1f} "
            f"{float(row['error_deg']):+8.1f}"
        )


def compute_offsets_from_reference(
    selected: list[str],
    current_abs: dict[str, float],
    reference_abs: dict[str, float],
    follower_ranges_abs: dict[str, tuple[float, float]],
    tolerance_deg: float = 0.0,
) -> dict[str, float]:
    offsets: dict[str, float] = {}
    tol = max(0.0, float(tolerance_deg))
    for name in selected:
        lo, hi = follower_ranges_abs[name]
        offsets[name] = strict_equivalent_deg(
            current_abs[name] - float(reference_abs[name]),
            lo - float(reference_abs[name]) - tol,
            hi - float(reference_abs[name]) + tol,
            context=name,
        )
    return offsets


def sample_startup_positions(
    selected: list[str],
    left_bus: RobStrideAtMotorsBus | None,
    right_bus: RobStrideAtMotorsBus | None,
    left_motors: dict[str, Motor],
    right_motors: dict[str, Motor],
    allow_low_disturbance_sample: bool = False,
) -> dict[str, float]:
    feedback: dict[str, float] = {}
    missing = set(selected)
    deadline = time.monotonic() + 1.2
    while missing and time.monotonic() < deadline:
        progressed = False
        for name in list(missing):
            bus = left_bus if name.startswith("left_") else right_bus
            motors = left_motors if name.startswith("left_") else right_motors
            if bus is None:
                continue
            state = bus.bus.latest_feedback(motors[name].id, max_age_s=1.0)
            if state is None:
                continue
            feedback[name] = math.degrees(float(state["position_rad"]) - bus._zero_offsets_rad.get(name, 0.0))
            missing.remove(name)
            progressed = True
        if missing:
            time.sleep(0.02 if progressed else 0.05)
    if missing and allow_low_disturbance_sample:
        for name in list(sorted(missing)):
            bus = left_bus if name.startswith("left_") else right_bus
            motors = left_motors if name.startswith("left_") else right_motors
            if bus is None:
                continue
            motor_id = motors[name].id
            state = sample_follower_low_disturbance(bus, motor_id)
            if state is None:
                continue
            feedback[name] = math.degrees(float(state["position_rad"]) - bus._zero_offsets_rad.get(name, 0.0))
            missing.remove(name)

    if missing:
        raise RuntimeError(
            "Missing passive follower feedback: "
            + ", ".join(sorted(missing))
            + ". Refuse to enable torque before seeing fresh follower state."
            + (
                " Pass --allow-startup-low-disturbance-sample to read startup position with brief enable/disable."
                if not allow_low_disturbance_sample
                else ""
            )
        )
    return feedback


def read_latest_frame_positions(
    selected: list[str],
    left_bus: RobStrideAtMotorsBus | None,
    right_bus: RobStrideAtMotorsBus | None,
    left_motors: dict[str, Motor],
    right_motors: dict[str, Motor],
    max_age_s: float,
    retry_count: int = 0,
    retry_sleep_s: float = 0.0,
) -> dict[str, float]:
    out, missing = try_read_latest_frame_positions(
        selected,
        left_bus,
        right_bus,
        left_motors,
        right_motors,
        max_age_s,
        retry_count=retry_count,
        retry_sleep_s=retry_sleep_s,
    )
    if missing:
        raise RuntimeError("Missing fresh follower feedback: " + ", ".join(missing))
    return out


def try_read_latest_frame_positions(
    selected: list[str],
    left_bus: RobStrideAtMotorsBus | None,
    right_bus: RobStrideAtMotorsBus | None,
    left_motors: dict[str, Motor],
    right_motors: dict[str, Motor],
    max_age_s: float,
    retry_count: int = 0,
    retry_sleep_s: float = 0.0,
) -> tuple[dict[str, float], list[str]]:
    out: dict[str, float] = {}
    missing = list(selected)
    attempts = max(0, int(retry_count)) + 1
    for attempt_idx in range(attempts):
        next_missing = []
        for name in missing:
            bus = left_bus if name.startswith("left_") else right_bus
            motors = left_motors if name.startswith("left_") else right_motors
            if bus is None:
                next_missing.append(name)
                continue
            state = bus.bus.latest_feedback(motors[name].id, max_age_s=max_age_s)
            if state is None:
                next_missing.append(name)
                continue
            out[name] = math.degrees(float(state["position_rad"]) - bus._zero_offsets_rad.get(name, 0.0))
        missing = next_missing
        if not missing:
            break
        if attempt_idx < attempts - 1 and retry_sleep_s > 0:
            time.sleep(retry_sleep_s)
    return out, missing


def sample_follower_low_disturbance(bus: RobStrideAtMotorsBus, motor_id: int):
    for attempt in range(4):
        since = time.monotonic()
        state = bus.bus.enable(motor_id, timeout=0.30)
        if state is None:
            state = wait_fresh_feedback(bus, motor_id, since=since, timeout=0.20)
        if state is not None:
            bus.bus.disable_confirmed(motor_id, clear_error=True, timeout=0.20, retries=4, inter_retry_delay=0.03)
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
        bus.bus.disable_confirmed(motor_id, clear_error=True, timeout=0.20, retries=4, inter_retry_delay=0.03)
        time.sleep(0.06 + attempt * 0.04)
        if state is not None:
            return state
    return None


def wait_fresh_feedback(bus: RobStrideAtMotorsBus, motor_id: int, since: float, timeout: float):
    frame = bus.bus.wait_for(
        lambda item: item.comm_type == 2 and (item.area2 & 0xFF) == motor_id,
        timeout=timeout,
        since=since,
    )
    if frame is None:
        return None
    return bus.bus.latest_feedback(motor_id, max_age_s=1.0)


def clamp_absolute_target(target_deg: float, range_abs_deg: tuple[float, float]) -> float:
    lo, hi = range_abs_deg
    return clamp(target_deg, lo, hi)


def compute_live_step_limit_deg(
    *,
    name: str,
    tracking_error_deg: float,
    base_step_deg: float,
    catchup_step_deg: float,
    wrist_step_deg: float,
    wrist_catchup_step_deg: float,
    gripper_step_deg: float,
    gripper_catchup_step_deg: float,
    catchup_start_error_deg: float,
    catchup_full_error_deg: float,
) -> float:
    if name.endswith("gripper"):
        return compute_group_step_limit_deg(
            tracking_error_deg=tracking_error_deg,
            base_step_deg=gripper_step_deg,
            catchup_step_deg=gripper_catchup_step_deg if gripper_catchup_step_deg > 0 else gripper_step_deg,
            catchup_start_error_deg=catchup_start_error_deg,
            catchup_full_error_deg=catchup_full_error_deg,
        )

    if "_wrist_" in name:
        wrist_base_step_deg = wrist_step_deg if wrist_step_deg > 0 else base_step_deg
        wrist_catchup_cap_deg = wrist_catchup_step_deg if wrist_catchup_step_deg > 0 else catchup_step_deg
        return compute_group_step_limit_deg(
            tracking_error_deg=tracking_error_deg,
            base_step_deg=wrist_base_step_deg,
            catchup_step_deg=wrist_catchup_cap_deg,
            catchup_start_error_deg=catchup_start_error_deg,
            catchup_full_error_deg=catchup_full_error_deg,
        )

    return compute_group_step_limit_deg(
        tracking_error_deg=tracking_error_deg,
        base_step_deg=base_step_deg,
        catchup_step_deg=catchup_step_deg,
        catchup_start_error_deg=catchup_start_error_deg,
        catchup_full_error_deg=catchup_full_error_deg,
    )


def compute_group_step_limit_deg(
    *,
    tracking_error_deg: float,
    base_step_deg: float,
    catchup_step_deg: float,
    catchup_start_error_deg: float,
    catchup_full_error_deg: float,
) -> float:
    catchup_cap = max(base_step_deg, catchup_step_deg)
    if catchup_cap <= base_step_deg:
        return base_step_deg

    error_abs = abs(tracking_error_deg)
    if error_abs <= catchup_start_error_deg:
        return base_step_deg
    if error_abs >= catchup_full_error_deg:
        return catchup_cap

    ramp_span = catchup_full_error_deg - catchup_start_error_deg
    if ramp_span <= 0:
        return catchup_cap

    ramp = (error_abs - catchup_start_error_deg) / ramp_span
    return base_step_deg + ramp * (catchup_cap - base_step_deg)


def clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def strict_equivalent_deg(value: float, lo: float, hi: float, context: str = "") -> float:
    candidates = [float(value) + 360.0 * k for k in range(-4, 5)]
    inside = [candidate for candidate in candidates if lo <= candidate <= hi]
    if not inside:
        prefix = f"{context}: " if context else ""
        raise RuntimeError(
            f"{prefix}value {float(value):+.2f} has no equivalent within [{float(lo):+.2f}, {float(hi):+.2f}]"
        )
    return min(inside, key=lambda candidate: abs(candidate - value))


def send_frame_targets(
    selected: list[str],
    frame_targets: dict[str, float],
    left_bus: RobStrideAtMotorsBus | None,
    right_bus: RobStrideAtMotorsBus | None,
    lerobot_calibration: dict,
    default_kp: float,
    default_kd: float,
) -> None:
    left_commands: dict[str, tuple[float, float, float, float, float]] = {}
    right_commands: dict[str, tuple[float, float, float, float, float]] = {}
    for name in selected:
        position_deg = frame_targets[name]
        kp, kd, torque = joint_gains(name, default_kp, default_kd, lerobot_calibration)
        command = (kp, kd, position_deg, 0.0, torque)
        if name.startswith("left_"):
            left_commands[name] = command
        else:
            right_commands[name] = command
    if left_bus is not None and left_commands and right_bus is not None and right_commands:
        errors: list[BaseException] = []

        def send_side(bus: RobStrideAtMotorsBus, commands: dict[str, tuple[float, float, float, float, float]]) -> None:
            try:
                bus._mit_control_batch(commands)
            except BaseException as exc:
                errors.append(exc)

        left_thread = threading.Thread(target=send_side, args=(left_bus, left_commands), daemon=True)
        right_thread = threading.Thread(target=send_side, args=(right_bus, right_commands), daemon=True)
        left_thread.start()
        right_thread.start()
        left_thread.join()
        right_thread.join()
        if errors:
            raise errors[0]
        return

    if left_bus is not None and left_commands:
        left_bus._mit_control_batch(left_commands)
    if right_bus is not None and right_commands:
        right_bus._mit_control_batch(right_commands)


def enable_with_immediate_hold(
    selected: list[str],
    frame_targets: dict[str, float],
    left_bus: RobStrideAtMotorsBus | None,
    right_bus: RobStrideAtMotorsBus | None,
    lerobot_calibration: dict,
    default_kp: float,
    default_kd: float,
    log_handle: TextIO | None = None,
) -> None:
    groups: list[tuple[str, RobStrideAtMotorsBus, list[str]]] = []
    if left_bus is not None:
        groups.append(("left", left_bus, [name for name in selected if name.startswith("left_")]))
    if right_bus is not None:
        groups.append(("right", right_bus, [name for name in selected if name.startswith("right_")]))

    startup_enable_start = time.monotonic()
    enabled_names_by_side: dict[str, list[str]] = {side: [] for side, _, _ in groups}
    write_log_event(
        log_handle,
        "startup_enable_begin",
        selected=selected,
        startup_targets_deg=float_dict({name: frame_targets[name] for name in selected}),
    )
    max_group_len = max((len(names) for _, _, names in groups), default=0)
    sequence_idx = 0
    for name_idx in range(max_group_len):
        for bus_label, bus, names in groups:
            if name_idx >= len(names):
                continue
            name = names[name_idx]
            motor_id = bus.motors[name].id
            kp, kd, torque = joint_gains(name, default_kp, default_kd, lerobot_calibration)
            write_log_event(
                log_handle,
                "startup_enable_motor",
                sequence_idx=int(sequence_idx),
                side=bus_label,
                name=name,
                motor_id=int(motor_id),
                target_deg=float(frame_targets[name]),
                kp=float(kp),
                kd=float(kd),
                torque_nm=float(torque),
                phase="before_enable",
                elapsed_s=float(time.monotonic() - startup_enable_start),
            )
            bus.bus.send_raw(robstride_can_id(0x03, bus.bus.host_id, motor_id), bytes(8))
            bus._mit_control(name, kp=kp, kd=kd, position_degrees=frame_targets[name], torque=torque)
            write_log_event(
                log_handle,
                "startup_enable_motor",
                sequence_idx=int(sequence_idx),
                side=bus_label,
                name=name,
                motor_id=int(motor_id),
                target_deg=float(frame_targets[name]),
                kp=float(kp),
                kd=float(kd),
                torque_nm=float(torque),
                phase="after_hold",
                elapsed_s=float(time.monotonic() - startup_enable_start),
            )
            enabled_names_by_side[bus_label].append(name)
            sequence_idx += 1
        for bus_label, bus, _ in groups:
            active_names = enabled_names_by_side.get(bus_label, [])
            if not active_names:
                continue
            send_frame_targets(
                active_names,
                frame_targets,
                bus if bus_label == "left" else None,
                bus if bus_label == "right" else None,
                lerobot_calibration,
                default_kp,
                default_kd,
            )
            write_log_event(
                log_handle,
                "startup_hold_refresh",
                phase="interleave",
                side=bus_label,
                enabled_names=list(active_names),
                enabled_count=len(active_names),
                elapsed_s=float(time.monotonic() - startup_enable_start),
            )
    for bus_label, bus, names in groups:
        if not names:
            continue
        send_frame_targets(
            names,
            frame_targets,
            bus if names[0].startswith("left_") else None,
            bus if names[0].startswith("right_") else None,
            lerobot_calibration,
            default_kp,
            default_kd,
        )
        write_log_event(
            log_handle,
            "startup_hold_seed",
            side=bus_label,
            names=names,
            positions_deg=float_dict({name: frame_targets[name] for name in names}),
            elapsed_s=float(time.monotonic() - startup_enable_start),
        )
    for warmup_idx in range(3):
        for bus_label, bus, names in groups:
            if not names:
                continue
            send_frame_targets(
                names,
                frame_targets,
                bus if bus_label == "left" else None,
                bus if bus_label == "right" else None,
                lerobot_calibration,
                default_kp,
                default_kd,
            )
            write_log_event(
                log_handle,
                "startup_hold_refresh",
                phase="warmup",
                cycle_idx=int(warmup_idx),
                side=bus_label,
                enabled_names=list(names),
                enabled_count=len(names),
                elapsed_s=float(time.monotonic() - startup_enable_start),
            )
        time.sleep(0.01)
    write_log_event(
        log_handle,
        "startup_enable_complete",
        elapsed_s=float(time.monotonic() - startup_enable_start),
        motors_enabled=int(sequence_idx),
    )


def unwrap_12bit_delta(delta: int) -> int:
    if delta > 2048:
        return delta - 4096
    if delta < -2048:
        return delta + 4096
    return delta


class LeaderReader:
    def __init__(self, port: str, calibration: dict) -> None:
        self.port_name = port
        self.calibration = calibration
        self.port = PortHandler(port)
        self.packet = PacketHandler(0)
        self.connected = False

    def connect(self) -> None:
        if self.connected:
            return
        if not self.port.openPort():
            raise RuntimeError(f"Failed to open leader port {self.port_name}")
        if not self.port.setBaudRate(1_000_000):
            raise RuntimeError("Failed to set leader baudrate")
        self.connected = True

    def disconnect(self) -> None:
        if self.connected:
            self.port.closePort()
            self.connected = False

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
        names = selected_motors or list(self.calibration)
        out: dict[str, int] = {}
        for name in names:
            item = self.calibration[name]
            out[name] = self._read_raw_position(int(item["id"]))
        return out

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
