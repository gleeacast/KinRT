# DIYRobot Calibration and Zeroing

Calibration files describe one physical robot at one point in time. They are
not portable model parameters. Generate them on the target mechanism and keep
their hashes with every collection or evaluation session.

## Safety Prerequisites

Before running any command in this document:

- Assign a qualified operator to the physical power disconnect.
- Support both arms against an unexpected torque release.
- Clear the complete swept volume.
- Verify `/dev/diyrobot/*` mappings by physical role.
- Place leader and follower near a conservative rest pose.
- Stop camera, recording, teleoperation, and policy processes.
- Select one joint or one side for the first pass.

The follower feedback tools may briefly enable and disable a RobStride motor.
They do not issue a pose target, but they are not hardware-inert.

## Joint Identity

The strict recorder and policy client use this order:

```text
left_shoulder_pan, left_shoulder_lift, left_elbow_flex,
left_wrist_pitch, left_wrist_flex, left_wrist_roll, left_gripper,
right_shoulder_pan, right_shoulder_lift, right_elbow_flex,
right_wrist_pitch, right_wrist_flex, right_wrist_roll, right_gripper
```

Leader IDs are 1-7 on the left and 8-14 on the right. Follower IDs are
1-7 on the left and 11-17 on the right. Confirm the physical joint attached to
each ID before recording calibration.

The generic robot class historically enumerated the wrist names differently
from the strict path. Do not share a calibration until joint names, IDs, order,
units, and bus topology all match.

## Calibration Artifacts

| File | Purpose | Publication rule |
| --- | --- | --- |
| `diyrobot_rest_range_calibration.json` | Absolute rest pose and safe follower ranges | Robot-specific; do not publish as transferable |
| `diyrobot_lerobot_calibration.json` | Middle/range, drive mode, scaling, validation status | Robot-specific; do not publish as transferable |
| `leader_calibration.json` | Leader export used by the driver | Robot-specific |
| `follower_calibration.json` | Follower zero/range export | Robot-specific |
| `teleop_session_zero.json` | Historical session-zero compatibility | Do not use as the sole safety map |
| `diyrobot_joint_mapping.json` | Per-joint mapping samples | Diagnostic evidence |
| `diyrobot_calibration_poses.json` | Four named pose records | Reference only, not proof of safe range |

Back up the full set before recalibration. Never hand-edit one field to make a
preflight pass; repeat the relevant physical measurement instead.

## 1. Record Strict Rest and Ranges

Use the split follower buses. Start with one side and keep the other disabled.

```bash
cd /path/to/lerobot/src/lerobot/robots/diyrobot

python diyrobot_rest_range_calibrate.py rest --device both --side left
python diyrobot_rest_range_calibrate.py range --device follower --side left \
  --duration 20
python diyrobot_rest_range_calibrate.py show --device both --side left
```

During range capture, move each selected joint by hand through a conservative
working range. Do not contact or use a mechanical stop as the calibration
limit. Repeat for the right side only after reviewing the left-side output.

```bash
python diyrobot_rest_range_calibrate.py rest --device both --side right
python diyrobot_rest_range_calibrate.py range --device follower --side right \
  --duration 20
python diyrobot_rest_range_calibrate.py show --device both --side right
```

Export only after every selected joint contains leader rest, follower rest,
range minimum, and range maximum:

```bash
python diyrobot_rest_range_calibrate.py export --device both --side both \
  --update-follower-calibration
```

## 2. Build LeRobot Direction and Scale Records

The LeRobot-style tool accepts one follower port per invocation. On the active
split-bus layout, explicitly select the matching side and port.

```bash
python diyrobot_lerobot_calibrate.py middle --device both --side left \
  --follower-port /dev/diyrobot/follower-left
python diyrobot_lerobot_calibrate.py range --device both --side left \
  --follower-port /dev/diyrobot/follower-left --duration 20

python diyrobot_lerobot_calibrate.py middle --device both --side right \
  --follower-port /dev/diyrobot/follower-right
python diyrobot_lerobot_calibrate.py range --device both --side right \
  --follower-port /dev/diyrobot/follower-right --duration 20

python diyrobot_lerobot_calibrate.py show
python diyrobot_lerobot_calibrate.py export
```

Do not use `--allow-partial` for benchmark collection or policy motion.

## 3. Validate Mapping Per Joint

Whole-arm middle, zero, rotation, and rest poses cannot establish direction,
wrap continuity, scale, and safe range for every joint. Validate one joint at
a time under a separate physical test plan.

The mapping sampler records human-placed base, positive, and negative samples:

```bash
python diyrobot_joint_mapping_calibrate.py record \
  --side left --joint shoulder_pan --sample base \
  --follower-port /dev/diyrobot/follower-left
python diyrobot_joint_mapping_calibrate.py record \
  --side left --joint shoulder_pan --sample positive \
  --follower-port /dev/diyrobot/follower-left
python diyrobot_joint_mapping_calibrate.py record \
  --side left --joint shoulder_pan --sample negative \
  --follower-port /dev/diyrobot/follower-left
python diyrobot_joint_mapping_calibrate.py show
```

Repeat for all 14 joints. After an independently supervised, tiny-motion test
confirms direction and range, record the validation state:

```bash
python diyrobot_lerobot_calibrate.py validate \
  --device both --motors left_shoulder_pan
```

The `validate` command records an operator decision; it does not perform the
physical validation itself.

## 4. Review Coverage Before Use

A release-ready target robot must have, for every active joint:

- Correct leader and follower ID.
- Rest measurement.
- Conservative absolute follower range.
- Direction and scaling evidence.
- Explicit physical teleoperation validation.
- Calibration timestamp, operator, robot ID, and SHA-256 digest.

Run software-only policy validation before opening buses:

```bash
OFFLINE_SELF_TEST=1 ./start_diyrobot_pi05_policy_client.sh
```

Then run the strict startup preflight with motion disabled. A mismatch is a
hard stop. Investigate pose, device identity, joint order, calibration age,
and mechanical changes instead of widening thresholds.

## Lift Zeroing

The observed lift contract is:

| Parameter | Value |
| --- | ---: |
| Homing direction | `-1` |
| Homing step | `2 deg` |
| Homing gains | `kp=8`, `kd=0.3` |
| Settle interval | `0.2 s` |
| Software zero | `0 deg` |
| Low/high thresholds | `30 deg` / `90 deg` |

Automatic homing is disabled by default. Do not enable it until the optical
limit state is fresh, the direction is physically confirmed, the lift is
supported, and the stop path has been tested. A stale limit reader requires
holding lift position and aborting motion.

## Recalibration Triggers

Repeat the affected calibration after motor replacement, encoder or coupler
service, link or gripper changes, cable rerouting that changes motion, bus-ID
changes, firmware changes, mechanical impact, unexplained drift, or any
startup mismatch that cannot be explained by pose placement.
