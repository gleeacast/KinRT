# DIYRobot Lower-Host Read-Only Audit

Audit date: 2026-08-18

Scope: the active DIYRobot implementation under the legacy
`src/lerobot/robots/hi_arm` path on the lower host supplied for this review.

## Write and execution boundary

The audit used read-only shell operations to list files, count lines, search
symbols, and display bounded source ranges. It did not modify any file on the
lower host. It did not install software, open cameras, connect serial or CAN
devices, start a WebUI, collect data, enable torque, or command motion.

The previous RoboTwin comparison was not fully read-only. Its remote-write
record is preserved separately in `CHANGELOG_AND_SOURCE_MANIFEST.md`. That
earlier activity is not attributed to this DIYRobot audit.

## Active files reviewed

Core robot layer:

- `hi_arm.py`
- `config_hi_arm.py`
- `hi_arm.yaml`
- `robstride_at_bus.py`
- `vendor_usbcan/vendor_usbcan.py`

Current arm workflows:

- `dual_arm_teleop_strict_v22.py`
- `hiarm_leader_follower_teleop.py`
- `hiarm_pi05_record.py`
- `start_hiarm_pi05_record.sh`
- `start_hiarm_pi05_record_manual.sh`
- `hiarm_pi05_policy_client.py`
- `start_hiarm_pi05_policy_client.sh`
- `hiarm_pi05_policy_tasks_v3.sh`
- `hiarm_three_camera_webui.py`
- `remote_camera_webui.py`
- `hiarm_rest_range_calibrate.py`
- `hiarm_lerobot_calibrate.py`
- `hiarm_four_pose_calibrate.py`
- `hiarm_joint_mapping_calibrate.py`

Operational references:

- `HIARM_PI05_PIPELINE.md`
- `HIARM_HANDOFF_MANUAL.md`
- `HIARM_LEROBOT_CALIBRATION.md`
- `HIARM_SAFE_TELEOP_CALIBRATION.md`
- `HIARM_FOUR_POSE_CALIBRATION.md`
- `ROBSTRIDE_COMM_PROTOCOL.md`
- `README_WEBUI.md`

Backups, logs, cache directories, calibration backups, and historical strict
teleop versions were excluded unless needed to identify an active reference.

## Verified hardware contract

| Subsystem | Active source contract |
| --- | --- |
| Leader | 14 Feetech STS3215 servos; left IDs 1-7 and right IDs 8-14 |
| Followers | 14 RobStride joints; left IDs 1-7 and right IDs 11-17 |
| Base | Damiao wheel IDs 0x21, 0x22, and 0x23 |
| Lift | Damiao ID 0x24 plus a serial limit reader |
| Cameras | Right gripper, left gripper, and overhead RGB streams |
| Arm state | 14 absolute follower positions in degrees |
| Arm action | 14 absolute follower targets in degrees |

The generic DIYRobot class supports leader, follower, chassis, lift, limit
reader, and camera features through one robot abstraction. The current strict
teleop, recorder, and policy-client path uses separate left and right follower
ports. These are distinct connection models and must not be combined without a
code-level integration review.

## Joint order

The recorder, policy client, and DIYRobot OpenPI configuration agree on:

```text
left_shoulder_pan
left_shoulder_lift
left_elbow_flex
left_wrist_pitch
left_wrist_flex
left_wrist_roll
left_gripper
right_shoulder_pan
right_shoulder_lift
right_elbow_flex
right_wrist_pitch
right_wrist_flex
right_wrist_roll
right_gripper
```

The generic `hi_arm.py` class declares a different internal enumeration order
for the three wrist names. Its IDs are derived from that list. A deployment
must therefore use the joint order defined by the selected execution path and
must validate physical motor identity before sharing calibration or datasets
between the generic class and strict recorder/client path.

## Camera naming

Collection uses:

```text
right_gripper
left_gripper
overhead
```

The current policy launcher uses:

```text
cam_right_wrist
cam_left_wrist
cam_high
```

The DIYRobot training configs explicitly map collection fields to the policy
names. Older documents that describe `front`, `left_wrist`, or `right_wrist`
as direct dataset keys are not authoritative for the current pipeline.

## Dataset contract

The active recorder defaults to LeRobot v2.1 and writes:

- `observation.state`: follower feedback at the current frame;
- `action`: the follower target actually sent at the current frame;
- three `observation.images.*` streams;
- a natural-language task.

It supports automatic-duration episodes and manual keyboard-controlled
episodes. Resume mode validates whether a task is intended to reuse an
existing task index or append a new one. Strict-teleop JSONL logs are diagnostic
and are not compatible substitutes for the LeRobot dataset.

## Policy contract and safety gates

The current client accepts OpenPI WebSocket, a project-specific TCP protocol,
or an in-memory no-op policy. The lower host retains exclusive ownership of
cameras and follower buses.

Observed fail-closed checks include:

- explicit `--allow-motion` before commands are sent;
- refusal of unbounded live motion unless separately authorized;
- startup pose comparison against recorded follower rest;
- immediate hold after torque enable and drift monitoring during the guard;
- calibrated range clamps and joint-specific per-frame step limits;
- timeout rejection for stale policy responses;
- termination on stale follower feedback beyond the grace period;
- action shape and finiteness validation;
- torque disable and bus cleanup in the finalization path.

The launcher overrides some parser defaults. In particular, it currently uses
a 60-second policy timeout and executes up to 20 actions from each returned
chunk. A first physical validation should set one executed action per request
and a short duration. This is a deployment recommendation, not a change made
to the audited source.

## Documentation inconsistencies found

1. Historical manuals contain USB paths from different physical topologies.
   Stable device paths must be discovered on the current host.
2. Older policy documentation names collection camera keys as inference keys.
   The active policy launcher now uses `cam_high`, `cam_left_wrist`, and
   `cam_right_wrist`.
3. An older handoff section states that one action per chunk is the default.
   The current launcher overrides the parser with 20 steps.
4. The generic robot class and strict recorder/client path use different
   follower-port layouts and wrist enumeration orders.
5. The supplied paper table labels a DIYRobot column `Pull Bottle`, while the
   active prompt-v3 launcher specifies pulling a pill box onto a black pad.

These inconsistencies are documented instead of being resolved by modifying
the lower-host source, as the audit was explicitly read-only.

## Reproduction conclusion

The reviewed lower-host source is now distributed under
`hardware/diyrobot/lower_host`, with calibration, camera-alignment, and
provenance documentation. Robot-specific calibration and device identities are
deliberately excluded and must be generated on the target mechanism. CAD,
mechanical dimensions, wiring drawings, and task-mat geometry remain pending;
until those are released, this is a complete software integration release but
not a mechanically self-contained hardware distribution.
