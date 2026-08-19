# DIYRobot Source and Release Manifest

Initial read-only audit: 2026-08-18

Dependency confirmation: 2026-08-19
Remote scope: `~/lerobot/src/lerobot/robots/hi_arm`

## Remote-Write Statement

The DIYRobot source host was accessed only to list, read, hash, and copy files.
No remote source, documentation, configuration, calibration, log, dataset, or
system file was modified. No camera, serial/CAN bus, motor, WebUI, collection,
calibration, teleoperation, or policy process was started.

## Included Source

| Area | Included files | Purpose |
| --- | --- | --- |
| LeRobot integration | `__init__.py`, `config_diyrobot.py`, `diyrobot.py`, `diyrobot.yaml` | Robot/config registration, arms, base, lift, cameras |
| Bus drivers | `robstride_at_bus.py`, `limit_reader.py`, `vendor_usbcan/*` | RobStride AT transport, optical limits, Damiao vendor CDC |
| Teleoperation | `diyrobot_dual_arm_teleop_strict.py`, `diyrobot_leader_follower_teleop.py` | Strict and basic leader/follower control |
| Calibration | `diyrobot_rest_range_calibrate.py`, `diyrobot_lerobot_calibrate.py`, `diyrobot_four_pose_calibrate.py`, `diyrobot_joint_mapping_calibrate.py` | Rest/range, LeRobot, pose, and per-joint records |
| Collection | `diyrobot_pi05_record.py`, `start_diyrobot_pi05_record.sh`, `start_diyrobot_pi05_record_manual.sh` | LeRobot v2.1 automatic and manual episodes |
| Policy deployment | `diyrobot_pi05_policy_client.py`, `start_diyrobot_pi05_policy_client.sh`, `diyrobot_pi05_policy_tasks_v3.sh` | Observation construction, policy transport, guarded action execution |
| Camera preview | `remote_camera_webui.py`, `diyrobot_three_camera_webui.py`, `start_diyrobot_three_camera_webui.sh` | Three-view inspection and overhead reference overlay |

`hiarm_four_pose_calibrate.py` was fetched after the initial directory copy
because `hiarm_joint_mapping_calibrate.py` imports it. Its presence on the
source host was confirmed read-only on 2026-08-19.

## Naming Boundary

DIYRobot is the only public platform and runtime name. The release source uses
`DIYRobot` / `DIYRobotConfig`, the `diyrobot` registration key,
`diyrobot_tcp`, and `diyrobot_*` filenames. The old internal name appears only
in the audited remote path, original-source hashes, and immutable historical
checkpoint identifiers. These provenance strings do not define a second
platform or a public runtime alias.

## Release-Copy Changes

Control algorithms, motor IDs, gains, safety limits, timing, joint order, task
prompts, data schema shapes, and camera dimensions were retained. The public
copy differs from the source-host snapshot only in these release concerns:

1. Public source filenames, Python APIs, config keys, launchers, and UI strings use DIYRobot.
2. Python comments, docstrings, shell comments, and WebUI text are English.
3. Personal home and virtual-environment paths were replaced by script-relative
   paths, `PYTHON_BIN`, `LEROBOT_SRC`, `DATA_ROOT`, `/data/diyrobot`, or
   `/tmp/diyrobot` defaults. Vendor transport monitoring now uses the public
   `DIYROBOT_VENDOR_MONITOR_LOG` environment variable.
4. The private policy-server default was removed; network policy modes now
   require `POLICY_SERVER`.
5. Camera serial numbers and machine-specific USB topology were replaced by
   semantic `/dev/diyrobot/*` paths created on the target host.
6. Dataset IDs, log prefixes, robot-type metadata, and calibration filenames
   use `diyrobot`.
7. The release entry points were renamed to `diyrobot_*`; source-host names are
   retained only in the original hash manifest.
8. Calibration comments now disclose brief follower enable/disable and
   zero-gain feedback probing instead of describing those tools as hardware-inert.
9. Five Chinese WebUI labels were translated to concise English.
10. New public documentation and deterministic hash manifests were added.

## Deliberately Excluded

- `__pycache__`, bytecode, logs, backups, process files, and temporary output.
- Robot-specific calibration and overhead-calibration JSON.
- Datasets, checkpoints, normalization assets, router labels, and telemetry.
- Camera serial numbers, internal network defaults, user home paths, SSH data,
  and device-topology values from the source host.
- Older local manuals containing Chinese text, stale port maps, or internal
  machine details.
- Mechanical dimensions, CAD, wiring drawings, and task-mat geometry that have
  not yet been approved for public release.

## Known Source Boundaries

- The generic robot class uses one follower-port abstraction; the active strict
  recorder/client uses separate left and right follower buses.
- The generic class historically enumerates wrist joints differently from the
  strict 14-D policy path. Joint identity must be verified before reuse.
- The overhead WebUI reference JSON is not consumed by collection or policy
  preprocessing; current reproduction requires physical pixel alignment.
- Model checkpoints, LeRobot source revision, and mechanical release artifacts
  remain external.

## Hashes

`REMOTE_SOURCE_SHA256.txt` records the unmodified source-host snapshot.
`RELEASE_SOURCE_SHA256.txt` records files in `lower_host/` after the documented
sanitization. A differing hash is expected for every edited release file and
is not evidence of an undisclosed control change; use the numbered list above
and Git diff for review.
