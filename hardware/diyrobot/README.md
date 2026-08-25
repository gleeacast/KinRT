# DIYRobot Lower-Host Release

DIYRobot is the physical bimanual platform used for the KinRT real-world
experiments. The earlier internal name refers to this same platform, not a
second robot. Public source filenames, Python APIs, data identifiers, launchers,
UI text, and documentation now use DIYRobot. The old name remains only in
immutable provenance records for the audited source and historical checkpoints.

This directory contains a reviewed copy of the active lower-host source. The
source host was inspected and copied with read-only operations. No remote file
was changed and no camera, serial bus, actuator, calibration, collection, or
policy process was started during the audit.

## Release Status

Included now:

- Lower-host robot, bus, teleoperation, recording, policy-client, calibration,
  and camera-preview source.
- Verified electrical addressing, control units, camera roles, and safety
  thresholds.
- A public parameter register, core electromechanical BOM, mechanical
  provenance boundary, and third-party notices.
- Device-mapping, zeroing, camera-alignment, collection, and deployment
  procedures.
- Original-snapshot and public-release SHA-256 manifests.

Pending a later hardware release:

- Verified RobStride-adapted arm CAD, printable parts, and mechanical drawings.
- Original chassis and lift CAD, manufacturing drawings, and complete dimensions.
- Arm-link, frame, work-mat, camera-mount, and wiring dimensions.
- Manufacturer part numbers not already present in the core BOM.
- A printable task-mat reference and camera-intrinsic calibration.
- Transferable checkpoints and robot-specific calibration files.

Do not infer missing dimensions from photographs. Current software and
protocol reproduction is possible, but mechanically identical reconstruction
requires the pending artifacts.

## Safety Boundary

Real-robot execution requires a qualified operator and a tested physical power
disconnect. Source availability is not evidence that a command is safe for a
different assembly.

1. Keep one person responsible for the emergency stop or power disconnect.
2. Clear and support the full swept volume before any torque is enabled.
3. Discover device identities without sending motor scans or write commands.
4. Generate calibration on the target robot; never copy calibration JSON from
   another mechanism.
5. Pass software-only, passive-feedback, dry-policy, and bounded-motion gates
   in that order.
6. Stop on unexpected direction, drift, stale feedback, wrong camera mapping,
   invalid policy output, or collision risk.

Several calibration tools do not issue position targets but briefly enable
and disable a selected RobStride motor to obtain feedback. Treat them as
hardware-active tools.

## Verified Platform Contract

| Subsystem | Verified release contract |
| --- | --- |
| Leader arms | 14 Feetech STS3215 servos; left IDs 1-7, right IDs 8-14 |
| Follower arms | 14 RobStride O3 joints; left IDs 1-7, right IDs 11-17 |
| Mobile base | Three Damiao DM4310 motors; IDs `0x21`, `0x22`, `0x23` |
| Lift | Damiao DM4310 ID `0x24` with an ESP32 optical-limit reader |
| Cameras | Logitech C930e right wrist, Logitech C920 left wrist, ARKMICRO overhead |
| Camera stream | RGB, `640 x 480`, 30 FPS |
| Policy state/action | 14 absolute joint positions/targets in degrees |
| Collection rate | 20 FPS |
| Wheel radius | `0.05 m` |
| Base radius | `0.125 m` center-to-wheel |
| Maximum wheel speed | `100 RPM` |

The measured values above are software parameters from the active source, not
a complete mechanical specification.

## Mechanical Design and Attribution

DIYRobot combines a derived arm mechanism with an original mobile platform:

| Assembly | Public provenance | Current release state |
| --- | --- | --- |
| Follower arm | Derived from the Apache-2.0-licensed [TRLC-DK1 Follower](https://github.com/robot-learning-co/trlc-dk1) v0.2.0 working design and adapted for RobStride O3 actuators | Source-verified actuator contract and modification inventory released; final CAD validation pending |
| Three-wheel mobile base | Original DIYRobot mechanical design | Kinematics, addressing, control parameters, and safety limits released; approved CAD and complete manufacturing dimensions pending |
| Lift integration | Original DIYRobot system integration | Motor, limit sensing, homing, and base/lift interlocks released; travel, load rating, and mechanical drawings pending |

The current TRLC-DK1 repository publishes v0.3.0. It is a useful upstream
reference but is not asserted to be geometrically identical to the local
v0.2.0-derived DIYRobot assembly. The internal NX workspace also contains a
pre-adaptation STEP and generated import duplicates, so it is not uploaded as a
finished hardware release.

Released hardware references:

- [Public parameter register](PARAMETERS.md)
- [Core electromechanical BOM](BOM.csv)
- [Mechanical release and CAD audit](mechanics/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

All released derivative CAD must preserve TRLC-DK1 attribution and carry a
prominent modification notice. The chassis originality statement applies only
to the DIYRobot mechanical chassis and lift integration; purchased hardware and
third-party software retain their respective ownership and licenses.

## Control Topology

The lower host owns all cameras and actuator buses. The GPU host runs the KinRT
policy server. Observations travel from the lower host to the server; returned
14-D action chunks are validated and bounded on the lower host before motion.

The active strict recorder and policy client use separate left and right
follower ports. The generic `DIYRobot` class also retains a single-follower-bus
mode. Select one topology and keep its device map, calibration, and launcher
consistent; do not merge the two layouts implicitly.

## Install Into LeRobot

Use the LeRobot revision validated for the platform. This release does not
claim compatibility with every LeRobot version.

```bash
git clone <compatible-lerobot-repository> lerobot
cd lerobot
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m pip install pyserial opencv-python scservo-sdk

cp -a /path/to/KinRT/hardware/diyrobot/lower_host \
  src/lerobot/robots/diyrobot

python -c "from lerobot.robots.diyrobot import DIYRobot, DIYRobotConfig; print(DIYRobot.name)"
```

Install the OpenPI client used by the policy launcher:

```bash
python -m pip install -e /path/to/KinRT/policy/pi05/packages/openpi-client
```

The custom LeRobot revision may already provide some dependencies. Use one
environment and resolve import errors before connecting hardware.

## Create Stable Device Names

The release uses semantic paths under `/dev/diyrobot/` and deliberately omits
the original host's serial numbers and USB topology. On the target host, first
inspect devices without opening them:

```bash
ls -l /dev/serial/by-id /dev/serial/by-path
ls -l /dev/v4l/by-id /dev/v4l/by-path
lsusb
udevadm info --query=property --name=/dev/ttyUSB0
udevadm info --query=property --name=/dev/video0
```

Create `/etc/udev/rules.d/99-diyrobot.rules` using unique properties observed
on the target robot. Replace every placeholder; do not publish real serial
numbers in a shared configuration.

```udev
SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="<leader-serial>", SYMLINK+="diyrobot/leader", GROUP="dialout", MODE="0660"
SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="<left-follower-serial>", SYMLINK+="diyrobot/follower-left", GROUP="dialout", MODE="0660"
SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="<right-follower-serial>", SYMLINK+="diyrobot/follower-right", GROUP="dialout", MODE="0660"
SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="<chassis-serial>", SYMLINK+="diyrobot/chassis", GROUP="dialout", MODE="0660"
SUBSYSTEM=="tty", ENV{ID_SERIAL_SHORT}=="<limit-reader-serial>", SYMLINK+="diyrobot/lift-limits", GROUP="dialout", MODE="0660"
SUBSYSTEM=="video4linux", ENV{ID_SERIAL_SHORT}=="<right-camera-serial>", ATTR{index}=="0", SYMLINK+="diyrobot/camera-right-wrist", GROUP="video", MODE="0660"
SUBSYSTEM=="video4linux", ENV{ID_SERIAL_SHORT}=="<left-camera-serial>", ATTR{index}=="0", SYMLINK+="diyrobot/camera-left-wrist", GROUP="video", MODE="0660"
SUBSYSTEM=="video4linux", ENV{ID_PATH}=="<overhead-camera-path>", ATTR{index}=="0", SYMLINK+="diyrobot/camera-overhead", GROUP="video", MODE="0660"
```

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
find -L /dev/diyrobot -maxdepth 1 -type l -print -exec readlink -f {} \;
```

A device name is accepted only after the operator confirms its physical role.
Do not identify an actuator bus by trial motion.

## Reproduction Gates

### Gate A: software only

This must not open cameras or serial/CAN devices:

```bash
cd /path/to/lerobot/src/lerobot/robots/diyrobot
OFFLINE_SELF_TEST=1 ./start_diyrobot_pi05_policy_client.sh
```

Expected output reports 14 motors and an action shape of `(1, 14)` while
stating that hardware was not opened.

### Gate B: calibrate and align

Follow [CALIBRATION.md](CALIBRATION.md) on the target robot. Then use
[CAMERA_ALIGNMENT.md](CAMERA_ALIGNMENT.md) to verify the three semantic views
and the overhead pixel reference.

### Gate C: collect one bounded episode

```bash
TASK="Put the pill box into the center of the notebook." \
REPO_ID="local/diyrobot_smoke" ROOT="/data/diyrobot/smoke" \
EPISODES=1 EPISODE_TIME=20 RESET_TIME=0 FPS=20 \
./start_diyrobot_pi05_record.sh
```

Inspect metadata, parquet rows, all three videos, 14-D state/action values,
prompt identity, and episode outcome before collecting the benchmark dataset.

### Gate D: dry policy and bounded motion

Start with camera and passive-feedback validation without motion:

```bash
POLICY_SERVER="<gpu-host>:<port>" TASK_ID=0 \
  ./start_diyrobot_pi05_policy_client.sh
```

Only after the qualified operator approves every prior gate:

```bash
POLICY_SERVER="<gpu-host>:<port>" TASK_ID=0 \
ALLOW_MOTION=1 DURATION=30 ACTION_CHUNK_STEPS=1 \
  ./start_diyrobot_pi05_policy_client.sh
```

The first bounded trial should execute one action from each returned chunk.
Increase duration or chunk execution only after reviewing logs and physical
behavior.

## Data and Evaluation

The recorder writes LeRobot v2.1 with three RGB observations, 14-D follower
state, the 14-D target actually sent, and task text. The five canonical prompts
are in `lower_host/diyrobot_pi05_policy_tasks_v3.sh`.

The standard benchmark uses five tasks and 50 trials per task in the original
environment. An optional 100-episode training dataset was collected under
altered illumination. Results trained with that dataset must be reported as a
separate condition and evaluated with the same 50-trial-per-task protocol.

## Files and Provenance

See [SOURCE_MANIFEST.md](SOURCE_MANIFEST.md) for file purposes, exclusions,
compatibility identifiers, and every release-copy change. Verify hashes with:

```bash
python tools/validate_release.py --check-manifests
```

`REMOTE_SOURCE_SHA256.txt` applies to the original snapshot names and content;
use `RELEASE_SOURCE_SHA256.txt` to verify this sanitized public copy.

## Further Reading

- [Public parameter register](PARAMETERS.md)
- [Core electromechanical BOM](BOM.csv)
- [Mechanical release and CAD audit](mechanics/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Calibration and zeroing](CALIBRATION.md)
- [Camera and workspace alignment](CAMERA_ALIGNMENT.md)
- [Source and release manifest](SOURCE_MANIFEST.md)
- [KinRT DIYRobot web guide](../../docs/diyrobot.html)
