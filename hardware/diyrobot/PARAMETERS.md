# DIYRobot Public Parameter Register

This register separates values verified in the released DIYRobot source from
mechanical values that still require CAD or physical measurement. It is the
normative public parameter reference for the current hardware release.

## Evidence Levels

| Level | Meaning |
| --- | --- |
| Verified | Read directly from the active lower-host source or confirmed source audit. |
| Upstream reference | Published by TRLC-DK1; useful for provenance, but not asserted as final DIYRobot geometry. |
| Pending | Not present in the audited source and not safe to infer from photographs. |

## Design Provenance

| Assembly | Design status | Public statement |
| --- | --- | --- |
| Follower-arm mechanism | Derived design | DIYRobot adapts the TRLC-DK1 Follower mechanism for RobStride O3 actuators. The local CAD filename identifies TRLC-DK1 v0.2.0 as the mechanical starting point. |
| Motor adapters and affected arm links | DIYRobot modifications | Adapter, shaft-extension, bearing-extension, link, cable-cover, and mount parts were modified for the actuator integration. Final release geometry is pending assembly verification. |
| Three-wheel mobile base | Original DIYRobot mechanical design | The mobile chassis geometry and system integration were designed by the DIYRobot team. Only the source-verified kinematic dimensions are released in this revision. |
| Lift integration | Original DIYRobot system integration | The lift actuation, base-speed interlock, and optical-limit integration are part of the DIYRobot platform. Mechanical travel and load ratings are pending. |
| Lower-host software | Mixed provenance | The release integrates LeRobot interfaces and vendor transports with DIYRobot-specific control, safety, collection, and deployment code. See `THIRD_PARTY_NOTICES.md`. |

The currently audited TRLC-DK1 repository is Apache-2.0 licensed and publishes
Follower CAD, printable parts, and a URDF. The current upstream v0.3.0 geometry
must not be presented as the exact source revision or geometry of the local
v0.2.0-derived DIYRobot assembly.

## System Contract

| Parameter | Released value | Evidence |
| --- | --- | --- |
| Platform | Bimanual mobile manipulator with lift | Verified |
| Leader joints | 14 Feetech STS3215 servos | Verified |
| Follower joints | 14 RobStride O3 actuators | Verified |
| Policy state/action | 14 absolute joint positions/targets | Verified |
| Arm units | Degrees | Verified |
| Policy and collection loop | 20 Hz | Verified |
| Camera capture | RGB, 640 x 480, 30 FPS | Verified |
| Standard real-world dataset | Five tasks, 500 demonstrations total | Verified release protocol |
| Standard evaluation | 50 trials per task | Verified release protocol |
| Optional OOD-lighting data | 100 additional training episodes | Verified release protocol |

## Arm Addressing and Order

The active recorder and policy client use the following 14-D order. This order
is part of the checkpoint contract and must not be changed implicitly.

```text
left_shoulder_pan, left_shoulder_lift, left_elbow_flex,
left_wrist_pitch, left_wrist_flex, left_wrist_roll, left_gripper,
right_shoulder_pan, right_shoulder_lift, right_elbow_flex,
right_wrist_pitch, right_wrist_flex, right_wrist_roll, right_gripper
```

| Side | Leader IDs | Follower IDs | Joint order |
| --- | --- | --- | --- |
| Left | 1-7 | 1-7 | shoulder pan, shoulder lift, elbow flex, wrist pitch, wrist flex, wrist roll, gripper |
| Right | 8-14 | 11-17 | shoulder pan, shoulder lift, elbow flex, wrist pitch, wrist flex, wrist roll, gripper |

The generic `DIYRobot` class retains a different wrist enumeration inherited
from its compatibility path. Use the strict recorder/client order above for
KinRT datasets and checkpoints.

## Arm Control and Policy Guards

| Parameter | Generic robot class | Active KinRT policy client |
| --- | ---: | ---: |
| Follower position `kp` | 20.0 | 18.0 |
| Follower damping `kd` | 0.5 | 0.8 |
| Nominal arm step cap | Configuration-dependent | 0.75 deg/frame |
| Arm catch-up cap | Configuration-dependent | 2.40 deg/frame |
| Wrist step / catch-up cap | Configuration-dependent | 1.20 / 4.20 deg/frame |
| Gripper step / catch-up cap | Configuration-dependent | 4.5 / 10.0 deg/frame |
| Catch-up error interval | Configuration-dependent | 0.15 to 0.8 deg |
| Startup mismatch limit | Configuration-dependent | 7.0 deg |
| Startup hold drift limit | Configuration-dependent | 1.0 deg |
| Feedback maximum age | Configuration-dependent | 0.30 s |
| Policy response timeout | Not applicable | 0.50 s |
| Returned chunk steps executed by default | Not applicable | 1 |
| Follower transmit minimum gap | Transport-dependent | 0.003 s |

These are software defaults, not permission to apply them to an unverified
assembly. Joint ranges and zero offsets are target-specific calibration
artifacts and must be generated on each robot.

## Mobile Base

The custom base uses three equally spaced omni-wheel modules. The released
kinematic transform uses body-frame commands `(x, y, theta)` where `x` and `y`
are metres per second and `theta` is degrees per second.

| Parameter | Released value | Evidence |
| --- | ---: | --- |
| Wheel motors | 3 x Damiao DM4310 | Verified |
| Command IDs | left `0x21`, back `0x22`, right `0x23` | Verified |
| Receive IDs | left `0x31`, back `0x32`, right `0x33` | Verified |
| Wheel radius | 0.050 m | Verified software kinematics |
| Center-to-wheel radius | 0.125 m | Verified software kinematics |
| Wheel-center azimuths | left 240 deg, back 0 deg, right 120 deg | Verified software kinematics |
| Tangential drive-axis angles | left 150 deg, back -90 deg, right 30 deg | Verified software kinematics |
| Wheel speed cap | 100 RPM (10.472 rad/s) | Verified |
| Maximum nominal rim speed | 0.524 m/s | Derived from released radius and speed cap |
| CAN bitrate | 1,000,000 bit/s | Verified |
| CAN FD | Disabled | Verified |
| Wheel `kp` | 0.0 | Verified |
| Wheel `kd` | 3.0 | Verified |
| Wheel torque feed-forward | 3.0 | Verified source value; vendor units apply |
| Test velocity | 5.0 rad/s | Verified |
| Direction signs | left +1, back +1, right +1 | Verified release default |

The overall chassis outline, plate thickness, wheel model, mounting-hole
pattern, ground clearance, mass, payload, and printable/manufacturing files are
pending because no approved chassis CAD was present in the audited materials.

## Lift

| Parameter | Released value | Evidence |
| --- | ---: | --- |
| Lift motor | Damiao DM4310 | Verified |
| Command / receive ID | `0x24` / `0x34` | Verified |
| Position controller `kp` / `kd` | 20.0 / 0.5 | Verified generic class |
| Automatic homing | Disabled by default | Verified |
| Homing direction | -1 | Verified |
| Homing step | 2.0 deg | Verified |
| Homing `kp` / `kd` | 8.0 / 0.3 | Verified |
| Homing settle time | 0.2 s | Verified |
| Zero position | 0.0 deg | Verified release default |
| Limit reader | ESP32 with top and bottom optical sensors | Verified |
| Limit serial rate | 115200 bit/s | Verified |
| Limit-data stale timeout | 1.0 s | Verified |
| Action on stale limit data | Hold current lift target | Verified |
| Base speed scale below 30 deg | 1.0 | Verified |
| Base speed scale from 30 to 90 deg | 0.6 | Verified |
| Base speed scale at or above 90 deg | 0.3 | Verified |
| Fast-base threshold | 0.2 m/s | Verified |

When planar base speed exceeds 0.2 m/s, the controller blocks a command that
would raise the lift. Top and bottom optical limits block further motion toward
the asserted limit.

## Cameras and Workspace

| View | Hardware | Collection key | Policy key | Format |
| --- | --- | --- | --- | --- |
| Overhead | ARKMICRO | `overhead` | `cam_high` | 640 x 480 at 30 FPS |
| Left wrist | Logitech C920 | `left_gripper` | `cam_left_wrist` | 640 x 480 at 30 FPS |
| Right wrist | Logitech C930e | `right_gripper` | `cam_right_wrist` | 640 x 480 at 30 FPS |

The current overhead reference rectangle is `(128,151)` to `(458,416)` at
640 x 480. It is a browser alignment overlay, not a metric calibration and not
an inference-time homography. Camera intrinsics, extrinsics, exposure settings,
mount dimensions, task-mat dimensions, and a printable reference remain
pending.

## Mechanical Values Still Required for a Complete Build

- Final verified STEP and printable STL/3MF files for the RobStride-adapted arm.
- Link dimensions, actuator-interface tolerances, fastener map, and print settings.
- Original chassis CAD, overall dimensions, plate material/thickness, wheel SKU,
  mounting pattern, ground clearance, mass, and payload.
- Lift rail/screw or transmission specification, stroke, limit positions, speed,
  rated load, and mount drawings.
- Camera-mount geometry, intrinsics, extrinsics, and task-mat dimensions.
- A complete procurement BOM with supplier part numbers for custom and standard parts.

Until these fields are released, the repository supports software and protocol
reproduction but not a claim of mechanically identical reconstruction.
