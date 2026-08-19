# DIYRobot Camera and Workspace Alignment

KinRT uses three RGB streams at `640 x 480` and 30 FPS:

| Semantic view | Hardware observed on the audited robot | Collection key | Policy key |
| --- | --- | --- | --- |
| Right wrist | Logitech C930e | `right_gripper` | `cam_right_wrist` |
| Left wrist | Logitech C920 | `left_gripper` | `cam_left_wrist` |
| Overhead | ARKMICRO | `overhead` | `cam_high` |

Device serial numbers are intentionally absent from the public release. Map
each camera to a stable `/dev/diyrobot/*` path on the target host and verify the
semantic view visually before collection or evaluation.

## What the Current Code Does

The camera preview displays a movable four-point reference overlay and can save
normalized points to `overhead_calibration.json`. The recorder and policy
client do not read that file. They do not undistort, rectify, crop, or apply a
homography to the overhead image.

Therefore, the current checkpoint contract depends on physical alignment.
Matching camera height alone is insufficient: yaw, pitch, roll, lens field of
view, principal point, exposure, task-mat pose, and image-device mapping also
affect the observation.

## Current Overhead Pixel Reference

At `640 x 480`, the audited preview uses this canonical rectangle:

| Quantity | Pixels | Normalized coordinate |
| --- | ---: | ---: |
| Left | `128` | `0.200000` |
| Top | `151` | `0.314583` |
| Right | `458` | `0.715625` |
| Bottom | `416` | `0.866667` |
| Width | `330` | `0.515625` |
| Height | `265` | `0.552083` |

Corner order is top-left `(128,151)`, top-right `(458,151)`, bottom-right
`(458,416)`, and bottom-left `(128,416)`.

These values reproduce the browser overlay only. They are not a metric camera
calibration and are not applied to training or inference frames.

## Alignment Procedure for the Current Checkpoint

1. Rigidly mount all three cameras and lock focus, resolution, frame rate, and
   orientation where the hardware permits.
2. Start the preview only while collection and policy clients are stopped:

   ```bash
   ./start_diyrobot_three_camera_webui.sh
   ```

3. Confirm left/right wrist identity, upright orientation, frame freshness,
   focus, exposure, occlusion, and gripper visibility.
4. Open the overhead calibration page and display the canonical rectangle.
5. Place the task mat so its published reference boundary coincides with all
   four rectangle corners. Adjust the physical camera mount and mat pose; do
   not digitally stretch only the preview.
6. Capture and archive a reference frame with the mount revision, camera model,
   resolution, exposure settings, date, and operator.
7. Stop the preview before recording or policy inference so the camera devices
   have a single owner.

The physical task-mat dimensions and printable reference image are pending the
mechanical release. Until those are published, another laboratory can match the
pixel rectangle but cannot prove metric workspace equivalence.

## Acceptance Checks

Before each session, compare a fresh still image against the approved reference:

- All four task-mat reference corners agree within the declared pixel tolerance.
- No camera has been mirrored, rotated, or swapped.
- The gripper and tool occupy the expected image region at the rest pose.
- Exposure and white balance do not clip task objects or indicators.
- The three frames are current and acquired at the declared format.
- The exact same preprocessing keys are used for training and inference.

The current source does not define a numeric pixel tolerance. Record the
observed error and do not claim equivalence until the project publishes one.

## Recommended Geometry Release

For reproduction independent of a specific camera mount, a future release
should include:

- Camera intrinsic matrices and distortion coefficients.
- Fiducial type, IDs, physical dimensions, and mat coordinates.
- Camera-to-base or camera-to-workspace extrinsics.
- A planar homography to a versioned canonical image.
- Canonical output size, interpolation, crop, and color conversion.
- One shared rectification implementation used by collection and inference.
- Reference images and an automated reprojection-error test.

Without these artifacts, physical pixel alignment is the only checkpoint-
compatible method supported by the current code.

## OOD Illumination Protocol

The project provides 100 additional trials under changed illumination as OOD
scenes. Keep camera geometry, task layout, checkpoint, and success criteria
fixed while changing only the declared lighting condition. Report these 100
trials separately from the five-task standard-lighting result.
