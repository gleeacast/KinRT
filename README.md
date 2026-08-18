# KinRT Reproduction Package

KinRT learns a global Top-1 router from action-derived kinematic labels and
uses only the current visual, language, and proprioceptive observation at
inference time. FULL and LoRA are parameterization choices for this one KinRT
method. They do not define different routing algorithms.

This release contains the audited KinRT policy sources, the confirmed paper
configuration registry, provenance records, validation tooling, and a static
documentation website. It does not include datasets, base checkpoints,
fine-tuned checkpoints, or a complete RoboTwin simulator checkout.

## Read this first

Reproduction has four independently checkable layers:

1. **KinRT method**: generate router labels, compute normalization statistics,
   train, and serve a checkpoint.
2. **RoboTwin**: place the KinRT policy overlay into a compatible RoboTwin 2.0
   checkout, then run clean and randomized evaluation.
3. **DIYRobot**: collect or obtain the five-task LeRobot dataset, use the
   retained paper configuration, and evaluate on the real platform.
4. **HiArm hardware**: discover ports, calibrate, pass preflight, run an
   offline policy check, then perform a supervised short-motion test.

Do not begin with real motion. Complete the software-only and dry policy gates
first. Keep a physical emergency stop or power disconnect within reach during
every powered test.

## Package layout

```text
KinRT/
|-- policy/pi05/                # Unified OpenPI/KinRT training workspace
|-- script/                     # RoboTwin evaluation integration
|-- configs/PAPER_MODELS.json   # Machine-readable Table 1 mapping
|-- docs/                       # GitHub Pages-ready documentation site
|-- manifests/                  # Source provenance and SHA-256 inventory
|-- tools/validate_release.py   # Source and website validation
|-- docs/FULL_VS_LORA.md        # Focused parameterization comparison
|-- docs/KINRT_IMPLEMENTATION.md # Routing implementation reference
|-- docs/PAPER_MODEL_CONFIG_MAP.md # Paper model evidence and mapping
|-- REAL_ROBOT_AUDIT.md         # Read-only HiArm implementation audit
`-- CHANGELOG_AND_SOURCE_MANIFEST.md
```

## External artifacts

Create or obtain these resources before training:

| Artifact | Required for | Verification |
| --- | --- | --- |
| Compatible RoboTwin 2.0 checkout | Simulation evaluation | `envs/`, `task_config/`, and `script/` exist |
| LeRobot dataset | Labels, norm stats, training | `meta/info.json`, `data/`, and image data exist |
| PI0.5 or PI0 base parameters | Initialization | The configured `params` path exists |
| `router_labels.npy` | KinRT training | One label slot exists for every global frame index |
| Normalization statistics | Training and inference | Stored under the config's asset ID |
| Fine-tuned checkpoint | Evaluation | The requested checkpoint step exists |
| HiArm lower-host source and calibration | Real-robot execution | Preflight reports all selected joints as valid |

The release intentionally uses placeholder or source-host paths in
`src/openpi/training/config.py`. Replace them with local paths. Never publish a
private host path as if it were a downloadable artifact.

## 1. Prepare a GPU environment

The pinned project requires Linux, Python 3.11 or newer, a CUDA 12-compatible
NVIDIA driver, Git, Git LFS, `uv`, and enough GPU memory for the selected
parameterization. Start with LoRA if hardware is limited.

```bash
cd /path/to/KinRT/policy/pi05
uv sync --frozen
uv run python -c "import jax; print(jax.devices())"
uv run python -c "import openpi; print('openpi import OK')"
```

Expected result: JAX lists at least one GPU and both commands exit with status
zero. FULL and LoRA are selected by configuration name in this same workspace.

## 2. Select the configuration

Use one name consistently for normalization, training, serving, and
evaluation.

| Target | Config | Parameterization |
| --- | --- | --- |
| RoboTwin, full parameters | `kinrt_full` | FULL |
| RoboTwin, LoRA | `kinrt_lora` | LoRA |
| DIYRobot, PI0.5 full | `kinrt_full_diyrobot` | FULL |
| DIYRobot, PI0.5 LoRA | `kinrt_lora_diyrobot` | LoRA |
| DIYRobot, PI0 full | `kinrt_full_pi0_diyrobot` | FULL |
| DIYRobot, PI0 LoRA | `kinrt_lora_pi0_diyrobot` | LoRA |
| DIYRobot, AdaMoE backbone | `kinrt_adamoe_diyrobot` | AdaMoE |

The remaining retained `*_diyrobot` entries are the four PI0/PI0.5 baselines.
See `docs/PAPER_MODEL_CONFIG_MAP.md` for the complete 14-row paper mapping. Five
paper rows use external frameworks and are not fabricated as OpenPI configs.

## 3. Verify the dataset contract

KinRT expects a LeRobot repository with a global integer `index`, episode and
frame indices, `observation.state`, `action`, a task/prompt, and three image
streams. For DIYRobot, the raw capture names are remapped during training:

```text
observation.images.overhead       -> cam_high
observation.images.left_gripper   -> cam_left_wrist
observation.images.right_gripper  -> cam_right_wrist
```

The state and action are 14 absolute follower targets in degrees, ordered as:

```text
left_shoulder_pan, left_shoulder_lift, left_elbow_flex,
left_wrist_pitch, left_wrist_flex, left_wrist_roll, left_gripper,
right_shoulder_pan, right_shoulder_lift, right_elbow_flex,
right_wrist_pitch, right_wrist_flex, right_wrist_roll, right_gripper
```

Inspect the dataset before label generation:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/data/lerobot/my_dataset")
info = json.loads((root / "meta" / "info.json").read_text())
print("episodes", info["total_episodes"])
print("frames", info["total_frames"])
print("data_path", info["data_path"])
print("features", sorted(info["features"]))
PY
```

Stop if the action dimension, joint order, units, or image mapping does not
match the configuration. Do not repair a semantic mismatch with normalization.

## 4. Generate KinRT router labels

Run from the chosen `policy/pi05` directory:

```bash
uv run python scripts/generate_router_labels.py \
  --repo-root /data/lerobot/my_dataset \
  --output-dir /data/lerobot/my_dataset/meta/router_labels_k4 \
  --action-horizon 50 \
  --feature-mode chunk_velocity \
  --pca-components 64 \
  --num-clusters 4 \
  --seed 0
```

Expected files include `router_labels.npy`, `sample_indices.npy`,
`router_label_model.joblib`, and `summary.json`. Check the summary and verify
that no selected sample has label `-1`:

```bash
python - <<'PY'
import json
import numpy as np
from pathlib import Path

root = Path("/data/lerobot/my_dataset/meta/router_labels_k4")
summary = json.loads((root / "summary.json").read_text())
labels = np.load(root / "router_labels.npy")
indices = np.load(root / "sample_indices.npy")
print(summary["cluster_counts"])
print("unlabeled selected frames", int((labels[indices] < 0).sum()))
PY
```

Cluster numbers are empirical KMeans identifiers, not fixed semantic class
names. Preserve the scaler, PCA model, KMeans model, seed, and dataset identity
with every result.

## 5. Point the config to local artifacts

Edit only the selected `TrainConfig` block in
`src/openpi/training/config.py`. Replace:

- `repo_id` with the LeRobot repository identity or local directory name;
- `router_labels_path` with the generated `router_labels.npy`;
- `weight_loader` with the matching PI0.5 or PI0 base `params` directory;
- `assets_dir` and `asset_id` with the normalization-statistics location;
- `checkpoint_base_dir` when the default output location is unsuitable.

Keep `action_horizon=50`, four experts, Top-1 routing, pooled router input, and
router-supervision coefficient `0.05` for the reported KinRT setting.

Confirm the registry after editing:

```bash
uv run python - <<'PY'
from openpi.training import config

cfg = config.get_config("kinrt_lora")
print(cfg.name, cfg.model.action_expert_num_moe_experts)
print(cfg.data.repo_id, cfg.num_train_steps, cfg.batch_size)
PY
```

Change `kinrt_lora` to the selected config name.

## 6. Compute normalization statistics and train

```bash
uv run python scripts/compute_norm_stats.py --config-name kinrt_lora
uv run python scripts/train.py kinrt_lora --exp-name kinrt_reproduction
```

Use `--resume` only when the exact experiment directory already contains a
compatible checkpoint. A first run should not silently resume an unrelated
experiment. The reported RoboTwin configs run 10,000 steps with global batch
size 32; the retained DIYRobot paper configs run 8,000 steps with batch size
32. GPU count is part of each config and may require adjustment for available
hardware, which changes throughput but not the routing definition.

Verify at the first checkpoint:

- the total loss and router cross-entropy are finite;
- all four classes occur in the training stream;
- router accuracy is above random only after learning begins;
- checkpoint parameters and matching normalization assets are both present.

## 7. Reproduce RoboTwin

Obtain a compatible RoboTwin 2.0 checkout and install its simulator assets.
The release is a policy overlay, not a complete simulator copy. Use a fresh
RoboTwin worktree for each experiment:

```bash
export ROBOTWIN_ROOT=/work/RoboTwin
cp -a /path/to/KinRT/policy/. "$ROBOTWIN_ROOT/policy/"
cp -a /path/to/KinRT/script/. "$ROBOTWIN_ROOT/script/"
cd "$ROBOTWIN_ROOT/policy/pi05"
uv sync --frozen
```

Configure `policy/pi05/deploy_policy.yml` from
`deploy_policy.example.yml`. At minimum set `task_name`, `task_config`,
`train_config_name`, `model_name`, `checkpoint_id`, `checkpoint_dir`, GPU IDs,
host, port, `test_num`, and output directories.

Single-process evaluation:

```bash
cd "$ROBOTWIN_ROOT/policy/pi05"
bash eval.sh <task_name> <task_config> kinrt_full \
  kinrt_reproduction 0 0
```

Remote evaluation uses one process for the model and one for the simulator:

```bash
# GPU/model host
bash serve_remote_model.sh deploy_policy.yml

# RoboTwin evaluator host
MODEL_SERVER_HOST=<reachable-ip> bash eval_remote.sh <task_name> 0
```

Run 100 trials for every clean and randomized task when reproducing the paper
table. Record the task config, seeds, checkpoint step, success count, and router
telemetry directory. See `docs/robotwin.html` for the eight-task checklist and
failure isolation rules.

## 8. Reproduce DIYRobot

The current recorder writes LeRobot v2.1 directly. On the HiArm lower host,
discover device paths first, complete calibration, and run one short smoke
episode:

```bash
cd /path/to/lerobot/src/lerobot/robots/hi_arm
ls -l /dev/serial/by-path
ls -l /dev/v4l/by-id

TASK="Put the pill box into the center of the notebook." \
REPO_ID="local/hiarm_smoke" \
ROOT="/data/hiarm_smoke" \
EPISODES=1 EPISODE_TIME=20 RESET_TIME=0 FPS=20 \
./start_hiarm_pi05_record.sh
```

Expected layout:

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
meta/episodes_stats.jsonl
data/chunk-000/episode_000000.parquet
videos/chunk-000/observation.images.<camera>/episode_000000.mp4
```

After video and action/state inspection, collect the five tasks, transfer the
unchanged dataset directory to the GPU host, select
`kinrt_lora_diyrobot` or `kinrt_full_diyrobot`, and repeat Steps 4-6. The
paper uses 50 evaluation trials per DIYRobot task. Never mix failed episodes
into a success-only dataset unless the training protocol explicitly requires
them.

## 9. Deploy on HiArm

The real-robot source was audited read-only. It is not duplicated or modified
by this package. The observed stable entry points are:

```text
dual_arm_teleop_strict_v22.py
hiarm_pi05_record.py
start_hiarm_pi05_record.sh
start_hiarm_pi05_record_manual.sh
hiarm_pi05_policy_client.py
start_hiarm_pi05_policy_client.sh
```

First run the lower-host software-only gate. It opens no cameras, serial ports,
CAN interfaces, or motors:

```bash
OFFLINE_SELF_TEST=1 ./start_hiarm_pi05_policy_client.sh
```

Then serve the trained checkpoint on the GPU host and run a dry policy check.
Omit `ALLOW_MOTION`; the client opens cameras and follower feedback, performs
preflight, requests one action, and prints clamped targets without sending
motor commands:

```bash
POLICY_SERVER="<gpu-host>:<port>" \
TASK="Put the pill box into the center of the notebook." \
./start_hiarm_pi05_policy_client.sh
```

Only after calibration, preflight, camera validation, server-contract
validation, and dry policy output have been reviewed by a qualified operator,
run a short supervised motion test:

```bash
POLICY_SERVER="<gpu-host>:<port>" \
TASK="Put the pill box into the center of the notebook." \
ALLOW_MOTION=1 DURATION=30 \
./start_hiarm_pi05_policy_client.sh
```

The client rejects stale policy responses, stale follower feedback, startup
mismatch, invalid action dimensions, and unbounded live duration. It clamps
targets to calibrated ranges and applies per-frame step limits. These checks do
not replace physical supervision or an emergency stop.

See `REAL_ROBOT_AUDIT.md` and `docs/real-robot.html` before connecting hardware.

## 10. Validate and open the website

```bash
python tools/validate_release.py --check-manifests
python -m http.server 8080 --directory docs
```

Open `http://127.0.0.1:8080/`. The site has no build step or third-party
runtime dependency and can be published from `docs/` with GitHub Pages.

## Reproducibility boundary

A source audit can validate configuration identity, syntax, path contracts,
and evaluation protocol. It cannot reproduce benchmark numbers without the
original or equivalently constructed datasets, base weights, simulator assets,
fine-tuned checkpoints, and physical hardware. Every public result should name
the exact config, source revision, dataset digest, router-label digest,
normalization assets, checkpoint step, seeds, and evaluation condition.
