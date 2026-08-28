# RoboDojo Adaptation

KinRT integrates with RoboDojo through its XPolicyLab policy interface. The initial integration targets the benchmark's `arx_x5` environment in joint-control mode.

## Compatibility

| Interface | RoboDojo | KinRT |
| --- | --- | --- |
| Embodiment | Dual ARX X5 | Dual-arm Aloha-style input |
| State/action | `6 + 1 + 6 + 1` | 14 dimensions |
| Head camera | `cam_head` | `cam_high` |
| Wrist cameras | `cam_left_wrist`, `cam_right_wrist` | Same names |
| Observation frequency | 25 Hz | Converted with explicit metadata policy |
| Control type | `joint` | Supported |

RoboDojo and KinRT use the same packed order: left arm joints, left gripper, right arm joints, right gripper. `adapt_to_pi=False` prevents Aloha-specific joint and gripper remapping.

## Training Configurations

The OpenPI configuration registry provides:

- `pi05_lora_robodojo` (matched baseline without MoE)
- `kinrt_lora_robodojo`
- `kinrt_full_robodojo`

Both are KinRT. They share the same four experts, observation-conditioned top-1 router, action-and-velocity labels, supervised router loss, sampling policy, data transforms, action horizon, and batch size. The LoRA configuration freezes dense base parameters and trains LoRA, router, and expert parameters; the Full configuration updates all model parameters.

The dataset defaults to `RoboDojo-KinRT-arx_x5-joint`. Override it with `KINRT_ROBODOJO_REPO_ID`. Router labels default to `<HF_LEROBOT_HOME>/<repo_id>/meta/router_labels_k4/router_labels.npy` and can be overridden with `KINRT_ROBODOJO_ROUTER_LABELS_PATH`.

The first RoboDojo candidate artifacts use the shared dataset id
`RoboDojo-KinRT-stack_bowls100-arx_x5-joint` and controlled 10,000-step
overrides for both `pi05_lora_robodojo` and `kinrt_lora_robodojo`:

```text
num_train_steps=10000
batch_size=32
num_workers=8
fsdp_devices=4
save_interval=1000
seed=0
```

The registry keeps a 60,000-step default for future full-budget runs. Results
from the 10,000-step candidate must be labeled as such and must not be mixed
with a later 60,000-step checkpoint.

## XPolicyLab Adapter

On the RoboDojo host, install the adapter at:

```text
RoboDojo/XPolicyLab/policy/KinRT/
```

The adapter provides data conversion, label generation, normalization, LoRA/Full training, checkpoint resolution, policy serving, and debug/simulator evaluation. Its README is the operational guide and records the exact environment variables and commands.

## Temporal Convention

RoboDojo collects observations at 25 Hz. Its official Pi 0.5 converter writes 50 FPS into LeRobot metadata and uses the next recorded state as the action target. The KinRT adapter preserves that behavior by default so Pi 0.5 and KinRT remain directly comparable. Consequently, a 50-step chunk spans two seconds of source motion even though the metadata represents one second.

Experiments that set the metadata to 25 FPS must use a distinct dataset and result name. They answer a different temporal-ablation question and should not be mixed with the 50-FPS baseline.

## Validation Order

1. Load both RoboDojo KinRT configurations and verify their shared routing/data fields.
2. Validate one converted episode: three RGB cameras, 14-D state, 14-D next-state action, and a non-empty task string.
3. Generate K=4 labels and verify every selected global frame index has a valid label.
4. Compute normalization statistics for the selected LoRA or Full configuration.
5. Run XPolicyLab static checks and the debug closed loop, including encoded observations.
6. Run paired simulator evaluation over the same layouts for the baseline and KinRT.
7. Report three evaluation seeds with mean and standard deviation, then submit the exact artifact for hidden-layout verification.

Do not interpret an interface-only debug run with a checkpoint from another embodiment as a RoboDojo performance result.
