# KinRT RoboTwin Implementation Reference

Status: implementation reference for the active `kinrt_full` / `kinrt_lora` comparison

Reviewed: 2026-08-17

## Scope and terminology

This document uses **KinRT** to denote the implementation present in this
repository: offline action-motion clusters supervise a learned router that
selects residual feed-forward experts in the PI0.5 action-expert transformer.
No expansion of the KinRT acronym is inferred here because the source tree does
not define one.

The authoritative implementation path is:

1. `scripts/generate_router_labels.py` constructs frame-aligned action-motion labels.
2. `src/openpi/training/data_loader.py` aligns labels with LeRobot global indices and applies balanced sampling.
3. `src/openpi/models/pi0.py` derives routing context and adds supervised routing loss to flow-matching loss.
4. `src/openpi/models/gemma.py` computes router decisions and the residual mixture-of-experts branch.
5. `pi_model.py`, `script/policy_model_server.py`, and `script/eval_policy_client.py` execute and record evaluation.

## Offline label construction

For each dataset frame, the generator builds a future action chunk of horizon
50. Short episode tails are padded by repeating the final action. In the active
`chunk_velocity` mode, the feature vector concatenates the flattened action
chunk and its first temporal difference:

```text
feature = concat(flatten(a[t:t+50]), flatten(diff(a[t:t+50])))
```

For a 14-dimensional action vector this produces 1,386 features. Features are
standardized, reduced to 64 dimensions with Incremental PCA, and clustered with
KMeans using four clusters. The output `router_labels.npy` is indexed by the
LeRobot global `index` field, not by local dataloader position.

The label artifacts used by both compared configurations are identical:

| Property | Value |
| --- | --- |
| Episodes | 800 |
| Frames | 162,545 |
| Action horizon | 50 |
| Raw feature dimension | 1,386 |
| PCA dimension | 64 |
| Number of clusters | 4 |
| Cluster counts | [59,018, 74,748, 7,182, 21,597] |
| SHA-256 | `bceda7104ee949d37c9872a50c06842a111387ad5b63eb9011d50e19b33b256a` |

Cluster identifiers are empirical categories, not stable semantic class names.
Their numerical identities depend on the fitted KMeans model and must not be
reinterpreted without inspecting the associated trajectories and model file.

## Dataset alignment and sampling

The dataloader wraps each sample with its hard router label and a class weight.
A label of `-1` is invalid and is excluded from supervised routing loss. The
active configurations use uniform class-loss weights.

Both configurations enable replacement sampling with
`router_sampling_alpha=0.5`. For a sample in class `c`, the unnormalized
sampling weight is:

```text
w(c) = count(c)^(-0.5)
```

This reduces class imbalance without forcing a uniform sampled distribution.
It is independent of the class weight used inside router cross-entropy.

## Routing computation

PI0.5 first encodes the observation prefix, including visual inputs, language,
and the discretized state representation used by PI0.5. Valid prefix tokens are
masked-mean pooled to one context vector per sample.

For the active configuration:

- router type: dense;
- router input: pooled prefix context;
- experts: 4;
- selected experts: Top-1;
- action horizon: 50;
- router supervision coefficient: 0.05;
- load-balance, entropy, contrastive, dead-expert, and action-loss weighting coefficients: 0.

The dense router projects the pooled context, produces four logits, selects the
highest-scoring expert, and broadcasts that decision across the action tokens
and action-expert transformer depth. Each routed transformer block retains its
ordinary dense feed-forward path. The selected extra expert output is added as a
residual branch; it does not replace the original feed-forward computation.

Training minimizes the PI0.5 flow-matching action loss plus 0.05 times masked
router cross-entropy. Inference has no access to future actions or offline
cluster labels. It predicts the expert solely from the current observation
prefix, which is the intended train/inference information asymmetry.

## Configuration identity

The same configuration name has different optimization semantics on the two
hosts and must not be treated as an equivalent ablation:

| Setting | 4L20: `/private/RoboTwin` | L20yz: `/private/yth/projects/RoboTwin` |
| --- | --- | --- |
| PaliGemma variant | `gemma_2b` | `gemma_2b_lora`, rank 32 |
| Action-expert variant | `gemma_300m` | `gemma_300m_lora`, rank 64 |
| Trainable base parameters | All | Frozen by LoRA filter |
| Trainable router/MoE parameters | Yes | Yes |
| Training steps | 10,000 | 10,000 |
| Global batch size | 32 | 32 |
| FSDP devices | 1 | 2 |
| Save interval | 500 (default) | 1,000 |
| EMA | Disabled | Disabled |

The 4L20 entry is the canonical full-parameter RoboTwin KinRT configuration.
The L20yz entry is a historical LoRA port whose name is retained for checkpoint
compatibility. Comparisons must report this distinction explicitly.

## Evaluation contract

The model server shares immutable heavyweight policy weights and creates an
independent `PI0Session` for each socket connection. Observation windows,
language state, action queues, and router telemetry directories therefore do
not leak between clients.

The YAML fields `model_server_host`, `server_bind_host`, `port`,
`test_num`, `eval_result_root`, `router_info_root`, and
`record_router_info` are part of the evaluation contract. Each episode writes
a CSV-compatible result row with success, seed, elapsed time, action count, and
router telemetry directory. Numpy arrays are transferred with explicit dtype
and shape metadata; bfloat16 arrays are converted to float32 for portable JSON
transport.

L20yz additionally emits `eval_done` only after all requested episodes have a
result record. Its evaluation loop treats a model-transport failure as terminal
and treats other inference or simulation exceptions as a failed episode so the
remaining seeds can still be evaluated.

## Implementation differences beyond the active configuration

4L20 contains the broader routing implementation: per-layer and layer-group
routing, hard or soft labels, CSV label ingestion, expert-count validation,
straight-through action-gradient routing, and optional feed-forward activation
telemetry.

L20yz contains checkpoint-oriented extensions: automatic LoRA/MoE structure
detection, a `state_only` checkpoint layout, synchronous checkpoint options,
and offline experiment logging support.

Two L20yz fields require compatibility review before behavioral changes:

- `router_sampling_mix_beta` is present in configuration objects but is not
  consumed by the dataloader. Implementing it would change sampling for many
  existing experiments.
- `action_expert_moe_mlp_dim` reaches the Gemma configuration, but routed
  experts currently use `mlp_dim`. Changing this alters parameter shapes and
  may invalidate existing checkpoints.

Neither field is used to reinterpret the active KinRT result.
