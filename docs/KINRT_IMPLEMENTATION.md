# KinRT Implementation Reference

Reviewed: 2026-08-17

## Definition and scope

In this package, KinRT denotes the implemented mechanism in which offline
action-motion clusters supervise a learned router that selects residual
feed-forward experts in the PI0.5 action-expert transformer. The source does not
define an expansion of the acronym, so this document does not infer one.

The authoritative path is:

1. Generate offline frame labels from future action chunks.
2. Align labels by the LeRobot global frame index.
3. Pool valid observation-prefix tokens into one routing context per sample.
4. Predict a discrete expert from that context.
5. Add the selected expert output to the ordinary dense FFN output.
6. Train with flow-matching loss plus masked router cross-entropy.

## Offline labels

For each frame, the active generator builds a 50-step future action chunk.
Episode tails are padded by repeating the final action. With 14 action
dimensions and `chunk_velocity` mode, the feature is:

```text
concat(flatten(action_chunk), flatten(diff(action_chunk)))
```

The resulting 1,386-dimensional feature is standardized, reduced to 64
dimensions with Incremental PCA, and clustered with four-cluster KMeans. Labels
are written by the dataset global `index`, not by dataloader position.

Both primary parameterizations use the same audited label artifact:

| Property | Value |
| --- | --- |
| Episodes | 800 |
| Frames | 162,545 |
| Action horizon | 50 |
| Raw feature dimension | 1,386 |
| PCA dimension | 64 |
| Clusters | 4 |
| Counts | `[59018, 74748, 7182, 21597]` |
| SHA-256 | `bceda7104ee949d37c9872a50c06842a111387ad5b63eb9011d50e19b33b256a` |

Cluster identifiers are empirical KMeans categories. Their numeric values are
not stable semantic class names and must not be assigned semantics without
examining the fitted artifacts and associated trajectories.

## Dataset alignment and sampling

The dataloader attaches a hard router label and class weight to each sample. A
label of `-1` is invalid and is excluded from supervised router loss. The active
configurations use uniform class-loss weights.

Both configurations use replacement sampling with
`router_sampling_alpha=0.5`. For class `c`, the unnormalized sample weight is:

```text
w(c) = count(c)^(-0.5)
```

This reduces imbalance without forcing the sampled class distribution to be
uniform. Sampling weights and cross-entropy class weights are separate
mechanisms.

## Routing and expert computation

PI0.5 encodes images, language, and its discretized state representation as an
observation prefix. A masked mean over valid prefix tokens produces one context
vector per sample. The active dense router maps this vector to four logits,
selects the Top-1 expert, and broadcasts that choice across action tokens and
routed transformer depth.

Each routed transformer block retains the ordinary dense FFN. The selected
expert is an additional residual branch; it does not replace the dense FFN.
The active auxiliary coefficients for load balance, entropy, contrastive loss,
dead-expert loss, and action-loss weighting are zero. The router supervision
coefficient is `0.05`.

Training therefore minimizes:

```text
flow_matching_loss + 0.05 * masked_router_cross_entropy
```

Offline labels are training targets only. At inference time, the router receives
the current observation prefix and has no access to future actions or cluster
labels.

## Evaluation state

Heavyweight policy parameters are shared, while a `PI0Session` owns mutable
observation windows, language state, action queues, and telemetry state. The
model server creates connection-local sessions when supported so client state
does not leak between connections.

Numpy responses carry dtype and shape metadata. Bfloat16 values are converted
to float32 for portable JSON transport. Episode records include success, seed,
elapsed time, action count, and router telemetry location.

## Compatibility constraints

Two compatibility fields are intentionally documented without behavior changes:

- `router_sampling_mix_beta` is represented in configuration objects but is
  not consumed by the dataloader.
- `action_expert_moe_mlp_dim` reaches model configuration, but routed experts
  continue to use `mlp_dim`.

Implementing either field would alter established sampling behavior or
checkpoint parameter shapes. Such changes require a separate compatibility
decision and are outside this source-preservation release.
