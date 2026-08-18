# KinRT FULL and LoRA Parameterizations

KinRT has one routing implementation and one configuration registry. FULL and
LoRA are training choices, not separate source directories or different routing
methods.

## Primary RoboTwin configurations

| Property | `kinrt_full` | `kinrt_lora` |
| --- | --- | --- |
| Source workspace | `policy/pi05` | `policy/pi05` |
| PaliGemma branch | `gemma_2b` | `gemma_2b_lora`, rank 32 |
| Action expert | `gemma_300m` | `gemma_300m_lora`, rank 64 |
| Dense base parameters | Trainable | Frozen by `freeze_filter` |
| Router and routed experts | Trainable | Trainable |
| Experts / Top-K | 4 / 1 | 4 / 1 |
| Router context | Masked-mean pooled prefix | Masked-mean pooled prefix |
| Router type | Dense | Dense |
| Router supervision | 0.05 | 0.05 |
| Balanced sampling | `count(c)^(-0.5)` | `count(c)^(-0.5)` |
| Steps / batch | 10,000 / 32 | 10,000 / 32 |
| FSDP devices | 1 | 2 |
| Save interval | 500 | 1,000 |
| EMA | Disabled | Disabled |

Both configurations use action-derived K=4 labels and observation-only
inference. Results must identify the configuration and checkpoint because FULL
and LoRA produce separate trained parameter sets.

## Implementation contract

The shared source provides label alignment, balanced sampling, pooled-prefix
routing, router loss integration, routed residual FFNs, checkpoint structure
detection, per-connection policy sessions, and RoboTwin telemetry. Selecting a
FULL or LoRA config changes parameterization and resource settings only.

The distinction is expressed in
`policy/pi05/src/openpi/training/config.py`: FULL omits a freeze filter and uses
dense Gemma variants; LoRA selects LoRA variants and applies the corresponding
freeze filter. No alternate KinRT model file is selected.

## Evaluation contract

Use the same evaluation client, policy server, and policy session code for both
parameterizations. Keep config names, experiment directories, checkpoint IDs,
normalization assets, and telemetry directories distinct. Evaluating a FULL
checkpoint with a LoRA config, or the reverse, is invalid even though the
source implementation is shared.
