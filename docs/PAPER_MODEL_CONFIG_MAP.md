# Paper Model and Configuration Map

This audit covers all 14 model labels in Table 1. Every row has an identified
evaluation checkpoint. Nine rows are native OpenPI configurations retained in
this release. Five rows use independent training frameworks and therefore are
not represented as `TrainConfig` objects.

## Evidence levels

- **Confirmed OpenPI config**: the complete configuration block, training run,
  checkpoint, and evaluation alias agree.
- **Confirmed external run**: the external launch script, run directory, and
  evaluation checkpoint agree.
- **External checkpoint confirmed**: the framework entry point and evaluated
  checkpoint are known, but the current 4L20 tree does not preserve a complete
  immutable snapshot of every launch override.

## Table 1 mapping

| Paper model | Evidence | Release config or external entry | Evaluated checkpoint |
| --- | --- | --- | --- |
| OpenVLA | Confirmed external run | OpenVLA `train_oft_robotwin_l20.sh` | `hiarm_new_button_openvla_lora_r32_bs16_8000/8000` |
| RDT-1B | External checkpoint confirmed | RDT-1B `finetune.sh` | `rdt1b_hiarm_new_button_v3_bs32_8000/checkpoint-8000` |
| pi0-Full | Confirmed OpenPI config | `pi0_full_diyrobot` | `hiarm_500_new_button_pi0_full_8000_bs32/8000` |
| pi0-LoRA | Confirmed OpenPI config | `pi0_lora_diyrobot` | `hiarm_500_new_button_pi0_lora_8000_bs32/8000` |
| pi0.5-Full | Confirmed OpenPI config | `pi05_full_diyrobot` | `hiarm_500_new_button_full_base_8000_bs32/8000` |
| pi0.5-LoRA | Confirmed OpenPI config | `pi05_lora_diyrobot` | `hiarm_500_new_button_lora_8000_bs32/8000` |
| Hi-MoE | External checkpoint confirmed | Hi-MoE `scripts/train.py` | `hiarm_new_button_official_pd8_ga4_4gpu_8000/checkpoint-8000` |
| AdaMoE | External checkpoint confirmed | AdaMoE `scripts/train.py` | `adamoe_hiarm_500_new_button_bs32_8000/8000` |
| KinRT-OpenVLA | Confirmed external run | OpenVLA `train_oft_robotwin_l20.sh` with K4 routing | `hiarm_new_button_openvla_kinrt_k4_lora_r32_bs16_8000/8000` |
| KinRT-Full (pi0) | Confirmed OpenPI config | `kinrt_full_pi0_diyrobot` | `hiarm_500_new_button_pi0_full_moe_k4_8000_bs32/8000` |
| KinRT-LoRA (pi0) | Confirmed OpenPI config | `kinrt_lora_pi0_diyrobot` | `hiarm_500_new_button_pi0_lora_moe_k4_8000_bs32/8000` |
| KinRT-AdaMoE | Confirmed OpenPI config | `kinrt_adamoe_diyrobot` | `hiarm_500_new_button_adamoe_kinrt_k4_8000_bs32/8000` |
| KinRT-Full | Confirmed OpenPI config | `kinrt_full_diyrobot` | `hiarm_500_new_button_full_moe_k4_8000_bs32/8000` |
| KinRT-LoRA | Confirmed OpenPI config | `kinrt_lora_diyrobot` | `hiarm_500_new_button_lora_moe_k4_8000_bs32/8000` |

The machine-readable form is `configs/PAPER_MODELS.json`. The original
evaluation mapping and the normalized OpenPI training-run mapping are retained
under `manifests`.

## What 4L20 actually contains

The current 4L20 OpenPI `config.py` contains ten general configurations, not
the complete nine-configuration DIYRobot paper suite. Its two unambiguous
RoboTwin KinRT entries were retained and renamed:

| Legacy name | Release name | Meaning |
| --- | --- | --- |
| `pi05_800_full` | `kinrt_full` | Full-parameter pi0.5 with KinRT K4 routing |
| `pi0_800_full` | `kinrt_full_pi0` | Full-parameter pi0 with KinRT K4 routing |

4L20 also contains OpenVLA, RDT-1B, Hi-MoE, and AdaMoE framework trees and
OpenVLA run directories. This confirms framework availability, but does not
make those frameworks valid OpenPI `TrainConfig` entries.

## What L20yz contributes

L20yz contains the nine exact OpenPI blocks used by the DIYRobot paper runs.
The release renames them to the identifiers in the table above. It also retains
the 800-episode RoboTwin LoRA entry as `kinrt_lora`. All other experimental and
ablation configurations were removed from the release registry.

For `KinRT-LoRA (pi0)`, an older manifest used the nonexistent suffix
`lora_moe_k4`. The executable source block is
`pi0_hiarm_manual_500_promptv3_new_button_moe_k4`; its LoRA variants and freeze
filter make the adaptation regime unambiguous. The release mapping records the
executable source name.
