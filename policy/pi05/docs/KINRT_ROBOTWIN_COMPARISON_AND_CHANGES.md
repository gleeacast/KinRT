# KinRT RoboTwin Comparison and Change Record

Date: 2026-08-17

Scope: `/private/RoboTwin` on 4L20 and `/private/yth/projects/RoboTwin` on L20yz

## Executive conclusion

The `kinrt_full` and `kinrt_lora` entries use the same 800-episode dataset, identical
offline router labels, and the same active routing objective. They are not the
same training experiment. 4L20 performs full-parameter fine-tuning, whereas
L20yz performs LoRA fine-tuning with rank 32 in PaliGemma and rank 64 in the
action expert. Results from these checkpoints must therefore be labeled as
different optimization regimes.

Before this change, the L20yz remote-evaluation shell scripts passed `--host`
to programs that did not accept the argument. The YAML evaluation count and
output paths were also ignored, model sessions were shared across connections,
and server-side errors were not propagated to the client. Those interface
defects have been corrected without changing the KinRT model or training loss.

## Reviewed code

The review covered the complete active path rather than only the named
configuration block:

| Area | Files |
| --- | --- |
| Training configuration | `src/openpi/training/config.py`, optimizer and training entry points |
| Dataset and labels | `src/openpi/training/data_loader.py`, transforms, label artifacts, label generator |
| Model configuration | `src/openpi/models/pi0_config.py` |
| Routing and experts | `src/openpi/models/gemma.py`, `src/openpi/models/pi0.py` |
| Checkpoints | weight loaders and checkpoint managers on both hosts |
| Policy inference | `pi_model.py`, policy construction |
| Remote evaluation | `script/eval_policy_client.py`, `script/policy_model_server.py` |
| Launch configuration | `compute_ns.sh`, `finetune.sh`, `eval_remote.sh`, `serve_remote_model.sh`, `deploy_policy.yml` |
| Existing documentation | KinRT implementation and remote-inference notes |

Repository baselines at review time:

| Host | Commit | Worktree observation |
| --- | --- | --- |
| 4L20 | `56fc2c4597aeee30fa52c67e3d66995d712cac7d` | Existing uncommitted `config.py` change enables the active full-parameter KinRT entry; unrelated untracked visualization files were left untouched. |
| L20yz | `bf44be51cf5717a5595ce59447f2cf5263d2aa95` | Extensive pre-existing research changes were preserved; no reset, checkout, or broad replacement was performed. |

## Training comparison

| Property | 4L20 `kinrt_full` | L20yz `kinrt_lora` |
| --- | --- | --- |
| Fine-tuning regime | Full parameter | LoRA plus router/MoE parameters |
| PaliGemma | `gemma_2b` | `gemma_2b_lora`, rank 32 |
| Action expert | `gemma_300m` | `gemma_300m_lora`, rank 64 |
| Experts / Top-K | 4 / 1 | 4 / 1 |
| Router input | Masked-mean pooled prefix | Masked-mean pooled prefix |
| Router type | Dense | Dense |
| Supervised router coefficient | 0.05 | 0.05 |
| Balanced sampling | Inverse frequency to power 0.5 | Inverse frequency to power 0.5 |
| Class-loss weights | Uniform | Explicitly uniform |
| Steps / batch | 10,000 / 32 | 10,000 / 32 |
| FSDP devices | 1 | 2 |
| Save interval | 500 | 1,000 |

The label files are byte-identical with SHA-256
`bceda7104ee949d37c9872a50c06842a111387ad5b63eb9011d50e19b33b256a`.
They contain 162,545 frame labels with counts
`[59018, 74748, 7182, 21597]`.

## Implementation differences retained

4L20 remains the more general routing implementation. It supports per-layer
and layer-group routers, hard and soft labels, CSV ingestion, label/expert
validation, straight-through action-gradient routing, and feed-forward
activation telemetry.

L20yz retains checkpoint structure auto-detection, `state_only` checkpoints,
synchronous-save controls, offline logging support, and its evaluation
completion protocol. These features were not copied to 4L20 because they are
independent of the requested KinRT comparison and may affect established
workflows.

No behavior was added for `router_sampling_mix_beta`; the field remains unused.
No parameter-shape change was made for `action_expert_moe_mlp_dim`; routed
experts continue to use `mlp_dim`. Both points are documented for a separate
compatibility decision.

## Changes common to both repositories

- Added `policy/pi05/docs/KINRT_ROBOTWIN_IMPLEMENTATION.md` as the authoritative
  English implementation reference.
- Added `policy/pi05/scripts/generate_router_labels.py`, a portable source
  version of the actual action-chunk, velocity, PCA, and KMeans label pipeline.
  `--repo-root` is required so the script does not encode a host-specific path.
- Added precise English router and residual-expert docstrings in `gemma.py`
  and `pi0.py`.
- Added explicit configuration comments distinguishing the 4L20 full-parameter
  entry from the L20yz LoRA entry.
- Replaced CJK comments in active PI05 Python, shell, and YAML files with
  technical English. Runtime behavior was not changed by these translations.
- Added this comparison and change record.

4L20 also received comment-only cleanup in the tracked historical variants
`gemma_org.py`, `gemma_yth.py`, `pi0_config_yth.py`, `pi_org.py`, and
`pi_yth.py` so active code files under `policy/pi05` satisfy the same
comment-language rule.

## L20yz evaluation corrections

- Restored `--host` parsing in the evaluation client and model server.
- Honored YAML `test_num`, `eval_result_root`, `router_info_root`, and
  `record_router_info`.
- Added per-episode CSV records, elapsed times, seeds, action counts, and router
  telemetry paths.
- Propagated structured server errors to the client.
- Added portable bfloat16 response handling by serializing as float32.
- Created one model session per socket connection to isolate observation,
  language, action-queue, and telemetry state.
- Preserved episode-level exception isolation. Non-transport failures are
  recorded as failed episodes; connection loss terminates evaluation.
- Preserved `eval_done`, but emit it only when every requested episode has a
  result record.
- Excluded the obsolete MoE design draft because its Top-2, token-plus-prefix,
  and automatic class-weight descriptions do not match the active configs.
- Translated CJK comments in the L20yz `state_only` extraction and sequential
  single-task launch helpers.

## Files intentionally not changed

- Router label arrays, PCA/KMeans artifacts, datasets, checkpoints, and
  normalization statistics.
- Existing experiment names and checkpoint directory layout.
- KinRT loss coefficients or sampling behavior.
- Unrelated RoboTwin tasks, visualization scripts, images, and untracked files.
- Historical backup files matching `*.bak*`.

## Validation

Local validation completed:

- Python syntax compilation for every modified Python file: passed.
- YAML parsing for both deployment files: passed.
- CJK scan of active `*.py`, `*.sh`, `*.yml`, and `*.yaml` files under
  `policy/pi05` plus the two evaluation programs: no matches.
- The local host lacks `pyarrow`, so the generator dependency/import check is
  deferred to the project environments on the remote hosts.

Remote configuration import, CLI, shell syntax, and final Git-diff checks are
complete:

- Both project virtual environments compiled all modified Python files.
- 4L20 resolved `kinrt_full` as `gemma_2b` / `gemma_300m`, no LoRA ranks,
  four experts, Top-1 pooled routing, coefficient 0.05, 10,000 steps, batch 32,
  one FSDP device, save interval 500, and no freeze filter.
- L20yz resolved `kinrt_lora` as `gemma_2b_lora` rank 32 /
  `gemma_300m_lora` rank 64, four experts, Top-1 pooled routing, coefficient
  0.05, 10,000 steps, batch 32, two FSDP devices, save interval 1,000, and an
  active freeze filter.
- The label generator imported successfully and exposed its CLI on both hosts.
- Shell syntax checks passed for the relevant training and evaluation launchers.
- Static CLI checks confirmed `--host` in both evaluation programs.
- A socket-level test confirmed per-connection session isolation and continued
  request handling after a structured model error.
- Remote Unicode scans reported zero CJK-containing active code/config files.
- `git diff --check` passed for both scoped change sets.
- SHA-256 comparison found no mismatch between all 32 remote files and their
  locally reviewed copies.
