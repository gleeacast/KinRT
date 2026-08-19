# KinRT Consolidated Change and Source Record

Audit date: 2026-08-18

Website revision date: 2026-08-19

This file records the source provenance, remote-write boundary, package
consolidation, comment audit, and documentation changes made for the public
KinRT release.

## Direct answer about earlier remote changes

The earlier RoboTwin comparison did modify both remote source hosts.

- The two remote `config.py` files were rewritten during English comment
  cleanup. AST comparison showed no executable-structure change in that
  comment-only operation.
- `policy/pi05/docs/KINRT_ROBOTWIN_IMPLEMENTATION.md` and
  `policy/pi05/docs/KINRT_ROBOTWIN_COMPARISON_AND_CHANGES.md` were added on both
  hosts on 2026-08-17.
- The recorded 4L20 diff also contained active full-parameter KinRT routing
  changes. Those changes already existed in the dirty snapshot used for the
  comparison and cannot be attributed to the comment rewrite alone.
- The recorded L20yz worktree contained broader research and evaluation
  changes, including corrections in `script/eval_policy_client.py`,
  `script/policy_model_server.py`, and `policy/pi05/pi_model.py`.

It is therefore inaccurate to describe the earlier remote comparison as fully
read-only or entirely comment-only.

The current consolidation and website pass did not write either RoboTwin host.
The 2026-08-18 DIYRobot audit of the legacy `hi_arm` path was also strictly read-only: no lower-host file was
modified and no camera, bus, motor, installation, collection, or policy-motion
process was started.

## Source provenance

| Contribution | Source | Recorded commit |
| --- | --- | --- |
| Canonical KinRT/OpenPI and evaluation implementation | L20yz `/private/yth/projects/RoboTwin` | `bf44be51cf5717a5595ce59447f2cf5263d2aa95` |
| Primary FULL RoboTwin config blocks | 4L20 `/private/RoboTwin` | `56fc2c4597aeee30fa52c67e3d66995d712cac7d` |

Both source repositories were dirty research worktrees. No reset, rollback, or
history rewrite was performed.

## Single-source consolidation

The release now contains one executable source tree:

```text
KinRT/
|-- policy/pi05/
`-- script/
```

The former release directories `KINRT_FULL/` and `KINRT_LORA/` are not present.
The L20yz snapshot is the canonical implementation because it contains the
final checkpoint/evaluation handling and all confirmed DIYRobot/OpenPI paper
configs. The two primary FULL RoboTwin configs from 4L20 were added to the same
registry.

No model-routing implementation was duplicated. FULL and LoRA now differ only
through named configuration entries in
`policy/pi05/src/openpi/training/config.py`:

- FULL trains the dense base, router, and expert parameters.
- LoRA freezes the dense base and trains LoRA, router, and expert parameters.
- Both use four experts, Top-1 routing, masked-mean pooled observation context,
  dense routing, router-supervision coefficient `0.05`, balanced sampling with
  exponent `-0.5`, 10,000 RoboTwin steps, batch size 32, and no EMA.

The unified registry contains exactly these 12 entries:

```text
kinrt_full
kinrt_full_pi0
kinrt_lora
pi05_full_diyrobot
pi05_lora_diyrobot
pi0_full_diyrobot
pi0_lora_diyrobot
kinrt_full_diyrobot
kinrt_lora_diyrobot
kinrt_full_pi0_diyrobot
kinrt_lora_pi0_diyrobot
kinrt_adamoe_diyrobot
```

## Files changed for consolidation

- `policy/pi05/src/openpi/training/config.py`: added the two primary FULL
  entries to the canonical registry and added a short objective description of
  the FULL/LoRA trainable-parameter difference.
- `configs/PAPER_MODELS.json`: changed native OpenPI entry points to the unified
  `policy/pi05/scripts/train.py` path.
- `README.md`: replaced split-tree setup, training, and overlay instructions
  with one-workspace commands.
- `docs/getting-started.html`, `docs/robotwin.html`, `docs/diyrobot.html`, and
  `docs/diyrobot.html`: consolidated hardware, calibration, data, and evaluation.
- `docs/FULL_VS_LORA.md`, `docs/KINRT_IMPLEMENTATION.md`, and
  `docs/OPEN_SOURCE_NOTES.md`: aligned the technical description with one
  implementation and one config registry.
- `tools/validate_release.py`: changed exact config checking and manifest
  generation from two variants to one source tree.
- `policy/pi05/scripts/convert_diyrobot_to_lerobot.py`: renamed the public
  real-platform data converter and its internal symbols from the legacy
  platform name without changing the conversion schema.
- `manifests/KINRT_SOURCE_SHA256.txt`: added a deterministic inventory of all
  files under `policy/` and `script/`.

Obsolete generated artifacts removed from the consolidated package:

```text
manifests/KINRT_FULL_SHA256.txt
manifests/KINRT_LORA_SHA256.txt
manifests/FULL_VS_LORA_FILES.tsv
```

## Website changes

- Replaced the dense release-dashboard homepage with a concise project page:
  name and method statement, method figure, three-stage explanation, compact
  FULL/LoRA comparison, RoboTwin/DIYRobot tracks, and minimal quick start.
- Increased the base, navigation, sidebar, table, code, caption, and mobile
  text sizes across the documentation site.
- Reused the project pipeline, cluster, platform, hardware, and task images.
- Retained detailed beginner instructions on dedicated pages instead of
  placing audit detail on the homepage.
- Kept the site dependency-free and directly publishable from `docs/`.

## Earlier registry and comment cleanup retained in this release

- Renamed 4L20 `pi05_800_full` to `kinrt_full`.
- Renamed L20yz `pi05_800_full` to `kinrt_lora`.
- Renamed 4L20 `pi0_800_full` to `kinrt_full_pi0`.
- Retained only the two unambiguous primary FULL RoboTwin configs, the primary
  LoRA RoboTwin config, and the nine confirmed DIYRobot/OpenPI configs.
- Removed discarded-config launchers, one obsolete non-active Chinese MoE
  design draft, and one unretained LIBERO launcher from the packaged snapshot.
- Preserved router coefficients, expert count, Top-K, training steps, batch
  size, LoRA ranks, and freeze-filter behavior.

## Paper mapping

`configs/PAPER_MODELS.json` covers all 14 Table 1 rows. Nine are executable
OpenPI configurations in the unified registry. OpenVLA, RDT-1B, Hi-MoE,
AdaMoE, and KinRT-OpenVLA remain external-framework records and are not
fabricated as `TrainConfig` entries.

The `KinRT-LoRA (pi0)` mapping uses the executable source identity
`pi0_hiarm_manual_500_promptv3_new_button_moe_k4`; the older `lora_moe_k4`
suffix did not exist in the audited registry.

## Comment and packaging audit

- Python comments and docstrings are tokenized during validation.
- Python, shell, YAML, TOML, JSON, HTML, CSS, and JavaScript files are scanned
  for CJK text; active code and configuration must contain none.
- Retained comments use short English descriptions of active behavior.
- Private credentials, host markers, caches, logs, checkpoints, datasets,
  labels, normalization assets, and virtual environments are excluded.
- Source paths for external datasets and checkpoints remain placeholders or
  provenance references and must be mapped before execution.

## DIYRobot lower-host release

The DIYRobot source under the legacy `src/lerobot/robots/hi_arm` path was
reviewed remotely without writes and is now redistributed as a sanitized public
copy under `hardware/diyrobot/lower_host`. `hardware/diyrobot/SOURCE_MANIFEST.md`
records included files, original hashes, release-copy changes, exclusions,
compatibility identifiers, and unresolved mechanical artifacts.

Public classes, source filenames, newly generated data IDs, logs, calibration
outputs, UI text, and documentation use DIYRobot. Source-host names remain only
in provenance and historical dataset/checkpoint records. Personal paths, internal network
defaults, camera serial numbers, caches, logs, and robot-specific calibration
were excluded.

The released vendor transport uses `DIYROBOT_VENDOR_MONITOR_LOG` and defaults
to `/tmp/diyrobot/vendor_usbcan_monitor.jsonl`. The old internal environment
variable and its machine-specific default path are not public aliases.

`docs/diyrobot.html` now contains one end-to-end physical-platform guide.
`docs/real-robot.html` remains only as a redirect for old links.

One paper/source discrepancy remains unresolved: the paper table says `Pull
Bottle`, while the active prompt-v3 launcher describes pulling a pill box onto
a black pad. The release documents the discrepancy instead of selecting an
unsupported interpretation.

## Website and README revision

The 2026-08-19 presentation pass made the following documentation-only
changes:

- Removed the duplicated standalone `arXiv:2607.26807 [cs.RO]` line from the
  README header while retaining the linked paper badge.
- Replaced the sparse environment list with a platform support matrix and
  locally stored Simple Icons for the public toolchain.
- Corrected DIYRobot OOD wording throughout the README, website, and hardware
  documentation. The 100-episode altered-lighting artifact is optional
  training data; evaluation remains 50 trials per task in the original
  environment.
- Reduced the README parameter table to the requested two-column reference.
- Added direct arXiv and GitHub links with recognizable icons to the project
  homepage; the GitHub README already links back to the project page.
- Moved the executable K=4 router-label command out of the RoboTwin and
  DIYRobot benchmark guides and into the KinRT method page. Benchmark pages
  now link to that method procedure.
- Added paired RoboTwin Easy/Hard scene examples to the method page and
  explicitly labeled them as benchmark evidence rather than cluster semantics.

No policy, training, evaluation, simulator, or DIYRobot runtime source was
changed in this presentation pass.

## Validation

Run:

```bash
python tools/validate_release.py --check-manifests
```

The validator checks Python syntax, comment language, JSON/TOML parsing, exact
12-config membership, the 14-row paper mapping, secrets and forbidden
artifacts, website links/fragments/assets, and the deterministic unified source
manifest. It does not start training, simulation, cameras, buses, or motors.
