# Open-Source Packaging Notes

## Included

- Current `src`, `scripts`, `packages`, `examples`, and documentation in the
  unified PI0.5 workspace.
- Current PI0.5 project metadata, lock files, launchers, and public deployment
  templates.
- RoboTwin evaluation client, model server, evaluator, dependency list, and
  public task-configuration template.
- Existing OpenPI Apache-2.0 and RoboTwin MIT license texts.
- A portable router-label generator requiring an explicit dataset root.

## Excluded

- Checkpoints, datasets, normalization assets, router-label binaries, videos,
  images, experiment results, W&B state, logs, and process identifiers.
- Virtual environments, Python bytecode, pytest caches, compiled objects, and
  copied upstream `third_party` worktrees.
- Files named as backups, rejected patches, or originals such as `*.bak*`,
  `*.orig`, and `*.rej`.
- SSH keys, host addresses, passwords, tokens, and local user paths.
- Host-specific `deploy_policy.yml` files. Public
  `deploy_policy.example.yml` templates replace them without changing copied
  executable code.

The upstream submodule declarations remain as provenance. The submodule
worktrees are not copied into this source package.

## Preserved host-specific paths

The active training configurations still contain absolute dataset, label, and
base-checkpoint paths from their source hosts. They are part of executable
configuration and were deliberately not rewritten during this comment-only
release pass. Users must map them to local resources before running training.

## Behavioral scope

The consolidation does not change the shared model-routing or evaluation
implementation. It merges the two primary FULL configuration blocks into the
canonical registry and updates comments and release paths to describe one
source tree. Other comment cleanup retains executable AST structure.

Earlier work on the remote repositories did include evaluation behavior changes
on the source used as the canonical release implementation. Those changes are disclosed in
`CHANGELOG_AND_SOURCE_MANIFEST.md` and must not be described as comment-only.
