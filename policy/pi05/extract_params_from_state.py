"""Extract a params directory from a state_only checkpoint without JAX or a GPU.

The operation hard-links params.* subdirectories and rewrites _METADATA.

Usage:
  python extract_params_from_state.py \
      --ckpt_dir /path/to/checkpoints/name/name \
      [--steps 10000 9500 ...]   # By default, process every step containing state/.
"""
import argparse
import ast
import json
import os
import shutil
from pathlib import Path


def extract_one_step(step_dir: Path):
    state_dir = step_dir / "state"
    params_dir = step_dir / "params"

    if params_dir.exists():
        print(f"  [skip] params/ already exists: {params_dir}")
        return

    if not state_dir.exists():
        print(f"  [skip] no state/ dir: {step_dir}")
        return

    print(f"  Extracting {step_dir.name} ...")

    # 1. Read _METADATA
    meta_path = state_dir / "_METADATA"
    with open(meta_path) as f:
        meta = json.load(f)

    tree_meta = meta["tree_metadata"]

    # Keys in tree_metadata are string-repr of tuples, e.g. "('params', 'action_in_proj', 'bias', 'value')"
    # Keep only entries whose first element is 'params'
    params_tree_meta = {}
    for k, v in tree_meta.items():
        try:
            parts = ast.literal_eval(k)
        except Exception:
            continue
        if isinstance(parts, tuple) and parts and parts[0] == "params":
            params_tree_meta[k] = v

    print(f"    params leaves: {len(params_tree_meta)}")

    # 2. Create params/ dir
    params_dir.mkdir()

    # 3. For each params leaf, hardlink or copy the tensor directory
    for k in params_tree_meta:
        parts = ast.literal_eval(k)
        # The file name in state/ is the dot-joined path, e.g. "params.action_in_proj.bias.value"
        state_file = state_dir / ".".join(parts)
        params_file = params_dir / ".".join(parts)

        if not state_file.exists():
            print(f"    WARNING: missing file {state_file}")
            continue

        if state_file.is_dir():
            # Directory containing tensor data (file "0" inside)
            params_file.mkdir(exist_ok=True)
            for child in state_file.iterdir():
                dst = params_file / child.name
                try:
                    os.link(child, dst)  # hardlink (same filesystem)
                except OSError:
                    shutil.copy2(child, dst)  # fallback: copy
        else:
            try:
                os.link(state_file, params_file)
            except OSError:
                shutil.copy2(state_file, params_file)

    # 4. Write new _METADATA with only params keys
    new_meta = {
        "tree_metadata": params_tree_meta,
        "use_zarr3": meta.get("use_zarr3", False),
        "store_array_data_equal_to_fill_value": meta.get("store_array_data_equal_to_fill_value", False),
        "custom_metadata": meta.get("custom_metadata", {}),
    }
    with open(params_dir / "_METADATA", "w") as f:
        json.dump(new_meta, f)

    print(f"    Done → {params_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Checkpoint base dir (contains step subdirs)")
    parser.add_argument("--steps", nargs="*", type=int, default=None)
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)

    if args.steps:
        step_dirs = [ckpt_dir / str(s) for s in args.steps]
    else:
        step_dirs = sorted(
            [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda d: int(d.name)
        )

    for step_dir in step_dirs:
        extract_one_step(step_dir)

    print("All done.")


if __name__ == "__main__":
    main()
