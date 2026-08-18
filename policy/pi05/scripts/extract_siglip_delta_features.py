"""Extract SigLiP visual delta embeddings for clustering (v3 pipeline).

For each frame t in the primary repo, computes:
    delta_t = mean_pool(SigLiP(image_{t+H})) - mean_pool(SigLiP(image_t))

where H = horizon (default 24).  The delta captures *what the action window
changes* in the visual scene, which is far more discriminative for skill
clustering than the absolute embedding:

  - "Approach empty-handed"   → wrist-cam changes from table → object close-up
  - "Transport with object"   → cam_high shows object position shifting
  - "Place and release"       → wrist-cam changes from object → empty
  - "Bimanual coordination"   → both wrist-cams change simultaneously

Two cameras are used:
  • cam_left_wrist  (primary: captures gripper/object contact state)
  • cam_high        (secondary: captures workspace context / object location)

Output (per-frame, aligned to global LeRobot index):
  <output_dir>/siglip_delta_left_wrist.npy   shape (total_frames, embed_dim)
  <output_dir>/siglip_delta_cam_high.npy     shape (total_frames, embed_dim)
  <output_dir>/summary.json

Frames not covered by the strided sample (or without a valid t+H partner)
are left as zero vectors and marked invalid in the summary.

Usage (run from pi05 repo root):
  .venv/bin/python scripts/extract_siglip_delta_features.py \\
      --repo-root  /path/to/demo_mixed_repo \\
      --extra-repo /path/to/demo_randomized_repo \\
      --extra-episodes 400 \\
      --stride 8 --batch-size 48 --horizon 24
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from transformers import SiglipVisionModel, SiglipImageProcessor

SIGLIP_MODEL_ID = "google/siglip-base-patch16-224"

CAM_KEYS = {
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_high":       "observation.images.cam_high",
}


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_path(repo_root: Path, pattern: str, episode_index: int, chunks_size: int) -> Path:
    return repo_root / pattern.format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )


def _load_episode_images(
    parquet_path: Path,
    cam_col: str,
    stride: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (global_indices, frame_indices, images_hwc) for strided frames."""
    table = pq.read_table(
        parquet_path,
        columns=[cam_col, "index", "frame_index"],
    )
    n = len(table)
    # Sampled positions: 0, stride, 2*stride, ...
    sample_pos = np.arange(0, n, stride, dtype=np.int64)
    # For delta we also need pos + horizon; keep only pairs where both exist.
    valid_mask = (sample_pos + horizon) < n
    sample_pos = sample_pos[valid_mask]
    future_pos = sample_pos + horizon

    global_idx = np.asarray(table["index"].combine_chunks().to_pylist(), dtype=np.int64)
    frame_idx  = np.asarray(table["frame_index"].combine_chunks().to_pylist(), dtype=np.int64)

    cam_col_data = table[cam_col].combine_chunks()

    def _decode(pos: int) -> np.ndarray:
        val = cam_col_data[int(pos)].as_py()
        img = Image.open(io.BytesIO(val["bytes"])).convert("RGB")
        return np.array(img, dtype=np.uint8)

    imgs_cur    = np.stack([_decode(p) for p in sample_pos], axis=0)   # (S, H, W, 3)
    imgs_future = np.stack([_decode(p) for p in future_pos], axis=0)   # (S, H, W, 3)

    return (
        global_idx[sample_pos],
        global_idx[future_pos],
        frame_idx[sample_pos],
        imgs_cur,
        imgs_future,
    )


@torch.no_grad()
def _embed_batch(
    images: np.ndarray,  # (B, H, W, 3) uint8
    processor: SiglipImageProcessor,
    model: SiglipVisionModel,
    device: torch.device,
) -> np.ndarray:
    """Return (B, embed_dim) float32 mean-pooled embeddings."""
    pil_imgs = [Image.fromarray(img) for img in images]
    inputs = processor(images=pil_imgs, return_tensors="pt").to(device)
    out = model(**inputs)
    # last_hidden_state: (B, n_patches, D) → mean pool
    pooled = out.last_hidden_state.mean(dim=1).cpu().float().numpy()
    return pooled


def _process_repo(
    repo_root: Path,
    episode_indices: list[int],
    *,
    processor: SiglipImageProcessor,
    model: SiglipVisionModel,
    device: torch.device,
    stride: int,
    horizon: int,
    batch_size: int,
    total_frames: int,
    cam_keys: dict[str, str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Process episodes and return per-camera dict of:
      (global_indices_cur, delta_embeddings)
    """
    info = read_json(repo_root / "meta" / "info.json")
    pattern = info["data_path"]
    chunks_size = int(info["chunks_size"])
    embed_dim: int | None = None

    # Accumulate across episodes
    all_cur_idx:    dict[str, list[np.ndarray]] = {k: [] for k in cam_keys}
    all_deltas:     dict[str, list[np.ndarray]] = {k: [] for k in cam_keys}

    for ep_i, episode_index in enumerate(episode_indices):
        if ep_i % 50 == 0:
            print(f"  episode {ep_i}/{len(episode_indices)} (repo={repo_root.name})")
        pq_path = episode_path(repo_root, pattern, episode_index, chunks_size)

        for cam_name, cam_col in cam_keys.items():
            try:
                cur_idx, _fut_idx, _frame_idx, imgs_cur, imgs_fut = _load_episode_images(
                    pq_path, cam_col, stride=stride, horizon=horizon
                )
            except Exception as e:
                print(f"    WARNING: could not load {cam_col} for ep {episode_index}: {e}")
                continue
            if len(cur_idx) == 0:
                continue

            # Embed both sets in batches, then delta
            embeds_cur  = []
            embeds_fut  = []
            for start in range(0, len(imgs_cur), batch_size):
                end = start + batch_size
                embeds_cur.append(_embed_batch(imgs_cur[start:end],  processor, model, device))
                embeds_fut.append(_embed_batch(imgs_fut[start:end], processor, model, device))
            ec = np.concatenate(embeds_cur,  axis=0)  # (S, D)
            ef = np.concatenate(embeds_fut, axis=0)  # (S, D)
            if embed_dim is None:
                embed_dim = ec.shape[1]

            delta = ef - ec  # visual change over [t, t+H]
            all_cur_idx[cam_name].append(cur_idx)
            all_deltas[cam_name].append(delta)

    results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cam_name in cam_keys:
        if all_cur_idx[cam_name]:
            idx = np.concatenate(all_cur_idx[cam_name], axis=0)
            dlt = np.concatenate(all_deltas[cam_name],  axis=0).astype(np.float32)
            results[cam_name] = (idx, dlt)
    return results, int(embed_dim or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_mixed_repo"),
    )
    parser.add_argument(
        "--extra-repo",
        type=Path,
        default=Path("/private/yth/projects/cache/huggingface/lerobot/demo_randomized_repo"),
    )
    parser.add_argument("--extra-episodes", type=int, default=400)
    parser.add_argument("--stride",    type=int, default=8,  help="Frame stride for sampling")
    parser.add_argument("--horizon",   type=int, default=24, help="Action horizon (frames ahead)")
    parser.add_argument("--batch-size",type=int, default=48, help="Images per GPU batch")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--model-id",  type=str, default=SIGLIP_MODEL_ID)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading SigLiP model: {args.model_id}")
    processor = SiglipImageProcessor.from_pretrained(args.model_id)
    model = SiglipVisionModel.from_pretrained(args.model_id).to(device).eval()
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    print(f"Model loaded.  embed_dim={base_model.config.hidden_size}")

    repo_root = args.repo_root.resolve()
    info = read_json(repo_root / "meta" / "info.json")
    total_episodes = int(info["total_episodes"])
    total_frames   = int(info["total_frames"])

    output_dir = args.output_dir or (
        repo_root / "meta" / f"siglip_delta_h{args.horizon}_s{args.stride}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Primary repo ──────────────────────────────────────────────────────
    print(f"\nProcessing primary repo: {repo_root}  ({total_episodes} episodes)")
    primary_results, embed_dim = _process_repo(
        repo_root,
        list(range(total_episodes)),
        processor=processor,
        model=model,
        device=device,
        stride=args.stride,
        horizon=args.horizon,
        batch_size=args.batch_size,
        total_frames=total_frames,
        cam_keys=CAM_KEYS,
    )

    # Save dense arrays indexed by global LeRobot index
    saved_files: dict[str, str] = {}
    cam_sample_counts: dict[str, int] = {}
    for cam_name, (idx, delta) in primary_results.items():
        arr = np.zeros((total_frames, embed_dim), dtype=np.float32)
        arr[idx] = delta
        fname = f"siglip_delta_{cam_name}.npy"
        np.save(output_dir / fname, arr)
        # Also save the valid mask
        mask = np.zeros(total_frames, dtype=bool)
        mask[idx] = True
        np.save(output_dir / f"siglip_valid_{cam_name}.npy", mask)
        saved_files[cam_name] = fname
        cam_sample_counts[cam_name] = int(mask.sum())
        print(f"  {cam_name}: {int(mask.sum())} sampled frames → {fname}")

    # ── Extra repo (for fitting PCA / scaler) ─────────────────────────────
    extra_results: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if args.extra_repo is not None:
        extra_repo = args.extra_repo.resolve()
        extra_info = read_json(extra_repo / "meta" / "info.json")
        total_extra = int(extra_info["total_episodes"])
        rng = np.random.default_rng(args.seed)
        ep_idx = np.sort(rng.choice(total_extra, min(args.extra_episodes, total_extra), replace=False))
        print(f"\nProcessing extra repo: {extra_repo}  (sampling {len(ep_idx)} episodes)")
        extra_results, _ = _process_repo(
            extra_repo,
            ep_idx.tolist(),
            processor=processor,
            model=model,
            device=device,
            stride=args.stride,
            horizon=args.horizon,
            batch_size=args.batch_size,
            total_frames=0,         # not needed (no global array)
            cam_keys=CAM_KEYS,
        )
        for cam_name, (idx, delta) in extra_results.items():
            fname = f"siglip_delta_extra_{cam_name}.npy"
            np.save(output_dir / fname, delta)
            np.save(output_dir / f"siglip_extra_count_{cam_name}.npy", np.array([len(idx)]))
            saved_files[f"extra_{cam_name}"] = fname
            print(f"  extra {cam_name}: {len(idx)} samples → {fname}")

    summary = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "extra_repo": str(args.extra_repo) if args.extra_repo else None,
        "model_id": args.model_id,
        "embed_dim": embed_dim,
        "horizon": args.horizon,
        "stride": args.stride,
        "total_primary_frames": total_frames,
        "cam_sample_counts": cam_sample_counts,
        "cameras": list(CAM_KEYS.keys()),
        "files": saved_files,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved to: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
