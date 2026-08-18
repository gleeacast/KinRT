from __future__ import annotations

import asyncio
import concurrent.futures as futures
import dataclasses
import logging
from typing import Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
from orbax.checkpoint._src.path.atomicity import CommitFileTemporaryPath

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    async_checkpointing: bool = True,
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            # use_ocdbt=False avoids OCDBT manifest file locking, which fails with
            # ENOLCK on NFS filesystems. Restore auto-detects format via
            # is_ocdbt_checkpoint(), so existing OCDBT checkpoints still load fine.
            "state": ocp.PyTreeCheckpointHandler(use_ocdbt=False),
            "train_state": ocp.PyTreeCheckpointHandler(use_ocdbt=False),
            "params": ocp.PyTreeCheckpointHandler(use_ocdbt=False),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            cleanup_tmp_directories=True,
            enable_async_checkpointing=async_checkpointing,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
            # Use CommitFileTemporaryPath instead of AtomicRenameTemporaryPath to
            # avoid rename() calls that trigger ENOLCK on NFS (EPC) filesystems.
            # Instead of renaming tmp->final, orbax writes a commit_success file
            # in-place, which does not require file locking.
            temporary_path_class=CommitFileTemporaryPath,
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    if resuming and tuple(mngr.all_steps()) in [(), (0,)]:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    save_mode: str = "split",
):
    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)

    if save_mode == "state_only":
        items = {
            "assets": save_assets,
            "state": state,
        }
    else:
        # Split params that can be used for inference into a separate item.
        with at.disable_typechecking():
            train_state, params = _split_params(state)
        items = {
            "assets": save_assets,
            "train_state": train_state,
            "params": {"params": params},
        }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        restore_step = step if step is not None else checkpoint_manager.latest_step()
        if restore_step is None:
            raise FileNotFoundError(f"No steps found in {checkpoint_manager.directory}.")

        step_dir = checkpoint_manager.directory / str(restore_step)
        if (step_dir / "state").exists():
            restored = checkpoint_manager.restore(
                restore_step,
                items={
                    "state": state,
                },
            )
            return restored["state"]

        # Older checkpoints split inference params into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            restore_step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
        return _merge_params(restored["train_state"], restored["params"])


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        await asyncio.to_thread(self.save, directory, args)
        return []

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Reverse `_split_params`; a `params` item indicates that the split included EMA parameters.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
