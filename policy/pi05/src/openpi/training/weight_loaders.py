import dataclasses
import hashlib
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)

_CHECKPOINT_MISSING_REGEX = (
    r".*(lora|moe|router_context_proj|global_router|global_scale_adapter|prototype_router).*"
)
_MOE_EXPERT_COPY_NOISE_STD_SCALE = 1e-3


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add newly initialized adapter/MoE weights that are absent from the base checkpoint.
        return _merge_params(loaded_params, params, missing_regex=_CHECKPOINT_MISSING_REGEX)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            initialized = _init_missing_param_from_loaded(k, flat_loaded=flat_loaded, flat_ref=flat_ref)
            result[k] = initialized if initialized is not None else flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _init_missing_param_from_loaded(k: str, *, flat_loaded: dict[str, at.Array], flat_ref: dict[str, at.Array]) -> at.Array | None:
    """Initializes new MoE experts from the corresponding shared FFN when possible."""
    source_key = _shared_ffn_key_for_moe_expert(k)
    if source_key is None or source_key not in flat_loaded:
        return None

    source = flat_loaded[source_key]
    ref = flat_ref[k]
    if source.shape != ref.shape:
        return None
    source_f32 = source.astype(np.float32)
    noise_std = np.std(source_f32) * _MOE_EXPERT_COPY_NOISE_STD_SCALE
    initialized = source_f32 + _deterministic_gaussian(k, source.shape) * noise_std
    return initialized.astype(ref.dtype) if initialized.dtype != ref.dtype else initialized


def _deterministic_gaussian(k: str, shape: tuple[int, ...]) -> np.ndarray:
    digest = hashlib.sha256(k.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little") % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape).astype(np.float32)


def _shared_ffn_key_for_moe_expert(k: str) -> str | None:
    match = re.fullmatch(r"(.*?/layers/)moe(_\d+)?/expert_\d+/(gating_einsum|linear)", k)
    if match is None:
        return None
    prefix, stream_suffix, param_name = match.groups()
    return f"{prefix}mlp{stream_suffix or ''}/{param_name}"
