import dataclasses
import logging
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing import Literal
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

_logger = logging.getLogger("openpi")

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # LoRA rank override for attention and feed-forward modules in both model branches.
    # None preserves the variant defaults: rank 16 for gemma_2b_lora and rank 32
    # for gemma_300m_lora. An explicit value supports controlled rank comparisons.
    lora_rank: int | None = None

    # Per-branch overrides take precedence over lora_rank. None falls back first to
    # lora_rank and then to the variant default.
    paligemma_lora_rank: int | None = None      # Variant default: 16.
    action_expert_lora_rank: int | None = None  # Variant default: 32.

    action_expert_num_moe_experts: int | None = None
    action_expert_moe_mlp_dim: int | None = None
    action_expert_num_experts_per_tok: int | None = None
    action_expert_router_aux_loss_coef: float | None = None
    action_expert_router_entropy_loss_coef: float | None = None
    action_expert_router_temperature: float | None = None
    action_expert_router_noise_std: float | None = None
    action_expert_moe_type: Literal["vanilla", "adamoe"] | None = None
    action_expert_router_input: Literal["token", "pooled", "pooled_time", "token_pooled", "token_pooled_time"] | None = None
    action_expert_moe_gamma: float | None = None
    action_expert_router_type: Literal["dense", "prototype"] | None = None
    action_expert_router_prototype_dim: int | None = None
    action_expert_routing_contrastive_temperature: float | None = None
    action_expert_routing_contrastive_loss_coef: float | None = None
    action_expert_dead_expert_loss_coef: float | None = None
    action_expert_dead_expert_min_usage: float | None = None
    action_expert_router_supervised_loss_coef: float = 0.0
    action_expert_router_action_loss_weight_coef: float = 0.0

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

        def _check(field: str, *, ge: float | None = None, gt: float | None = None):
            v = getattr(self, field)
            if v is None:
                return
            if ge is not None and v < ge:
                raise ValueError(f"{field} must be >= {ge}")
            if gt is not None and v <= gt:
                raise ValueError(f"{field} must be > {gt}")

        _check("action_expert_num_experts_per_tok", ge=1)
        _check("action_expert_num_moe_experts", ge=0)
        _check("action_expert_moe_mlp_dim", ge=1)
        _check("action_expert_router_aux_loss_coef", ge=0)
        _check("action_expert_router_entropy_loss_coef", ge=0)
        _check("action_expert_router_temperature", gt=0)
        _check("action_expert_router_noise_std", ge=0)
        _check("action_expert_router_prototype_dim", ge=1)
        _check("action_expert_routing_contrastive_temperature", gt=0)
        _check("action_expert_routing_contrastive_loss_coef", ge=0)
        _check("action_expert_dead_expert_loss_coef", ge=0)
        _check("action_expert_dead_expert_min_usage", ge=0)
        _check("action_expert_router_supervised_loss_coef", ge=0)
        _check("action_expert_router_action_loss_weight_coef", ge=0)

        if self.action_expert_moe_type is not None and self.action_expert_moe_type not in ("vanilla", "adamoe"):
            raise ValueError(f"Unsupported action_expert_moe_type: {self.action_expert_moe_type}")
        if self.action_expert_router_type is not None and self.action_expert_router_type not in ("dense", "prototype"):
            raise ValueError(f"Unsupported action_expert_router_type: {self.action_expert_router_type}")
        if self.action_expert_router_input is not None and self.action_expert_router_input not in (
            "token", "pooled", "pooled_time", "token_pooled", "token_pooled_time",
        ):
            raise ValueError(f"Unsupported action_expert_router_input: {self.action_expert_router_input}")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    @override
    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "_model.BaseModel":
        ae_lora_rank = self.action_expert_lora_rank if self.action_expert_lora_rank is not None else self.lora_rank
        overrides: dict = {}
        overrides.update(_detect_lora_variants(params, self.paligemma_variant, self.action_expert_variant))
        # Use the (possibly updated) variant names when detecting MoE, so ae_width lookup is correct
        ae_variant = overrides.get("action_expert_variant", self.action_expert_variant)
        overrides.update(_detect_action_expert_moe(params, ae_variant, self.action_expert_num_moe_experts, lora_rank=ae_lora_rank))
        if overrides:
            adjusted = dataclasses.replace(self, **overrides)
            print(
                f"[pi0_config] Checkpoint structure auto-detected — adjusting config:\n"
                f"  paligemma_variant : {self.paligemma_variant!r} → {adjusted.paligemma_variant!r}\n"
                f"  action_expert_variant : {self.action_expert_variant!r} → {adjusted.action_expert_variant!r}\n"
                f"  action_expert_num_moe_experts : {self.action_expert_num_moe_experts} → {adjusted.action_expert_num_moe_experts}\n"
                f"  overrides applied: {overrides}"
            )
        else:
            adjusted = self
            print(
                f"[pi0_config] Checkpoint structure matches config — no adjustment needed "
                f"(paligemma={self.paligemma_variant!r}, ae={self.action_expert_variant!r}, moe={self.action_expert_num_moe_experts})"
            )
        # Call BaseModelConfig.load directly to avoid infinite recursion via the override
        return _model.BaseModelConfig.load(adjusted, params, remove_extra_params=remove_extra_params)

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            filters.append(
                nnx.Not(
                    nnx_utils.PathRegex(
                        ".*(lora|global_router|prototype_router|router_context_proj|global_scale_adapter).*"
                    )
                ),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)


def _detect_action_expert_moe(
    params: at.Params,
    action_expert_variant: str,
    config_num_experts: int | None,
    *,
    lora_rank: int | None = None,
) -> dict:
    """Inspect checkpoint params and return Pi0Config field overrides to match checkpoint's MoE structure.

    Returns an empty dict if no adjustment is needed.
    """
    llm_params = params.get("PaliGemma", {}).get("llm", {})
    ckpt_has_moe = "global_router_1" in llm_params
    config_has_moe = (config_num_experts or 0) > 0

    # ── Case 1: both agree on no-MoE ─────────────────────────────────────────
    if not ckpt_has_moe and not config_has_moe:
        return {}

    # ── Case 2: config says MoE, checkpoint doesn't have it ──────────────────
    if not ckpt_has_moe and config_has_moe:
        return {"action_expert_num_moe_experts": 0}

    # ── Case 3: checkpoint has MoE — detect structural params ────────────────
    router_params = llm_params["global_router_1"]

    # Count experts from scan-stacked layer params
    moe_layer_params = llm_params.get("layers", {}).get("moe_1", {})
    num_experts = sum(1 for k in moe_layer_params if k.startswith("expert_"))

    # Router type: prototype router has a 'prototypes' parameter
    router_type: _gemma.RouterType = "prototype" if "prototypes" in router_params else "dense"

    overrides: dict = {
        "action_expert_num_moe_experts": num_experts,
        "action_expert_router_type": router_type,
    }

    if router_type == "dense":
        # Detect router_input from context_proj presence and router kernel input dim
        has_context_proj = "context_proj" in router_params
        if not has_context_proj:
            overrides["action_expert_router_input"] = "token"
        else:
            router_kernel = router_params.get("router", {}).get("kernel", None)
            if router_kernel is not None and hasattr(router_kernel, "shape"):
                ae_width = _gemma.get_config(action_expert_variant, lora_rank=lora_rank).width
                input_dim = router_kernel.shape[0]
                if input_dim == ae_width:
                    overrides["action_expert_router_input"] = "pooled"
                elif input_dim == 3 * ae_width:
                    overrides["action_expert_router_input"] = "token_pooled_time"
                else:  # 2 * ae_width → token_pooled (most common) or pooled_time
                    overrides["action_expert_router_input"] = "token_pooled"
        # Detect adamoe from scale_adapter param
        if "scale_adapter" in router_params:
            overrides["action_expert_moe_type"] = "adamoe"

    # Checkpoint is the source of truth for MoE structure — always return detected overrides
    return overrides


def _detect_lora_variants(
    params: at.Params,
    paligemma_variant: str,
    action_expert_variant: str,
) -> dict:
    """Detect LoRA presence from checkpoint and return variant/rank overrides for Pi0Config.

    Returns an empty dict if no adjustment is needed.
    """
    attn_params = params.get("PaliGemma", {}).get("llm", {}).get("layers", {}).get("attn", {})
    overrides: dict = {}

    # PaliGemma (expert index 0) → q_einsum (no suffix)
    pg_q = attn_params.get("q_einsum", {})
    ckpt_pg_lora = "lora_a" in pg_q
    config_pg_lora = "lora" in paligemma_variant

    if config_pg_lora and not ckpt_pg_lora:
        # Config says LoRA, checkpoint has none → strip _lora from variant
        overrides["paligemma_variant"] = paligemma_variant.replace("_lora", "")
    elif not config_pg_lora and ckpt_pg_lora:
        # Checkpoint has LoRA, config doesn't → add _lora and detect rank
        lora_a = pg_q.get("lora_a")
        overrides["paligemma_variant"] = paligemma_variant + "_lora"
        if lora_a is not None and hasattr(lora_a, "shape"):
            overrides["paligemma_lora_rank"] = int(lora_a.shape[-1])

    # Action expert (expert index 1) → q_einsum_1
    ae_q = attn_params.get("q_einsum_1", {})
    ckpt_ae_lora = "lora_a" in ae_q
    config_ae_lora = "lora" in action_expert_variant

    if config_ae_lora and not ckpt_ae_lora:
        overrides["action_expert_variant"] = action_expert_variant.replace("_lora", "")
    elif not config_ae_lora and ckpt_ae_lora:
        lora_a = ae_q.get("lora_a")
        overrides["action_expert_variant"] = action_expert_variant + "_lora"
        if lora_a is not None and hasattr(lora_a, "shape"):
            overrides["action_expert_lora_rank"] = int(lora_a.shape[-1])

    return overrides
