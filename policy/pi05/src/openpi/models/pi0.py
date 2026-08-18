import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        # Rank precedence is per-branch override, shared lora_rank, then variant default.
        # The built-in defaults are 16 for PaliGemma and 32 for the action expert.
        pg_rank = config.paligemma_lora_rank if config.paligemma_lora_rank is not None else config.lora_rank
        ae_rank = config.action_expert_lora_rank if config.action_expert_lora_rank is not None else config.lora_rank
        paligemma_config = _gemma.get_config(config.paligemma_variant, lora_rank=pg_rank)
        action_expert_config = _gemma.get_config(config.action_expert_variant, lora_rank=ae_rank)
        _CONFIG_FIELD_MAP = {
            "action_expert_num_moe_experts": "num_local_experts",
            "action_expert_moe_mlp_dim": "moe_mlp_dim",
            "action_expert_num_experts_per_tok": "num_experts_per_tok",
            "action_expert_router_aux_loss_coef": "router_aux_loss_coef",
            "action_expert_router_entropy_loss_coef": "router_entropy_loss_coef",
            "action_expert_router_temperature": "router_temperature",
            "action_expert_router_noise_std": "router_noise_std",
            "action_expert_moe_type": "moe_type",
            "action_expert_router_input": "router_input",
            "action_expert_moe_gamma": "moe_gamma",
            "action_expert_router_type": "router_type",
            "action_expert_router_prototype_dim": "router_prototype_dim",
            "action_expert_routing_contrastive_temperature": "routing_contrastive_temperature",
            "action_expert_routing_contrastive_loss_coef": "routing_contrastive_loss_coef",
            "action_expert_dead_expert_loss_coef": "dead_expert_loss_coef",
            "action_expert_dead_expert_min_usage": "dead_expert_min_usage",
        }
        action_expert_overrides = {
            dst: getattr(config, src)
            for src, dst in _CONFIG_FIELD_MAP.items()
            if getattr(config, src) is not None
        }
        if action_expert_overrides:
            action_expert_config = dataclasses.replace(action_expert_config, **action_expert_overrides)
        self.action_expert_num_moe_experts = action_expert_config.num_local_experts
        self.action_expert_router_aux_loss_coef = action_expert_config.router_aux_loss_coef
        self.action_expert_router_entropy_loss_coef = action_expert_config.router_entropy_loss_coef
        self.action_expert_routing_contrastive_loss_coef = action_expert_config.routing_contrastive_loss_coef
        self.action_expert_dead_expert_loss_coef = action_expert_config.dead_expert_loss_coef
        self.action_expert_router_supervised_loss_coef = config.action_expert_router_supervised_loss_coef
        self.action_expert_router_action_loss_weight_coef = config.action_expert_router_action_loss_weight_coef
        self.action_expert_router_input = action_expert_config.router_input
        self.action_expert_uses_router_context = (
            self.action_expert_num_moe_experts > 0
            and (self.action_expert_router_input != "token" or action_expert_config.router_type == "prototype")
        )
        logger.info(
            "Pi0 effective config: pi05=%s, paligemma_variant=%s, action_expert_variant=%s, "
            "moe_experts=%s, moe_mlp_dim=%s, top_k=%s, aux_coef=%s, entropy_coef=%s, router_temp=%s, "
            "router_noise_std=%s, moe_type=%s, router_input=%s, moe_gamma=%s, "
            "router_type=%s, router_prototype_dim=%s, contrast_coef=%s, dead_coef=%s, "
            "router_supervised_coef=%s, action_class_weight_coef=%s, global_router_context=%s",
            config.pi05,
            config.paligemma_variant,
            config.action_expert_variant,
            action_expert_config.num_local_experts,
            action_expert_config.moe_mlp_dim or action_expert_config.mlp_dim,
            action_expert_config.num_experts_per_tok,
            action_expert_config.router_aux_loss_coef,
            action_expert_config.router_entropy_loss_coef,
            action_expert_config.router_temperature,
            action_expert_config.router_noise_std,
            action_expert_config.moe_type,
            action_expert_config.router_input,
            action_expert_config.moe_gamma,
            action_expert_config.router_type,
            action_expert_config.router_prototype_dim,
            action_expert_config.routing_contrastive_loss_coef,
            action_expert_config.dead_expert_loss_coef,
            config.action_expert_router_supervised_loss_coef,
            config.action_expert_router_action_loss_weight_coef,
            self.action_expert_uses_router_context,
        )
        # TODO: migrate Gemma to NNX; the bridge is required until then.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @staticmethod
    def _masked_mean_pool(tokens: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
        weights = mask.astype(tokens.dtype)[..., None]
        denom = jnp.clip(jnp.sum(weights, axis=1), min=1.0)
        return jnp.sum(tokens * weights, axis=1) / denom

    @staticmethod
    def _router_supervised_loss(
        router_metrics: dict[str, at.Array],
        router_label: at.Array | None,
        router_class_weight: at.Array | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return masked cross-entropy and accuracy for hard offline router labels."""
        if router_label is None or "router_route_logits" not in router_metrics:
            zero = jnp.array(0.0, dtype=jnp.float32)
            return zero, zero

        logits = router_metrics["router_route_logits"].astype(jnp.float32)
        if logits.ndim == 4:
            # Dense router stores [layers, batch, 1, experts]; prototype is the same shape.
            logits = jnp.mean(logits[:, :, 0, :], axis=0)
        elif logits.ndim == 3:
            logits = jnp.mean(logits[:, :, :], axis=0)
        if logits.ndim != 2:
            raise ValueError(f"Unsupported router logits shape for supervised loss: {logits.shape}")

        labels = jnp.asarray(router_label, dtype=jnp.int32)
        valid = labels >= 0
        labels = jnp.clip(labels, 0, logits.shape[-1] - 1)
        per_example = -jax.nn.log_softmax(logits, axis=-1)[jnp.arange(labels.shape[0]), labels]
        weights = jnp.ones_like(per_example, dtype=jnp.float32)
        if router_class_weight is not None:
            weights = jnp.asarray(router_class_weight, dtype=jnp.float32)
        weights = jnp.where(valid, weights, 0.0)
        loss = jnp.sum(per_example * weights) / jnp.clip(jnp.sum(weights), a_min=1e-6)
        accuracy = jnp.sum((jnp.argmax(logits, axis=-1) == labels) * valid) / jnp.clip(jnp.sum(valid), a_min=1)
        return loss.astype(jnp.float32), accuracy.astype(jnp.float32)

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _prefix_router_contexts(self, prefix_out: jnp.ndarray, prefix_mask: jnp.ndarray) -> jnp.ndarray | None:
        """Pool valid vision-language prefix tokens into the global routing context."""
        if not self.action_expert_uses_router_context:
            return None
        pooled = self._masked_mean_pool(prefix_out, prefix_mask)
        # Old per-layer router implementation expanded this to
        # (depth, batch, width) so each MoE layer could consume its own context.
        # return jnp.broadcast_to(pooled, (self.PaliGemma.llm.module.configs[0].depth,) + pooled.shape)
        # The global router now consumes one shared prefix context for all layers.
        return pooled

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        loss, _ = self.compute_loss_with_info(rng, observation, actions, train=train)
        return loss

    def compute_loss_with_info(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        preprocess_rng, noise_rng, time_rng, router_noise_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _, _, _, _, _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=prefix_positions,
        )
        router_contexts = self._prefix_router_contexts(jax.lax.stop_gradient(prefix_out), prefix_mask)

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _, router_aux_loss, _, _, router_metrics = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
            router_contexts=router_contexts,
            router_noise_rng=router_noise_rng,
            deterministic=not train,
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        action_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if (
            observation.router_class_weight is not None
            and self.action_expert_router_action_loss_weight_coef > 0.0
        ):
            class_weight = jnp.asarray(observation.router_class_weight, dtype=action_loss.dtype)
            action_weight = 1.0 + self.action_expert_router_action_loss_weight_coef * (class_weight - 1.0)
            action_loss = action_loss * action_weight[:, None]
        weighted_router_aux_loss = self.action_expert_router_aux_loss_coef * router_aux_loss
        router_entropy_loss = router_metrics["router_entropy_loss"]
        weighted_router_entropy_loss = self.action_expert_router_entropy_loss_coef * router_entropy_loss
        routing_contrastive_loss = router_metrics["routing_contrastive_loss"]
        dead_expert_loss = router_metrics["dead_expert_loss"]
        router_supervised_loss, router_supervised_accuracy = self._router_supervised_loss(
            router_metrics,
            observation.router_label,
            observation.router_class_weight,
        )
        weighted_router_supervised_loss = (
            self.action_expert_router_supervised_loss_coef * router_supervised_loss
        )
        weighted_routing_contrastive_loss = (
            self.action_expert_routing_contrastive_loss_coef * routing_contrastive_loss
        )
        weighted_dead_expert_loss = self.action_expert_dead_expert_loss_coef * dead_expert_loss
        total_loss = (
            action_loss
            + weighted_router_aux_loss
            + weighted_router_entropy_loss
            + weighted_routing_contrastive_loss
            + weighted_dead_expert_loss
            + weighted_router_supervised_loss
        )
        scalar_router_metrics = {
            k: v
            for k, v in router_metrics.items()
            if getattr(v, "ndim", 0) == 0
        }
        loss_info = {
            "action_loss": jnp.mean(action_loss),
            "router_aux_loss": router_aux_loss,
            "router_aux_loss_weighted": weighted_router_aux_loss,
            "router_entropy_loss_weighted": weighted_router_entropy_loss,
            "router_supervised_loss": router_supervised_loss,
            "router_supervised_loss_weighted": weighted_router_supervised_loss,
            "router_supervised_accuracy": router_supervised_accuracy,
            "routing_contrastive_loss": routing_contrastive_loss,
            "routing_contrastive_loss_weighted": weighted_routing_contrastive_loss,
            "dead_expert_loss": dead_expert_loss,
            "dead_expert_loss_weighted": weighted_dead_expert_loss,
            **scalar_router_metrics,
        }
        if "router_usage" in router_metrics:
            for expert_idx in range(router_metrics["router_usage"].shape[0]):
                loss_info[f"router_usage_{expert_idx}"] = router_metrics["router_usage"][expert_idx]
        return total_loss, loss_info

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        x_0, _ = self._sample_actions_core(rng, observation, num_steps=num_steps, noise=noise, return_router_info=False)
        return x_0

    def sample_actions_with_router_info(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> tuple[_model.Actions, dict[str, at.Array]]:
        observation = _model.preprocess_observation(None, observation, train=False)
        return self._sample_actions_core(rng, observation, num_steps=num_steps, noise=noise, return_router_info=True)

    def _sample_actions_core(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        return_router_info: bool = False,
    ) -> tuple[_model.Actions, dict[str, at.Array] | None]:
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache, _, _, _, _ = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        router_contexts = self._prefix_router_contexts(prefix_out, prefix_mask)

        def denoise_step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _, _, router_info, _, _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                router_contexts=router_contexts if self.action_expert_uses_router_context else None,
                return_router_info=return_router_info,
            )
            if router_info is not None:
                router_info = {
                    **router_info,
                    "router_timestep": jnp.broadcast_to(time, (batch_size,)),
                }
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return (x_t + dt * v_t, time + dt), router_info

        if return_router_info:
            def scan_step(carry, _):
                return denoise_step(carry)

            (x_0, _), router_info = jax.lax.scan(scan_step, (noise, 1.0), xs=None, length=num_steps)
            return x_0, router_info

        def while_step(carry):
            carry, _ = denoise_step(carry)
            return carry

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, while_step, (noise, 1.0))
        return x_0, None
