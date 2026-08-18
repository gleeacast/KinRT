# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma adaptation for Pi, taken from big_vision.

We follow this einsum axis naming convention:
  B: batch
  T: query length
  S: k/v length
  N: num query heads
  K: num k/v heads
  G: num query heads per k/v head
  H: head dim
  D: d_model ("features")
"""

from collections.abc import Sequence
import dataclasses
from typing import Literal, TypeAlias

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.lora as lora
import openpi.shared.array_typing as at
import openpi.training.sharding as sharding

PALIGEMMA_VOCAB_SIZE = 257_152

MoeType = Literal["vanilla", "adamoe"]
RouterInput = Literal["token", "pooled", "pooled_time", "token_pooled", "token_pooled_time"]
RouterType = Literal["dense", "prototype"]


@dataclasses.dataclass
class Config:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    moe_mlp_dim: int | None = None
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)
    num_local_experts: int = 0
    num_experts_per_tok: int = 2
    router_aux_loss_coef: float = 0.0
    router_entropy_loss_coef: float = 0.0
    router_temperature: float = 1.0
    router_noise_std: float = 0.0
    moe_type: MoeType = "vanilla"
    router_input: RouterInput = "token_pooled"
    moe_gamma: float = 0.01
    router_type: RouterType = "dense"
    router_prototype_dim: int = 512
    routing_contrastive_temperature: float = 0.07
    routing_contrastive_loss_coef: float = 0.0
    dead_expert_loss_coef: float = 0.0
    dead_expert_min_usage: float = 0.03


Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "moe_gemma"]


def get_config(variant: Variant, lora_rank: int | None = None) -> Config:
    """Returns config for specified gemma variant.

    Args:
        variant: Model variant name.
        lora_rank: LoRA rank to use for lora variants. If None, falls back to the
            per-variant default (gemma_2b_lora → 16, gemma_300m_lora → 32).
            Pass an explicit value to override (e.g. lora_rank=8) when tuning rank
            across experiments. alpha is always set equal to rank so scaling = 1.0.
    """
    if variant == "dummy":
        return Config(
            width=64,
            depth=4,
            mlp_dim=128,
            num_heads=8,
            num_kv_heads=1,
            head_dim=16,
        )
    if variant == "gemma_300m":
        # 311M params
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
        )
    if variant == "gemma_2b_lora":
        # The default rank is 16; lora_rank may override it for controlled comparisons.
        rank = lora_rank if lora_rank is not None else 16
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=rank, alpha=float(rank)), "ffn": lora.LoRAConfig(rank=rank, alpha=float(rank))},
        )
    if variant == "gemma_300m_lora":
        # 311M parameters; the default rank is 32 and lora_rank may override it.
        rank = lora_rank if lora_rank is not None else 32
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=rank, alpha=float(rank)), "ffn": lora.LoRAConfig(rank=rank, alpha=float(rank))},
        )
    if variant == "moe_gemma":
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            num_local_experts=4,
            num_experts_per_tok=2,
            router_aux_loss_coef=0.0,
            router_input="token_pooled",
        )
    raise ValueError(f"Unknown variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x, cond):
        dtype = x.dtype  # original dtype, could be half-precision
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)  # compute variance in float32
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))  # compute normalization in float32
        if cond is None:
            # regular RMSNorm
            scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
            normed_inputs = normed_inputs * (
                1 + scale
            )  # scale by learned parameter in float32 (matches Flax implementation)
            return normed_inputs.astype(dtype), None  # return in original dtype

        # adaptive RMSNorm
        modulation = nn.Dense(x.shape[-1] * 3, kernel_init=nn.initializers.zeros, dtype=dtype)(cond)
        scale, shift, gate = jnp.split(modulation[:, None, :], 3, axis=-1)
        normed_inputs = normed_inputs * (1 + scale) + shift  # scale and shift in float32
        return normed_inputs.astype(dtype), gate


@at.typecheck
class Embedder(nn.Module):
    """Embedder module."""

    vocab_size: int
    embed_dim: int

    def setup(self):
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        x = self.input_embedding_table[(x,)]
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Attention module."""

    configs: Sequence[Config]

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        # all experts must share the same head dim, num heads, and num kv heads for self-attention to work
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        dtype = next(x.dtype for x in xs if x is not None)  # original dtype, could be half-precision

        qkvs = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue
            if config.num_kv_heads == config.num_heads:
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),
                    name=_name("qkv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
                    lora_config=config.lora_configs.get("attn"),
                )
                q = q_einsum("BTD,NDH->BTNH", x)
                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        q = _apply_rope(q, positions=positions)
        q *= self.configs[0].head_dim ** -0.5

        k = _apply_rope(k, positions=positions)

        # should still be half-precision here (if input was half-precision)
        assert q.dtype == k.dtype == v.dtype == dtype

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)
            v = jnp.concatenate([cache_v, v], axis=1)

        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask with shape {attn_mask.shape} but shapes for q and k are: {q.shape} and {k.shape}"
            )

        # big_neg = jnp.finfo(logits.dtype).min
        big_neg = -2.3819763e38  # See gemma/modules.py
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        out = []
        start = 0
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                end = start + x.shape[1]
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),
                    name=_name("attn_vec_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),
                )
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end
            else:
                out.append(None)

        return out, (k, v)


@at.typecheck
class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # original dtype, could be half-precision
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)
        ff_gate = jnp.dot(x, w_gating[0])
        gate_value = nn.gelu(ff_gate)

        ff1 = jnp.dot(x, w_gating[1])
        activations = gate_value * ff1

        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype
        return outputs


def _topk_routing(
    router_logits: jnp.ndarray,
    top_k: int,
    *,
    temperature: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    temperature = max(float(temperature), 1e-6)
    probs = jax.nn.softmax(router_logits / temperature, axis=-1)
    top_weights, top_indices = jax.lax.top_k(probs, k=top_k)
    if top_k > 1:
        top_weights = top_weights / jnp.clip(jnp.sum(top_weights, axis=-1, keepdims=True), a_min=1e-6)
    else:
        top_weights = jnp.ones_like(top_weights)
    one_hot = jax.nn.one_hot(top_indices, router_logits.shape[-1], dtype=router_logits.dtype)
    combine_weights = jnp.sum(one_hot * top_weights[..., None], axis=-2)
    return combine_weights, probs, top_indices, top_weights


def _l2_normalize(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    return x * jax.lax.rsqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=True) + 1e-6)


def _broadcast_batch_vector(vector: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    if vector.ndim == 3:
        vector = jnp.mean(vector, axis=0)
    return jnp.broadcast_to(vector[:, None, :], target.shape[:2] + vector.shape[-1:])


def _zeros_batch_vector_like(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.zeros((x.shape[0], x.shape[-1]), dtype=x.dtype)


def _masked_mean_pool(tokens: jnp.ndarray, mask: jnp.ndarray | None) -> jnp.ndarray:
    if mask is None:
        return jnp.mean(tokens, axis=1)
    weights = mask.astype(tokens.dtype)[..., None]
    denom = jnp.maximum(jnp.sum(weights, axis=1), 1.0)
    return jnp.sum(tokens * weights, axis=1) / denom


def _project_router_context(
    router_context: jnp.ndarray | None,
    *,
    features: int,
    dtype: jnp.dtype,
    name: str,
) -> jnp.ndarray | None:
    if router_context is None:
        return None
    return nn.Dense(features=features, dtype=dtype, name=name)(router_context)


def _build_router_input(
    x: jnp.ndarray,
    config: Config,
    router_context: jnp.ndarray | None,
    time_context: jnp.ndarray | None,
) -> jnp.ndarray:
    if router_context is None:
        router_context = _zeros_batch_vector_like(x)
    if time_context is None:
        time_context = _zeros_batch_vector_like(x)

    pooled_tokens = _broadcast_batch_vector(router_context.astype(x.dtype), x)
    time_tokens = _broadcast_batch_vector(time_context.astype(x.dtype), x)

    if config.router_input == "token":
        return x
    if config.router_input == "pooled":
        return pooled_tokens
    if config.router_input == "pooled_time":
        return jnp.concatenate([pooled_tokens, time_tokens], axis=-1)
    if config.router_input == "token_pooled":
        return jnp.concatenate([x, pooled_tokens], axis=-1)
    if config.router_input == "token_pooled_time":
        return jnp.concatenate([x, pooled_tokens, time_tokens], axis=-1)
    raise ValueError(f"Unsupported MoE router input: {config.router_input}")


def _load_balance_loss(router_probs: jnp.ndarray, top_indices: jnp.ndarray, top_k: int) -> jnp.ndarray:
    num_experts = router_probs.shape[-1]
    selected = jax.nn.one_hot(top_indices, num_experts, dtype=router_probs.dtype)
    selected = jnp.sum(selected, axis=-2)
    tokens_per_expert = jnp.sum(selected, axis=1) * (num_experts / (router_probs.shape[1] * top_k))
    prob_per_expert = jnp.mean(router_probs, axis=1)
    return jnp.mean(jnp.sum(tokens_per_expert * prob_per_expert, axis=-1))


def _router_entropy_loss(router_probs: jnp.ndarray) -> jnp.ndarray:
    entropy = -jnp.sum(router_probs * jnp.log(jnp.clip(router_probs, a_min=1e-9)), axis=-1)
    return jnp.mean(entropy).astype(jnp.float32)


def _routing_contrastive_loss(
    router_embedding: jnp.ndarray,
    prototypes: jnp.ndarray,
    top_indices: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    num_experts = prototypes.shape[-2]
    top1 = top_indices[..., 0]
    selected = jax.nn.one_hot(top1, num_experts, dtype=router_embedding.dtype)
    counts = jnp.sum(selected, axis=0)
    cluster_means = jnp.einsum("be,bd->ed", selected, router_embedding) / jnp.clip(counts[:, None], a_min=1.0)
    cluster_means = _l2_normalize(cluster_means)
    prototypes = _l2_normalize(prototypes.astype(jnp.float32))
    logits = jnp.einsum("ed,fd->ef", prototypes, cluster_means) / temperature
    labels = jnp.arange(num_experts)
    per_expert_loss = -jax.nn.log_softmax(logits, axis=-1)[labels, labels]
    active = counts > 0
    return jnp.sum(jnp.where(active, per_expert_loss, 0.0)) / jnp.clip(jnp.sum(active), a_min=1.0)


def _dead_expert_loss(
    top_indices: jnp.ndarray,
    router_probs: jnp.ndarray,
    num_experts: int,
    min_usage: float,
) -> jnp.ndarray:
    top1 = top_indices[..., 0]
    hard_usage = jnp.mean(jax.nn.one_hot(top1, num_experts, dtype=jnp.float32), axis=0)
    soft_usage = jnp.mean(router_probs.astype(jnp.float32), axis=0)
    usage = jax.lax.stop_gradient(hard_usage - soft_usage) + soft_usage
    return jnp.mean(jnp.square(jnp.maximum(0.0, min_usage - usage)))


def _router_metrics(moe_routings: Sequence[dict[str, jnp.ndarray] | None]) -> dict[str, jnp.ndarray]:
    probs = [r["router_probs"].astype(jnp.float32) for r in moe_routings if r is not None]
    if not probs:
        zero = jnp.array(0.0, dtype=jnp.float32)
        return {
            "router_top1_prob": zero,
            "router_entropy_norm": zero,
            "router_top1_margin": zero,
            "router_usage_max": zero,
            "router_usage": jnp.array([], dtype=jnp.float32),
        }

    router_probs = jnp.concatenate([jnp.reshape(p, (-1, p.shape[-1])) for p in probs], axis=0)
    top1 = jnp.max(router_probs, axis=-1)
    top1_prob = jnp.mean(top1)
    masked_probs = jnp.where(router_probs == top1[:, None], -jnp.inf, router_probs)
    top2 = jnp.max(masked_probs, axis=-1)
    top1_margin = jnp.mean(jnp.where(jnp.isfinite(top2), top1 - top2, 0.0))
    entropy = -jnp.sum(router_probs * jnp.log(jnp.clip(router_probs, a_min=1e-9)), axis=-1)
    entropy_denom = jnp.maximum(jnp.log(jnp.asarray(router_probs.shape[-1], dtype=jnp.float32)), 1e-9)
    entropy_norm = jnp.mean(entropy / entropy_denom)

    top_indices = [jnp.reshape(r["top_indices"], (-1, r["top_indices"].shape[-1])) for r in moe_routings if r is not None]
    top_indices = jnp.concatenate(top_indices, axis=0)
    hard_usage = jnp.mean(jax.nn.one_hot(top_indices, router_probs.shape[-1], dtype=jnp.float32), axis=(0, 1))
    first_routing = next(r for r in moe_routings if r is not None)
    return {
        "router_top1_prob": top1_prob,
        "router_entropy_norm": entropy_norm,
        "router_top1_margin": top1_margin,
        "router_usage_max": jnp.max(hard_usage),
        "router_usage": hard_usage,
        "router_route_logits": first_routing["route_logits"],
        "router_route_probs": first_routing["route_probs"],
        "router_route_top_indices": first_routing["route_top_indices"],
    }


MoeRouting: TypeAlias = dict[str, jnp.ndarray]


@at.typecheck
class DenseMoeRouter(nn.Module):
    """Compute global expert scores from the configured action-expert routing context.

    KinRT supplies one masked-mean prefix vector per sample.
    Top-k selection is discrete; normalized selected probabilities determine the
    mixture weights applied to every action token in the routed layer.
    """

    config: Config
    embed_dtype: str

    @nn.compact
    def __call__(
        self,
        x,
        router_contexts: jnp.ndarray | None,
        time_context: jnp.ndarray | None,
        router_noise_rng: jax.Array | None,
        deterministic: bool,
    ) -> MoeRouting:
        config = self.config
        num_experts = config.num_local_experts
        layer_router_context = (
            nn.Dense(features=config.width, dtype=jnp.dtype(self.embed_dtype), name="context_proj")(router_contexts)
            if config.router_input != "token" and router_contexts is not None
            else None
        )
        router_input = _build_router_input(x, config, layer_router_context, time_context)
        route_input = router_input[:, :1, :]
        router_logits = nn.Dense(features=num_experts, use_bias=False, dtype=jnp.float32, name="router")(route_input)
        if not deterministic and config.router_noise_std > 0.0:
            if router_noise_rng is None:
                raise ValueError("router_noise_rng must be provided when router noise is enabled")
            noise = jax.random.normal(router_noise_rng, router_logits.shape, dtype=router_logits.dtype)
            router_logits = router_logits + noise * config.router_noise_std

        top_k = config.num_experts_per_tok
        combine_weights, router_probs, top_indices, top_weights = _topk_routing(
            router_logits,
            top_k,
            temperature=config.router_temperature,
        )
        combine_weights = jnp.broadcast_to(combine_weights, x.shape[:2] + (num_experts,))
        router_probs = jnp.broadcast_to(router_probs, x.shape[:2] + (num_experts,))
        top_indices = jnp.broadcast_to(top_indices, x.shape[:2] + top_indices.shape[-1:])
        top_weights = jnp.broadcast_to(top_weights, x.shape[:2] + top_weights.shape[-1:])
        aux_loss = _load_balance_loss(router_probs, top_indices, top_k).astype(jnp.float32)
        entropy_loss = _router_entropy_loss(router_probs)

        if config.moe_type == "adamoe":
            scale_logits = nn.Dense(
                features=num_experts,
                use_bias=True,
                dtype=jnp.float32,
                kernel_init=nn.initializers.zeros,
                bias_init=nn.initializers.zeros,
                name="scale_adapter",
            )(router_input)
            scale_weights = jax.nn.sigmoid(scale_logits)
            top_weights = top_weights * jnp.take_along_axis(scale_weights, top_indices, axis=-1)
            one_hot = jax.nn.one_hot(top_indices, num_experts, dtype=router_logits.dtype)
            combine_weights = jnp.sum(one_hot * top_weights[..., None], axis=-2)
        elif config.moe_type != "vanilla":
            raise ValueError(f"Unsupported MoE type: {config.moe_type}")

        return {
            "combine_weights": jnp.broadcast_to(combine_weights, (config.depth,) + combine_weights.shape),
            "router_probs": jnp.broadcast_to(router_probs, (config.depth,) + router_probs.shape),
            "top_indices": jnp.broadcast_to(top_indices, (config.depth,) + top_indices.shape),
            "top_weights": jnp.broadcast_to(top_weights, (config.depth,) + top_weights.shape),
            "route_input": jnp.broadcast_to(route_input, (config.depth,) + route_input.shape),
            "route_logits": jnp.broadcast_to(router_logits, (config.depth,) + router_logits.shape),
            "route_probs": jnp.broadcast_to(router_probs[:, :1, :], (config.depth,) + router_probs[:, :1, :].shape),
            "route_top_indices": jnp.broadcast_to(top_indices[:, :1, :], (config.depth,) + top_indices[:, :1, :].shape),
            "route_top_weights": jnp.broadcast_to(top_weights[:, :1, :], (config.depth,) + top_weights[:, :1, :].shape),
            "aux_loss": jnp.broadcast_to(aux_loss, (config.depth,)),
            "entropy_loss": jnp.broadcast_to(entropy_loss, (config.depth,)),
            "routing_contrastive_loss": jnp.zeros((config.depth,), dtype=jnp.float32),
            "dead_expert_loss": jnp.zeros((config.depth,), dtype=jnp.float32),
        }


@at.typecheck
class PrototypeMoeRouter(nn.Module):
    """Route each sample to the nearest learned expert prototype in projected context space."""

    config: Config

    @nn.compact
    def __call__(
        self,
        x,
        router_contexts: jnp.ndarray | None,
        time_context: jnp.ndarray | None,
        router_noise_rng: jax.Array | None,
        deterministic: bool,
    ) -> MoeRouting:
        del time_context, router_noise_rng, deterministic
        config = self.config
        num_experts = config.num_local_experts
        if router_contexts is None:
            raise ValueError("Prototype MoE routing requires router_contexts from the prefix encoder.")
        if config.num_experts_per_tok != 1:
            raise ValueError("Prototype MoE routing uses hard top-1 routing; set num_experts_per_tok=1.")
        if config.moe_type == "adamoe":
            raise ValueError("Prototype MoE routing does not support adamoe scale adapters.")
        if config.moe_type != "vanilla":
            raise ValueError(f"Unsupported MoE type: {config.moe_type}")

        route_input = nn.Dense(features=config.router_prototype_dim, dtype=jnp.float32, name="proj")(router_contexts)
        route_input = _l2_normalize(route_input.astype(jnp.float32))
        prototypes = self.param(
            "prototypes",
            nn.initializers.normal(stddev=0.02),
            (config.depth, num_experts, config.router_prototype_dim),
            jnp.float32,
        )
        prototypes = _l2_normalize(prototypes.astype(jnp.float32))
        router_logits = jnp.einsum("bd,led->lbe", route_input, prototypes)
        route_probs = jax.nn.softmax(router_logits / config.routing_contrastive_temperature, axis=-1)
        route_top_indices = jnp.argmax(router_logits, axis=-1)[..., None]
        route_top_weights = jnp.ones(route_top_indices.shape, dtype=router_logits.dtype)
        combine_weights = jax.nn.one_hot(route_top_indices[..., 0], num_experts, dtype=router_logits.dtype)

        contrastive_loss = jax.vmap(
            lambda p, ti: _routing_contrastive_loss(
                route_input,
                p,
                ti,
                config.routing_contrastive_temperature,
            )
        )(prototypes, route_top_indices)
        dead_loss = jax.vmap(
            lambda ti, rp: _dead_expert_loss(ti, rp, num_experts, config.dead_expert_min_usage)
        )(route_top_indices, route_probs)

        route_input = jnp.broadcast_to(route_input[None, :, None, :], (config.depth, x.shape[0], 1, route_input.shape[-1]))
        router_logits = router_logits[:, :, None, :]
        route_probs = route_probs[:, :, None, :]
        route_top_indices = route_top_indices[:, :, None, :]
        route_top_weights = route_top_weights[:, :, None, :]
        return {
            "combine_weights": jnp.broadcast_to(
                combine_weights[:, :, None, :],
                (config.depth, x.shape[0], x.shape[1], num_experts),
            ),
            "router_probs": jnp.broadcast_to(route_probs, (config.depth, x.shape[0], x.shape[1], num_experts)),
            "top_indices": jnp.broadcast_to(route_top_indices, (config.depth, x.shape[0], x.shape[1], 1)),
            "top_weights": jnp.broadcast_to(route_top_weights, (config.depth, x.shape[0], x.shape[1], 1)),
            "route_input": route_input,
            "route_logits": router_logits,
            "route_probs": route_probs,
            "route_top_indices": route_top_indices,
            "route_top_weights": route_top_weights,
            "aux_loss": jnp.zeros((config.depth,), dtype=jnp.float32),
            "entropy_loss": jnp.zeros((config.depth,), dtype=jnp.float32),
            "routing_contrastive_loss": contrastive_loss.astype(jnp.float32),
            "dead_expert_loss": dead_loss.astype(jnp.float32),
        }


@at.typecheck
class RoutedMoeFeedForward(nn.Module):
    """Evaluate extra action-expert FFNs and combine them using router weights.

    This module produces the routed residual branch; the transformer block retains
    its original dense feed-forward path. Global routing weights are broadcast over
    action tokens, so one expert decision governs an entire action chunk.
    """

    config: Config

    @nn.compact
    def __call__(
        self,
        x,
        routing: MoeRouting,
        return_router_info: bool = False,  # noqa: FBT002
    ):
        num_extra_experts = self.config.num_local_experts
        dtype = x.dtype
        combine_weights = routing["combine_weights"]
        aux_loss = routing["aux_loss"]

        if self.config.moe_type not in ("vanilla", "adamoe"):
            raise ValueError(f"Unsupported MoE type: {self.config.moe_type}")

        routed_outputs = []
        for expert_idx in range(num_extra_experts):
            expert_out = lora.FeedForward(
                features=self.config.width,
                hidden_dim=self.config.mlp_dim,
                name=f"expert_{expert_idx}",
                lora_config=self.config.lora_configs.get("ffn"),
            )(x)
            routed_outputs.append(expert_out)
        routed_outputs = jnp.stack(routed_outputs, axis=-2)
        expert_diag = {}
        if return_router_info:
            expert_outputs_f32 = routed_outputs.astype(jnp.float32)
            expert_norms = jnp.linalg.norm(expert_outputs_f32, axis=-1)
            expert_unit = expert_outputs_f32 / jnp.clip(expert_norms[..., None], a_min=1e-6)
            expert_diag = {
                "expert_norms": expert_norms,
                "expert_cosine": jnp.einsum("bted,btfd->btef", expert_unit, expert_unit),
            }
        routed = jnp.einsum("bte,bted->btd", combine_weights, routed_outputs)
        router_info = (
            {
                **expert_diag,
            }
            if return_router_info
            else None
        )
        return routed.astype(dtype), aux_loss, router_info


@at.typecheck
class Block(nn.Module):
    """Transformer block."""

    configs: tuple[Config, ...]

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(
        self,
        xs,
        kv_cache,
        positions,
        attn_mask,
        adarms_cond,
        router_contexts,
        router_context_mask,
        moe_routings,
        deterministic=True,  # noqa: FBT002
        return_router_info: bool = False,  # noqa: FBT002
    ):
        xs = sharding.activation_sharding_constraint(xs)
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        attn = Attention(configs=self.configs, name="attn")

        pre_attn = []
        gates = []
        for i, x in enumerate(xs):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_attention_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
            pre_attn.append(x)
            gates.append(gate if x is not None else None)

        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, post_attn, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        out = []
        gates = []
        moe_aux_loss = jnp.array(0.0, dtype=jnp.float32)
        router_info = None
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                x, gate = RMSNorm(name=_name("pre_ffw_norm", i))(x, adarms_cond[i])  # noqa: PLW2901
                if config.num_local_experts > 0:
                    shared = lora.FeedForward(
                        features=config.width,
                        hidden_dim=config.mlp_dim,
                        name=_name("mlp", i),
                        lora_config=config.lora_configs.get("ffn"),
                    )(x)
                    routed, aux_loss, info = RoutedMoeFeedForward(config=config, name=_name("moe", i))(
                        x,
                        routing=moe_routings[i],
                        return_router_info=return_router_info,
                    )
                    moe_aux_loss = moe_aux_loss + aux_loss
                    moe_output = 0.5 * shared + 0.5 * routed
                    if info is not None:
                        shared_f32 = shared.astype(jnp.float32)
                        routed_f32 = routed.astype(jnp.float32)
                        moe_output_f32 = moe_output.astype(jnp.float32)

                        shared_norm = jnp.linalg.norm(shared_f32, axis=-1)
                        routed_norm = jnp.linalg.norm(routed_f32, axis=-1)
                        moe_output_norm = jnp.linalg.norm(moe_output_f32, axis=-1)
                        shared_routed_denom = jnp.clip(shared_norm * routed_norm, a_min=1e-6)
                        routing = moe_routings[i]
                        info = {
                            **info,
                            "router_input": routing["route_input"],
                            "router_logits": routing["route_logits"],
                            "router_probs": routing["route_probs"],
                            "top_indices": routing["route_top_indices"],
                            "top_weights": routing["route_top_weights"],
                            "shared_norm": shared_norm,
                            "routed_moe_norm": routed_norm,
                            "moe_output_norm": moe_output_norm,
                            "routed_moe_shared_ratio": routed_norm / jnp.clip(shared_norm, a_min=1e-6),
                            "shared_routed_moe_cosine": jnp.sum(shared_f32 * routed_f32, axis=-1)
                            / shared_routed_denom,
                        }
                        router_info = info
                    x = moe_output  # noqa: PLW2901
                else:
                    x = lora.FeedForward(  # noqa: PLW2901
                        features=config.width,
                        hidden_dim=config.mlp_dim,
                        name=_name("mlp", i),
                        lora_config=config.lora_configs.get("ffn"),
                    )(x)
            out.append(x)
            gates.append(gate if x is not None else None)

        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        xs = [_gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
        xs = sharding.activation_sharding_constraint(xs)

        return xs, (kv_cache, moe_aux_loss, router_info)


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """Transformer model, supporting a mixture of different weights for different tokens."""

    configs: Sequence[Config]  # list of configs, one for each expert
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()  # Every float is dropped independently.
    adarms: bool = False

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )
        self.moe_routers = [
            (
                DenseMoeRouter(config=config, embed_dtype=self.embed_dtype, name=_name("global_router", i))
                if config.router_type == "dense"
                else PrototypeMoeRouter(config=config, name=_name("prototype_router", i))
            )
            if config.num_local_experts > 0
            else None
            for i, config in enumerate(self.configs)
        ]

        block_cls = nn.remat(
            Block,
            prevent_cse=False,
            static_argnums=(9, 10),  # 0=self, 9=deterministic, 10=return_router_info
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            in_axes=(
                0,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                nn.broadcast,
                0,
                nn.broadcast,
                nn.broadcast,
            ),
            length=self.configs[0].depth,
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    def _global_router_context(
        self,
        embedded: Sequence[jnp.ndarray | None],
        router_contexts: jnp.ndarray | None,
        router_context_mask: jnp.ndarray | None,
    ) -> jnp.ndarray | None:
        if router_contexts is not None:
            return router_contexts
        needs_router_context = any(
            c.num_local_experts > 0 and (c.router_input != "token" or c.router_type == "prototype")
            for c in self.configs
        )
        if not needs_router_context:
            return None
        if router_context_mask is not None and embedded[0] is not None:
            return _masked_mean_pool(embedded[0], router_context_mask)
        return None

    def _global_moe_routings(
        self,
        embedded: Sequence[jnp.ndarray | None],
        router_contexts: jnp.ndarray | None,
        adarms_cond: Sequence[jnp.ndarray | None],
        router_noise_rng: jax.Array | None,
        deterministic: bool,
    ) -> tuple[list[MoeRouting | None], dict[str, jnp.ndarray]]:
        routings = []
        for i, (x, config) in enumerate(zip(embedded, self.configs, strict=True)):
            if x is None or config.num_local_experts <= 0:
                routings.append(None)
                continue

            layer_rng = None
            if router_noise_rng is not None:
                router_noise_rng, layer_rng = jax.random.split(router_noise_rng)
            routings.append(
                self.moe_routers[i](
                    x,
                    router_contexts,
                    adarms_cond[i],
                    layer_rng,
                    deterministic,
                )
            )

        active = [r for r in routings if r is not None]
        zero = jnp.array(0.0, dtype=jnp.float32)
        extra_losses = {
            "router_entropy_loss": (
                jnp.mean(jnp.concatenate([r["entropy_loss"].reshape(-1) for r in active])).astype(jnp.float32)
                if active
                else zero
            ),
            "routing_contrastive_loss": (
                jnp.mean(jnp.concatenate([r["routing_contrastive_loss"].reshape(-1) for r in active])).astype(jnp.float32)
                if active
                else zero
            ),
            "dead_expert_loss": (
                jnp.mean(jnp.concatenate([r["dead_expert_loss"].reshape(-1) for r in active])).astype(jnp.float32)
                if active
                else zero
            ),
        }
        return routings, extra_losses

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        router_contexts: at.Float[at.Array, "b _d"] | None = None,
        router_context_mask: at.Bool[at.Array, "b _t"] | None = None,
        router_noise_rng: at.KeyArrayLike | None = None,
        deterministic: bool = True,
        return_router_info: bool = False,
    ) -> tuple[
        Sequence[at.Float[at.Array, "b _t _d"] | None],
        KVCache,
        at.Float[at.Array, ""],
        dict[str, at.Array] | None,
        at.Float[at.Array, "b _d"] | None,
        dict[str, at.Array],
    ]:
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        mask = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)
        router_contexts = self._global_router_context(embedded, router_contexts, router_context_mask)
        moe_routings, router_extra_losses = self._global_moe_routings(
            embedded,
            router_contexts,
            adarms_cond,
            router_noise_rng,
            deterministic,
        )
        router_metrics = _router_metrics(moe_routings)

        embedded, (kv_cache, moe_aux_losses, router_info) = self.layers(
            embedded,
            kv_cache,
            positions,
            mask,
            adarms_cond,
            router_contexts,
            router_context_mask,
            moe_routings,
            deterministic,
            return_router_info,
        )
        # Each routed block reports the same global auxiliary loss, so the mean counts it once.
        moe_aux_loss = jnp.mean(moe_aux_losses)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        router_metrics = {
            **router_metrics,
            **router_extra_losses,
        }
        return [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, embedded, adarms_cond, strict=True)
        ], kv_cache, moe_aux_loss, router_info, router_contexts, router_metrics

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters, necessary due to the quirks of linen."""
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        needs_router_context = any(
            c.num_local_experts > 0 and (c.router_input != "token" or c.router_type == "prototype")
            for c in self.configs
        )
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)],
            router_context_mask=jnp.ones((1, 1), dtype=bool) if needs_router_context else None,
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Applies RoPE positions [B, L] to x [B, L, H, D]."""
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]
    assert radians.dtype == jnp.float32
    # radians.shape = [...,L,1,d=D/2]
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    x1, x2 = jnp.split(x, 2, axis=-1)
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32
    # The original bigvision impl allows RoPE to upcast to float32. It is then immediately downcast again to the cache
    # dtype when in inference mode (but not in training mode). I don't think any of this was intentional. Based on the
    # original DeepMind impl, as well as the widely-used transformers impl, it is ok to always downcast back to bfloat16
    # here.
    return res.astype(x.dtype)


def _name(name, i):
    # we name layers like this because we want the first expert's weights to have no suffix (e.g., "attn"), so that they
    # can be loaded seamlessly from the existing PaliGemma checkpoint. subsequent experts will have a suffix (e.g.,
    # "attn_1") and their weights will be initialized from scratch. in practice, we only use two experts -- PaliGemma,
    # and the action expert.
    if i == 0:
        return name
    return f"{name}_{i}"


def _gated_residual(x, y, gate):
    assert (x is None) == (y is None)
    if x is None:
        return None
    if gate is None:
        return x + y
    return x + y * gate
