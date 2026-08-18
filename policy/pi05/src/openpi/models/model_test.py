import dataclasses

from flax import nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.gemma as _gemma
from openpi.models import pi0_config
from openpi.shared import nnx_utils
from openpi.training import weight_loaders


def test_pi0_dense_moe_router():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_expert_num_moe_experts=2,
        action_expert_num_experts_per_tok=2,
        action_expert_router_aux_loss_coef=2e-4,
        action_expert_router_entropy_loss_coef=1e-5,
        action_expert_router_temperature=0.7,
        action_expert_router_noise_std=0.05,
        action_expert_router_input="pooled_time",
        action_horizon=4,
        action_dim=8,
        max_token_len=8,
        pi05=True,
    )
    model = config.create(key)
    model_state = nnx.state(model)

    context_proj_params = model_state.filter(nnx_utils.PathRegex(".*/llm/global_router_1/context_proj/.*")).flat_state()
    router_kernel_state = model_state.filter(nnx_utils.PathRegex(".*/llm/global_router_1/router/kernel")).flat_state()
    router_kernel = next(iter(router_kernel_state.values())).value
    assert len(context_proj_params) == 2
    assert router_kernel.shape == (128, config.action_expert_num_moe_experts)
    assert len(model_state.filter(nnx_utils.PathRegex(".*/llm/prototype_router_1/.*")).flat_state()) == 0
    assert len(model_state.filter(nnx_utils.PathRegex(".*/llm/layers/moe_1/expert_0/.*")).flat_state()) == 2
    assert len(model_state.filter(nnx_utils.PathRegex(".*/llm/layers/moe_1/expert_1/.*")).flat_state()) == 2

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss, loss_info = nnx_utils.module_jit(model.compute_loss_with_info)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert {
        "action_loss",
        "router_aux_loss",
        "router_aux_loss_weighted",
        "router_entropy_loss",
        "router_entropy_loss_weighted",
        "routing_contrastive_loss",
        "dead_expert_loss",
    }.issubset(loss_info)

    actions, router_info = nnx_utils.module_jit(
        model.sample_actions_with_router_info, static_argnames=("num_steps",)
    )(key, obs, num_steps=2)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
    assert router_info["router_probs"].shape == (2, 4, batch_size, 1, 2)
    assert router_info["top_indices"].shape == (2, 4, batch_size, 1, config.action_expert_num_experts_per_tok)
    assert router_info["top_weights"].shape == (2, 4, batch_size, 1, config.action_expert_num_experts_per_tok)
    assert router_info["router_input"].shape == (2, 4, batch_size, 1, 128)
    assert router_info["router_logits"].shape == (2, 4, batch_size, 1, 2)
    assert router_info["router_timestep"].shape == (2, batch_size)
    assert router_info["expert_norms"].shape == (2, 4, batch_size, config.action_horizon, 2)
    assert router_info["expert_cosine"].shape == (2, 4, batch_size, config.action_horizon, 2, 2)
    assert router_info["routed_moe_shared_ratio"].shape == (2, 4, batch_size, config.action_horizon)
    assert router_info["shared_routed_moe_cosine"].shape == (2, 4, batch_size, config.action_horizon)


def test_pi0_prototype_moe_router():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_expert_num_moe_experts=2,
        action_expert_num_experts_per_tok=1,
        action_expert_router_aux_loss_coef=0.0,
        action_expert_router_input="pooled",
        action_expert_router_type="prototype",
        action_expert_router_prototype_dim=16,
        action_expert_routing_contrastive_loss_coef=0.05,
        action_expert_dead_expert_loss_coef=1e-4,
        action_horizon=4,
        action_dim=8,
        max_token_len=8,
        pi05=True,
    )
    model = config.create(key)
    model_state = nnx.state(model)

    assert len(model_state.filter(nnx_utils.PathRegex(".*/llm/global_router_1/router/kernel")).flat_state()) == 0
    assert len(model_state.filter(nnx_utils.PathRegex(".*/llm/prototype_router_1/proj/.*")).flat_state()) == 2
    prototype_state = model_state.filter(nnx_utils.PathRegex(".*/llm/prototype_router_1/prototypes")).flat_state()
    prototypes = next(iter(prototype_state.values())).value
    assert prototypes.shape == (4, config.action_expert_num_moe_experts, config.action_expert_router_prototype_dim)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss, loss_info = nnx_utils.module_jit(model.compute_loss_with_info)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)
    assert {
        "routing_contrastive_loss",
        "routing_contrastive_loss_weighted",
        "dead_expert_loss",
        "dead_expert_loss_weighted",
    }.issubset(loss_info)

    actions, router_info = nnx_utils.module_jit(
        model.sample_actions_with_router_info, static_argnames=("num_steps",)
    )(key, obs, num_steps=2)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
    assert router_info["router_probs"].shape == (2, 4, batch_size, 1, 2)
    assert router_info["top_indices"].shape == (2, 4, batch_size, 1, 1)
    assert router_info["top_weights"].shape == (2, 4, batch_size, 1, 1)


def test_llm_suffix_router_uses_global_vlm_context():
    key = jax.random.key(0)
    depth = 3
    batch_size = 1
    prefix_len = 2
    suffix_len = 2
    width = 8
    base_config = _gemma.Config(
        width=width,
        depth=depth,
        mlp_dim=16,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
    )
    action_config = dataclasses.replace(
        base_config,
        num_local_experts=2,
        num_experts_per_tok=1,
        router_input="pooled",
    )
    llm = nnx_bridge.ToNNX(_gemma.Module(configs=[base_config, action_config], embed_dtype="float32"))
    llm.lazy_init(rngs=nnx.Rngs(key), method="init", use_adarms=[False, False])

    prefix_tokens = jnp.arange(batch_size * prefix_len * width, dtype=jnp.float32).reshape(
        batch_size, prefix_len, width
    )
    prefix_mask = jnp.ones((batch_size, prefix_len), dtype=jnp.bool_)
    prefix_attn_mask = jnp.ones((batch_size, prefix_len, prefix_len), dtype=jnp.bool_)
    prefix_positions = jnp.broadcast_to(jnp.arange(prefix_len), (batch_size, prefix_len))
    (_, _), kv_cache, _, _, router_contexts, _ = llm(
        [prefix_tokens, None],
        positions=prefix_positions,
        mask=prefix_attn_mask,
        router_context_mask=prefix_mask,
    )
    assert router_contexts.shape == (batch_size, width)

    suffix_tokens = jnp.full((batch_size, suffix_len, width), 0.25, dtype=jnp.float32)
    suffix_positions = jnp.broadcast_to(jnp.arange(prefix_len, prefix_len + suffix_len), (batch_size, suffix_len))
    suffix_attn_mask = jnp.ones((batch_size, suffix_len, prefix_len + suffix_len), dtype=jnp.bool_)

    def suffix_router_probs(contexts):
        (_, _), _, _, router_info, _, _ = llm(
            [None, suffix_tokens],
            positions=suffix_positions,
            mask=suffix_attn_mask,
            kv_cache=kv_cache,
            router_contexts=contexts,
            return_router_info=True,
        )
        return router_info["router_probs"]

    router_contexts = jnp.zeros_like(router_contexts)
    base_probs = suffix_router_probs(router_contexts)
    perturbed_probs = suffix_router_probs(router_contexts + 1_000_000.0)
    per_layer_delta = np.max(np.abs(np.asarray(perturbed_probs - base_probs)), axis=(1, 2, 3))

    assert np.all(per_layer_delta > 1e-4)
    np.testing.assert_allclose(per_layer_delta, np.broadcast_to(per_layer_delta[:1], per_layer_delta.shape), atol=1e-6)


def test_moe_experts_are_initialized_from_shared_ffn_when_merging_params():
    source_gate = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    source_linear = np.arange(6, dtype=np.float32).reshape(3, 2)
    ref_gate = jnp.zeros_like(source_gate, dtype=jnp.bfloat16)
    ref_linear = jnp.zeros_like(source_linear, dtype=jnp.bfloat16)
    ref_router = jnp.ones((2, 2), dtype=jnp.bfloat16)
    loaded_params = {
        "PaliGemma": {
            "llm": {
                "global_router_1": {"router": {"kernel": ref_router}},
                "layers": {
                    "mlp_1": {
                        "gating_einsum": source_gate,
                        "linear": source_linear,
                    },
                },
            },
        },
    }
    params = {
        "PaliGemma": {
            "llm": {
                "global_router_1": {"router": {"kernel": ref_router}},
                "layers": {
                    "mlp_1": {
                        "gating_einsum": ref_gate,
                        "linear": ref_linear,
                    },
                    "moe_1": {
                        "expert_0": {
                            "gating_einsum": ref_gate,
                            "linear": ref_linear,
                        },
                        "expert_1": {
                            "gating_einsum": ref_gate + 1,
                            "linear": ref_linear + 1,
                        },
                    },
                },
            },
        },
    }

    merged = weight_loaders._merge_params(loaded_params, params, missing_regex=r".*(lora|moe|global_router).*")  # noqa: SLF001
    for expert_name in ("expert_0", "expert_1"):
        expert = merged["PaliGemma"]["llm"]["layers"]["moe_1"][expert_name]
        gate_noise_atol = 4 * np.std(source_gate) * weight_loaders._MOE_EXPERT_COPY_NOISE_STD_SCALE  # noqa: SLF001
        linear_noise_atol = 4 * np.std(source_linear) * weight_loaders._MOE_EXPERT_COPY_NOISE_STD_SCALE  # noqa: SLF001
        np.testing.assert_allclose(
            np.asarray(expert["gating_einsum"], dtype=np.float32),
            source_gate,
            atol=gate_noise_atol,
        )
        np.testing.assert_allclose(np.asarray(expert["linear"], dtype=np.float32), source_linear, atol=linear_noise_atol)
    assert not np.array_equal(
        np.asarray(merged["PaliGemma"]["llm"]["layers"]["moe_1"]["expert_0"]["gating_einsum"], dtype=np.float32),
        np.asarray(merged["PaliGemma"]["llm"]["layers"]["moe_1"]["expert_1"]["gating_einsum"], dtype=np.float32),
    )
    np.testing.assert_array_equal(merged["PaliGemma"]["llm"]["global_router_1"]["router"]["kernel"], ref_router)


def test_checkpoint_merge_keeps_router_params():
    loaded_params = {"PaliGemma": {"llm": {"layers": {"mlp_1": {"linear": np.ones((2, 2), dtype=np.float32)}}}}}
    params = {
        "PaliGemma": {
            "llm": {
                "global_router_1": {
                    "context_proj": {
                        "kernel": jnp.full((2, 3), 2.0, dtype=jnp.bfloat16),
                        "bias": jnp.full((3,), 3.0, dtype=jnp.bfloat16),
                    },
                    "router": {
                        "kernel": jnp.full((3, 4), 4.0, dtype=jnp.bfloat16),
                    },
                },
                "prototype_router_1": {
                    "proj": {
                        "kernel": jnp.full((2, 3), 5.0, dtype=jnp.bfloat16),
                        "bias": jnp.full((3,), 6.0, dtype=jnp.bfloat16),
                    },
                    "prototypes": jnp.full((4, 2, 3), 7.0, dtype=jnp.bfloat16),
                },
                "layers": {
                    "mlp_1": {"linear": jnp.zeros((2, 2), dtype=jnp.bfloat16)},
                },
            }
        },
    }

    merged = weight_loaders._merge_params(  # noqa: SLF001
        loaded_params,
        params,
        missing_regex=weight_loaders._CHECKPOINT_MISSING_REGEX,  # noqa: SLF001
    )

    np.testing.assert_array_equal(
        merged["PaliGemma"]["llm"]["global_router_1"]["context_proj"]["kernel"],
        params["PaliGemma"]["llm"]["global_router_1"]["context_proj"]["kernel"],
    )
    np.testing.assert_array_equal(
        merged["PaliGemma"]["llm"]["global_router_1"]["router"]["kernel"],
        params["PaliGemma"]["llm"]["global_router_1"]["router"]["kernel"],
    )
    np.testing.assert_array_equal(
        merged["PaliGemma"]["llm"]["prototype_router_1"]["prototypes"],
        params["PaliGemma"]["llm"]["prototype_router_1"]["prototypes"],
    )
