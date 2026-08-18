import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_router_supervised_loss_uses_labels_and_class_weights():
    router_metrics = {
        "router_route_logits": jnp.asarray(
            [
                [
                    [[2.0, 0.0, -1.0]],
                    [[0.0, 3.0, -1.0]],
                ],
                [
                    [[1.0, 0.0, -1.0]],
                    [[0.0, 2.0, -1.0]],
                ],
            ],
            dtype=jnp.float32,
        ),
    }
    labels = jnp.asarray([0, 2], dtype=jnp.int32)
    class_weights = jnp.asarray([1.0, 2.0], dtype=jnp.float32)

    loss, accuracy = _pi0.Pi0._router_supervised_loss(router_metrics, labels, class_weights)

    logits = np.asarray(jnp.mean(router_metrics["router_route_logits"][:, :, 0, :], axis=0))
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=-1, keepdims=True))
    expected_loss = -(log_probs[0, 0] * 1.0 + log_probs[1, 2] * 2.0) / 3.0
    assert np.allclose(loss, expected_loss, atol=1e-6)
    assert np.allclose(accuracy, 0.5)
