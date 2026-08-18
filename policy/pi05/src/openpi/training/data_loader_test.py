import dataclasses

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _IndexDataset:
    def __init__(self, indices):
        self.hf_dataset = {"index": np.asarray(indices, dtype=np.int64)}

    def __len__(self):
        return len(self.hf_dataset["index"])


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_router_balanced_sampler_uses_normalized_inverse_frequency_weights():
    dataset = _IndexDataset(range(4))
    labels = np.array([0, 0, 0, 1], dtype=np.int32)

    sampler = _data_loader._make_router_balanced_sampler(
        dataset,
        labels,
        alpha=1.0,
        mix_beta=1.0,
        seed=0,
    )

    np.testing.assert_allclose(
        sampler.weights.numpy(),
        np.array([2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 2.0], dtype=np.float64),
    )


def test_router_balanced_sampler_mix_beta_interpolates_with_uniform_sampling():
    dataset = _IndexDataset(range(4))
    labels = np.array([0, 0, 0, 1], dtype=np.int32)

    sampler = _data_loader._make_router_balanced_sampler(
        dataset,
        labels,
        alpha=1.0,
        mix_beta=0.5,
        seed=0,
    )

    np.testing.assert_allclose(
        sampler.weights.numpy(),
        np.array([5.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.5], dtype=np.float64),
    )
