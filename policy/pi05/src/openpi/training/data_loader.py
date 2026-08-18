from collections.abc import Iterator, Sequence
import logging
import multiprocessing
import os
from pathlib import Path
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import datasets.features.features as hf_features
import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)

# Some locally converted LeRobot parquet files were written by newer
# HuggingFace Datasets versions that serialize fixed-size lists as "List".
# Datasets 3.6.0 only registers "Sequence", but the constructor is compatible.
hf_features._FEATURE_TYPES.setdefault("List", hf_features.Sequence)

try:
    from datasets.arrow_dataset import Column as _HfDatasetColumn
except ImportError:  # pragma: no cover - depends on datasets internals.
    _HfDatasetColumn = ()

_ORIGINAL_TORCH_STACK = torch.stack


def _torch_stack_hf_column_compatible(tensors, *args, **kwargs):
    if _HfDatasetColumn and isinstance(tensors, _HfDatasetColumn):
        source = tensors.source
        if hasattr(source, "with_format"):
            source = source.with_format(None)
        values = list(source[tensors.column_name])
        if not values or not isinstance(values[0], torch.Tensor):
            return torch.as_tensor(values, **kwargs)
        tensors = values
    return _ORIGINAL_TORCH_STACK(tensors, *args, **kwargs)


if torch.stack is not _torch_stack_hf_column_compatible:
    torch.stack = _torch_stack_hf_column_compatible


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class RouterLabelDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        labels: np.ndarray,
        class_weights: np.ndarray,
    ):
        self._dataset = dataset
        self._labels = labels.astype(np.int32, copy=False)
        self._class_weights = class_weights.astype(np.float32, copy=False)

    def __getitem__(self, index: SupportsIndex) -> dict:
        item = dict(self._dataset[index])
        global_index = int(np.asarray(item["index"]))
        label = int(self._labels[global_index])
        item["router_label"] = np.asarray(label, dtype=np.int32)
        item["router_class_weight"] = np.asarray(
            self._class_weights[label] if label >= 0 else 0.0,
            dtype=np.float32,
        )
        return item

    def __len__(self) -> int:
        return len(self._dataset)


def _load_router_labels(data_config: _config.DataConfig) -> tuple[np.ndarray, np.ndarray] | None:
    if data_config.router_labels_path is None:
        return None
    labels = np.load(data_config.router_labels_path)
    valid = labels >= 0
    if not np.any(valid):
        raise ValueError(f"No valid router labels found in {data_config.router_labels_path}")
    num_classes = int(labels[valid].max()) + 1
    counts = np.bincount(labels[valid], minlength=num_classes).astype(np.float64)
    if data_config.router_class_weights is not None:
        class_weights = np.asarray(data_config.router_class_weights, dtype=np.float32)
        if class_weights.shape != (num_classes,):
            raise ValueError(
                f"router_class_weights has shape {class_weights.shape}, expected {(num_classes,)}"
            )
    else:
        total = float(np.sum(counts))
        class_weights = np.sqrt(total / np.clip(num_classes * counts, a_min=1.0, a_max=None)).astype(np.float32)
    logging.info(
        "Loaded router labels from %s: counts=%s, class_weights=%s",
        data_config.router_labels_path,
        counts.astype(np.int64).tolist(),
        class_weights.tolist(),
    )
    return labels.astype(np.int32, copy=False), class_weights


def _make_router_balanced_sampler(
    dataset: Dataset,
    labels: np.ndarray,
    *,
    alpha: float,
    seed: int,
) -> torch.utils.data.WeightedRandomSampler:
    global_indices = _get_dataset_global_indices(dataset)
    sample_labels = labels[global_indices]
    valid = sample_labels >= 0
    if not np.any(valid):
        raise ValueError("Balanced router sampling requires at least one valid router label.")
    if not np.all(valid):
        logging.warning(
            "Router-balanced sampler will ignore %s samples without valid router labels.",
            int(np.sum(~valid)),
        )
    num_classes = int(sample_labels[valid].max()) + 1
    counts = np.bincount(sample_labels[valid], minlength=num_classes).astype(np.float64)
    inv_freq = 1.0 / np.clip(counts, a_min=1.0, a_max=None)
    weights = np.zeros(sample_labels.shape, dtype=np.float64)
    weights[valid] = inv_freq[sample_labels[valid]] ** alpha
    generator = torch.Generator()
    generator.manual_seed(seed)
    logging.info("Using router-balanced sampler: counts=%s, alpha=%s", counts.astype(np.int64).tolist(), alpha)
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(sample_labels),
        replacement=True,
        generator=generator,
    )


def _get_dataset_global_indices(dataset: Dataset) -> np.ndarray:
    current = dataset
    while hasattr(current, "_dataset"):
        current = current._dataset  # noqa: SLF001
    if hasattr(current, "hf_dataset"):
        hf_dataset = current.hf_dataset.with_format(None)
        return np.asarray(hf_dataset["index"], dtype=np.int64)
    raise ValueError(
        "Router-balanced sampling requires access to the underlying LeRobot hf_dataset index column."
    )


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    filtered_episodes = list(data_config.episodes) if data_config.episodes is not None else None
    lerobot_root = None
    hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if hf_lerobot_home is not None:
        candidate_root = Path(hf_lerobot_home) / data_config.repo_id
        if candidate_root.exists():
            lerobot_root = candidate_root

    # Initialize metadata only after resolving the local root to avoid an unnecessary Hub request.
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=lerobot_root)

    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=lerobot_root,
        # Restrict loading to an explicit episode subset when requested, for example for single-task LoRA runs.
        # A None value loads the complete dataset.
        episodes=filtered_episodes,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        video_backend="pyav",  # torchcodec requires FFmpeg libs not available in this env
    )

    # LeRobot stores episode_data_index by positional index 0..N-1, while __getitem__ uses the
    # original episode_index (for example, 502). Remap the table so original episode identifiers
    # are valid indices when a non-contiguous subset is loaded.
    if filtered_episodes is not None:
        pos_from = dataset.episode_data_index["from"]   # Shape: [N], indexed by position 0..N-1.
        pos_to   = dataset.episode_data_index["to"]     # shape: [N]
        max_ep   = max(filtered_episodes)
        new_from = torch.zeros(max_ep + 1, dtype=torch.long)
        new_to   = torch.zeros(max_ep + 1, dtype=torch.long)
        for pos, ep_idx in enumerate(filtered_episodes):
            new_from[ep_idx] = pos_from[pos]
            new_to[ep_idx]   = pos_to[pos]
        dataset.episode_data_index = {"from": new_from, "to": new_to}

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        filter_dict_path=data_config.filter_dict_path,
    )


def _get_norm_stats(data_config: _config.DataConfig, skip_norm_stats: bool) -> dict:
    if data_config.repo_id == "fake" or skip_norm_stats:
        return {}
    if data_config.norm_stats is None:
        raise ValueError(
            "Normalization stats not found. "
            "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
        )
    return data_config.norm_stats


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = _get_norm_stats(data_config, skip_norm_stats)
    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = _get_norm_stats(data_config, skip_norm_stats)
    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    router_label_data = _load_router_labels(data_config)
    if router_label_data is not None:
        labels, class_weights = router_label_data
        dataset = RouterLabelDataset(dataset, labels, class_weights)
    sampler_source_dataset = dataset
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if data_config.router_balanced_sampling:
        if router_label_data is None:
            raise ValueError("router_balanced_sampling=True requires router_labels_path")
        if framework == "pytorch" and torch.distributed.is_initialized():
            raise NotImplementedError("Router-balanced sampling is not implemented for PyTorch DDP.")
        sampler = _make_router_balanced_sampler(
            sampler_source_dataset,
            router_label_data[0],
            alpha=data_config.router_sampling_alpha,
            seed=seed,
        )
    if framework == "pytorch":
        if sampler is None and torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
