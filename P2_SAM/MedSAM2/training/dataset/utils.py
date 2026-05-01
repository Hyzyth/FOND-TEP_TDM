"""
training/dataset/utils.py
===========================
Dataset wrappers for repeat-factor sampling (adapted from Detectron2).
"""

from typing import Iterable

import torch
from torch.utils.data import (
    ConcatDataset as TorchConcatDataset,
    Dataset,
    Subset as TorchSubset,
)


class ConcatDataset(TorchConcatDataset):
    """Concatenation of datasets that supports repeat-factor sampling.

    Propagates ``repeat_factors`` and ``set_epoch`` across sub-datasets.
    """

    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)
        self.repeat_factors = torch.cat([d.repeat_factors for d in datasets])

    def set_epoch(self, epoch: int) -> None:
        for dataset in self.datasets:
            if hasattr(dataset, "epoch"):
                dataset.epoch = epoch
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)


class Subset(TorchSubset):
    """Subset of a dataset with repeat-factor support."""

    def __init__(self, dataset, indices) -> None:
        super().__init__(dataset, indices)
        self.repeat_factors = dataset.repeat_factors[indices]


class RepeatFactorWrapper(Dataset):
    """Wrap a dataset to apply repeat-factor oversampling.

    On each call to :meth:`set_epoch` a new list of indices is drawn
    stochastically so that the long-run frequency of each sample matches
    its ``repeat_factor`` value.

    Parameters
    ----------
    dataset : Dataset
        Must expose a ``repeat_factors`` FloatTensor attribute.
    seed : int
        Base random seed.
    """

    def __init__(self, dataset, seed: int = 0) -> None:
        self.dataset = dataset
        self.epoch_ids = None
        self._seed = seed
        self._int_part = torch.trunc(dataset.repeat_factors)
        self._frac_part = dataset.repeat_factors - self._int_part

    def _get_epoch_indices(self, generator) -> torch.Tensor:
        """Stochastic rounding to generate per-epoch index list."""
        rands = torch.rand(len(self._frac_part), generator=generator)
        rep_factors = self._int_part + (rands < self._frac_part).float()
        indices = []
        for dataset_index, rep_factor in enumerate(rep_factors):
            indices.extend([dataset_index] * int(rep_factor.item()))
        return torch.tensor(indices, dtype=torch.int64)

    def __len__(self) -> int:
        if self.epoch_ids is None:
            raise RuntimeError("Call set_epoch() before using RepeatFactorWrapper.")
        return len(self.epoch_ids)

    def set_epoch(self, epoch: int) -> None:
        g = torch.Generator()
        g.manual_seed(self._seed + epoch)
        self.epoch_ids = self._get_epoch_indices(g)
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def __getitem__(self, idx):
        if self.epoch_ids is None:
            raise RuntimeError("Call set_epoch() before iterating RepeatFactorWrapper.")
        return self.dataset[self.epoch_ids[idx]]
