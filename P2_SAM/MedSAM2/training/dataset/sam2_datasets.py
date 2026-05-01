"""
training/dataset/sam2_datasets.py
====================================
Mixed-dataset loader for the SAM2 training pipeline.
Combines multiple :class:`~torch.utils.data.DataLoader` instances and
samples from them according to dataset-proportional probabilities.
"""

import logging
import math
from typing import Callable, Iterable, List, Optional

import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, IterableDataset, Subset
from torch.utils.data.distributed import DistributedSampler


class MixedDataLoader:
    """Interleave multiple DataLoaders with given mixing probabilities.

    Parameters
    ----------
    dataloaders : list[DataLoader]
    mixing_prob : FloatTensor  probability per dataloader (must sum to 1)
    """

    def __init__(self, dataloaders: List[DataLoader], mixing_prob: torch.FloatTensor) -> None:
        assert len(dataloaders) == mixing_prob.shape[0]
        self.dataloaders = dataloaders
        self.mixing_prob = mixing_prob
        self._iter_dls = None
        self._iter_mixing_prob = None
        self.random_generator = torch.Generator()

    def __len__(self):
        return sum(len(d) for d in self.dataloaders)

    def __iter__(self):
        self.random_generator.manual_seed(42)
        self._iter_dls = [iter(loader) for loader in self.dataloaders]
        self._iter_mixing_prob = self.mixing_prob.clone()
        return self

    def __next__(self):
        if self._iter_dls is None:
            raise TypeError(f"{type(self).__name__} is not an iterator")
        while self._iter_mixing_prob.any():
            idx = self._iter_mixing_prob.multinomial(1, generator=self.random_generator).item()
            try:
                return next(self._iter_dls[idx])
            except StopIteration:
                self._iter_mixing_prob[idx] = 0
            except Exception as e:
                logging.error(e)
                raise
        raise StopIteration


class TorchTrainMixedDataset:
    """Training dataset that mixes multiple datasets with configurable batch sizes.

    Parameters
    ----------
    datasets : list[Dataset]
    batch_sizes : list[int]   one entry per dataset
    num_workers : int
    shuffle : bool
    pin_memory : bool
    drop_last : bool
    collate_fn : callable or None
    worker_init_fn : callable or None
    phases_per_epoch : int   split one epoch into this many phases
    dataset_prob : list[float] or None  sampling probabilities; inferred if None
    """

    def __init__(
        self,
        datasets: List[Dataset],
        batch_sizes: List[int],
        num_workers: int,
        shuffle: bool,
        pin_memory: bool,
        drop_last: bool,
        collate_fn: Optional[Callable] = None,
        worker_init_fn: Optional[Callable] = None,
        phases_per_epoch: int = 1,
        dataset_prob: Optional[List[float]] = None,
    ) -> None:
        self.datasets = datasets
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        assert len(datasets) > 0
        for dataset in datasets:
            assert not isinstance(dataset, IterableDataset)
            self._set_dataset_epoch(dataset, 0)
        self.phases_per_epoch = phases_per_epoch
        self.chunks = [None] * len(datasets)

        if dataset_prob is None:
            dataset_lens = [
                (math.floor(len(d) / bs) if drop_last else math.ceil(len(d) / bs))
                for d, bs in zip(datasets, batch_sizes)
            ]
            total_len = sum(dataset_lens)
            dataset_prob = torch.tensor([dl / total_len for dl in dataset_lens])
        else:
            dataset_prob = torch.tensor(dataset_prob)

        logging.info("Dataset mixing probabilities: %s", dataset_prob.tolist())
        assert abs(dataset_prob.sum().item() - 1.0) < 1e-6
        self.dataset_prob = dataset_prob

    def _set_dataset_epoch(self, dataset, epoch: int) -> None:
        if hasattr(dataset, "epoch"):
            dataset.epoch = epoch
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def get_loader(self, epoch: int) -> Iterable:
        """Build and return a :class:`MixedDataLoader` for *epoch*."""
        dataloaders = []
        for d_idx, (dataset, batch_size) in enumerate(zip(self.datasets, self.batch_sizes)):
            if self.phases_per_epoch > 1:
                main_epoch = epoch // self.phases_per_epoch
                local_phase = epoch % self.phases_per_epoch
                if local_phase == 0 or self.chunks[d_idx] is None:
                    self._set_dataset_epoch(dataset, main_epoch)
                    g = torch.Generator()
                    g.manual_seed(main_epoch)
                    self.chunks[d_idx] = torch.chunk(
                        torch.randperm(len(dataset), generator=g), self.phases_per_epoch
                    )
                dataset = Subset(dataset, self.chunks[d_idx][local_phase])
            else:
                self._set_dataset_epoch(dataset, epoch)

            sampler = DistributedSampler(dataset, shuffle=self.shuffle)
            sampler.set_epoch(epoch)
            batch_sampler = BatchSampler(sampler, batch_size, drop_last=self.drop_last)
            dataloaders.append(DataLoader(
                dataset, num_workers=self.num_workers, pin_memory=self.pin_memory,
                batch_sampler=batch_sampler, collate_fn=self.collate_fn,
                worker_init_fn=self.worker_init_fn,
            ))
        return MixedDataLoader(dataloaders, self.dataset_prob)
