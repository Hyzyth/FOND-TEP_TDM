"""
lr_scheduler.py  —  Learning rate schedulers for DualwaveSAM
=============================================================
Inlined from SwinCross/optimizers/lr_scheduler.py so that the
DualwaveSAM project is fully self-contained without importing
from the models/swincross source tree.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """
    Linear warm-up followed by cosine annealing to zero.

    Parameters
    ----------
    optimizer      : torch.optim.Optimizer
    warmup_epochs  : int   — epochs to linearly ramp LR from 0 to base_lr
    max_epochs     : int   — total training epochs
    warmup_start_lr: float — LR at epoch 0  (default 0)
    eta_min        : float — minimum LR at the end of cosine decay (default 0)
    last_epoch     : int   — last completed epoch (for resume)
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs:    int,
        warmup_start_lr: float = 0.0,
        eta_min:         float = 0.0,
        last_epoch:      int   = -1,
    ):
        self.warmup_epochs   = warmup_epochs
        self.max_epochs      = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min         = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        ep = self.last_epoch

        if ep < self.warmup_epochs:
            # Linear warm-up
            alpha = ep / max(self.warmup_epochs, 1)
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        # Cosine annealing
        progress = (ep - self.warmup_epochs) / max(
            self.max_epochs - self.warmup_epochs, 1
        )
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + cosine * (base_lr - self.eta_min)
            for base_lr in self.base_lrs
        ]
