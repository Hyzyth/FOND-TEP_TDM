"""
losses.py  —  Loss functions for 3-class DualwaveSAM
======================================================

Primary loss  : FocalTverskyLoss3Class
  - Per-class Tversky loss (alpha=0.3 FP penalty, beta=0.7 FN penalty)
  - Focal modulation (gamma=0.75) to focus on hard examples
  - Computed only over foreground classes (1=GTVp, 2=GTVn) by default,
    or over all classes including background.

Auxiliary loss: same FocalTverskyLoss3Class applied to the auxiliary head.

Combined loss:  0.8 * primary + 0.2 * auxiliary (when aux head is active)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ────────────────────────────────────────────────────────────────────

def one_hot_2d(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert (B, H, W) int64 label map → (B, num_classes, H, W) float one-hot.
    Values outside [0, num_classes) are silently mapped to 0.
    """
    labels = labels.long().clamp(0, num_classes - 1)
    B, H, W = labels.shape
    oh = torch.zeros(B, num_classes, H, W,
                     dtype=torch.float32, device=labels.device)
    oh.scatter_(1, labels.unsqueeze(1), 1.0)
    return oh


# ── Tversky index (soft, differentiable) ──────────────────────────────────────

def tversky_loss_per_class(
    probs:    torch.Tensor,   # (B, C, H, W)  softmax probabilities
    targets:  torch.Tensor,   # (B, H, W)     integer labels
    num_classes: int,
    alpha:    float = 0.3,    # FP weight
    beta:     float = 0.7,    # FN weight
    smooth:   float = 1e-5,
) -> torch.Tensor:
    """
    Return per-class Tversky loss tensor of shape (C,).
    background class (index 0) is included but can be masked out upstream.
    """
    oh      = one_hot_2d(targets, num_classes).to(probs.device)   # (B, C, H, W)
    dims    = (0, 2, 3)   # reduce over batch and spatial dims

    tp  = (probs * oh).sum(dim=dims)
    fp  = (probs * (1 - oh)).sum(dim=dims)
    fn  = ((1 - probs) * oh).sum(dim=dims)

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky   # (C,)


# ── Focal Tversky loss ─────────────────────────────────────────────────────────

class FocalTverskyLoss3Class(nn.Module):
    """
    Focal Tversky loss for 3-class 2D segmentation.

    Parameters
    ----------
    alpha            : FP penalty weight (default 0.3)
    beta             : FN penalty weight (default 0.7)
    gamma            : focal exponent (default 0.75)
    smooth           : numerical stability (default 1e-5)
    include_bg       : include background class in loss (default False)
    class_weights    : optional per-class weight tensor of shape (num_classes,)
    """

    def __init__(
        self,
        alpha:         float = 0.3,
        beta:          float = 0.7,
        gamma:         float = 0.75,
        smooth:        float = 1e-5,
        include_bg:    bool  = False,
        num_classes:   int   = 3,
        class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.alpha       = alpha
        self.beta        = beta
        self.gamma       = gamma
        self.smooth      = smooth
        self.include_bg  = include_bg
        self.num_classes = num_classes

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        logits:  torch.Tensor,   # (B, C, H, W) — raw logits
        targets: torch.Tensor,   # (B, H, W)    — integer labels 0/1/2
    ) -> torch.Tensor:

        probs = F.softmax(logits, dim=1)   # (B, C, H, W)

        per_class_loss = tversky_loss_per_class(
            probs, targets,
            num_classes=self.num_classes,
            alpha=self.alpha,
            beta=self.beta,
            smooth=self.smooth,
        )   # (C,)

        # Focal modulation
        per_class_loss = torch.clamp(per_class_loss, min=1e-7) ** self.gamma

        # Optionally exclude background
        if not self.include_bg:
            per_class_loss = per_class_loss[1:]   # drop class 0

        # Optional class weighting
        if self.class_weights is not None:
            w = self.class_weights
            if not self.include_bg:
                w = w[1:]
            per_class_loss = per_class_loss * w.to(per_class_loss.device)

        return per_class_loss.mean()


# ── Combined primary + auxiliary loss ─────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Primary Focal Tversky + optional auxiliary Focal Tversky.
    weight_primary + weight_aux should sum to 1.
    """

    def __init__(
        self,
        alpha:          float = 0.3,
        beta:           float = 0.7,
        gamma:          float = 0.75,
        smooth:         float = 1e-5,
        include_bg:     bool  = False,
        num_classes:    int   = 3,
        weight_primary: float = 0.8,
        weight_aux:     float = 0.2,
    ):
        super().__init__()
        self.weight_primary = weight_primary
        self.weight_aux     = weight_aux

        loss_kwargs = dict(
            alpha=alpha, beta=beta, gamma=gamma,
            smooth=smooth, include_bg=include_bg,
            num_classes=num_classes,
        )
        self.primary_loss = FocalTverskyLoss3Class(**loss_kwargs)
        self.aux_loss     = FocalTverskyLoss3Class(**loss_kwargs)

    def forward(
        self,
        logits:     torch.Tensor,           # (B, C, H, W)
        targets:    torch.Tensor,           # (B, H, W)
        aux_logits: torch.Tensor = None,    # (B, C, H, W) | None
    ) -> torch.Tensor:

        loss = self.weight_primary * self.primary_loss(logits, targets)

        if aux_logits is not None:
            loss = loss + self.weight_aux * self.aux_loss(aux_logits, targets)

        return loss
