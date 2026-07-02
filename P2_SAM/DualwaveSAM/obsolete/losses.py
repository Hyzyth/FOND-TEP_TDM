import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# CHANGE: removed unused `import numpy as np` duplicate — already imported above.
# cos_loss is a pure-NumPy evaluation utility and cannot propagate gradients;
# added a clear docstring warning so it is not accidentally used in training.


# =========================
# 2D Cross-Entropy Loss
# =========================
def cross_entropy_2D(input, target, weight=None, size_average=True):
    """
    2D pixel-wise cross entropy via log-softmax + NLL.

    Args:
        input:        logits [N, C, H, W]
        target:       labels [N, H, W]
        weight:       optional class weights
        size_average: normalise by element count
    """
    n, c, h, w = input.size()
    log_p      = F.log_softmax(input, dim=1)
    log_p      = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target     = target.view(target.numel())
    loss       = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss


# =========================
# Dice Loss
# =========================
class DiceLoss(nn.Module):
    """Soft Dice loss over all spatial dimensions."""

    def __init__(self):
        super().__init__()
        self.smooth = 0.001

    def forward(self, input, target):
        axes     = tuple(range(1, input.dim()))
        intersect = (input * target).sum(dim=axes)
        union     = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        return (1 - (2 * intersect + self.smooth) / (union + self.smooth)).mean()


# =========================
# Binary Tversky Loss
# =========================
class BinaryTverskyLossV2(nn.Module):
    """
    Tversky loss for binary segmentation.
    alpha controls FP penalty, beta controls FN penalty.
    """

    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean'):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth       = 10
        self.reduction    = reduction
        s = alpha + beta
        self.alpha = alpha / s
        self.beta  = beta  / s

    def forward(self, output, target, mask=None):
        batch_size = output.size(0)
        bg_target  = 1 - target

        if self.ignore_index is not None:
            valid     = (target != self.ignore_index).float()
            output    = output.float().mul(valid)
            target    = target.float().mul(valid)
            bg_target = bg_target.float().mul(valid)

        output    = torch.sigmoid(output).view(batch_size, -1)
        target    = target.view(batch_size, -1)
        bg_target = bg_target.view(batch_size, -1)

        P_G  = torch.sum(output * target,    1)
        P_NG = torch.sum(output * bg_target, 1)
        NP_G = torch.sum((1 - output) * target, 1)

        tversky = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth)
        loss    = 1.0 - tversky

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        return torch.mean(loss)


# =========================
# Focal Loss
# =========================
class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced segmentation."""

    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.eps   = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)
        loss  = -(
            target       * torch.pow(1 - input, self.gamma) * torch.log(input) +
            (1 - target) * torch.pow(input,     self.gamma) * torch.log(1 - input)
        )
        return loss.mean()


# =========================
# Dice + Focal combined
# =========================
class Dice_and_FocalLoss(nn.Module):
    """Hybrid Dice + Focal loss with optional delayed weighting."""

    def __init__(self, gamma=2, alpha=0.99, num=0):
        super().__init__()
        self.dice_loss  = DiceLoss()
        self.focal_loss = FocalLoss(gamma)
        self.delay      = alpha ** num

    def forward(self, input, target):
        if self.delay == 1:
            return self.dice_loss(input, target) + self.focal_loss(input, target)
        return (
            self.delay       * self.dice_loss(input, target) +
            (1 - self.delay) * self.focal_loss(input, target)
        )


# =========================
# Cosine similarity (evaluation only)
# =========================
def cos_loss(prediction, target):
    """
    Cosine similarity-based metric for 3D volumes, computed slice-by-slice.

    CHANGE: added explicit warning — this function detaches tensors and
    converts to NumPy, so it CANNOT propagate gradients. Use only for
    evaluation / logging, never as a training loss.

    Args:
        prediction: Tensor [B, 1, H, W, D]
        target:     Tensor [B, 1, H, W, D]

    Returns:
        scalar float
    """
    loss = []
    for b in range(target.shape[0]):
        cos_total = 0.0
        for s in range(target.shape[-1]):
            pred = prediction[b, 0, :, :, s].detach().cpu().numpy()
            gt   = target[b,    0, :, :, s].detach().cpu().numpy()

            re  = np.dot(pred.T, gt)
            x   = np.linalg.norm(pred, axis=0)
            y   = np.linalg.norm(gt,   axis=0).reshape(-1, 1)
            x[x == 0] = 1
            y[y == 0] = 1
            diag       = np.diagonal(re / (x * y)) * 0.5 + 0.5
            cos_total += diag.mean()

        loss.append(1 - cos_total / target.shape[-1])
    return np.mean(loss)


# =========================
# Weighted Cross Entropy
# =========================
class WeightedCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross entropy with optional class weighting. Input must be raw logits."""

    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight

    def forward(self, inp, target):
        target      = target.long()
        num_classes = inp.size(1)

        # Permute channel dim to last position
        i0, i1 = 1, 2
        while i1 < len(inp.shape):
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp    = inp.contiguous().view(-1, num_classes)
        target = target.view(-1)
        return nn.CrossEntropyLoss(weight=self.weight)(inp, target)
