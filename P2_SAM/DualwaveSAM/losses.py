import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


# =========================
# 2D Cross-Entropy Loss
# =========================
def cross_entropy_2D(input, target, weight=None, size_average=True):
    """
    2D pixel-wise cross entropy loss using log-softmax + NLL loss.

    Args:
        input: model logits [N, C, H, W]
        target: ground truth labels [N, H, W]
        weight: class weighting (optional)
        size_average: whether to normalize by number of elements

    Returns:
        scalar loss
    """
    n, c, h, w = input.size()

    log_p = F.log_softmax(input, dim=1)

    # reshape for NLL loss
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())

    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)

    if size_average:
        loss /= float(target.numel())

    return loss


# =========================
# Dice Loss (soft formulation)
# =========================
class DiceLoss(nn.Module):
    """
    Soft Dice loss computed over spatial dimensions.
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 0.001

    def forward(self, input, target):
        axes = tuple(range(1, input.dim()))

        # Intersection between prediction and target
        intersect = (input * target).sum(dim=axes)

        # L2-based union formulation
        union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)

        loss = 1 - (2 * intersect + self.smooth) / (union + self.smooth)
        return loss.mean()


# =========================
# Binary Tversky Loss
# =========================
class BinaryTverskyLossV2(nn.Module):
    """
    Tversky loss for binary segmentation.

    Controls trade-off between false positives and false negatives.
    """

    def __init__(self, alpha=0.3, beta=0.7, ignore_index=None, reduction='mean'):
        super(BinaryTverskyLossV2, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.ignore_index = ignore_index
        self.smooth = 10
        self.reduction = reduction

        # Normalize alpha/beta to sum to 1
        s = self.beta + self.alpha
        if s != 1:
            self.beta = self.beta / s
            self.alpha = self.alpha / s

    def forward(self, output, target, mask=None):
        batch_size = output.size(0)

        bg_target = 1 - target

        # Optional ignore mask
        if self.ignore_index is not None:
            valid_mask = (target != self.ignore_index).float()
            output = output.float().mul(valid_mask)
            target = target.float().mul(valid_mask)
            bg_target = bg_target.float().mul(valid_mask)

        output = torch.sigmoid(output).view(batch_size, -1)
        target = target.view(batch_size, -1)
        bg_target = bg_target.view(batch_size, -1)

        # True positive, false positive, false negative
        P_G = torch.sum(output * target, 1)
        P_NG = torch.sum(output * bg_target, 1)
        NP_G = torch.sum((1 - output) * target, 1)

        tversky_index = P_G / (P_G + self.alpha * P_NG + self.beta * NP_G + self.smooth)

        loss = 1. - tversky_index

        if self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return torch.mean(loss)


# =========================
# Focal Loss
# =========================
class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance in segmentation.
    """

    def __init__(self, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-3

    def forward(self, input, target):
        input = input.clamp(self.eps, 1 - self.eps)

        loss = - (
            target * torch.pow((1 - input), self.gamma) * torch.log(input) +
            (1 - target) * torch.pow(input, self.gamma) * torch.log(1 - input)
        )

        return loss.mean()


# =========================
# Dice + Focal combined loss
# =========================
class Dice_and_FocalLoss(nn.Module):
    """
    Hybrid loss combining Dice loss and Focal loss.

    Optionally applies delayed weighting between the two.
    """

    def __init__(self, gamma=2, alpha=0.99, num=0):
        super(Dice_and_FocalLoss, self).__init__()

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma)
        self.BinaryTverskyLoss = BinaryTverskyLossV2()
        self.ce = nn.CrossEntropyLoss()

        # Delay factor controlling interpolation between losses
        self.delay = alpha ** num

    def forward(self, input, target):
        if self.delay == 1:
            loss = self.dice_loss(input, target) + self.focal_loss(input, target)
        else:
            loss = (
                self.delay * self.dice_loss(input, target) +
                (1 - self.delay) * self.focal_loss(input, target)
            )
        return loss


# =========================
# Cosine similarity loss (3D slices)
# =========================
def cos_loss(prediction, target):
    """
    Cosine similarity-based loss for 3D volumes.

    Computes similarity slice-by-slice along depth dimension.
    """

    loss = []

    for b in range(target.shape[0]):
        cos_total = 0

        for s in range(target.shape[-1]):
            pred = prediction[b, 0, :, :, s]
            gt = target[b, 0, :, :, s]

            pred = pred.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()

            re = np.dot(pred.T, gt)

            x = np.linalg.norm(pred, axis=0)
            y = np.linalg.norm(gt, axis=0).reshape(-1, 1)

            x[x == 0] = 1
            y[y == 0] = 1

            mul = x * y

            res = re / mul

            diag = np.diagonal(res) * 0.5 + 0.5

            cos_total += diag.mean()

        cos_s = cos_total / target.shape[-1]
        loss.append(1 - cos_s)

    return np.mean(loss)


# =========================
# Weighted Cross Entropy
# =========================
class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    Cross entropy loss with optional class weighting.

    Note:
        Input must NOT have activation applied.
    """

    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inp, target):
        target = target.long()
        num_classes = inp.size()[1]

        # Permute to move channel dimension last
        i0 = 1
        i1 = 2
        while i1 < len(inp.shape):
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)

        wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        return wce_loss(inp, target)
