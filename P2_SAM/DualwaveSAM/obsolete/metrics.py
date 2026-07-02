import numpy as np
import torch.nn as nn
from scipy.spatial import cKDTree
from scipy.ndimage import label, sum as ndi_sum


# =========================
# Dice loss (PyTorch module)
# =========================
class Dice(nn.Module):
    """
    Soft Dice coefficient implemented as a PyTorch loss/module.

    Computes overlap between prediction and target over all spatial dimensions.
    """

    def __init__(self):
        super(Dice, self).__init__()
        self.smooth = 0.001

    def forward(self, input, target):
        axes = tuple(range(0, input.dim()))

        # Intersection between prediction and ground truth
        intersect = (input * target).sum(dim=axes)

        # Union (soft formulation)
        union = input.sum(dim=axes) + target.sum(dim=axes)

        dic = 2 * intersect / (union + self.smooth)
        return dic.mean()


# =========================
# Binary Dice (NumPy)
# =========================
def dice(mask_gt, mask_seg):
    """
    Compute Dice similarity coefficient for binary masks.

    Args:
        mask_gt: ground truth binary mask
        mask_seg: predicted binary mask

    Returns:
        Dice score (float)
    """
    return 2 * np.sum(np.logical_and(mask_gt, mask_seg)) / (
        np.sum(mask_gt) + np.sum(mask_seg) + 1e-8
    )


# =========================
# IoU (Jaccard index)
# =========================
def iou(pred, target):
    """
    Intersection over Union (IoU) metric.

    Args:
        pred: predicted binary mask
        target: ground truth mask

    Returns:
        IoU score
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    eps = 1e-8
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    return (intersection + eps) / (union + eps)


# =========================
# VOE (Volume Overlap Error)
# =========================
def voe(pred, target):
    """
    Volume Overlap Error = 1 - IoU
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    eps = 1e-8
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()

    return 1.0 - (intersection + eps) / (union + eps)


# =========================
# Relative Volume Difference
# =========================
def rvd(pred, target):
    """
    Relative Volume Difference between prediction and ground truth.
    """
    pred = pred.astype(bool)
    target = target.astype(bool)

    eps = 1e-8
    pred_vol = pred.sum()
    target_vol = target.sum()

    return abs(pred_vol - target_vol) / (target_vol + eps)


# =========================
# Dice variants (thresholded)
# =========================
def dice_1(input, target):
    """
    Dice score with threshold = 0.5
    """
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)

    score = 2 * intersect / (union + 1e-3)
    return score.mean()


def dice_2(input, target):
    """
    Dice score with threshold = 0.6
    """
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.6).float()

    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)

    score = 2 * intersect / (union + 1e-3)
    return score.mean()


# =========================
# Recall metric
# =========================
def recall(input, target):
    """
    Recall (sensitivity) over batch dimension.
    """
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positives = target.sum(dim=axes)

    recall = true_positives / all_positives
    return recall.mean()


# =========================
# Precision metric
# =========================
def precision(input, target):
    """
    Precision over batch dimension.
    """
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positive_calls = binary_input.sum(dim=axes)

    precision = true_positives / all_positive_calls
    return precision.mean()


# =========================
# Hausdorff Distance
# =========================
def hausdorff_distance(image0, image1):
    """
    Standard Hausdorff distance between two binary masks.

    Handles empty sets explicitly.
    """
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(
        max(cKDTree(a_points).query(b_points, k=1)[0]),
        max(cKDTree(b_points).query(a_points, k=1)[0])
    )


# =========================
# Hausdorff Distance 95
# =========================
def hausdorff_distance_95(image0, image1):
    """
    95th percentile Hausdorff distance (HD95).

    More robust than standard HD to outliers.
    """
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    if len(a_points) == 0:
        return 0.0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    dist_a_to_b = cKDTree(a_points).query(b_points, k=1)[0]
    dist_b_to_a = cKDTree(b_points).query(a_points, k=1)[0]

    hd95_a = np.percentile(dist_a_to_b, 95)
    hd95_b = np.percentile(dist_b_to_a, 95)

    return max(hd95_a, hd95_b)


# =========================
# Lesion-wise metrics
# =========================
def compute_lesion_metrics(pred_mask, gt_mask):
    """
    Compute lesion-level Precision, Recall, and F1-score.

    Uses connected-component analysis to evaluate lesion detection
    instead of pixel-wise overlap.
    """
    pred_mask = (pred_mask > 0).astype(int)
    gt_mask = (gt_mask > 0).astype(int)

    # Connected components
    pred_labeled, n_pred_lesions = label(pred_mask)
    gt_labeled, n_gt_lesions = label(gt_mask)

    # Edge cases: empty volumes
    if n_gt_lesions == 0 and n_pred_lesions == 0:
        return 1.0, 1.0, 1.0
    if n_gt_lesions == 0 or n_pred_lesions == 0:
        return 0.0, 0.0, 0.0

    # Recall: how many GT lesions are detected
    gt_overlaps = ndi_sum(pred_mask, gt_labeled, index=range(1, n_gt_lesions + 1))
    tp_gt = np.count_nonzero(gt_overlaps)
    recall = tp_gt / n_gt_lesions

    # Precision: how many predicted lesions are correct
    pred_overlaps = ndi_sum(gt_mask, pred_labeled, index=range(1, n_pred_lesions + 1))
    tp_pred = np.count_nonzero(pred_overlaps)
    precision = tp_pred / n_pred_lesions

    # F1-score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1
