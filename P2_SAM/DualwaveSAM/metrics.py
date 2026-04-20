import numpy as np
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree


class Dice(nn.Module):
    def __init__(self):
        super(Dice, self).__init__()
        self.smooth = 0.001

    def forward(self, input, target):
        c = input.dim()
        axes = tuple(range(0, input.dim()))
        intersect = (input * target).sum(dim=axes)
        # union = torch.pow(input, 2).sum(dim=axes) + torch.pow(target, 2).sum(dim=axes)
        union = input.sum(dim=axes) + target.sum(dim=axes)
        dic = 2 * intersect / (union + self.smooth)
        return dic.mean()

def dice(mask_gt, mask_seg):
    # print(mask_gt,mask_seg)
    # print(np.logical_and(mask_gt, mask_seg))
    return 2 * np.sum(np.logical_and(
        mask_gt, mask_seg)) / (np.sum(mask_gt) + np.sum(mask_seg)+1e-8)



def iou(pred, target):
    # Ensure binary masks are boolean
    pred = pred.astype(bool)
    target = target.astype(bool)
    eps = 1e-8  # small epsilon for numerical stability
    # Compute intersection and union
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    # IoU is intersection over union
    return (intersection + eps) / (union + eps)

def voe(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    eps = 1e-8
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    # VOE is 1 - intersection/union
    return 1.0 - (intersection + eps) / (union + eps)

def rvd(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)
    eps = 1e-8
    pred_vol = pred.sum()
    target_vol = target.sum()
    # RVD as absolute volume difference relative to ground truth volume
    return abs(pred_vol - target_vol) / (target_vol + eps)


def dice_1(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.5).float()
    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)
    return score.mean()


def dice_2(input, target):
    axes = tuple(range(1, input.dim()))
    bin_input = (input > 0.6).float()
    intersect = (bin_input * target).sum(dim=axes)
    union = bin_input.sum(dim=axes) + target.sum(dim=axes)
    score = 2 * intersect / (union + 1e-3)
    return score.mean()


def recall(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positives = target.sum(dim=axes)
    recall = true_positives / all_positives

    return recall.mean()


def precision(input, target):
    axes = tuple(range(1, input.dim()))
    binary_input = (input > 0.5).float()

    true_positives = (binary_input * target).sum(dim=axes)
    all_positive_calls = binary_input.sum(dim=axes)
    precision = true_positives / all_positive_calls

    return precision.mean()



def hausdorff_distance(image0, image1):
    """Code copied from
    https://github.com/scikit-image/scikit-image/blob/main/skimage/metrics/set_metrics.py#L7-L54
    for compatibility reason with python 3.6
    """
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    # Handle empty sets properly:
    # - if both sets are empty, return zero
    # - if only one set is empty, return infinity
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))




def hausdorff_distance_95(image0, image1):
    a_points = np.transpose(np.nonzero(image0))
    b_points = np.transpose(np.nonzero(image1))

    # 处理空集（与原始函数一致）
    if len(a_points) == 0:
        return 0.0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf

    # 计算双向距离并取95%分位数
    dist_a_to_b = cKDTree(a_points).query(b_points, k=1)[0]
    dist_b_to_a = cKDTree(b_points).query(a_points, k=1)[0]
    
    # 取95%分位数（HD95定义）
    hd95_a = np.percentile(dist_a_to_b, 95)
    hd95_b = np.percentile(dist_b_to_a, 95)
    
    # 返回两个方向的最大分位数
    return max(hd95_a, hd95_b)




import numpy as np
from scipy.ndimage import label, sum as ndi_sum

def compute_lesion_metrics(pred_mask, gt_mask):
    """
    计算 Lesion-wise Precision, Recall 和 F1-score
    :param pred_mask: 预测的二值掩膜 (numpy array), 0为背景, 1为前景
    :param gt_mask: 真实的二值掩膜 (numpy array), 0为背景, 1为前景
    :return: precision, recall, f1
    """
    # 确保输入是布尔或0/1整数
    pred_mask = (pred_mask > 0).astype(int)
    gt_mask = (gt_mask > 0).astype(int)

    # 1. 连通域标记 (Label Connected Components)
    # structure定义连通性，默认是十字连通(connectivity=1)
    pred_labeled, n_pred_lesions = label(pred_mask)
    gt_labeled, n_gt_lesions = label(gt_mask)


    if n_gt_lesions == 0 and n_pred_lesions == 0:
        return 1.0, 1.0, 1.0
    if n_gt_lesions == 0 or n_pred_lesions == 0:
        return 0.0, 0.0, 0.0

    # 2. 计算 Recall (Sensitivity)
    gt_overlaps = ndi_sum(pred_mask, gt_labeled, index=range(1, n_gt_lesions + 1))
    tp_gt = np.count_nonzero(gt_overlaps) # 被检出的 GT 病灶数量
    recall = tp_gt / n_gt_lesions

    # 3. 计算 Precision
    pred_overlaps = ndi_sum(gt_mask, pred_labeled, index=range(1, n_pred_lesions + 1))
    tp_pred = np.count_nonzero(pred_overlaps) # 正确的预测病灶数量
    precision = tp_pred / n_pred_lesions

    # 4. 计算 F1-score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1