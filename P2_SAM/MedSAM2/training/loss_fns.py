"""
training/loss_fns.py
======================
Multi-step multi-mask loss combining focal, Dice, and IoU objectives.
Used during fine-tuning on HECKTOR and other medical segmentation tasks.
"""

from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.trainer import CORE_LOSS_KEY
from training.utils.distributed import get_world_size, is_dist_avail_and_initialized


# ──────────────────────────────────────────────────────────────────────────────
# Component losses
# ──────────────────────────────────────────────────────────────────────────────

def dice_loss(inputs: torch.Tensor, targets: torch.Tensor,
              num_objects: float, loss_on_multimask: bool = False) -> torch.Tensor:
    """Generalised DICE loss.

    Parameters
    ----------
    inputs : Tensor  raw logits, shape (N, [M,] H, W)
    targets : Tensor binary GT, same shape as inputs
    num_objects : float  normalisation constant
    loss_on_multimask : bool  keep the multi-mask channel dim in output

    Returns
    -------
    Tensor  scalar (or (N, M) if loss_on_multimask)
    """
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss / num_objects if loss_on_multimask else loss.sum() / num_objects


def sigmoid_focal_loss(inputs: torch.Tensor, targets: torch.Tensor,
                       num_objects: float, alpha: float = 0.25, gamma: float = 2,
                       loss_on_multimask: bool = False) -> torch.Tensor:
    """Sigmoid focal loss (RetinaNet).

    Parameters
    ----------
    inputs : Tensor  raw logits
    targets : Tensor  binary GT
    num_objects : float
    alpha : float  class balancing weight
    gamma : float  focusing parameter
    loss_on_multimask : bool

    Returns
    -------
    Tensor  scalar (or (N, M))
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if loss_on_multimask:
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects
    return loss.mean(1).sum() / num_objects


def iou_loss(inputs: torch.Tensor, targets: torch.Tensor,
             pred_ious: torch.Tensor, num_objects: float,
             loss_on_multimask: bool = False,
             use_l1_loss: bool = False) -> torch.Tensor:
    """IoU prediction loss (L1 or MSE between predicted and actual IoU).

    Parameters
    ----------
    inputs : Tensor  raw mask logits
    targets : Tensor  binary GT
    pred_ious : Tensor  model's IoU prediction
    num_objects : float
    loss_on_multimask : bool
    use_l1_loss : bool  L1 instead of MSE

    Returns
    -------
    Tensor  scalar (or (N, M))
    """
    pred_mask = inputs.flatten(2) > 0
    gt_mask   = targets.flatten(2) > 0
    area_i = (pred_mask & gt_mask).float().sum(-1)
    area_u = (pred_mask | gt_mask).float().sum(-1)
    actual_ious = area_i / area_u.clamp(min=1.0)
    loss = (F.l1_loss if use_l1_loss else F.mse_loss)(pred_ious, actual_ious, reduction="none")
    return loss / num_objects if loss_on_multimask else loss.sum() / num_objects


# ──────────────────────────────────────────────────────────────────────────────
# Combined loss
# ──────────────────────────────────────────────────────────────────────────────

class MultiStepMultiMasksAndIous(nn.Module):
    """Combined focal + Dice + IoU loss over multi-step interactive predictions.

    Parameters
    ----------
    weight_dict : dict  keys ``loss_mask``, ``loss_dice``, ``loss_iou``, ``loss_class``
    focal_alpha : float
    focal_gamma : float
    supervise_all_iou : bool  back-prop IoU loss for all mask candidates
    iou_use_l1_loss : bool
    pred_obj_scores : bool
    focal_gamma_obj_score : float
    focal_alpha_obj_score : float
    """

    def __init__(self, weight_dict, focal_alpha=0.25, focal_gamma=2,
                 supervise_all_iou=False, iou_use_l1_loss=False,
                 pred_obj_scores=False, focal_gamma_obj_score=0.0,
                 focal_alpha_obj_score=-1) -> None:
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in weight_dict and "loss_dice" in weight_dict and "loss_iou" in weight_dict
        self.weight_dict.setdefault("loss_class", 0.0)
        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores

    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor) -> Dict:
        assert len(outs_batch) == len(targets_batch)
        num_objects = torch.tensor(targets_batch.shape[1], device=targets_batch.device, dtype=torch.float)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_objects)
        num_objects = num_objects.clamp(min=1) / get_world_size()

        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            for k, v in self._forward(outs, targets, num_objects.item()).items():
                losses[k] += v
        return losses

    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects: float) -> Dict:
        target_masks = targets.unsqueeze(1).float()   # (N, 1, H, W)
        src_masks_list  = outputs["multistep_pred_multimasks_high_res"]
        ious_list       = outputs["multistep_pred_ious"]
        obj_score_list  = outputs["multistep_object_score_logits"]

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, obj_score_logits in zip(src_masks_list, ious_list, obj_score_list):
            self._update_losses(losses, src_masks, target_masks, ious, num_objects, obj_score_logits)
        losses[CORE_LOSS_KEY] = self._reduce_loss(losses)
        return losses

    def _update_losses(self, losses, src_masks, target_masks, ious, num_objects, obj_score_logits):
        target_masks = target_masks.expand_as(src_masks)
        loss_multimask = sigmoid_focal_loss(src_masks, target_masks, num_objects,
                                            self.focal_alpha, self.focal_gamma, loss_on_multimask=True)
        loss_multidice = dice_loss(src_masks, target_masks, num_objects, loss_on_multimask=True)

        if not self.pred_obj_scores:
            loss_class = torch.tensor(0.0, dtype=loss_multimask.dtype, device=loss_multimask.device)
            target_obj = torch.ones(loss_multimask.shape[0], 1, dtype=loss_multimask.dtype, device=loss_multimask.device)
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[..., None].float()
            loss_class = sigmoid_focal_loss(obj_score_logits, target_obj, num_objects,
                                            alpha=self.focal_alpha_obj_score, gamma=self.focal_gamma_obj_score)

        loss_multiiou = iou_loss(src_masks, target_masks, ious, num_objects,
                                 loss_on_multimask=True, use_l1_loss=self.iou_use_l1_loss)

        if loss_multimask.size(1) > 1:
            loss_combo = (loss_multimask * self.weight_dict["loss_mask"]
                          + loss_multidice * self.weight_dict["loss_dice"])
            best_inds = loss_combo.argmin(dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_inds].unsqueeze(1)
            loss_iou  = (loss_multiiou.mean(dim=-1).unsqueeze(1) if self.supervise_all_iou
                         else loss_multiiou[batch_inds, best_inds].unsqueeze(1))
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou  = loss_multiiou

        losses["loss_mask"]  += (loss_mask  * target_obj).sum()
        losses["loss_dice"]  += (loss_dice  * target_obj).sum()
        losses["loss_iou"]   += (loss_iou   * target_obj).sum()
        losses["loss_class"] += loss_class

    def _reduce_loss(self, losses: Dict) -> torch.Tensor:
        total = 0.0
        for key, weight in self.weight_dict.items():
            if key not in losses:
                raise ValueError(f"{type(self).__name__} does not compute '{key}'")
            if weight != 0:
                total = total + losses[key] * weight
        return total
