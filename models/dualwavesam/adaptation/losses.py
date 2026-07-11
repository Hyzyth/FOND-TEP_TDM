"""
losses.py  -  Loss functions for 3-class DualwaveSAM
======================================================

Primary loss  : Dice + Focal + λ1*MAE + λ2*MSE
  - L_main = L_Dice + L_Focal + 0.01*L_MAE + 0.1*L_MSE
  - MAE and MSE are computed pixel-wise between softmax probs and one-hot targets.

Auxiliary loss: Dice + Focal (applied to the Tiny Pseudo Decoder)
  - L_aux = L_Dice + L_Focal

Combined loss : L_total = L_main + L_aux
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ────────────────────────────────────────────────────────────────────

def one_hot_2d(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Convert (B, H, W) int64 label map -> (B, num_classes, H, W) float one-hot.
    Values outside [0, num_classes) are silently mapped to 0.
    """
    labels = labels.long().clamp(0, num_classes - 1)
    B, H, W = labels.shape
    oh = torch.zeros(B, num_classes, H, W,
                     dtype=torch.float32, device=labels.device)
    oh.scatter_(1, labels.unsqueeze(1), 1.0)
    return oh


# ── Standard Dice Loss ─────────────────────────────────────────────────────────

def dice_loss_multiclass(
    probs: torch.Tensor,     # (B, C, H, W) softmax probabilities
    targets_oh: torch.Tensor,# (B, C, H, W) one-hot encoded targets
    smooth: float = 1e-5,
    ignore_bg: bool = True,
    class_weights: list = None  # Added weighting parameter
) -> torch.Tensor:
    """
    Standard multi-class Dice loss.
    """
    dims = (0, 2, 3)  # Reduce over batch and spatial dims
    
    tp = (probs * targets_oh).sum(dim=dims)
    fp = (probs * (1 - targets_oh)).sum(dim=dims)
    fn = ((1 - probs) * targets_oh).sum(dim=dims)

    dice_score = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    dice_loss_per_class = 1.0 - dice_score

    if ignore_bg:
        dice_loss_per_class = dice_loss_per_class[1:]  # Drop class 0 (background)
        
        # Apply class weighting if provided (e.g., [0.25, 0.75])
        if class_weights is not None:
            weights = torch.tensor(class_weights, device=dice_loss_per_class.device)
            weights = weights / weights.sum() # Normalize weights
            return (dice_loss_per_class * weights).sum()

    return dice_loss_per_class.mean()


# ── Categorical Focal Loss ─────────────────────────────────────────────────────

def focal_loss_categorical(
    logits: torch.Tensor,    # (B, C, H, W) raw logits
    targets: torch.Tensor,   # (B, H, W) integer targets
    gamma: float = 2.0,
    ignore_bg: bool = True
) -> torch.Tensor:
    """
    Standard categorical focal loss: -alpha * (1 - pt)^gamma * log(pt)
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none') # (B, H, W)
    pt = torch.exp(-ce_loss)
    focal_loss = ((1 - pt) ** gamma) * ce_loss
    
    if ignore_bg:
        # Create a mask to zero out background contributions
        bg_mask = (targets != 0).float()
        focal_loss = focal_loss * bg_mask
        # Average only over foreground pixels to avoid dilution
        return focal_loss.sum() / (bg_mask.sum() + 1e-5)
    
    return focal_loss.mean()


# ── Combined Paper Loss Formulation ────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Implements the exact loss formulation from the HECKTOR DualwaveSAM paper:
    L_main  = L_Dice + L_Focal + λ1 * L_MAE + λ2 * L_MSE
    L_aux   = L_Dice + L_Focal
    L_total = L_main + L_aux
    """

    def __init__(
        self,
        lambda1: float = 0.01,  # MAE weight
        lambda2: float = 0.1,   # MSE weight
        gamma: float = 2.0,     # Focal exponent
        smooth: float = 1e-5,
        num_classes: int = 3,
        ignore_bg: bool = True,
        **kwargs # Catch-all for CLI arguments passed from train.py (alpha, beta, etc.)
    ):
        super().__init__()
        self.lambda1     = lambda1
        self.lambda2     = lambda2
        self.gamma       = gamma
        self.smooth      = smooth
        self.num_classes = num_classes
        self.ignore_bg   = ignore_bg

    def forward(
        self,
        logits: torch.Tensor,           # (B, C, H, W) - Main SAM Decoder
        targets: torch.Tensor,          # (B, H, W)
        aux_logits: torch.Tensor = None # (B, C, H, W) - Tiny Pseudo Decoder
    ) -> torch.Tensor:
        
        # Ensure targets are int64 for PyTorch cross_entropy
        targets = targets.long()

        # 1. Prepare representations
        probs = F.softmax(logits, dim=1)
        targets_oh = one_hot_2d(targets, self.num_classes).to(logits.device)

        # 2. Main Loss Components (Dice + Focal)
        # Apply class weighting: 0.4 to Tumor, 0.6 to Nodule
        dice_main = dice_loss_multiclass(probs, targets_oh, self.smooth, self.ignore_bg, class_weights=[0.4, 0.6])
        focal_main = focal_loss_categorical(logits, targets, self.gamma, self.ignore_bg)

        # 3. Regularization Components (MAE + MSE)
        if self.ignore_bg:
            # Create a spatial mask for foreground pixels
            bg_mask = (targets != 0).float().unsqueeze(1) # (B, 1, H, W)
            
            # Compute pixel-wise absolute and squared errors
            l1_pixel = torch.abs(probs - targets_oh)
            mse_pixel = (probs - targets_oh) ** 2
            
            # Average ONLY over the foreground pixels to prevent background dilution
            # Adding 1e-5 prevents division by zero if a batch has no foreground
            mae_loss = (l1_pixel * bg_mask).sum() / (bg_mask.sum() * self.num_classes + 1e-5)
            mse_loss = (mse_pixel * bg_mask).sum() / (bg_mask.sum() * self.num_classes + 1e-5)
        else:
            mae_loss = F.l1_loss(probs, targets_oh)
            mse_loss = F.mse_loss(probs, targets_oh)

        # 4. Assemble Main Loss
        L_main = dice_main + focal_main + (self.lambda1 * mae_loss) + (self.lambda2 * mse_loss)

        # 5. Assemble Auxiliary Loss (if present)
        L_total = L_main
        if aux_logits is not None:
            aux_probs = F.softmax(aux_logits, dim=1)
            # Apply the same class weights to the auxiliary head
            dice_aux  = dice_loss_multiclass(aux_probs, targets_oh, self.smooth, self.ignore_bg, class_weights=[0.4, 0.6])
            focal_aux = focal_loss_categorical(aux_logits, targets, self.gamma, self.ignore_bg)
            L_aux     = dice_aux + focal_aux
            
            L_total = L_main + L_aux

        return L_total
