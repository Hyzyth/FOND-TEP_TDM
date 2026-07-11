"""
ex_transforms.py
Augmentation and preprocessing transforms for the DualwaveSAM pipeline.

All transforms operate on a sample dict:
    {
        'id':     int,
        'input':  np.ndarray  (H, W, C)   float32
        'target': np.ndarray  (H, W, 1)   float32
    }

ToTensor converts both arrays to (C, H, W) torch.Tensors.
ProbsToLabels is a callable applied to raw model output (numpy array).
"""

import random
import numpy as np
import torch


# ============================================================
# Compose
# ============================================================
class Compose:
    """Chain multiple transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


# ============================================================
# ToTensor
# ============================================================
class ToTensor:
    """
    Convert numpy sample arrays to torch.Tensor.

    input:  (H, W, C) float32  →  (C, H, W) FloatTensor
    target: (H, W, 1) float32  →  (1, H, W) FloatTensor
    """

    def __call__(self, sample):
        img = sample["input"].astype(np.float32)    # (H, W, C)
        tgt = sample["target"].astype(np.float32)   # (H, W, 1)

        sample["input"]  = torch.from_numpy(img).permute(2, 0, 1)   # → (C, H, W)
        sample["target"] = torch.from_numpy(tgt).permute(2, 0, 1)   # → (1, H, W)

        return sample


# ============================================================
# Spatial augmentations
# ============================================================
class Horizontal_Mirroring:
    """Random horizontal flip (left ↔ right)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            # flip along W axis (axis=1 for H×W×C layout)
            sample["input"]  = np.flip(sample["input"],  axis=1).copy()
            sample["target"] = np.flip(sample["target"], axis=1).copy()
        return sample


class Vertical_Mirroring:
    """Random vertical flip (top ↔ bottom)."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            # flip along H axis (axis=0 for H×W×C layout)
            sample["input"]  = np.flip(sample["input"],  axis=0).copy()
            sample["target"] = np.flip(sample["target"], axis=0).copy()
        return sample


class Random90Rotation:
    """
    Random 90° rotation (k ∈ {0, 1, 2, 3}).

    Optional extra augmentation; not used in the default pipeline.
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            k = random.randint(1, 3)
            # rotate in the H×W plane (axes 0,1)
            sample["input"]  = np.rot90(sample["input"],  k, axes=(0, 1)).copy()
            sample["target"] = np.rot90(sample["target"], k, axes=(0, 1)).copy()
        return sample


# ============================================================
# Post-processing: probability maps → binary labels
# ============================================================
class ProbsToLabels:
    """
    Convert raw model probability outputs to binary segmentation masks.

    Applies sigmoid if values are outside [0, 1], then thresholds.

    Args:
        threshold (float): binarisation threshold (default 0.5)
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def __call__(self, probs: np.ndarray) -> np.ndarray:
        """
        Args:
            probs: numpy array of any shape (raw logits or probabilities)

        Returns:
            Binary mask as float32 array (same shape)
        """
        # If values lie outside [0,1] assume logits → apply sigmoid
        if probs.min() < 0.0 or probs.max() > 1.0:
            probs = 1.0 / (1.0 + np.exp(-probs))

        return (probs > self.threshold).astype(np.float32)
