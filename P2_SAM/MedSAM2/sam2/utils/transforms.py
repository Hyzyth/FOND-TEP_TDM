"""
sam2/utils/transforms.py
==========================
Pre- and post-processing transforms for SAM2 image prediction.
"""

import warnings
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize, Resize, ToTensor


class SAM2Transforms(nn.Module):
    """Resize + normalise pipeline for SAM2 image input.

    Parameters
    ----------
    resolution : int    target (square) size
    mask_threshold : float
    max_hole_area : float  fill holes smaller than this (0 = disabled)
    max_sprinkle_area : float  remove isolated foreground smaller than this
    """

    def __init__(self, resolution: int, mask_threshold: float,
                 max_hole_area: float = 0.0, max_sprinkle_area: float = 0.0) -> None:
        super().__init__()
        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std  = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = torch.jit.script(nn.Sequential(
            Resize((resolution, resolution)),
            Normalize(self.mean, self.std),
        ))

    def __call__(self, x):
        return self.transforms(self.to_tensor(x))

    def forward_batch(self, img_list):
        return torch.stack([self.transforms(self.to_tensor(img)) for img in img_list], dim=0)

    def transform_coords(self, coords: torch.Tensor, normalize: bool = False,
                         orig_hw=None) -> torch.Tensor:
        """Scale coordinates to the model's internal resolution."""
        if normalize:
            assert orig_hw is not None
            h, w = orig_hw
            coords = coords.clone()
            coords[..., 0] = coords[..., 0] / w
            coords[..., 1] = coords[..., 1] / h
        return coords * self.resolution

    def transform_boxes(self, boxes: torch.Tensor, normalize: bool = False,
                        orig_hw=None) -> torch.Tensor:
        return self.transform_coords(boxes.reshape(-1, 2, 2), normalize, orig_hw)

    def postprocess_masks(self, masks: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
        """Optionally fill holes / remove sprinkles, then resize to *orig_hw*."""
        from sam2.utils.misc import get_connected_components
        masks = masks.float()
        input_masks = masks
        mask_flat = masks.flatten(0, 1).unsqueeze(1)
        try:
            if self.max_hole_area > 0:
                labels, areas = get_connected_components(mask_flat <= self.mask_threshold)
                is_hole = (labels > 0) & (areas <= self.max_hole_area)
                masks = torch.where(is_hole.reshape_as(masks), self.mask_threshold + 10.0, masks)
            if self.max_sprinkle_area > 0:
                labels, areas = get_connected_components(mask_flat > self.mask_threshold)
                is_sprinkle = (labels > 0) & (areas <= self.max_sprinkle_area)
                masks = torch.where(is_sprinkle.reshape_as(masks), self.mask_threshold - 10.0, masks)
        except Exception as e:
            warnings.warn(f"Post-processing skipped: {e}", UserWarning, stacklevel=2)
            masks = input_masks
        return F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)
