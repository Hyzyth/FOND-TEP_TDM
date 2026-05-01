"""
sam2/modeling/sam2_utils.py
============================
Shared utility classes and functions for the SAM2 model.
"""

import copy
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.utils.misc import mask_to_box


def select_closest_cond_frames(frame_idx, cond_frame_outputs, max_cond_frame_num):
    """Select up to *max_cond_frame_num* conditioning frames closest to *frame_idx*.

    Returns ``(selected_outputs, unselected_outputs)`` dicts.
    """
    if max_cond_frame_num == -1 or len(cond_frame_outputs) <= max_cond_frame_num:
        return cond_frame_outputs, {}
    assert max_cond_frame_num >= 2
    selected = {}
    idx_before = max((t for t in cond_frame_outputs if t < frame_idx), default=None)
    if idx_before is not None:
        selected[idx_before] = cond_frame_outputs[idx_before]
    idx_after = min((t for t in cond_frame_outputs if t >= frame_idx), default=None)
    if idx_after is not None:
        selected[idx_after] = cond_frame_outputs[idx_after]
    num_remain = max_cond_frame_num - len(selected)
    extras = sorted(
        (t for t in cond_frame_outputs if t not in selected),
        key=lambda x: abs(x - frame_idx),
    )[:num_remain]
    selected.update((t, cond_frame_outputs[t]) for t in extras)
    unselected = {t: v for t, v in cond_frame_outputs.items() if t not in selected}
    return selected, unselected


def get_1d_sine_pe(pos_inds: torch.Tensor, dim: int, temperature: float = 10000) -> torch.Tensor:
    """1-D sinusoidal positional embedding."""
    pe_dim = dim // 2
    dim_t = torch.arange(pe_dim, dtype=torch.float32, device=pos_inds.device)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    pos_embed = pos_inds.unsqueeze(-1) / dim_t
    return torch.cat([pos_embed.sin(), pos_embed.cos()], dim=-1)


def get_activation_fn(activation: str):
    if activation == "relu":  return F.relu
    if activation == "gelu":  return F.gelu
    if activation == "glu":   return F.glu
    raise RuntimeError(f"Unknown activation: {activation}")


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DropPath(nn.Module):
    """Stochastic depth per sample (applied in the residual path)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            rand.div_(keep_prob)
        return x * rand


class MLP(nn.Module):
    """Multi-layer perceptron with optional sigmoid output."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: nn.Module = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class LayerNorm2d(nn.Module):
    """Channel-first Layer Normalisation (identical to image_encoder.LayerNorm2d)."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def sample_box_points(masks, noise=0.1, noise_bound=20, top_left_label=2, bottom_right_label=3):
    """Sample noisy bounding-box corner points from binary masks."""
    device = masks.device
    box_coords = mask_to_box(masks)
    B, _, H, W = masks.shape
    box_labels = torch.tensor([top_left_label, bottom_right_label], dtype=torch.int, device=device).repeat(B)
    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)
        bbox_w = box_coords[..., 2] - box_coords[..., 0]
        bbox_h = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(bbox_w * noise, noise_bound)
        max_dy = torch.min(bbox_h * noise, noise_bound)
        box_noise = 2 * torch.rand(B, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)
        box_coords = box_coords + box_noise
        img_bounds = torch.tensor([W, H, W, H], device=device) - 1
        box_coords.clamp_(torch.zeros_like(img_bounds), img_bounds)
    return box_coords.reshape(-1, 2, 2), box_labels.reshape(-1, 2)


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    """Sample correction points uniformly from FP/FN error regions."""
    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    B, _, H_im, W_im = gt_masks.shape
    device = gt_masks.device
    fp_masks = ~gt_masks & pred_masks
    fn_masks = gt_masks & ~pred_masks
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)[..., None, None]
    pts_noise = torch.rand(B, num_pt, H_im, W_im, 2, device=device)
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % W_im
    pts_y = pts_idx // W_im
    return torch.stack([pts_x, pts_y], dim=2).float(), labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    """Sample the point farthest from the error region boundary (RITM strategy)."""
    import cv2
    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    B, _, _, W_im = gt_masks.shape
    device = gt_masks.device
    fp_masks = (~gt_masks & pred_masks).cpu().numpy()
    fn_masks = (gt_masks & ~pred_masks).cpu().numpy()
    points = torch.zeros(B, 1, 2, dtype=torch.float)
    labels = torch.ones(B, 1, dtype=torch.int32)
    for b in range(B):
        fn = fn_masks[b, 0]
        fp = fp_masks[b, 0]
        if padding:
            fn = np.pad(fn, 1, "constant")
            fp = np.pad(fp, 1, "constant")
        fn_dt = cv2.distanceTransform(fn.astype(np.uint8), cv2.DIST_L2, 0)
        fp_dt = cv2.distanceTransform(fp.astype(np.uint8), cv2.DIST_L2, 0)
        if padding:
            fn_dt = fn_dt[1:-1, 1:-1]
            fp_dt = fp_dt[1:-1, 1:-1]
        fn_flat, fp_flat = fn_dt.reshape(-1), fp_dt.reshape(-1)
        fn_arg, fp_arg = np.argmax(fn_flat), np.argmax(fp_flat)
        is_pos = fn_flat[fn_arg] > fp_flat[fp_arg]
        pt_idx = fn_arg if is_pos else fp_arg
        points[b, 0, 0] = pt_idx % W_im
        points[b, 0, 1] = pt_idx // W_im
        labels[b, 0] = int(is_pos)
    return points.to(device), labels.to(device)


def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    if method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    raise ValueError(f"Unknown sampling method: {method}")
