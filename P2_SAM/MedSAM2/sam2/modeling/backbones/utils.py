"""
sam2/modeling/backbones/utils.py
================================
Utility functions and modules for the Hiera backbone.

Includes windowed attention helpers and the patch-embedding layer.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window_size: int):
    """Partition *x* into non-overlapping windows, padding if necessary.

    Parameters
    ----------
    x : Tensor  (B, H, W, C)
    window_size : int

    Returns
    -------
    windows : Tensor  (B*num_windows, window_size, window_size, C)
    (Hp, Wp) : padded height and width
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """Reverse :func:`window_partition`, removing padding.

    Parameters
    ----------
    windows : Tensor  (B*num_windows, window_size, window_size, C)
    window_size : int
    pad_hw : (Hp, Wp)  padded height and width
    hw : (H, W)  original height and width

    Returns
    -------
    Tensor  (B, H, W, C)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


class PatchEmbed(nn.Module):
    """Image-to-patch embedding via a strided convolution.

    Parameters
    ----------
    kernel_size : tuple[int, int]
    stride : tuple[int, int]
    padding : tuple[int, int]
    in_chans : int   input channels (3 for RGB)
    embed_dim : int  output embedding dimension
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (7, 7),
        stride: Tuple[int, int] = (4, 4),
        padding: Tuple[int, int] = (3, 3),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=kernel_size, stride=stride, padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) → (B, H', W', embed_dim)."""
        return self.proj(x).permute(0, 2, 3, 1)


def get_abs_pos(
    abs_pos: torch.Tensor,
    has_cls_token: bool,
    hw: Tuple[int, int],
) -> torch.Tensor:
    """Resize absolute positional embeddings to a new spatial size.

    Parameters
    ----------
    abs_pos : Tensor  (1, num_positions, C)
    has_cls_token : bool
    hw : (H, W)  target spatial size

    Returns
    -------
    Tensor  (1, H, W, C)
    """
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        mode = "bilinear" if (not torch.cuda.is_available() and torch.mps.is_available()) else "bicubic"
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w), mode=mode, align_corners=False,
        )
        return new_abs_pos.permute(0, 2, 3, 1)
    return abs_pos.reshape(1, h, w, -1)
