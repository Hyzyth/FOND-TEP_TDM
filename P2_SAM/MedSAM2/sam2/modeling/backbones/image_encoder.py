"""
sam2/modeling/backbones/image_encoder.py
=========================================
ImageEncoder (trunk + FPN neck) and related layers for SAM2.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-first Layer Normalisation for (B, C, H, W) tensors."""

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


class ImageEncoder(nn.Module):
    """Backbone image encoder: trunk → FPN neck → optional scalp.

    Parameters
    ----------
    trunk : nn.Module   Hiera backbone
    neck  : nn.Module   FPN neck
    scalp : int         Drop this many finest FPN levels (0 = keep all)
    """

    def __init__(self, trunk: nn.Module, neck: nn.Module, scalp: int = 0) -> None:
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"Channel mismatch: trunk {self.trunk.channel_list} vs neck {self.neck.backbone_channel_list}"
        )

    def forward(self, sample: torch.Tensor) -> dict:
        """Return vision features, positional encodings, and FPN maps.

        Parameters
        ----------
        sample : Tensor  (B, 3, H, W)

        Returns
        -------
        dict with keys ``vision_features``, ``vision_pos_enc``, ``backbone_fpn``
        """
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features, pos = features[:-self.scalp], pos[:-self.scalp]
        return {
            "vision_features": features[-1],
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }


class FpnNeck(nn.Module):
    """Feature Pyramid Network neck for the Hiera backbone.

    Parameters
    ----------
    position_encoding : nn.Module
    d_model : int            output channel dimension
    backbone_channel_list : list[int]  backbone output dims (coarse → fine)
    kernel_size, stride, padding : int  1×1 lateral convolution parameters
    fpn_interp_model : str   interpolation mode for top-down pathway
    fuse_type : str          ``'sum'`` or ``'avg'``
    fpn_top_down_levels : list[int]  which levels receive top-down features
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.position_encoding = position_encoding
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim, d_model, kernel_size, stride, padding))
            for dim in backbone_channel_list
        ])
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ("sum", "avg")
        self.fuse_type = fuse_type
        self.fpn_top_down_levels = (
            list(fpn_top_down_levels) if fpn_top_down_levels is not None
            else list(range(len(self.convs)))
        )

    def forward(self, xs: List[torch.Tensor]):
        n = len(self.convs)
        out = [None] * n
        pos = [None] * n
        prev = None
        for i in range(n - 1, -1, -1):
            lateral = self.convs[n - 1 - i](xs[i])
            if i in self.fpn_top_down_levels and prev is not None:
                td = F.interpolate(
                    prev.float(), scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=None if self.fpn_interp_model == "nearest" else False,
                    antialias=False,
                )
                prev = lateral + td
                if self.fuse_type == "avg":
                    prev = prev / 2
            else:
                prev = lateral
            out[i] = prev
            pos[i] = self.position_encoding(prev).to(prev.dtype)
        return out, pos
