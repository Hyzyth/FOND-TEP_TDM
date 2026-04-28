# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the LICENSE file.

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# LayerNorm2d is defined locally below (removed dependency on efficient_track_anything)
from torch import nn as _nn


class LayerNorm2d(nn.Module):
    """Channel-first layer norm."""
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
    def __init__(self, trunk: nn.Module, neck: nn.Module, scalp: int = 0):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert (
            self.trunk.channel_list == self.neck.backbone_channel_list
        ), f"Channel dims mismatch. Trunk: {self.trunk.channel_list}, neck: {self.neck.backbone_channel_list}"

    def forward(self, sample: torch.Tensor):
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features, pos = features[: -self.scalp], pos[: -self.scalp]
        src = features[-1]
        return {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }


class FpnNeck(nn.Module):
    """
    Feature Pyramid Network neck for the Hiera backbone (SAM2).
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
    ):
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(in_channels=dim, out_channels=d_model,
                          kernel_size=kernel_size, stride=stride, padding=padding),
            )
            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        prev_features = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            out[i] = prev_features
            pos[i] = self.position_encoding(prev_features).to(prev_features.dtype)
        return out, pos