# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# Licensed under the license found in the LICENSE file
# at the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Type


# ============================================================
# Feedforward MLP block (Transformer-style)
# ============================================================
class MLPBlock(nn.Module):
    """
    Standard 2-layer feedforward network used in Transformer blocks.

    Structure:
        Linear -> Activation -> Linear

    Args:
        embedding_dim (int): Input/output feature dimension.
        mlp_dim (int): Hidden layer dimension.
        act (Type[nn.Module]): Activation function class (default: GELU).
    """

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()

        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (..., embedding_dim)

        Returns:
            torch.Tensor: Output tensor of same shape as input embedding dimension
        """
        return self.lin2(self.act(self.lin1(x)))


# ============================================================
# 2D LayerNorm implementation (channel-first)
# Source references:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
# ============================================================
class LayerNorm2d(nn.Module):
    """
    Layer Normalization adapted for 2D feature maps.

    Normalization is applied over channel dimension (dim=1),
    assuming input shape: (N, C, H, W).

    Learnable parameters:
        weight: scale (gamma)
        bias: shift (beta)
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()

        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise LayerNorm for 2D feature maps.

        Args:
            x (torch.Tensor): Input tensor (N, C, H, W)

        Returns:
            torch.Tensor: Normalized tensor with same shape
        """

        # Compute mean across channel dimension
        u = x.mean(1, keepdim=True)

        # Compute variance across channel dimension
        s = (x - u).pow(2).mean(1, keepdim=True)

        # Normalize
        x = (x - u) / torch.sqrt(s + self.eps)

        # Apply learned scale parameter
        y = self.weight[:, None, None] * x

        # Optional equivalent formulation (kept for reference):
        # y = torch.mul(self.weight[:, None, None], x)

        # Apply learned bias parameter
        x = y + self.bias[:, None, None]

        return x
