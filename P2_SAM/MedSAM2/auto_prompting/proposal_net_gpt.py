"""
auto_prompting/proposal_net.py
================================
MONAI-based 3D UNet proposal network (shape-safe, production-ready).

Replaces custom U-Net implementation with MONAI UNet to eliminate
skip-connection shape mismatches and improve training stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.nets import UNet
from monai.networks.layers import Norm


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class Small3DUNet(nn.Module):
    """
    MONAI-based 3D UNet for tumor proposal generation.

    Input:
        (B, 2, D, H, W)  -> CT + PET (normalized)

    Output:
        (B, 1, D, H, W)  -> probability map
    """

    def __init__(
        self,
        in_channels: int = 2,
        base: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.base = base
        self.dropout = dropout

        self.net = UNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,
            channels=(base, base * 2, base * 4, base * 8),
            strides=(2, 2, 2),
            num_res_units=1,
            norm=Norm.INSTANCE,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 2, D, H, W)

        Returns
        -------
        torch.Tensor
            Shape (B, 1, D, H, W), values in [0, 1]
        """
        return torch.sigmoid(self.net(x))

    # -----------------------------------------------------------------
    # Persistence utilities
    # -----------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Save model weights + config for reproducibility.
        """
        torch.save(
            {
                "model_state": self.state_dict(),
                "config": {
                    "in_channels": self.in_channels,
                    "base": self.base,
                    "dropout": self.dropout,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "Small3DUNet":
        """
        Load model from checkpoint created with `save()`.
        """
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})

        model = cls(**cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()

        return model


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
