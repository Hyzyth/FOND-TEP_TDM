"""
auto_prompting/proposal_net.py
================================
Lightweight 3-D U-Net for coarse tumor probability estimation.

Architecture: 2 encoder blocks → bottleneck → 2 decoder blocks → sigmoid output.
  in_channels  = 2   (CT, PET in [0, 1])
  out_channels = 1   (tumor probability)
  base         = 16  (feature maps: 16 → 32 → 64)

Designed to be trained quickly (~30 epochs) on HECKTOR NPZ data as a
high-recall proposal network, not as a final segmentation model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv3d → InstanceNorm → LeakyReLU × 2, with Dropout3d."""

    def __init__(self, in_c: int, out_c: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Dropout3d(dropout),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_c),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Small3DUNet(nn.Module):
    """Lightweight 3-D U-Net for tumour proposal generation.

    Parameters
    ----------
    in_channels : int   default 2  (CT + PET)
    base        : int   base feature-map count (default 16)
    dropout     : float Dropout3d probability (default 0.1)
    """

    def __init__(self,
                 in_channels: int = 2,
                 base: int = 16,
                 dropout: float = 0.1) -> None:
        super().__init__()
        # Store for save / load round-trip
        self.in_channels = in_channels
        self.base        = base
        self.dropout     = dropout

        b = base
        self.enc1  = ConvBlock(in_channels, b,     dropout)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2  = ConvBlock(b,     b * 2, dropout)
        self.pool2 = nn.MaxPool3d(2)

        self.neck  = ConvBlock(b * 2, b * 4, dropout)

        self.up2   = nn.ConvTranspose3d(b * 4, b * 2, 2, stride=2)
        self.dec2  = ConvBlock(b * 4, b * 2, dropout)

        self.up1   = nn.ConvTranspose3d(b * 2, b,     2, stride=2)
        self.dec1  = ConvBlock(b * 2, b,     dropout)

        self.head  = nn.Conv3d(b, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, D, H, W) float tensor, values in [0, 1]

        Returns
        -------
        (B, 1, D, H, W) probability map in [0, 1]
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.neck(self.pool2(e2))

        d2 = self.up2(b)
        if d2.shape[-3:] != e2.shape[-3:]:
            d2 = F.interpolate(d2, size=e2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-3:] != e1.shape[-3:]:
            d1 = F.interpolate(d1, size=e1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.head(d1))

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save weights + constructor config to *path*."""
        torch.save(
            {
                "model_state": self.state_dict(),
                "config": {
                    "in_channels": self.in_channels,
                    "base":        self.base,
                    "dropout":     self.dropout,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "Small3DUNet":
        """Load a checkpoint saved with :meth:`save`."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", {})
        net  = cls(**cfg)
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        return net.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
