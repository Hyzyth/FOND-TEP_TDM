"""
auto_prompting/proposal_net.py
================================
MONAI-based 3D U-Net for coarse tumour probability estimation.

Why MONAI UNet over the custom implementation
----------------------------------------------
The custom U-Net had skip-connection shape mismatches when input spatial dims
were not perfectly divisible by 2^depth, requiring manual trilinear
interpolation patches.  MONAI's UNet handles upsampling padding internally,
is battle-tested on medical volumes, and supports residual units out of the box.

Architecture (default base=16)
-------------------------------
  Channels : 16 → 32 → 64 → 128
  Strides  : 2,  2,  2         (3 pooling steps; min spatial dim divisor = 8)
  Residual : 2 units per block (standard for medical segmentation)
  Norm     : InstanceNorm3d    (stable with batch_size=1)
  Act      : LeakyReLU
  Params   : ~1.2 M at base=16 (vs 340 K previously; still lightweight)

Input  : (B, 2, D, H, W) float32 in [0, 1]   — channels [CT, PET]
Output : (B, 1, D, H, W) float32 in [0, 1]   — tumour probability

Crop-size constraint: each spatial dim divisible by 8 at training time.
At inference MONAI pads internally, so arbitrary input sizes are accepted.
With the default crop_size=64,128,128 the constraint is trivially satisfied.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from monai.networks.nets import UNet as MonaiUNet
from monai.networks.layers import Norm


class Small3DUNet(nn.Module):
    """MONAI-based 3-D U-Net for tumour proposal generation.

    Parameters
    ----------
    in_channels : int   Input channels (default 2 = CT + PET).
    base        : int   Base feature-map width.  Channels are
                        (base, base×2, base×4, base×8).  Default 16.
    dropout     : float Dropout probability applied inside each block.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.base        = base
        self.dropout     = dropout

        self.net = MonaiUNet(
            spatial_dims  = 3,
            in_channels   = in_channels,
            out_channels  = 1,
            channels      = (base, base * 2, base * 4, base * 8),
            strides       = (2, 2, 2),
            num_res_units = 2,
            norm          = Norm.INSTANCE,
            act           = "LEAKYRELU",
            dropout       = dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, D, H, W) float tensor, values in [0, 1]

        Returns
        -------
        (B, 1, D, H, W) probability map in [0, 1]
        """
        return torch.sigmoid(self.net(x))

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
