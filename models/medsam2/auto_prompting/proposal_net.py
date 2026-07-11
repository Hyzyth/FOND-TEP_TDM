"""
auto_prompting/proposal_net.py
================================
MONAI-based 3D U-Net for coarse tumour probability estimation.

Architecture (default base=16)
-------------------------------
  Channels : 16 → 32 → 64 → 128
  Strides  : 2,  2,  2
  Residual : 2 units per block
  Norm     : InstanceNorm3d
  Act      : LeakyReLU
  Params   : ~1.2 M

Input  : (B, 2, D, H, W) float32 in [0, 1]   — channels [CT, PET]
Output : (B, 1, D, H, W) float32 in [0, 1]   — tumour probability

CHANGE — Prior-probability output bias
---------------------------------------
The MONAI UNet's final layer bias defaults to zero, which means
sigmoid(0) = 0.5 for all voxels at initialisation.  With a 99:1
background-to-foreground ratio the loss gradient is near-zero at p=0.5,
trapping the optimiser in a flat region for tens of epochs.

Setting output_bias = log(prior / (1-prior)) so that the model starts with
mean_pred ≈ prior (default 0.02) places it in a region of strong Tversky
gradient immediately.  This is the same technique used in RetinaNet
(Lin et al. 2017, §4.1) for class-imbalanced detection.

The bias is stored as a scalar nn.Parameter so it is learnable and
automatically saved/loaded with the rest of the state dict.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from monai.networks.nets import UNet as MonaiUNet
from monai.networks.layers import Norm


class Small3DUNet(nn.Module):
    """MONAI-based 3-D U-Net for tumour proposal generation.

    Parameters
    ----------
    in_channels : int    Input channels (default 2 = CT + PET).
    base        : int    Base feature-map width.
    dropout     : float  Dropout probability.
    prior_prob  : float  Expected foreground fraction at initialisation.
                         Sets the output bias so mean_pred ≈ prior_prob,
                         avoiding the p=0.5 saddle point.  Default 0.02.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base: int = 16,
        dropout: float = 0.1,
        prior_prob: float = 0.02,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.base        = base
        self.dropout     = dropout
        self.prior_prob  = prior_prob

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

        # Scalar bias added to logits before sigmoid.
        # Initialised so that sigmoid(bias) ≈ prior_prob:
        #   bias = log(prior / (1 - prior))
        # This moves the starting mean_pred from 0.5 down to ~prior_prob,
        # immediately giving strong gradient signal.
        bias_init = math.log(prior_prob / (1.0 - prior_prob))
        self.output_bias = nn.Parameter(torch.tensor(bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, D, H, W) float tensor, values in [0, 1]

        Returns
        -------
        (B, 1, D, H, W) probability map in [0, 1]
        """
        return torch.sigmoid(self.net(x) + self.output_bias)

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
                    "prior_prob":  self.prior_prob,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "Small3DUNet":
        """Load a checkpoint saved with :meth:`save`."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg  = ckpt.get("config", {})
        # Backward compatibility: old checkpoints lack prior_prob
        cfg.setdefault("prior_prob", 0.02)
        net  = cls(**cfg)
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        return net.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
