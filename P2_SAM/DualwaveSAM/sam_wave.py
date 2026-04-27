from argparse import Namespace
import torch
import numpy as np
from build_sam_wave import sam_model_registry
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from types import SimpleNamespace as Namespace


class DualwaveSAM(nn.Module):
    def __init__(self):
        """
        Initialize the DualwaveSAM model.

        - Sets up configuration arguments
        - Builds the SAM-based model with dual-wave input
        - Freezes selected submodules (prompt encoder and mask decoder)
        """
        super().__init__()
        args = Namespace()
        args.image_size = 256
        args.sam_checkpoint = None
        args.wavelet = 'haar'   # Supported wavelets: haar / db2 / sym4 / bior4.4

        # 1) Build the SAM model with dual-wave architecture
        self.model = sam_model_registry["dual_wave"](args)

        # Disable gradient updates for specific components (freeze weights)
        for p in self.model.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.model.mask_decoder.parameters():
            p.requires_grad = False

    # MODIFICATION: exposed points, boxes, masks arguments so SAM prompting
    # is actually accessible; previously all prompts were hardcoded to None.
    def forward(self, x,
                point_coords=None,
                point_labels=None,
                boxes=None,
                masks=None):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            point_coords (Tensor | None): (B, N, 2) point prompt coordinates
            point_labels (Tensor | None): (B, N) point labels (0=neg, 1=pos)
            boxes (Tensor | None): (B, 4) bounding box prompts [x1,y1,x2,y2]
            masks (Tensor | None): (B, 1, H, W) mask prompts

        Returns:
            masks (Tensor): Predicted segmentation masks
            aux_pseudo_logits (Tensor | None): Optional auxiliary outputs
        """
        # MODIFICATION: pass all prompt types through to the SAM model dict
        # instead of always passing an empty dict with no prompts.
        inputs = {
            'image':         x,
            'original_size': (x.shape[-2], x.shape[-1]),
            'point_coords':  point_coords,
            'point_labels':  point_labels,
            'boxes':         boxes,
            'masks':         masks,
        }

        outputs = self.model(inputs, False)

        if "aux_pseudo_logits" in outputs:
            return outputs["masks"], outputs["aux_pseudo_logits"]
        else:
            return outputs["masks"], None


def format_bytes(n):
    """Convert a byte value into a human-readable string (MB and GB)."""
    return f"{n/1024/1024:.1f} MB ({n/1024/1024/1024:.3f} GB)"


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    x = torch.randn(1, 2, 256, 256, device=device)

    model = DualwaveSAM().to(device)
    model.eval()

    stats = summary(model, input_data=x, verbose=0)
    print(f"Total Params: {stats.total_params / 1e6:.2f}M")
    print(f"Total FLOPs: {stats.total_mult_adds / 1e9:.2f}G")

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        with torch.no_grad():
            _ = model(x)
            torch.cuda.synchronize()

        peak_alloc    = torch.cuda.max_memory_allocated(device)
        peak_reserved = torch.cuda.max_memory_reserved(device)
        cur_alloc     = torch.cuda.memory_allocated(device)
        cur_reserved  = torch.cuda.memory_reserved(device)

        print("[GPU Mem] Peak allocated:", format_bytes(peak_alloc))
        print("[GPU Mem] Peak reserved :", format_bytes(peak_reserved))
        print("[GPU Mem] Current allocated:", format_bytes(cur_alloc))
        print("[GPU Mem] Current reserved :", format_bytes(cur_reserved))
    else:
        print("GPU not enabled, cannot measure memory usage.")

    y, _ = model(x)
    print("Output shape:", y.shape)
