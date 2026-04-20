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

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            masks (Tensor): Predicted segmentation masks
            aux_pseudo_logits (Tensor or None): Optional auxiliary outputs
        """
        # Prepare input dictionary expected by the SAM model
        inputs = {'image': x, 'original_size': (x.shape[-2], x.shape[-1])}

        # Perform inference (second argument disables multimask output)
        outputs = self.model(inputs, False)

        # Return auxiliary logits if available
        if "aux_pseudo_logits" in outputs:
            return outputs["masks"], outputs["aux_pseudo_logits"]
        else:
            return outputs["masks"], None

def format_bytes(n):
    """
    Convert a byte value into a human-readable string (MB and GB).
    """
    return f"{n/1024/1024:.1f} MB ({n/1024/1024/1024:.3f} GB)"

if __name__ == "__main__":
    # Select device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate dummy input data
    # Shape: (batch_size=1, channels=2, height=256, width=256)
    # Example channels: CT and PET images
    x = torch.randn(1, 2, 256, 256, device=device)

    # Initialize model and move it to the selected device
    model = DualwaveSAM().to(device)
    model.eval()  # Set model to evaluation mode

    # Compute model statistics: parameter count and FLOPs
    stats = summary(model, input_data=x, verbose=0)
    print(f"Total Params: {stats.total_params / 1e6:.2f}M")
    print(f"Total FLOPs: {stats.total_mult_adds / 1e9:.2f}G")

    # ===== GPU memory usage statistics =====
    if device.type == "cuda":
        # Clear cache and reset memory tracking
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

        # Run a forward pass without gradient tracking
        with torch.no_grad():   
            _ = model(x)
            torch.cuda.synchronize()

        # Retrieve memory usage metrics
        peak_alloc = torch.cuda.max_memory_allocated(device)   # Peak allocated memory
        peak_reserved = torch.cuda.max_memory_reserved(device) # Peak reserved memory (including cache)
        cur_alloc = torch.cuda.memory_allocated(device)        # Current allocated memory
        cur_reserved = torch.cuda.memory_reserved(device)      # Current reserved memory

        print("[GPU Mem] Peak allocated:", format_bytes(peak_alloc))
        print("[GPU Mem] Peak reserved :", format_bytes(peak_reserved))
        print("[GPU Mem] Current allocated:", format_bytes(cur_alloc))
        print("[GPU Mem] Current reserved :", format_bytes(cur_reserved))
    else:
        # GPU is not available
        print("GPU not enabled, cannot measure memory usage.")

    # Verify output shape
    y, _ = model(x)
    print("Output shape:", y.shape)
