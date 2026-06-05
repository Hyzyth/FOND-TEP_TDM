"""
model.py  —  DualwaveSAM adapted for 3-class HECKTOR segmentation
==================================================================

Design principle: preserve the original DualwaveSAM forward pass exactly.

Original forward pass (sam_wave.py + sam_model.py):
  x (B,2,H,W)
    → WaveEncoder            → image_embeddings (B, 256, h, w)
    → PseudoMaskHead         → aux_pseudo_logits (B, 1, H, W)   [auxiliary]
    → PromptEncoder (frozen) → sparse_emb, dense_emb
    → MaskDecoder   (frozen) → low_res_masks (B, K, h, w)
    → postprocess_masks      → masks (B, K, H, W)
    → select mask slice      → final masks (B, 1, H, W)         [primary]

What we change for 3-class output:
  - MaskDecoder: num_multimask_outputs 3 → 2  (so output has 3 channels:
    1 iou-token mask + 2 multimask = 3 total; we keep all 3 and map them
    to bg/GTVp/GTVn via a thin 1×1 conv adaptor)
  - PseudoMaskHead: out channels 1 → 3
  - Add a tiny ClassAdaptor (1×1 conv, 3→3) after postprocess_masks to
    re-map the 3 raw mask channels to (bg, GTVp, GTVn) logits.

Everything else — WaveEncoder, PromptEncoder, MaskDecoder internals,
postprocess_masks, Laplacian sharpening — is UNCHANGED.

Frozen components (as in original): PromptEncoder, MaskDecoder.
Trainable:  WaveEncoder, PseudoMaskHead3, ClassAdaptor.
"""

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make DualwaveSAM root importable
_HERE     = Path(__file__).resolve().parent
_SAM_ROOT = _HERE.parent
sys.path.insert(0, str(_SAM_ROOT))

from sam_modeling_wave.wave_encoder  import WaveEncoder
from sam_modeling_wave.mask_decoder  import MaskDecoder
from sam_modeling_wave.prompt_encoder import PromptEncoder
from sam_modeling_wave.transformer   import TwoWayTransformer


NUM_CLASSES = 3   # 0=background, 1=GTVp, 2=GTVn

# MaskDecoder channel arithmetic:
#   num_mask_tokens = num_multimask_outputs + 1  (iou-token mask + multimask)
#   multimask_output=True  → output slice(1, None) → num_multimask_outputs channels
#
# To get exactly NUM_CLASSES=3 output channels:
#   num_multimask_outputs = 3  → num_mask_tokens = 4
#   multimask_output=True      → slice(1,None) on (B,4,h,w) → (B,3,h,w)  ✓
_NUM_MULTIMASK = NUM_CLASSES   # = 3


# ── Auxiliary head (widened to 3 classes) ─────────────────────────────────────

class PseudoMaskHead3(nn.Module):
    """
    Identical to sam_model.PseudoMaskHead but outputs NUM_CLASSES channels.
    Takes image_embeddings (B, 256, h, w) → (B, NUM_CLASSES, H, W) logits.
    """

    def __init__(self, in_ch: int = 256, mid_ch: int = 64,
                 out_size: tuple = (256, 256),
                 num_classes: int = NUM_CLASSES):
        super().__init__()
        self.out_size = out_size

        self.dw    = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw    = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.norm2 = nn.GroupNorm(1, mid_ch)
        self.act   = nn.GELU()
        self.out   = nn.Conv2d(mid_ch, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        x = self.act(self.norm1(self.dw(x)))
        x = self.act(self.norm2(self.pw(x)))
        return self.out(x)   # (B, num_classes, H, W)


# ── Thin class adaptor ────────────────────────────────────────────────────────

class ClassAdaptor(nn.Module):
    """
    1×1 conv that re-maps the 3 raw MaskDecoder output channels to
    (background, GTVp, GTVn) logits.  This is the only new learned layer
    inserted into the original SAM output path.

    Input : (B, 3, H, W)  — postprocessed mask logits from MaskDecoder
    Output: (B, 3, H, W)  — class logits
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.proj = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=True)

        # Using Kaiming normal initialization to allow the optimizer 
        # to freely learn the mapping from SAM's multimask outputs 
        # to the specific biological classes.
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ── Laplacian sharpening (verbatim from sam_model.py) ─────────────────────────

def _laplacian_edge(mask_in: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
    ).view(1, 1, 3, 3).to(mask_in.device)
    # Apply per-channel (expand kernel to match channels)
    C = mask_in.shape[1]
    k = kernel.expand(C, 1, 3, 3)
    return torch.abs(F.conv2d(mask_in, k, padding=1, groups=C))


# ── Main model ─────────────────────────────────────────────────────────────────

class DualwaveSAM3Class(nn.Module):
    """
    DualwaveSAM adapted for 3-class segmentation (background / GTVp / GTVn).

    Forward pass (preserves original structure):
      x (B,2,H,W)
        → WaveEncoder            → image_embeddings (B,256,h,w)
        → PseudoMaskHead3        → aux_logits (B,3,H,W)      [auxiliary loss]
        → PromptEncoder (frozen) → sparse_emb, dense_emb
        → MaskDecoder   (frozen) → low_res_masks (B,3,h,w)   [3 = 1+2 multimask]
        → postprocess_masks      → masks (B,3,H,W)
        → ClassAdaptor           → logits (B,3,H,W)          [primary loss]

    Frozen: PromptEncoder, MaskDecoder
    Trainable: WaveEncoder, PseudoMaskHead3, ClassAdaptor
    """

    mask_threshold: float = 0.0

    def __init__(
        self,
        img_size:     int  = 256,
        n_filters:    int  = 16,
        wavelet:      str  = "haar",
        num_classes:  int  = NUM_CLASSES,
        use_aux_head: bool = True,
        use_laplacian: bool = True,
    ):
        super().__init__()

        self.img_size     = img_size
        self.num_classes  = num_classes
        self.use_aux      = use_aux_head
        self.use_laplacian = use_laplacian and use_aux_head

        # ── 1. WaveEncoder (trainable) ────────────────────────────────────
        self.image_encoder = WaveEncoder(
            in_channels=1,
            n_filters=n_filters,
            wavelet=wavelet,
        )
        enc_out_ch = 16 * n_filters   # 256 for n_filters=16

        # ── 2. Auxiliary pseudo-mask head (trainable) ─────────────────────
        if use_aux_head:
            self.pseudo_head = PseudoMaskHead3(
                in_ch=enc_out_ch,
                mid_ch=64,
                out_size=(img_size, img_size),
                num_classes=num_classes,
            )

        # ── 3. Prompt encoder (FROZEN) ────────────────────────────────────
        prompt_embed_dim     = 256
        image_embedding_size = img_size // 16

        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
        )

        # ── 4. Mask decoder (FROZEN) ──────────────────────────────────────
        # num_multimask_outputs=2 → decoder outputs 3 channels total
        # (1 iou-token mask + 2 multimask), matching num_classes=3
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=_NUM_MULTIMASK,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        # ── 5. Class adaptor (trainable) ──────────────────────────────────
        self.class_adaptor = ClassAdaptor(num_classes=num_classes)

        # Positional encoding buffer (mirrors sam_model.py)
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1), False
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1), False
        )

        # ── Freeze SAM components (Surgically) ────────────────────────────
        # 1. Keep the PromptEncoder completely frozen
        for p in self.prompt_encoder.parameters():
            p.requires_grad = False
            
        # 2. Freeze the heavy transformer to preserve SAM's edge-awareness
        for p in self.mask_decoder.transformer.parameters():
            p.requires_grad = False
            
        # 3. UNFREEZE the final rendering layers to break the hierarchy
        for p in self.mask_decoder.mask_tokens.parameters():
            p.requires_grad = True
        for p in self.mask_decoder.output_upscaling.parameters():
            p.requires_grad = True
        for p in self.mask_decoder.output_hypernetworks_mlps.parameters():
            p.requires_grad = True

        # 4. Freeze the IoU prediction head (we don't use it, saves memory)
        for p in self.mask_decoder.iou_token.parameters():
            p.requires_grad = False
        for p in self.mask_decoder.iou_prediction_head.parameters():
            p.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  DualwaveSAM3Class | trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M params")

    @property
    def device(self):
        return self.pixel_mean.device

    def postprocess_masks(
        self,
        masks:         torch.Tensor,
        input_size:    tuple,
        original_size: tuple,
    ) -> torch.Tensor:
        """Verbatim from sam_model.Sam.postprocess_masks."""
        masks = F.interpolate(
            masks, (self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks, original_size,
            mode="bilinear", align_corners=False,
        )
        return masks

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, 2, H, W)  — channel 0 = CT, channel 1 = PET

        Returns
        -------
        logits     : (B, num_classes, H, W)  — primary segmentation logits
        aux_logits : (B, num_classes, H, W) | None
        """
        B, C, H, W = x.shape
        original_size = (H, W)

        # ── Step 1: encode image ──────────────────────────────────────────
        image_embeddings = self.image_encoder(x)   # (B, 256, h, w)

        # ── Step 2: auxiliary pseudo-mask branch ──────────────────────────
        aux_logits = None
        if self.use_aux:
            aux_logits = self.pseudo_head(image_embeddings)
            if self.use_laplacian:
                edges      = _laplacian_edge(aux_logits)
                aux_logits = 0.8 * aux_logits + 0.2 * edges

        # ── Step 3: prompt encoder (no prompts — learned prior only) ──────
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None,
        )
        # prompt_encoder returns (1, 0, 256) for no prompts; expand to batch
        sparse_embeddings = sparse_embeddings.expand(B, -1, -1)
        dense_embeddings  = dense_embeddings.expand(B, -1, -1, -1)

        # ── Step 4: mask decoder ──────────────────────────────────────────
        # multimask_output=True → returns all _NUM_MULTIMASK masks
        # Together with the iou-token mask: total 3 channels
        low_res_masks, _ = self.mask_decoder(
            image_embeddings        = image_embeddings,
            image_pe                = self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings  = dense_embeddings,
            multimask_output        = True,
        )   # (B, 3, h_low, w_low)

        # ── Step 5: postprocess to original resolution ────────────────────
        masks = self.postprocess_masks(
            low_res_masks,
            input_size    = (H, W),
            original_size = original_size,
        )   # (B, 3, H, W)

        # ── Step 6: class adaptor ─────────────────────────────────────────
        logits = self.class_adaptor(masks)   # (B, 3, H, W)

        return logits, aux_logits


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DualwaveSAM3Class(img_size=256, n_filters=16).to(device)
    x      = torch.randn(2, 2, 256, 256, device=device)
    logits, aux = model(x)
    print(f"logits:     {logits.shape}")     # (2, 3, 256, 256)
    print(f"aux_logits: {aux.shape}")        # (2, 3, 256, 256)
