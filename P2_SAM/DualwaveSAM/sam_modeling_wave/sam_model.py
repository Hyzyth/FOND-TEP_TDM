import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Optional, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder


# ============================================================
# Auxiliary pseudo mask prediction head
# ============================================================
class PseudoMaskHead(nn.Module):
    """
    Lightweight decoder head that upsamples image embeddings
    and produces a pseudo segmentation mask.

    Input:
        x: feature map [B, 256, h, w]

    Output:
        logits: [B, 1, out_size, out_size]
    """

    def __init__(self, in_ch=256, mid_ch=64, out_size=(256, 256)):
        super().__init__()
        self.out_size = out_size

        self.dw   = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw   = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.norm2 = nn.GroupNorm(1, mid_ch)
        self.act  = nn.GELU()
        self.out  = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        x = self.act(self.norm1(self.dw(x)))
        x = self.act(self.norm2(self.pw(x)))
        return self.out(x)


# ============================================================
# Segment Anything Model (SAM)
# ============================================================
class Sam(nn.Module):
    """
    SAM model with optional pseudo-mask supervision branch.

    Args:
        image_encoder:   backbone returning [B, C, H, W] embeddings
        prompt_encoder:  PromptEncoder
        mask_decoder:    MaskDecoder
        pixel_mean/std:  image normalisation statistics
        img_size:        input resolution (required for WaveEncoder compatibility)
        use_pseudo_head: attach auxiliary PseudoMaskHead branch
        use_laplacian:   apply Laplacian sharpening to pseudo logits
    """

    mask_threshold: float = 0.0
    image_format:   str   = "RGB"

    def __init__(
        self,
        image_encoder,
        prompt_encoder: PromptEncoder,
        mask_decoder:   MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std:  List[float] = [58.395, 57.12, 57.375],
        img_size:   int = 1024,
        use_pseudo_head: bool = True,
        use_laplacian:   bool = True,
    ) -> None:
        super().__init__()

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder   = mask_decoder
        self.img_size       = img_size

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std",  torch.Tensor(pixel_std).view(-1, 1, 1),  False)

        self.use_pseudo_head = use_pseudo_head
        self.use_laplacian   = use_laplacian and use_pseudo_head

        if use_pseudo_head:
            self.pseudo_head = PseudoMaskHead(
                in_ch=256, mid_ch=64, out_size=(img_size, img_size)
            )

    def set_epoch(self, epoch: int):
        self.curr_epoch = int(epoch)

    # ============================================================
    # Edge detection utilities
    # ============================================================
    @staticmethod
    def sobel_edge_detection(mask_in: torch.Tensor) -> torch.Tensor:
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(mask_in.device)
        sobel_y = torch.tensor(
            [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(mask_in.device)
        return torch.sqrt(F.conv2d(mask_in, sobel_x, padding=1) ** 2
                        + F.conv2d(mask_in, sobel_y, padding=1) ** 2)

    @staticmethod
    def laplacian_edge_detection(mask_in: torch.Tensor) -> torch.Tensor:
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0).to(mask_in.device)
        return torch.abs(F.conv2d(mask_in, kernel, padding=1))

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # ============================================================
    # Forward
    # ============================================================
    def forward(
        self,
        batched_input: Dict[str, Any],
        multimask_output: bool,
    ) -> Dict[str, torch.Tensor]:

        input_images     = batched_input.get("image")
        image_embeddings = self.image_encoder(input_images)

        # MODIFICATION: read all prompt types from the input dict so that
        # points, boxes, and mask prompts are actually forwarded to the
        # prompt encoder (previously only point_coords was read, boxes/masks
        # were silently dropped).
        point_coords = batched_input.get("point_coords")
        point_labels = batched_input.get("point_labels")
        boxes        = batched_input.get("boxes")
        masks        = batched_input.get("masks")

        points = (point_coords, point_labels) if point_coords is not None else None

        # Auxiliary pseudo-mask branch
        pseudo_logits = None
        if self.use_pseudo_head:
            pseudo_logits = self.pseudo_head(image_embeddings)
            if self.use_laplacian:
                laplacian_edges = self.laplacian_edge_detection(pseudo_logits)
                pseudo_logits   = 0.8 * pseudo_logits + 0.2 * laplacian_edges

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,    # MODIFICATION: was hardcoded None
            masks=masks,    # MODIFICATION: was hardcoded None
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings       = image_embeddings,
            image_pe               = self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings  = dense_embeddings,
            multimask_output       = multimask_output,
        )

        masks = self.postprocess_masks(
            low_res_masks,
            input_size    = batched_input["image"].shape[-2:],
            original_size = batched_input["original_size"],
        )

        outputs = {
            "masks":           masks,
            "iou_predictions": iou_predictions,
            "low_res_logits":  low_res_masks,
        }

        if pseudo_logits is not None:
            outputs["aux_pseudo_logits"] = pseudo_logits

        return outputs

    # ============================================================
    # Post-processing
    # ============================================================
    def postprocess_masks(
        self,
        masks:         torch.Tensor,
        input_size:    Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks, (self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, original_size,
            mode="bilinear", align_corners=False,
        )
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise and pad input to square resolution."""
        x    = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x    = F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
        return x
