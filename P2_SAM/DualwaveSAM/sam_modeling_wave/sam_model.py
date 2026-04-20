import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple

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
        logits: [B, 1, 256, 256]
    """

    def __init__(self, in_ch=256, mid_ch=64, out_size=(256, 256)):
        super().__init__()
        self.out_size = out_size

        # Depthwise convolution for spatial refinement
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)

        # Pointwise projection
        self.pw = nn.Conv2d(in_ch, mid_ch, 1, bias=False)

        self.norm1 = nn.GroupNorm(1, in_ch)
        self.norm2 = nn.GroupNorm(1, mid_ch)
        self.act = nn.GELU()

        self.out = nn.Conv2d(mid_ch, 1, 1)

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
    SAM model:
    - encodes image via ViT backbone
    - encodes prompts (points/boxes/masks)
    - decodes segmentation masks

    Extended version includes optional pseudo-mask supervision.
    """

    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],

        # ====================================================
        # Optional auxiliary pseudo supervision components
        # ====================================================
        use_pseudo_head: bool = True,
        use_laplacian: bool = True,
    ) -> None:

        super().__init__()

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder

        # Normalization buffers (fixed statistics, not trainable)
        self.register_buffer(
            "pixel_mean",
            torch.Tensor(pixel_mean).view(-1, 1, 1),
            False,
        )
        self.register_buffer(
            "pixel_std",
            torch.Tensor(pixel_std).view(-1, 1, 1),
            False,
        )

        # ====================================================
        # Auxiliary pseudo-mask branch
        # ====================================================
        self.use_pseudo_head = use_pseudo_head
        self.use_laplacian = use_laplacian

        if use_pseudo_head:
            self.pseudo_head = PseudoMaskHead(
                in_ch=256,
                mid_ch=64,
                out_size=(256, 256),
            )

    # --------------------------------------------------------
    # Epoch tracker (used for scheduling / logging behavior)
    # --------------------------------------------------------
    def set_epoch(self, epoch: int):
        self.curr_epoch = int(epoch)

    # ============================================================
    # Edge detection utilities
    # ============================================================
    @staticmethod
    def sobel_edge_detection(mask_in: torch.Tensor) -> torch.Tensor:
        """
        Compute Sobel edge magnitude map.

        Returns:
            Edge intensity map (gradient magnitude)
        """
        sobel_x = torch.tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        sobel_y = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        sobel_x = sobel_x.to(mask_in.device)
        sobel_y = sobel_y.to(mask_in.device)

        grad_x = F.conv2d(mask_in, sobel_x, padding=1)
        grad_y = F.conv2d(mask_in, sobel_y, padding=1)

        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return edge_map

    @staticmethod
    def laplacian_edge_detection(mask_in: torch.Tensor) -> torch.Tensor:
        """
        Compute Laplacian edge map (second-order derivative).
        """
        laplacian_kernel = torch.tensor(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)

        laplacian_kernel = laplacian_kernel.to(mask_in.device)

        laplacian_map = F.conv2d(mask_in, laplacian_kernel, padding=1)

        # Absolute response emphasizes edges
        edge_map = torch.abs(laplacian_map)
        return edge_map

    # ============================================================
    # Device property
    # ============================================================
    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    # ============================================================
    # Forward pass
    # ============================================================
    def forward(
        self,
        batched_input: Dict[str, Any],
        multimask_output: bool,
    ) -> List[Dict[str, torch.Tensor]]:

        input_images = batched_input.get("image")
        image_embeddings = self.image_encoder(input_images)

        # Optional point prompts
        if "point_coords" in batched_input and batched_input["point_coords"] is not None:
            points = (
                batched_input["point_coords"],
                batched_input["point_labels"],
            )
        else:
            points = None

        # ====================================================
        # Auxiliary pseudo-mask computation branch
        # ====================================================
        if self.use_pseudo_head:
            pseudo_logits = self.pseudo_head(image_embeddings)

        if self.use_laplacian:
            laplacian_edges = self.laplacian_edge_detection(pseudo_logits)
            pseudo_logits = 0.8 * pseudo_logits + 0.2 * laplacian_edges

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        masks = self.postprocess_masks(
            low_res_masks,
            input_size=batched_input["image"].shape[-2:],
            original_size=batched_input["original_size"],
        )

        outputs = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }

        # Add auxiliary output if enabled
        if self.use_pseudo_head:
            outputs["aux_pseudo_logits"] = pseudo_logits

        return outputs

    # ============================================================
    # Mask post-processing
    # ============================================================
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:

        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        masks = masks[..., : input_size[0], : input_size[1]]

        masks = F.interpolate(
            masks,
            original_size,
            mode="bilinear",
            align_corners=False,
        )

        return masks

    # ============================================================
    # Preprocessing
    # ============================================================
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input image and pad to square resolution.
        """

        # Channel-wise normalization
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad to square size required by encoder
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w

        x = F.pad(x, (0, padw, 0, padh))
        return x
