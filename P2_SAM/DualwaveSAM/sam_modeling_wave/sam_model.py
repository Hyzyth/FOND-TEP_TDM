
import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder



############## hjx ##############
class PseudoMaskHead(nn.Module):
    def __init__(self, in_ch=256, mid_ch=64, out_size=(256,256)):
        super().__init__()
        self.out_size = out_size
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.norm1 = nn.GroupNorm(1, in_ch)
        self.norm2 = nn.GroupNorm(1, mid_ch)
        self.act = nn.GELU()
        self.out = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x):                          # x: [B,256,h,w]
        x = F.interpolate(x, size=self.out_size, mode="bilinear", align_corners=False)
        x = self.act(self.norm1(self.dw(x)))
        x = self.act(self.norm2(self.pw(x)))
        return self.out(x)                         # logits [B,1,256,256]
############## hjx ##############
    




class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],

        ############## hjx ##############
        # === pseudo 相关 ===
        use_pseudo_head=True,
        use_laplacian=True,                # 是否使用 laplacian 边缘算子
        ############## hjx ##############

    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    ############## hjx ##############
        # pseudo
        self.use_pseudo_head = use_pseudo_head
        self.use_laplacian = use_laplacian
        if use_pseudo_head:
            self.pseudo_head = PseudoMaskHead(in_ch=256, mid_ch=64, out_size=(256,256))

    def set_epoch(self, epoch: int):
        self.curr_epoch = int(epoch)


    @staticmethod
    def sobel_edge_detection(mask_in: torch.Tensor) -> torch.Tensor:
        # Sobel 滤波器（水平和垂直方向）
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 将 Sobel 滤波器应用于输入图像
        sobel_x = sobel_x.to(mask_in.device)
        sobel_y = sobel_y.to(mask_in.device)

        # 卷积操作，计算边缘（需要扩展维度）
        grad_x = F.conv2d(mask_in, sobel_x, padding=1)
        grad_y = F.conv2d(mask_in, sobel_y, padding=1)

        # 计算边缘强度
        edge_map = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # 边缘强度
        return edge_map

    @staticmethod
    def laplacian_edge_detection(mask_in: torch.Tensor) -> torch.Tensor:
        # Laplacian 滤波器（相当于第二阶导数）
        laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 将 Laplacian 滤波器应用于输入图像
        laplacian_kernel = laplacian_kernel.to(mask_in.device)

        # 卷积操作，计算 Laplacian 边缘
        laplacian_map = F.conv2d(mask_in, laplacian_kernel, padding=1)

        # 对 Laplacian 结果进行绝对值处理，突出边缘
        edge_map = torch.abs(laplacian_map)
        return edge_map

    
    ############## hjx ##############


    @property
    def device(self) -> Any:
        return self.pixel_mean.device
    

 
    def forward(self, batched_input: Dict[str, Any], multimask_output: bool) -> List[Dict[str, torch.Tensor]]:

        input_images = batched_input.get("image")
        image_embeddings = self.image_encoder(input_images)

        if "point_coords" in batched_input and batched_input["point_coords"] != None:
            points = (batched_input["point_coords"], batched_input["point_labels"])
        else:
            points = None


        ############## hjx ##############
            
        if self.use_pseudo_head:
            pseudo_logits = self.pseudo_head(image_embeddings)      # [B,1,256,256]

        if self.use_laplacian:
            laplacian_edges = self.laplacian_edge_detection(pseudo_logits)
            pseudo_logits = 0.8*pseudo_logits + 0.2*laplacian_edges

        ############## hjx ##############
                

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=None, # batched_input.get("boxes", None),
            masks=None # batched_input.get("mask_inputs", None),
        )  # sparse_embeddings:[2, 0, 256],  dense_embeddings:[2, 256, 64, 64]

        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),  # 1x(256)x(64)x(64)
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
        
        ############## hjx ##############
        if self.use_pseudo_head:
            outputs["aux_pseudo_logits"] = pseudo_logits
        ############## hjx ##############

        return outputs

    def postprocess_masks(self,masks: torch.Tensor, input_size: Tuple[int, ...],original_size: Tuple[int, ...],) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size), mode="bilinear", align_corners=False,)  #[1,1024,1024]

        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
