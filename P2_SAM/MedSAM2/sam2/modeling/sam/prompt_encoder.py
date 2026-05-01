"""
sam2/modeling/sam/prompt_encoder.py
=====================================
SAM-style prompt encoder: converts point coordinates, bounding boxes,
and dense mask inputs into sparse and dense embeddings for the mask decoder.
"""

from typing import Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.position_encoding import PositionEmbeddingRandom
from sam2.modeling.sam2_utils import LayerNorm2d


class PromptEncoder(nn.Module):
    """Encodes sparse (points, boxes) and dense (mask) prompts.

    Parameters
    ----------
    embed_dim : int              embedding dimension
    image_embedding_size : tuple (H, W) of the image embedding
    input_image_size : tuple     (H, W) of the padded input image
    mask_in_chans : int          hidden channels for the mask encoder
    activation : nn.Module
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # 4 learned embeddings: negative point, positive point, TL box corner, BR box corner.
        self.num_point_embeddings = 4
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """Positional encoding for the full image embedding grid."""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        points = points + 0.5  # shift to pixel centre
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        return self.mask_downscaling(masks)

    def _get_batch_size(self, points, boxes, masks) -> int:
        if points is not None:   return points[0].shape[0]
        if boxes is not None:    return boxes.shape[0]
        if masks is not None:    return masks.shape[0]
        return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(sparse_embeddings, dense_embeddings)``."""
        bs = self._get_batch_size(points, boxes, masks)
        sparse = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            sparse = torch.cat([sparse, self._embed_points(coords, labels, pad=(boxes is None))], dim=1)
        if boxes is not None:
            sparse = torch.cat([sparse, self._embed_boxes(boxes)], dim=1)
        dense = (
            self._embed_masks(masks) if masks is not None
            else self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        )
        return sparse, dense
