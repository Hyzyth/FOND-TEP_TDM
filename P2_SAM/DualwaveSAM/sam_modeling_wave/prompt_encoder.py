# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn
from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    """
    Encodes prompts (points, boxes, masks) for SAM mask decoder input.
    Produces:
    - sparse embeddings (points + boxes)
    - dense embeddings (masks or learned default)
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Args:
            embed_dim: embedding dimension for prompt tokens
            image_embedding_size: (H, W) size of image encoder output
            input_image_size: padded input image size (H, W)
            mask_in_chans: hidden channels for mask encoding
            activation: activation function used in mask encoder
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size

        # Random positional encoding module
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # 4 types of point embeddings:
        # positive point, negative point, and 2 box corners
        self.num_point_embeddings: int = 4
        self.point_embeddings = nn.ModuleList(
            [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        )

        # Embedding used for invalid / padding points
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Expected mask input resolution after downsampling
        self.mask_input_size = (
            4 * image_embedding_size[0],
            4 * image_embedding_size[1],
        )

        # Convolutional mask encoder (downsampling + projection)
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

        # Default embedding when no mask is provided
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns dense positional encoding used for mask decoding.

        Shape:
            1 x C x H x W
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool,
    ) -> torch.Tensor:
        """
        Encode point prompts into embeddings.

        Args:
            points: (B, N, 2) coordinates
            labels: (B, N) point labels (-1 ignore, 0 neg, 1 pos)
            pad: whether to add padding token

        Returns:
            (B, N(+1), embed_dim)
        """

        # Shift coordinates to pixel center
        points = points + 0.5

        if pad:
            padding_point = torch.zeros(
                (points.shape[0], 1, 2), device=points.device
            )
            padding_label = -torch.ones(
                (labels.shape[0], 1), device=labels.device
            )

            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)

        # Generate positional encoding for each point
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size
        )

        # Zero out ignored points
        point_embedding[labels == -1] = 0.0

        # Ensure embedding dtype consistency (kept as-is, although redundant)
        self.not_a_point_embed.weight = torch.nn.Parameter(
            self.not_a_point_embed.weight.to(point_embedding.dtype),
            requires_grad=True,
        )
        self.point_embeddings[0].weight = torch.nn.Parameter(
            self.point_embeddings[0].weight.to(point_embedding.dtype),
            requires_grad=True,
        )
        self.point_embeddings[1].weight = torch.nn.Parameter(
            self.point_embeddings[1].weight.to(point_embedding.dtype),
            requires_grad=True,
        )

        # Add learned embeddings depending on label type
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight

        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Encode bounding box prompts into embeddings.

        Args:
            boxes: (B, 2, 2) box coordinates

        Returns:
            (B, 2, embed_dim)
        """

        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)

        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size
        )

        # First corner = top-left, second = bottom-right
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight

        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Encode mask prompts via convolutional downsampling.

        Args:
            masks: (B, 1, H, W)

        Returns:
            (B, embed_dim, H', W')
        """
        return self.mask_downscaling(masks)

    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """
        Infer batch size from available prompts.
        """
        if points is not None:
            return points[0].shape[0]
        if boxes is not None:
            return boxes.shape[0]
        if masks is not None:
            return masks.shape[0]
        return 1

    def _get_device(self) -> torch.device:
        """Return device of prompt embeddings."""
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode all prompt types into:
        - sparse embeddings (points + boxes)
        - dense embeddings (masks or default embedding)

        Returns:
            sparse_embeddings: (B, N, embed_dim)
            dense_embeddings: (B, embed_dim, H, W)
        """

        bs = self._get_batch_size(points, boxes, masks)

        # Empty sparse tensor as base container
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim),
            device=self._get_device(),
        )

        # Encode point prompts
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(
                coords, labels, pad=(boxes is None)
            )
            sparse_embeddings = torch.cat(
                [sparse_embeddings, point_embeddings], dim=1
            )

        # Encode box prompts
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat(
                [sparse_embeddings, box_embeddings], dim=1
            )

        # Encode mask prompts or use default embedding
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(
                1, -1, 1, 1
            ).expand(
                bs,
                -1,
                self.image_embedding_size[0],
                self.image_embedding_size[1],
            )

        return sparse_embeddings, dense_embeddings


# ============================================================
# Random Fourier positional encoding
# ============================================================
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random Gaussian Fourier features.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        if scale is None or scale <= 0.0:
            scale = 1.0

        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature mapping to normalized coordinates.

        Args:
            coords: (..., 2) normalized in [0, 1]

        Returns:
            (..., C) positional embedding
        """

        # Map [0,1] → [-1,1]
        coords = 2 * coords - 1

        # Random Fourier projection
        coords = coords @ self.positional_encoding_gaussian_matrix.to(torch.float32)

        coords = 2 * np.pi * coords

        return torch.cat(
            [torch.sin(coords), torch.cos(coords)],
            dim=-1,
        )

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate dense positional encoding grid.

        Args:
            size: (H, W)

        Returns:
            (C, H, W)
        """
        h, w = size

        device: Any = self.positional_encoding_gaussian_matrix.device

        grid = torch.ones((h, w), device=device, dtype=torch.float32)

        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5

        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(
        self,
        coords_input: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Encode absolute pixel coordinates into positional embeddings.

        Args:
            coords_input: (B, N, 2) pixel coordinates
            image_size: (H, W)

        Returns:
            (B, N, C)
        """

        coords = coords_input.clone()

        # Normalize coordinates to [0,1]
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]

        return self._pe_encoding(coords.to(torch.float))
