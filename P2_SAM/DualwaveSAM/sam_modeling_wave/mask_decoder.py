# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under the LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 10,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Mask decoder module for SAM-style segmentation.

        This module predicts segmentation masks conditioned on:
        - image embeddings
        - sparse prompts (points / boxes)
        - dense prompt embeddings

        It uses a transformer to fuse information and a hypernetwork
        to generate pixel-wise masks.

        Args:
            transformer_dim: hidden dimension of transformer tokens
            transformer: transformer module used for reasoning
            num_multimask_outputs: number of candidate masks per input
            activation: activation function used in upscaling blocks
            iou_head_depth: depth of IoU prediction MLP
            iou_head_hidden_dim: hidden size of IoU prediction MLP
        """
        super().__init__()

        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        # Token representing IoU (mask quality estimation)
        self.iou_token = nn.Embedding(1, transformer_dim)

        # One token per mask + one IoU token
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # Upsampling decoder to restore spatial resolution
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )

        # Hypernetwork generating mask-specific weights
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )

        # IoU prediction head (mask quality estimation)
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,              # [B, 256, 64, 64]
        image_pe: torch.Tensor,                     # [1, 256, 64, 64]
        sparse_prompt_embeddings: torch.Tensor,     # [B, 3, 256]
        dense_prompt_embeddings: torch.Tensor,      # [B, 256, 64, 64]
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for mask prediction.

        Args:
            image_embeddings: encoder output features
            image_pe: positional encoding matching image features
            sparse_prompt_embeddings: point/box embeddings
            dense_prompt_embeddings: mask prompt embeddings
            multimask_output: whether to return multiple candidate masks

        Returns:
            masks: predicted segmentation masks
            iou_pred: predicted mask quality scores
        """

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select mask subset depending on inference mode
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Internal mask prediction pipeline using transformer + hypernetwork.

        Steps:
        1. Build output tokens (IoU + mask tokens)
        2. Fuse with prompt embeddings
        3. Run transformer
        4. Decode spatial masks via hypernetwork
        5. Predict IoU scores
        """

        # Concatenate IoU token and mask tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )  # [1+K, C]

        # Expand tokens for batch
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )

        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Fuse image embeddings with dense prompts
        src = image_embeddings
        src = src + dense_prompt_embeddings

        # Expand positional encoding
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)

        b, c, h, w = src.shape

        # Transformer reasoning stage
        hs, src = self.transformer(src, pos_src, tokens)

        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Decode spatial features
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)

        # Hypernetwork per mask token
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )

        hyper_in = torch.stack(hyper_in_list, dim=1)  # [B, K, C]

        b, c, h, w = upscaled_embedding.shape

        # Produce final masks
        masks = (
            hyper_in @ upscaled_embedding.view(b, c, h * w)
        ).view(b, -1, h, w)

        # Predict IoU (mask quality)
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# ============================================================
# Lightweight MLP used in hypernetworks and IoU prediction
# ============================================================
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        """
        Simple multi-layer perceptron.

        Used for:
        - hypernetwork mask generation
        - IoU quality prediction

        Args:
            input_dim: input feature size
            hidden_dim: hidden layer size
            output_dim: output feature size
            num_layers: number of linear layers
            sigmoid_output: apply sigmoid activation at output
        """
        super().__init__()

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)

        self.layers = nn.ModuleList(
            nn.Linear(n, k)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

        self.sigmoid_output = sigmoid_output
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        """
        Forward pass through MLP.

        Note:
        - Hidden layers use ReLU activation
        - Final layer is linear (optionally sigmoid)
        """

        for i, layer in enumerate(self.layers):
            if i < self.num_layers - 1:
                x = F.relu(layer(x))
            else:
                x = layer(x)

        if self.sigmoid_output:
            x = F.sigmoid(x)

        return x
