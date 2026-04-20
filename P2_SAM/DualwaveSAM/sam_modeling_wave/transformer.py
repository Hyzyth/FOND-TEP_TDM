# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Licensed under LICENSE in the root directory of this source tree.

import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import MLPBlock


# ============================================================
# Two-way Transformer (cross-attention between tokens & image)
# ============================================================
class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        Two-way transformer used in segmentation decoders.

        It alternates attention between:
        - sparse query tokens (points/masks)
        - dense image embeddings

        Args:
            depth: number of transformer blocks
            embedding_dim: token feature dimension
            num_heads: attention heads (must divide embedding_dim)
            mlp_dim: hidden dimension of MLP block
            activation: activation function for MLP
        """
        super().__init__()

        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim

        # Stack of two-way attention blocks
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )

        # Final token-to-image attention refinement
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

        # ===== hjx modification: positional encoding projection =====
        self.conv_image_pe = nn.Conv1d(256, 4096, kernel_size=1)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of two-way transformer.

        Args:
            image_embedding: B x C x H x W feature map
            image_pe: positional encoding matching image_embedding
            point_embedding: sparse token embeddings (B x N x C)

        Returns:
            updated point embeddings
            updated image embeddings (flattened tokens)
        """

        # Flatten spatial image features: BxCxHxW -> Bx(HW)xC
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        queries = point_embedding
        keys = image_embedding

        # Stack of transformer blocks
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # Final token refinement
        q = queries + point_embedding

        # ===== hjx: handle mismatched positional encoding length =====
        if image_pe.shape[1] != keys.shape[1]:
            image_pe = self.conv_image_pe(image_pe)

        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)

        queries = self.norm_final_attn(queries + attn_out)

        return queries, keys


# ============================================================
# Two-way attention block (sparse ↔ dense interaction)
# ============================================================
class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        Transformer block with:
        1. self-attention (tokens)
        2. cross-attention (tokens -> image)
        3. MLP refinement
        4. cross-attention (image -> tokens)
        """
        super().__init__()

        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

        # ===== hjx modification: positional encoding projection =====
        self.conv_key_pe = nn.Conv1d(256, 4096, kernel_size=1)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_pe: Tensor,
        key_pe: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        # -------------------------
        # 1. Self-attention block
        # -------------------------
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out

        queries = self.norm1(queries)

        # ---------------------------------------
        # 2. Cross-attention (tokens → image)
        # ---------------------------------------
        q = queries + query_pe

        # ===== hjx: adapt positional encoding if needed =====
        if key_pe.shape[1] != keys.shape[1]:
            key_pe = self.conv_key_pe(key_pe)

        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)

        queries = self.norm2(queries + attn_out)

        # -------------------------
        # 3. MLP block
        # -------------------------
        queries = self.norm3(queries + self.mlp(queries))

        # ---------------------------------------
        # 4. Cross-attention (image ← tokens)
        # ---------------------------------------
        q = queries + query_pe
        k = keys + key_pe

        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = self.norm4(keys + attn_out)

        return queries, keys


# ============================================================
# Multi-head attention module
# ============================================================
class Attention(nn.Module):
    """
    Multi-head attention with optional dimensional downsampling.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads

        assert self.internal_dim % num_heads == 0, \
            "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    # -------------------------
    # Split into attention heads
    # -------------------------
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    # -------------------------
    # Merge attention heads
    # -------------------------
    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """
        Standard scaled dot-product multi-head attention.
        """

        # Input projections (type-aligned for stability)
        q = self.q_proj(q.to(self.q_proj.weight.dtype))  # NOTE: dtype cast
        k = self.k_proj(k.to(self.k_proj.weight.dtype))  # NOTE: dtype cast
        v = self.v_proj(v.to(self.v_proj.weight.dtype))  # NOTE: dtype cast

        # q = self.q_proj(q)
        # k = self.k_proj(k)
        # v = self.v_proj(v)
        # (alternative implementation disabled intentionally)

        # Split into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Scaled dot-product attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Weighted aggregation
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out
