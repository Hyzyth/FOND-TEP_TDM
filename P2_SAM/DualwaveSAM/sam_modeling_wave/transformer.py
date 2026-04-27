# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import math
from typing import Tuple, Type

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .common import MLPBlock


# ============================================================
# Two-way Transformer
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
        Two-way transformer: alternates attention between sparse query
        tokens (points/masks) and dense image embeddings.
        """
        super().__init__()

        self.depth         = depth
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.mlp_dim       = mlp_dim

        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
            )
            for i in range(depth)
        ])

        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

        # CHANGE: replaced the hardcoded Conv1d(256, 4096, 1) with a lazily-built
        # projection so this works at any resolution, not just 64x64 (1024px input).
        # The correct target length is derived from the actual key tensor at runtime.
        self._pe_proj_cache: dict = {}   # maps (src_len, tgt_len) → Conv1d module

    def _get_pe_proj(self, src_len: int, tgt_len: int, device, dtype) -> nn.Module:
        """
        Return (and cache) a Conv1d that maps src_len → tgt_len tokens.
        Created lazily so any encoder resolution is handled automatically.
        """
        key = (src_len, tgt_len)
        if key not in self._pe_proj_cache:
            proj = nn.Conv1d(src_len, tgt_len, kernel_size=1).to(device=device, dtype=dtype)
            self._pe_proj_cache[key] = proj
        return self._pe_proj_cache[key]

    def forward(
        self,
        image_embedding: Tensor,
        image_pe:        Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # B,(HW),C
        image_pe        = image_pe.flatten(2).permute(0, 2, 1)         # B,(HW),C

        queries = point_embedding
        keys    = image_embedding

        for layer in self.layers:
            queries, keys = layer(
                queries=queries, keys=keys,
                query_pe=point_embedding, key_pe=image_pe,
            )

        q = queries + point_embedding

        # CHANGE: dynamic PE projection instead of hardcoded 4096-output Conv1d.
        # Handles 16x16 (WaveEncoder / 256px) and 64x64 (ViT / 1024px) equally.
        if image_pe.shape[1] != keys.shape[1]:
            proj     = self._get_pe_proj(image_pe.shape[1], keys.shape[1],
                                          device=image_pe.device, dtype=image_pe.dtype)
            image_pe = proj(image_pe)

        k        = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries  = self.norm_final_attn(queries + attn_out)

        return queries, keys


# ============================================================
# Two-way attention block
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
        super().__init__()

        self.self_attn  = Attention(embedding_dim, num_heads)
        self.norm1      = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp   = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)

        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.skip_first_layer_pe = skip_first_layer_pe

        # CHANGE: replaced hardcoded Conv1d(256, 4096, 1) with lazy cache (same
        # fix as TwoWayTransformer) so every block adapts to the actual key length.
        self._key_pe_proj_cache: dict = {}

    def _get_key_pe_proj(self, src_len: int, tgt_len: int, device, dtype) -> nn.Module:
        key = (src_len, tgt_len)
        if key not in self._key_pe_proj_cache:
            proj = nn.Conv1d(src_len, tgt_len, kernel_size=1).to(device=device, dtype=dtype)
            self._key_pe_proj_cache[key] = proj
        return self._key_pe_proj_cache[key]

    def forward(
        self,
        queries:  Tensor,
        keys:     Tensor,
        query_pe: Tensor,
        key_pe:   Tensor,
    ) -> Tuple[Tensor, Tensor]:

        # 1. Self-attention
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q        = queries + query_pe
            queries  = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)

        # 2. Cross-attention: tokens → image
        q = queries + query_pe

        # CHANGE: dynamic PE projection (replaces hardcoded 4096)
        if key_pe.shape[1] != keys.shape[1]:
            proj   = self._get_key_pe_proj(key_pe.shape[1], keys.shape[1],
                                            device=key_pe.device, dtype=key_pe.dtype)
            key_pe = proj(key_pe)

        k        = keys + key_pe
        queries  = self.norm2(queries + self.cross_attn_token_to_image(q=q, k=k, v=keys))

        # 3. MLP
        queries = self.norm3(queries + self.mlp(queries))

        # 4. Cross-attention: image ← tokens
        q      = queries + query_pe
        k      = keys    + key_pe
        keys   = self.norm4(keys + self.cross_attn_image_to_token(q=k, k=q, v=queries))

        return queries, keys


# ============================================================
# Multi-head attention
# ============================================================
class Attention(nn.Module):
    """Multi-head attention with optional dimensional downsampling."""

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim  = embedding_dim // downsample_rate
        self.num_heads     = num_heads
        assert self.internal_dim % num_heads == 0

        self.q_proj   = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj   = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj   = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, num_heads, c // num_heads).transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        return x.transpose(1, 2).reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q.to(self.q_proj.weight.dtype))
        k = self.k_proj(k.to(self.k_proj.weight.dtype))
        v = self.v_proj(v.to(self.v_proj.weight.dtype))

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        _, _, _, c_per_head = q.shape
        attn = (q @ k.permute(0, 1, 3, 2)) / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        out = self._recombine_heads(attn @ v)
        return self.out_proj(out)
