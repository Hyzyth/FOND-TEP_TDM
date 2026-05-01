"""
sam2/modeling/sam/transformer.py
==================================
Two-way transformer and attention modules for the SAM mask decoder,
including RoPE-augmented attention used in the memory cross-attention.
"""

import contextlib
import math
import warnings
from functools import partial
from typing import Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from sam2.modeling.position_encoding import apply_rotary_enc, compute_axial_cis
from sam2.modeling.sam2_utils import MLP
from sam2.utils.misc import get_sdpa_settings

warnings.simplefilter(action="ignore", category=FutureWarning)

OLD_GPU, USE_FLASH_ATTN, MATH_KERNEL_ON = get_sdpa_settings()
ALLOW_ALL_KERNELS = False


def sdp_kernel_context(dropout_p: float):
    """Select the best available scaled-dot-product attention kernel."""
    if ALLOW_ALL_KERNELS:
        return contextlib.nullcontext()
    return torch.backends.cuda.sdp_kernel(
        enable_flash=USE_FLASH_ATTN,
        enable_math=(OLD_GPU and dropout_p > 0.0) or MATH_KERNEL_ON,
        enable_mem_efficient=OLD_GPU,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Two-way transformer
# ──────────────────────────────────────────────────────────────────────────────

class TwoWayTransformer(nn.Module):
    """Transformer decoder that bi-directionally attends between point tokens
    and image embeddings.

    Parameters
    ----------
    depth : int               number of TwoWayAttentionBlock layers
    embedding_dim : int
    num_heads : int
    mlp_dim : int
    activation : nn.Module
    attention_downsample_rate : int
    """

    def __init__(self, depth, embedding_dim, num_heads, mlp_dim,
                 activation=nn.ReLU, attention_downsample_rate=2) -> None:
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList([
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim, num_heads=num_heads,
                mlp_dim=mlp_dim, activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(i == 0),
            )
            for i in range(depth)
        ])
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: Tensor, image_pe: Tensor,
                point_embedding: Tensor) -> Tuple[Tensor, Tensor]:
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries, keys = point_embedding, image_embedding
        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys,
                                  query_pe=point_embedding, key_pe=image_pe)
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm_final_attn(queries + attn_out)
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    """One block of the two-way transformer: SA → CA(tokens→img) → MLP → CA(img→tokens)."""

    def __init__(self, embedding_dim, num_heads, mlp_dim=2048, activation=nn.ReLU,
                 attention_downsample_rate=2, skip_first_layer_pe=False) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLP(embedding_dim, mlp_dim, embedding_dim, num_layers=2, activation=activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries, keys, query_pe, key_pe):
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            queries = queries + self.self_attn(q=q, k=q, v=queries)
        queries = self.norm1(queries)
        q = queries + query_pe
        k = keys + key_pe
        queries = queries + self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm2(queries)
        queries = queries + self.mlp(queries)
        queries = self.norm3(queries)
        q = queries + query_pe
        k = keys + key_pe
        keys = keys + self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = self.norm4(keys)
        return queries, keys


# ──────────────────────────────────────────────────────────────────────────────
# Attention
# ──────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    """Multi-head attention with optional embedding downscaling.

    Parameters
    ----------
    embedding_dim : int
    num_heads : int
    downsample_rate : int  internal dimension = embedding_dim // downsample_rate
    dropout : float
    kv_in_dim : int or None  key/value input dimension (None = embedding_dim)
    """

    def __init__(self, embedding_dim, num_heads, downsample_rate=1,
                 dropout=0.0, kv_in_dim=None) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.kv_in_dim = kv_in_dim if kv_in_dim is not None else embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        self.dropout_p = dropout

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        return x.reshape(b, n, num_heads, c // num_heads).transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        return x.transpose(1, 2).reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self._separate_heads(self.q_proj(q), self.num_heads)
        k = self._separate_heads(self.k_proj(k), self.num_heads)
        v = self._separate_heads(self.v_proj(v), self.num_heads)
        dropout_p = self.dropout_p if self.training else 0.0
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            warnings.warn(f"Flash Attention failed: {e}\nFalling back to all kernels.", UserWarning, stacklevel=2)
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        return self.out_proj(self._recombine_heads(out))


class RoPEAttention(Attention):
    """Attention with Rotary Position Encoding.

    Additional parameters
    ---------------------
    rope_theta : float
    rope_k_repeat : bool  repeat Q frequencies to match longer K sequences
    feat_sizes : tuple    (W, H) for stride-16 feature maps at target resolution
    """

    def __init__(self, *args, rope_theta=10000.0, rope_k_repeat=False,
                 feat_sizes=(32, 32), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compute_cis = partial(
            compute_axial_cis, dim=self.internal_dim // self.num_heads, theta=rope_theta
        )
        self.freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.rope_k_repeat = rope_k_repeat

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                num_k_exclude_rope: int = 0) -> Tensor:
        q = self._separate_heads(self.q_proj(q), self.num_heads)
        k = self._separate_heads(self.k_proj(k), self.num_heads)
        v = self._separate_heads(self.v_proj(v), self.num_heads)
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q, k[:, :, :num_k_rope], freqs_cis=self.freqs_cis,
            repeat_freqs_k=self.rope_k_repeat,
        )
        dropout_p = self.dropout_p if self.training else 0.0
        try:
            with sdp_kernel_context(dropout_p):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        except Exception as e:
            warnings.warn(f"Flash Attention failed: {e}\nFalling back.", UserWarning, stacklevel=2)
            global ALLOW_ALL_KERNELS
            ALLOW_ALL_KERNELS = True
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        return self.out_proj(self._recombine_heads(out))
