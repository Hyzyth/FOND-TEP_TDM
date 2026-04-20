# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Licensed under the license found in the LICENSE file at the repository root.

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock


# NOTE:
# This implementation and helper functions are lightly adapted from ViTDet:
# https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py  # noqa


# ============================================================
# Adapter module (channel + spatial refinement)
# ============================================================
class Adapter_Layer(nn.Module):
    """
    Lightweight adapter module inserted into transformer blocks.

    Combines:
    - Channel-wise gating (MLP-based squeeze/excitation style)
    - Spatial refinement (strided conv + deconv reconstruction)

    Designed to enhance feature adaptation without modifying backbone weights.
    """

    def __init__(self, embed_dim, mlp_ratio=0.25, norm_layer=nn.LayerNorm, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect

        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm = norm_layer(embed_dim)

        # Global context pooling for channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Channel attention branch
        self.channel = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim, bias=False),
            nn.Sigmoid(),
        )

        # Spatial refinement branch (downsample + upsample)
        self.spatial = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(),
        )

        # Kaiming initialization for convolution/linear layers
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        # Input: (B, H, W, C)
        # Convert to NCHW for conv operations
        x = x.permute(0, 3, 1, 2)

        B, C, _, _ = x.size()

        # Channel attention
        x_channel = self.channel(self.avg_pool(x).view(B, C)).view(B, C, 1, 1) * x

        # Spatial refinement
        x_spatial = self.spatial(x_channel)

        # Residual or replacement behavior
        if self.skip_connect:
            x = x + x_spatial
        else:
            x = x_spatial

        # Back to NHWC format
        x = x.permute(0, 2, 3, 1)

        return self.norm(x)


# ============================================================
# Vision Transformer Encoder (SAM backbone variant)
# ============================================================
class ImageEncoderViT(nn.Module):
    """
    Vision Transformer-based image encoder used in SAM.

    Produces dense feature maps from input images using:
    - Patch embedding
    - Positional encoding (optional absolute)
    - Transformer blocks
    - Convolutional projection neck
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        adapter_train=False,
    ) -> None:
        super().__init__()

        self.img_size = img_size

        # Patch embedding (image → token grid)
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # Absolute positional embedding (optional)
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=(img_size // patch_size, img_size // patch_size),
                    adapter=adapter_train,
                )
            )

        # Feature projection head (neck)
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)

        # Add positional embedding if enabled
        if self.pos_embed is not None:
            x = x + self.pos_embed

        # Transformer encoding
        for blk in self.blocks:
            x = blk(x)

        # Convert to NCHW for CNN neck
        x = self.neck(x.permute(0, 3, 1, 2))

        return x


# ============================================================
# Transformer block (window + adapter support)
# ============================================================
class Block(nn.Module):
    """
    Transformer block supporting:
    - Global / window attention
    - MLP feed-forward
    - Optional adapter branch
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        adapter: bool = False,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.adapter = adapter

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size

        # Optional adapter branch
        if self.adapter:
            self.Adapter = Adapter_Layer(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)

        # Window attention partitioning
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)

        # Reverse window partitioning
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        # First residual connection (attention)
        x = shortcut + x

        # Feed-forward + optional adapter
        if self.adapter:
            x_norm = self.norm2(x)
            x = x + self.mlp(x_norm) + self.Adapter(x_norm)
        else:
            x = x + self.mlp(self.norm2(x))

        return x


# ============================================================
# Multi-head Attention with relative position encoding
# ============================================================
class Attention(nn.Module):
    """Multi-head self-attention with optional relative positional encoding."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos

        # Relative position embeddings
        if self.use_rel_pos:
            assert input_size is not None, "Input size required for relative position encoding."
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape

        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(
                attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W)
            )

        attn = attn.softmax(dim=-1)

        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


# ============================================================
# Window partition utilities
# ============================================================
def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Split feature map into non-overlapping windows (with padding if needed).
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)

    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor,
    window_size: int,
    pad_hw: Tuple[int, int],
    hw: Tuple[int, int],
) -> torch.Tensor:
    """
    Reconstruct feature map from windowed representation and remove padding.
    """
    Hp, Wp = pad_hw
    H, W = hw

    B = windows.shape[0] // (Hp * Wp // window_size // window_size)

    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()

    return x


# ============================================================
# Relative position utilities
# ============================================================
def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Extract relative positional embeddings for given query/key sizes.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)

    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)

    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Add decomposed 2D relative positional bias (from MViTv2).

    Reference:
    https://github.com/facebookresearch/mvit/blob/main/mvit/models/attention.py  # noqa
    """
    q_h, q_w = q_size
    k_h, k_w = k_size

    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)

    # Mixed precision compatibility workaround
    r_q = r_q.to(Rh.dtype)

    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, None]
        + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


# ============================================================
# Patch embedding layer
# ============================================================
class PatchEmbed(nn.Module):
    """Converts image into patch embeddings using a convolutional projection."""

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # Convert NCHW → NHWC for transformer blocks
        x = x.permute(0, 2, 3, 1)
        return x
