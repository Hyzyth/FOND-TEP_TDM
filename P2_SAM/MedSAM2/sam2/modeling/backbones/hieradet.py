"""
sam2/modeling/backbones/hieradet.py
=====================================
Hiera hierarchical vision transformer backbone (SAM2.1 variant).

Reference: https://arxiv.org/abs/2306.00989
"""

import logging
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from iopath.common.file_io import g_pathmgr

from sam2.modeling.backbones.utils import PatchEmbed, window_partition, window_unpartition
from sam2.modeling.sam2_utils import DropPath, MLP


def do_pool(x: torch.Tensor, pool: Optional[nn.Module], norm: Optional[nn.Module] = None) -> torch.Tensor:
    """Apply optional spatial pooling and layer normalisation.

    Parameters
    ----------
    x : Tensor  (B, H, W, C)
    pool : nn.Module or None
    norm : nn.Module or None

    Returns
    -------
    Tensor  (B, H', W', C)
    """
    if pool is None:
        return x
    x = pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    if norm:
        x = norm(x)
    return x


class MultiScaleAttention(nn.Module):
    """Multi-head self-attention with optional query pooling for downsampling.

    Parameters
    ----------
    dim : int       input dimension
    dim_out : int   output dimension
    num_heads : int
    q_pool : nn.Module or None  pooling applied to queries at stage transitions
    """

    def __init__(self, dim: int, dim_out: int, num_heads: int, q_pool: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return self.proj(x.transpose(1, 2).reshape(B, H, W, -1))


class MultiScaleBlock(nn.Module):
    """Transformer block with windowed attention and optional Q-pooling.

    Parameters
    ----------
    dim : int           input channel dimension
    dim_out : int       output channel dimension (may differ at stage transitions)
    num_heads : int
    mlp_ratio : float
    drop_path : float   stochastic depth rate
    norm_layer : str or nn.Module
    q_stride : tuple or None  stride for Q-pooling (downsamples spatial dims)
    act_layer : nn.Module
    window_size : int   0 = global attention
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: Union[nn.Module, str] = "LayerNorm",
        q_stride: Optional[Tuple[int, int]] = None,
        act_layer: nn.Module = nn.GELU,
        window_size: int = 0,
    ) -> None:
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride, ceil_mode=False)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads=num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out, num_layers=2, activation=act_layer)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)
        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Hiera(nn.Module):
    """Hiera hierarchical ViT backbone for SAM2.

    Parameters
    ----------
    embed_dim : int         initial embedding dimension
    num_heads : int         initial number of attention heads
    drop_path_rate : float  stochastic depth
    q_pool : int            number of pooling stages
    q_stride : tuple        downsampling stride between stages
    stages : tuple          blocks per stage
    dim_mul : float         dimension multiplier at stage transitions
    head_mul : float        head multiplier at stage transitions
    window_pos_embed_bkg_spatial_size : tuple
    window_spec : tuple     window size per stage
    global_att_blocks : tuple  block indices that use global (non-windowed) attention
    return_interm_layers : bool  return features from every stage
    """

    def __init__(
        self,
        embed_dim: int = 96,
        num_heads: int = 1,
        drop_path_rate: float = 0.0,
        q_pool: int = 3,
        q_stride: Tuple[int, int] = (2, 2),
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        window_spec: Tuple[int, ...] = (8, 4, 14, 7),
        global_att_blocks: Tuple[int, ...] = (12, 16, 20),
        weights_path: Optional[str] = None,
        return_interm_layers: bool = True,
    ) -> None:
        super().__init__()
        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.global_att_blocks = global_att_blocks
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, window_spec[0], window_spec[0]))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            block = MultiScaleBlock(
                dim=embed_dim, dim_out=dim_out, num_heads=num_heads, drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )
            embed_dim = dim_out
            self.blocks.append(block)
        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers else [self.blocks[-1].dim_out]
        )
        if weights_path is not None:
            with g_pathmgr.open(weights_path, "rb") as f:
                chkpt = torch.load(f, map_location="cpu")
            logging.info("loading Hiera %s", self.load_state_dict(chkpt, strict=False))

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        h, w = hw
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + self.pos_embed_window.tile(
            [x // y for x, y in zip(pos_embed.shape, self.pos_embed_window.shape)]
        )
        return pos_embed.permute(0, 2, 3, 1)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                outputs.append(x.permute(0, 3, 1, 2))
        return outputs

    def get_layer_id(self, layer_name: str) -> int:
        num_layers = self.get_num_layers()
        if "rel_pos" in layer_name:
            return num_layers + 1
        if "pos_embed" in layer_name or "patch_embed" in layer_name:
            return 0
        if "blocks" in layer_name:
            return int(layer_name.split("blocks")[1].split(".")[1]) + 1
        return num_layers + 1

    def get_num_layers(self) -> int:
        return len(self.blocks)
