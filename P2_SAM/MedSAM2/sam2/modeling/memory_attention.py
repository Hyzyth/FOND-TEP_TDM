"""
sam2/modeling/memory_attention.py
===================================
Memory attention module: cross-attends current frame features to stored
memory tokens from past frames, enabling temporal propagation.
"""

from typing import Optional

import torch
from torch import nn, Tensor

from sam2.modeling.sam.transformer import RoPEAttention
from sam2.modeling.sam2_utils import get_activation_fn, get_clones


class MemoryAttentionLayer(nn.Module):
    """Single memory-attention layer: self-attention → cross-attention → FFN.

    Parameters
    ----------
    activation : str            FFN activation (``'relu'`` or ``'gelu'``)
    cross_attention : nn.Module cross-attention sub-layer
    d_model : int               model dimension
    dim_feedforward : int       FFN hidden dimension
    dropout : float
    pos_enc_at_attn : bool      add pos enc to self-attention Q/K
    pos_enc_at_cross_attn_keys : bool
    pos_enc_at_cross_attn_queries : bool
    self_attention : nn.Module
    """

    def __init__(self, activation, cross_attention, d_model, dim_feedforward,
                 dropout, pos_enc_at_attn, pos_enc_at_cross_attn_keys,
                 pos_enc_at_cross_attn_queries, self_attention) -> None:
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        return tgt + self.dropout1(tgt2)

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory, **kwds,
        )
        return tgt + self.dropout2(tgt2)

    def forward(self, tgt, memory, pos=None, query_pos=None, num_k_exclude_rope=0):
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        return tgt + self.dropout3(tgt2)


class MemoryAttention(nn.Module):
    """Stack of :class:`MemoryAttentionLayer` modules.

    Parameters
    ----------
    d_model : int
    pos_enc_at_input : bool  add a scaled pos enc at the input
    layer : nn.Module        one MemoryAttentionLayer (will be cloned)
    num_layers : int
    batch_first : bool
    """

    def __init__(self, d_model, pos_enc_at_input, layer, num_layers, batch_first=True) -> None:
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(self, curr, memory, curr_pos=None, memory_pos=None, num_obj_ptr_tokens=0):
        if isinstance(curr, list):
            assert isinstance(curr_pos, list) and len(curr) == len(curr_pos) == 1
            curr, curr_pos = curr[0], curr_pos[0]
        assert curr.shape[1] == memory.shape[1]
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos
        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)
        for layer in self.layers:
            kwds = {"num_k_exclude_rope": num_obj_ptr_tokens} if isinstance(layer.cross_attn_image, RoPEAttention) else {}
            output = layer(tgt=output, memory=memory, pos=memory_pos, query_pos=curr_pos, **kwds)
        normed = self.norm(output)
        if self.batch_first:
            normed = normed.transpose(0, 1)
        return normed
