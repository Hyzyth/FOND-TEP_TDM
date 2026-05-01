"""
sam2/modeling/sam/mask_decoder.py
===================================
SAM-style mask decoder: a transformer + upscaling layers that converts
image and prompt embeddings into binary segmentation masks and IoU scores.
"""

from typing import List, Optional, Tuple, Type

import torch
from torch import nn

from sam2.modeling.sam2_utils import LayerNorm2d, MLP


class MaskDecoder(nn.Module):
    """Predict masks from image and sparse/dense prompt embeddings.

    Parameters
    ----------
    transformer_dim : int
    transformer : nn.Module        TwoWayTransformer
    num_multimask_outputs : int    number of candidate masks (default 3)
    activation : nn.Module
    iou_head_depth, iou_head_hidden_dim : int
    use_high_res_features : bool
    iou_prediction_use_sigmoid : bool
    dynamic_multimask_via_stability : bool
    dynamic_multimask_stability_delta : float
    dynamic_multimask_stability_thresh : float
    pred_obj_scores : bool
    pred_obj_scores_mlp : bool
    use_multimask_token_for_obj_ptr : bool
    """

    def __init__(
        self, *, transformer_dim, transformer, num_multimask_outputs=3,
        activation=nn.GELU, iou_head_depth=3, iou_head_hidden_dim=256,
        use_high_res_features=False, iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False, dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98, pred_obj_scores=False,
        pred_obj_scores_mlp=False, use_multimask_token_for_obj_ptr=False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.pred_obj_scores = pred_obj_scores
        if pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4), activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, 1, 1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, 1, 1)
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ])
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens,
            iou_head_depth, sigmoid_output=iou_prediction_use_sigmoid,
        )
        if pred_obj_scores:
            self.pred_obj_score_head = (
                MLP(transformer_dim, transformer_dim, 1, 3)
                if pred_obj_scores_mlp
                else nn.Linear(transformer_dim, 1)
            )
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                dense_prompt_embeddings, multimask_output, repeat_image,
                high_res_features=None):
        masks, iou_pred, mask_tokens_out, obj_score_logits = self.predict_masks(
            image_embeddings, image_pe, sparse_prompt_embeddings,
            dense_prompt_embeddings, repeat_image, high_res_features,
        )
        s = 1 if self.pred_obj_scores else 0
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]
        sam_tokens_out = (
            mask_tokens_out[:, 1:] if multimask_output and self.use_multimask_token_for_obj_ptr
            else mask_tokens_out[:, 0:1]
        )
        return masks, iou_pred, sam_tokens_out, obj_score_logits

    def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                      dense_prompt_embeddings, repeat_image, high_res_features=None):
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat([self.obj_score_token.weight, self.iou_token.weight, self.mask_tokens.weight], dim=0)
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
        src = (
            torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
            if repeat_image else image_embeddings
        )
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : s + 1 + self.num_mask_tokens, :]
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled = act1(ln1(dc1(src) + feat_s1))
            upscaled = act2(dc2(upscaled) + feat_s0)
        hyper_in = torch.stack([
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)
        ], dim=1)
        b, c, h, w = upscaled.shape
        masks = (hyper_in @ upscaled.view(b, c, h * w)).view(b, -1, h, w)
        iou_pred = self.iou_prediction_head(iou_token_out)
        obj_score_logits = (
            self.pred_obj_score_head(hs[:, 0, :]) if self.pred_obj_scores
            else 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)
        )
        return masks, iou_pred, mask_tokens_out, obj_score_logits

    def _get_stability_scores(self, mask_logits):
        mask_logits = mask_logits.flatten(-2)
        delta = self.dynamic_multimask_stability_delta
        area_i = (mask_logits > delta).float().sum(-1)
        area_u = (mask_logits > -delta).float().sum(-1)
        return torch.where(area_u > 0, area_i / area_u, torch.ones_like(area_i))

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        multi_logits = all_mask_logits[:, 1:, :, :]
        multi_ious = all_iou_scores[:, 1:]
        best_inds = torch.argmax(multi_ious, dim=-1)
        batch_inds = torch.arange(multi_ious.size(0), device=all_iou_scores.device)
        best_multi_logits = multi_logits[batch_inds, best_inds].unsqueeze(1)
        best_multi_ious = multi_ious[batch_inds, best_inds].unsqueeze(1)
        single_logits = all_mask_logits[:, 0:1, :, :]
        single_ious = all_iou_scores[:, 0:1]
        stability = self._get_stability_scores(single_logits)
        is_stable = stability >= self.dynamic_multimask_stability_thresh
        mask_out = torch.where(is_stable[..., None, None].expand_as(single_logits), single_logits, best_multi_logits)
        iou_out = torch.where(is_stable.expand_as(single_ious), single_ious, best_multi_ious)
        return mask_out, iou_out
