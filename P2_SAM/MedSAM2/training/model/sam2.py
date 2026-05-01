"""
training/model/sam2.py
=======================
SAM2Train: extends SAM2Base with interactive point/box sampling and
multi-step iterative correction for training.

Used during HECKTOR fine-tuning via the training config
``sam2/configs/sam2.1_hiera_tiny_hecktor.yaml``.
"""

import logging

import numpy as np
import torch
import torch.distributed

from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.sam2_utils import (
    get_1d_sine_pe, get_next_point, sample_box_points, select_closest_cond_frames,
)
from sam2.utils.misc import concat_points
from training.utils.data_utils import BatchedVideoDatapoint


class SAM2Train(SAM2Base):
    """SAM2 model extended for interactive training with correction clicks.

    Additional parameters (all others are forwarded to SAM2Base)
    ------------------------------------------------------------
    prob_to_use_pt_input_for_train : float
        Probability of using point prompts instead of mask prompts during training.
    prob_to_use_pt_input_for_eval : float
    prob_to_use_box_input_for_train : float
        Given ``use_pt_input=True``, probability of using a box rather than a click.
    prob_to_use_box_input_for_eval : float
    num_frames_to_correct_for_train : int
        How many frames receive iterative correction clicks.
    num_frames_to_correct_for_eval : int
    rand_frames_to_correct_for_train : bool
    rand_frames_to_correct_for_eval : bool
    num_init_cond_frames_for_train : int
        Number of initial conditioning frames (always includes frame 0).
    num_init_cond_frames_for_eval : int
    rand_init_cond_frames_for_train : bool
    rand_init_cond_frames_for_eval : bool
    add_all_frames_to_correct_as_cond : bool
    num_correction_pt_per_frame : int
    pt_sampling_for_eval : str  ``'uniform'`` or ``'center'``
    prob_to_sample_from_gt_for_train : float
    use_act_ckpt_iterative_pt_sampling : bool
    forward_backbone_per_frame_for_eval : bool
    freeze_image_encoder : bool
    """

    def __init__(
        self,
        image_encoder,
        memory_attention=None,
        memory_encoder=None,
        prob_to_use_pt_input_for_train=0.0,
        prob_to_use_pt_input_for_eval=0.0,
        prob_to_use_box_input_for_train=0.0,
        prob_to_use_box_input_for_eval=0.0,
        num_frames_to_correct_for_train=1,
        num_frames_to_correct_for_eval=1,
        rand_frames_to_correct_for_train=False,
        rand_frames_to_correct_for_eval=False,
        num_init_cond_frames_for_train=1,
        num_init_cond_frames_for_eval=1,
        rand_init_cond_frames_for_train=True,
        rand_init_cond_frames_for_eval=False,
        add_all_frames_to_correct_as_cond=False,
        num_correction_pt_per_frame=7,
        pt_sampling_for_eval="center",
        prob_to_sample_from_gt_for_train=0.0,
        use_act_ckpt_iterative_pt_sampling=False,
        forward_backbone_per_frame_for_eval=False,
        freeze_image_encoder=False,
        **kwargs,
    ) -> None:
        super().__init__(image_encoder, memory_attention, memory_encoder, **kwargs)
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        self.rng = np.random.default_rng(seed=42)

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

    # ──────────────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward(self, input: BatchedVideoDatapoint):
        """Full training forward: encode all frames then track with prompts.

        Parameters
        ----------
        input : BatchedVideoDatapoint

        Returns
        -------
        list[dict]  one output dict per frame
        """
        if self.training or not self.forward_backbone_per_frame_for_eval:
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        return self.forward_tracking(backbone_out, input)

    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute backbone features on-the-fly for *img_ids*."""
        if img_ids.numel() > 1:
            unique_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_ids, inv_ids = img_ids, None
        image = img_batch[unique_ids]
        backbone_out = self.forward_image(image)
        _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]
        return image, vision_feats, vision_pos_embeds, feat_sizes

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt preparation
    # ──────────────────────────────────────────────────────────────────────────

    def prepare_prompt_inputs(self, backbone_out, input: BatchedVideoDatapoint, start_frame_idx=0):
        """Decide on mask vs point inputs and sample initial conditioning frames."""
        gt_masks_per_frame = {t: masks.unsqueeze(1) for t, masks in enumerate(input.masks)}
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input.num_frames
        backbone_out["num_frames"] = num_frames

        if self.training:
            use_pt = self.rng.random() < self.prob_to_use_pt_input_for_train
            p_box = self.prob_to_use_box_input_for_train
            n_correct = self.num_frames_to_correct_for_train
            rand_correct = self.rand_frames_to_correct_for_train
            n_init = self.num_init_cond_frames_for_train
            rand_init = self.rand_init_cond_frames_for_train
        else:
            use_pt = self.rng.random() < self.prob_to_use_pt_input_for_eval
            p_box = self.prob_to_use_box_input_for_eval
            n_correct = self.num_frames_to_correct_for_eval
            rand_correct = self.rand_frames_to_correct_for_eval
            n_init = self.num_init_cond_frames_for_eval
            rand_init = self.rand_init_cond_frames_for_eval

        if num_frames == 1:
            use_pt = True
            n_correct = 1
            n_init = 1

        if rand_init and n_init > 1:
            n_init = self.rng.integers(1, n_init, endpoint=True)
        if use_pt and rand_correct and n_correct > n_init:
            n_correct = self.rng.integers(n_init, n_correct, endpoint=True)

        backbone_out["use_pt_input"] = use_pt

        if n_init == 1:
            init_cond_frames = [start_frame_idx]
        else:
            extra = self.rng.choice(range(start_frame_idx + 1, num_frames), n_init - 1, replace=False).tolist()
            init_cond_frames = [start_frame_idx] + extra

        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
        ]
        backbone_out["mask_inputs_per_frame"] = {}
        backbone_out["point_inputs_per_frame"] = {}

        for t in init_cond_frames:
            if not use_pt:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
            else:
                if self.rng.random() < p_box:
                    points, labels = sample_box_points(gt_masks_per_frame[t])
                else:
                    points, labels = get_next_point(
                        gt_masks=gt_masks_per_frame[t], pred_masks=None,
                        method="uniform" if self.training else self.pt_sampling_for_eval,
                    )
                backbone_out["point_inputs_per_frame"][t] = {"point_coords": points, "point_labels": labels}

        if not use_pt:
            frames_to_correct = []
        elif n_correct == n_init:
            frames_to_correct = init_cond_frames
        else:
            extra_n = n_correct - n_init
            extra_frames = self.rng.choice(backbone_out["frames_not_in_init_cond"], extra_n, replace=False).tolist()
            frames_to_correct = init_cond_frames + extra_frames

        backbone_out["frames_to_add_correction_pt"] = frames_to_correct
        return backbone_out

    # ──────────────────────────────────────────────────────────────────────────
    # Tracking forward
    # ──────────────────────────────────────────────────────────────────────────

    def forward_tracking(self, backbone_out, input: BatchedVideoDatapoint, return_dict=False):
        """Run the tracking loop over all frames, applying correction clicks."""
        img_feats_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_computed:
            _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)

        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_correct = backbone_out["frames_to_add_correction_pt"]
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]

        output_dict = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}

        for stage_id in processing_order:
            img_ids = input.flat_obj_to_img_idx[stage_id]
            if img_feats_computed:
                cur_feats = [x[:, img_ids] for x in vision_feats]
                cur_pos   = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                _, cur_feats, cur_pos, feat_sizes = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )

            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=(stage_id in init_cond_frames),
                current_vision_feats=cur_feats,
                current_vision_pos_embeds=cur_pos,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(stage_id),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(stage_id),
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id),
                frames_to_add_correction_pt=frames_to_correct,
                output_dict=output_dict,
                num_frames=num_frames,
            )

            add_as_cond = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond and stage_id in frames_to_correct
            )
            if add_as_cond:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out

        if return_dict:
            return output_dict

        all_outputs = {**output_dict["cond_frame_outputs"], **output_dict["non_cond_frame_outputs"]}
        all_outputs = [all_outputs[t] for t in range(num_frames)]
        return [{k: v for k, v in d.items() if k != "obj_ptr"} for d in all_outputs]

    # ──────────────────────────────────────────────────────────────────────────
    # Track step (with correction)
    # ──────────────────────────────────────────────────────────────────────────

    def track_step(self, frame_idx, is_init_cond_frame, current_vision_feats,
                    current_vision_pos_embeds, feat_sizes, point_inputs, mask_inputs,
                    output_dict, num_frames, track_in_reverse=False, run_mem_encoder=True,
                    prev_sam_mask_logits=None, frames_to_add_correction_pt=None, gt_masks=None):
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []

        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx, is_init_cond_frame, current_vision_feats, current_vision_pos_embeds,
            feat_sizes, point_inputs, mask_inputs, output_dict, num_frames,
            track_in_reverse, prev_sam_mask_logits,
        )
        (low_res_multimasks, high_res_multimasks, ious,
         low_res_masks, high_res_masks, obj_ptr, object_score_logits) = sam_outputs

        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        if frame_idx in frames_to_add_correction_pt:
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame, point_inputs, gt_masks, high_res_features, pix_feat,
                low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks,
                object_score_logits, current_out,
            )
            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = final_sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        self._encode_memory_in_output(
            current_vision_feats, feat_sizes, point_inputs, run_mem_encoder,
            high_res_masks, object_score_logits, current_out,
        )
        return current_out

    def _iter_correct_pt_sampling(self, is_init_cond_frame, point_inputs, gt_masks,
                                    high_res_features, pix_feat, low_res_multimasks,
                                    high_res_multimasks, ious, low_res_masks, high_res_masks,
                                    object_score_logits, current_out):
        assert gt_masks is not None
        all_low, all_high = [low_res_masks], [high_res_masks]
        all_lm, all_hm = [low_res_multimasks], [high_res_multimasks]
        all_ious, all_pt, all_obj = [ious], [point_inputs], [object_score_logits]

        for _ in range(self.num_correction_pt_per_frame):
            sample_from_gt = (self.training and self.prob_to_sample_from_gt_for_train > 0
                               and self.rng.random() < self.prob_to_sample_from_gt_for_train)
            pred_for_new_pt = None if sample_from_gt else (high_res_masks > 0)
            new_pts, new_labels = get_next_point(
                gt_masks=gt_masks, pred_masks=pred_for_new_pt,
                method="uniform" if self.training else self.pt_sampling_for_eval,
            )
            point_inputs = concat_points(point_inputs, new_pts, new_labels)
            mask_inputs = low_res_masks
            multimask = self._use_multimask(is_init_cond_frame, point_inputs)

            if self.use_act_ckpt_iterative_pt_sampling and not multimask:
                sam_out = torch.utils.checkpoint.checkpoint(
                    self._forward_sam_heads,
                    backbone_features=pix_feat, point_inputs=point_inputs,
                    mask_inputs=mask_inputs, high_res_features=high_res_features,
                    multimask_output=multimask, use_reentrant=False,
                )
            else:
                sam_out = self._forward_sam_heads(
                    backbone_features=pix_feat, point_inputs=point_inputs,
                    mask_inputs=mask_inputs, high_res_features=high_res_features,
                    multimask_output=multimask,
                )

            lmm, hmm, new_ious, lm, hm, _, new_obj = sam_out
            all_low.append(lm);  all_high.append(hm)
            all_lm.append(lmm);  all_hm.append(hmm)
            all_ious.append(new_ious); all_pt.append(point_inputs); all_obj.append(new_obj)
            low_res_masks, high_res_masks = lm, hm

        current_out["multistep_pred_masks"] = torch.cat(all_low, dim=1)
        current_out["multistep_pred_masks_high_res"] = torch.cat(all_high, dim=1)
        current_out["multistep_pred_multimasks"] = all_lm
        current_out["multistep_pred_multimasks_high_res"] = all_hm
        current_out["multistep_pred_ious"] = all_ious
        current_out["multistep_point_inputs"] = all_pt
        current_out["multistep_object_score_logits"] = all_obj
        return point_inputs, sam_out
