"""
sam2/sam2_video_predictor_npz.py
==================================
Video predictor adapted for 3-D medical image inference.

Axial slices are treated as video frames.  The predictor propagates
a 2-D prompt (box or points on the key slice) forward and backward
through the volume using SAM2's memory mechanism.

For HECKTOR usage see ``inference/infer_hecktor.py``.
"""

import warnings
from collections import OrderedDict

import torch
from tqdm import tqdm

from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
from sam2.utils.misc import concat_points, fill_holes_in_mask_scores


class SAM2VideoPredictorNPZ(SAM2Base):
    """Slice-by-slice 3-D predictor for pre-loaded NPZ medical volumes.

    Parameters
    ----------
    fill_hole_area : int          fill background holes ≤ this area (0 = off)
    non_overlap_masks : bool      apply non-overlapping constraints to output
    clear_non_cond_mem_around_input : bool
    clear_non_cond_mem_for_multi_obj : bool
    add_all_frames_to_correct_as_cond : bool
    """

    def __init__(
        self,
        fill_hole_area=0,
        non_overlap_masks=False,
        clear_non_cond_mem_around_input=False,
        clear_non_cond_mem_for_multi_obj=False,
        add_all_frames_to_correct_as_cond=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond

    # ──────────────────────────────────────────────────────────────────────────
    # State initialisation
    # ──────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def init_state(self, images, video_height, video_width,
                   offload_video_to_cpu=False, offload_state_to_cpu=False,
                   async_loading_frames=False):
        """Initialise inference state for a new volume.

        Parameters
        ----------
        images : Tensor  (D, 3, H, W) normalised float
        video_height, video_width : int  original slice dimensions
        offload_video_to_cpu : bool
        offload_state_to_cpu : bool

        Returns
        -------
        dict  inference_state
        """
        compute_device = self.device
        state = {}
        state["images"] = images
        state["num_frames"] = len(images)
        state["offload_video_to_cpu"] = offload_video_to_cpu
        state["offload_state_to_cpu"] = offload_state_to_cpu
        state["video_height"] = video_height
        state["video_width"] = video_width
        state["device"] = compute_device
        state["storage_device"] = torch.device("cpu") if offload_state_to_cpu else compute_device
        state["point_inputs_per_obj"] = {}
        state["mask_inputs_per_obj"] = {}
        state["cached_features"] = {}
        state["constants"] = {}
        state["obj_id_to_idx"] = OrderedDict()
        state["obj_idx_to_id"] = OrderedDict()
        state["obj_ids"] = []
        state["output_dict"] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        state["output_dict_per_obj"] = {}
        state["temp_output_dict_per_obj"] = {}
        state["consolidated_frame_inds"] = {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()}
        state["tracking_has_started"] = False
        state["frames_already_tracked"] = {}
        # Warm up the backbone on frame 0.
        self._get_image_feature(state, frame_idx=0, batch_size=1)
        return state

    # ──────────────────────────────────────────────────────────────────────────
    # Object management
    # ──────────────────────────────────────────────────────────────────────────

    def _obj_id_to_idx(self, state, obj_id):
        idx = state["obj_id_to_idx"].get(obj_id)
        if idx is not None:
            return idx
        if state["tracking_has_started"]:
            raise RuntimeError(f"Cannot add new object {obj_id} after tracking started.")
        idx = len(state["obj_id_to_idx"])
        state["obj_id_to_idx"][obj_id] = idx
        state["obj_idx_to_id"][idx] = obj_id
        state["obj_ids"] = list(state["obj_id_to_idx"])
        state["point_inputs_per_obj"][idx] = {}
        state["mask_inputs_per_obj"][idx] = {}
        state["output_dict_per_obj"][idx] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        state["temp_output_dict_per_obj"][idx] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        return idx

    def _get_obj_num(self, state):
        return len(state["obj_idx_to_id"])

    # ──────────────────────────────────────────────────────────────────────────
    # Prompt input
    # ──────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def add_new_points_or_box(self, inference_state, frame_idx, obj_id,
                               points=None, labels=None, clear_old_points=True,
                               normalize_coords=True, box=None):
        """Add a point or box prompt to initialise tracking on *frame_idx*.

        Parameters
        ----------
        inference_state : dict
        frame_idx : int           slice index
        obj_id : int              client-side object ID
        points : ndarray or None  (N, 2) in (x, y) pixel coords
        labels : ndarray or None  (N,) int32
        clear_old_points : bool
        normalize_coords : bool   divide coords by (W, H) before scaling
        box : ndarray or None     [x_min, y_min, x_max, y_max]
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame  = inference_state["mask_inputs_per_obj"][obj_idx]

        if points is None and box is None:
            raise ValueError("At least one of points or box must be provided.")
        if (points is None) != (labels is None):
            raise ValueError("points and labels must both be provided or both None.")

        points = torch.zeros(0, 2, dtype=torch.float32) if points is None else torch.as_tensor(points, dtype=torch.float32)
        labels = torch.zeros(0, dtype=torch.int32) if labels is None else torch.as_tensor(labels, dtype=torch.int32)
        if points.dim() == 2: points = points.unsqueeze(0)
        if labels.dim() == 1: labels = labels.unsqueeze(0)

        if box is not None:
            if not clear_old_points:
                raise ValueError("box prompt requires clear_old_points=True.")
            box_t = torch.as_tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box_t.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device).reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            H = inference_state["video_height"]
            W = inference_state["video_width"]
            points = points / torch.tensor([W, H], dtype=points.dtype, device=points.device)
        points = points * self.image_size
        points = points.to(inference_state["device"])
        labels = labels.to(inference_state["device"])

        point_inputs = None if clear_old_points else point_inputs_per_frame.get(frame_idx)
        point_inputs = concat_points(point_inputs, points, labels)
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)

        is_init = frame_idx not in inference_state["frames_already_tracked"]
        reverse = False if is_init else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        is_cond = is_init or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        prev_sam_mask_logits = None
        obj_out_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        prev_out = (obj_temp_dict[storage_key].get(frame_idx)
                    or obj_out_dict["cond_frame_outputs"].get(frame_idx)
                    or obj_out_dict["non_cond_frame_outputs"].get(frame_idx))
        if prev_out is not None and prev_out.get("pred_masks") is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].to(inference_state["device"], non_blocking=True)
            prev_sam_mask_logits = prev_sam_mask_logits.clamp(-32.0, 32.0)

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_out_dict,
            frame_idx=frame_idx, batch_size=1,
            is_init_cond_frame=is_init,
            point_inputs=point_inputs, mask_inputs=None,
            reverse=reverse, run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        obj_temp_dict[storage_key][frame_idx] = current_out

        consolidated = self._consolidate_temp_output_across_obj(
            inference_state, frame_idx, is_cond=is_cond,
            run_mem_encoder=False, consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated["pred_masks_video_res"]
        )
        return frame_idx, inference_state["obj_ids"], video_res_masks

    @torch.inference_mode()
    def add_new_mask(self, inference_state, frame_idx, obj_id, mask):
        """Add a binary mask prompt for *frame_idx*."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        mask_H, mask_W = mask.shape
        mask_inputs = mask[None, None].float().to(inference_state["device"])
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False, antialias=True,
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = mask_inputs
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        is_init = frame_idx not in inference_state["frames_already_tracked"]
        reverse = False if is_init else inference_state["frames_already_tracked"][frame_idx]["reverse"]
        is_cond = is_init or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        obj_out_dict = inference_state["output_dict_per_obj"][obj_idx]
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state, output_dict=obj_out_dict,
            frame_idx=frame_idx, batch_size=1, is_init_cond_frame=is_init,
            point_inputs=None, mask_inputs=mask_inputs, reverse=reverse, run_mem_encoder=False,
        )
        inference_state["temp_output_dict_per_obj"][obj_idx][storage_key][frame_idx] = current_out

        consolidated = self._consolidate_temp_output_across_obj(
            inference_state, frame_idx, is_cond=is_cond,
            run_mem_encoder=False, consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated["pred_masks_video_res"]
        )
        return frame_idx, inference_state["obj_ids"], video_res_masks

    # ──────────────────────────────────────────────────────────────────────────
    # Propagation
    # ──────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def propagate_in_video(self, inference_state, start_frame_idx=None,
                            max_frame_num_to_track=None, reverse=False):
        """Propagate the prompt across all slices, yielding per-slice masks.

        Parameters
        ----------
        inference_state : dict
        start_frame_idx : int or None  default = earliest conditioning frame
        max_frame_num_to_track : int or None  default = all frames
        reverse : bool  track towards slice 0 instead of the last slice

        Yields
        ------
        (frame_idx, obj_ids, video_res_masks)
        """
        self._propagate_preflight(inference_state)
        output_dict = inference_state["output_dict"]
        consolidated_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        if not output_dict["cond_frame_outputs"]:
            raise RuntimeError("No prompts provided; call add_new_points_or_box first.")

        if start_frame_idx is None:
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            max_frame_num_to_track = num_frames

        if reverse:
            end = max(start_frame_idx - max_frame_num_to_track, 0)
            order = range(start_frame_idx, end - 1, -1) if start_frame_idx > 0 else []
        else:
            end = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            order = range(start_frame_idx, end + 1)

        clear_non_cond = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )
        for frame_idx in tqdm(order, desc="propagate in video"):
            if frame_idx in consolidated_inds["cond_frame_outputs"]:
                current_out = output_dict["cond_frame_outputs"][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond:
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_inds["non_cond_frame_outputs"]:
                current_out = output_dict["non_cond_frame_outputs"][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state, output_dict=output_dict,
                    frame_idx=frame_idx, batch_size=batch_size,
                    is_init_cond_frame=False, point_inputs=None, mask_inputs=None,
                    reverse=reverse, run_mem_encoder=True,
                )
                output_dict["non_cond_frame_outputs"][frame_idx] = current_out

            self._add_output_per_object(inference_state, frame_idx, current_out,
                                         "cond_frame_outputs" if frame_idx in consolidated_inds["cond_frame_outputs"] else "non_cond_frame_outputs")
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
            _, video_res_masks = self._get_orig_video_res_output(inference_state, pred_masks)
            yield frame_idx, obj_ids, video_res_masks

    def _propagate_preflight(self, inference_state):
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)
        temp_output_dict = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        consolidated_inds = inference_state["consolidated_frame_inds"]

        for is_cond in [False, True]:
            key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            frame_inds = set()
            for obj_temp in temp_output_dict.values():
                frame_inds.update(obj_temp[key].keys())
            consolidated_inds[key].update(frame_inds)
            for frame_idx in frame_inds:
                cons = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                output_dict[key][frame_idx] = cons
                self._add_output_per_object(inference_state, frame_idx, cons, key)
                if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            for obj_temp in temp_output_dict.values():
                obj_temp[key].clear()

        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
            for obj_od in inference_state["output_dict_per_obj"].values():
                obj_od["non_cond_frame_outputs"].pop(frame_idx, None)

    # ──────────────────────────────────────────────────────────────────────────
    # State reset
    # ──────────────────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Clear all tracking state (keeps cached image features)."""
        self._reset_tracking_results(inference_state)
        for attr in ("obj_id_to_idx", "obj_idx_to_id", "obj_ids",
                     "point_inputs_per_obj", "mask_inputs_per_obj",
                     "output_dict_per_obj", "temp_output_dict_per_obj"):
            inference_state[attr].clear()

    def _reset_tracking_results(self, state):
        for v in state["point_inputs_per_obj"].values():  v.clear()
        for v in state["mask_inputs_per_obj"].values():   v.clear()
        for v in state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        state["output_dict"]["cond_frame_outputs"].clear()
        state["output_dict"]["non_cond_frame_outputs"].clear()
        state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        state["tracking_has_started"] = False
        state["frames_already_tracked"].clear()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_image_feature(self, state, frame_idx, batch_size):
        image, backbone_out = state["cached_features"].get(frame_idx, (None, None))
        if backbone_out is None:
            device = state["device"]
            image = state["images"][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            state["cached_features"] = {frame_idx: (image, backbone_out)}
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_out = {
            "backbone_fpn": [f.expand(batch_size, -1, -1, -1) for f in backbone_out["backbone_fpn"]],
            "vision_pos_enc": [p.expand(batch_size, -1, -1, -1) for p in backbone_out["vision_pos_enc"]],
        }
        return (expanded_image,) + self._prepare_backbone_features(expanded_out)

    def _run_single_frame_inference(self, inference_state, output_dict, frame_idx,
                                     batch_size, is_init_cond_frame, point_inputs,
                                     mask_inputs, reverse, run_mem_encoder,
                                     prev_sam_mask_logits=None):
        _, _, current_vision_feats, current_vision_pos_embeds, feat_sizes = \
            self._get_image_feature(inference_state, frame_idx, batch_size)

        current_out = self.track_step(
            frame_idx=frame_idx, is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes, point_inputs=point_inputs, mask_inputs=mask_inputs,
            output_dict=output_dict, num_frames=inference_state["num_frames"],
            track_in_reverse=reverse, run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16).to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, self.fill_hole_area)
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        compact = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": current_out["obj_ptr"],
            "object_score_logits": current_out["object_score_logits"],
        }
        return compact, pred_masks_gpu

    def _get_maskmem_pos_enc(self, state, current_out):
        out_pos = current_out["maskmem_pos_enc"]
        if out_pos is None:
            return None
        constants = state["constants"]
        if "maskmem_pos_enc" not in constants:
            constants["maskmem_pos_enc"] = [x[0:1].clone() for x in out_pos]
        batch_size = out_pos[0].size(0)
        return [x.expand(batch_size, -1, -1, -1) for x in constants["maskmem_pos_enc"]]

    def _get_orig_video_res_output(self, state, any_res_masks):
        device = state["device"]
        H, W = state["video_height"], state["video_width"]
        masks = any_res_masks.to(device, non_blocking=True)
        if masks.shape[-2:] != (H, W):
            masks = torch.nn.functional.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)
        if self.non_overlap_masks:
            masks = self._apply_non_overlapping_constraints(masks)
        return any_res_masks, masks

    def _consolidate_temp_output_across_obj(self, state, frame_idx, is_cond,
                                              run_mem_encoder, consolidate_at_video_res=False):
        batch_size = self._get_obj_num(state)
        key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        if consolidate_at_video_res:
            H, W = state["video_height"], state["video_width"]
            mask_key = "pred_masks_video_res"
        else:
            H = W = self.image_size // 4
            mask_key = "pred_masks"

        consolidated = {
            "maskmem_features": None, "maskmem_pos_enc": None,
            mask_key: torch.full((batch_size, 1, H, W), NO_OBJ_SCORE,
                                  dtype=torch.float32, device=state["storage_device"]),
            "obj_ptr": torch.full((batch_size, self.hidden_dim), NO_OBJ_SCORE,
                                   dtype=torch.float32, device=state["device"]),
            "object_score_logits": torch.full((batch_size, 1), 10.0,
                                               dtype=torch.float32, device=state["device"]),
        }

        for obj_idx in range(batch_size):
            out = (state["temp_output_dict_per_obj"][obj_idx][key].get(frame_idx)
                   or state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"].get(frame_idx)
                   or state["output_dict_per_obj"][obj_idx]["non_cond_frame_outputs"].get(frame_idx))
            if out is None:
                continue
            obj_mask = out["pred_masks"]
            cp = consolidated[mask_key]
            if obj_mask.shape[-2:] == cp.shape[-2:]:
                cp[obj_idx:obj_idx+1] = obj_mask
            else:
                cp[obj_idx:obj_idx+1] = torch.nn.functional.interpolate(
                    obj_mask, size=cp.shape[-2:], mode="bilinear", align_corners=False
                )
            consolidated["obj_ptr"][obj_idx:obj_idx+1] = out["obj_ptr"]
            consolidated["object_score_logits"][obj_idx:obj_idx+1] = out["object_score_logits"]

        if run_mem_encoder:
            device = state["device"]
            high_res = torch.nn.functional.interpolate(
                consolidated["pred_masks"].to(device, non_blocking=True),
                size=(self.image_size, self.image_size), mode="bilinear", align_corners=False,
            )
            if self.non_overlap_masks_for_mem_enc:
                high_res = self._apply_non_overlapping_constraints(high_res)
            features, pos_enc = self._run_memory_encoder(
                state, frame_idx, batch_size, high_res,
                consolidated["object_score_logits"], is_mask_from_pts=True,
            )
            consolidated["maskmem_features"] = features
            consolidated["maskmem_pos_enc"] = pos_enc
        return consolidated

    def _run_memory_encoder(self, state, frame_idx, batch_size, high_res_masks,
                              object_score_logits, is_mask_from_pts):
        _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(state, frame_idx, batch_size)
        features, pos_enc = self._encode_new_memory(
            current_vision_feats, feat_sizes, high_res_masks, object_score_logits, is_mask_from_pts
        )
        storage_device = state["storage_device"]
        features = features.to(torch.bfloat16).to(storage_device, non_blocking=True)
        return features, self._get_maskmem_pos_enc(state, {"maskmem_pos_enc": pos_enc})

    def _add_output_per_object(self, state, frame_idx, current_out, storage_key):
        mf = current_out["maskmem_features"]
        mp = current_out["maskmem_pos_enc"]
        for obj_idx, obj_od in state["output_dict_per_obj"].items():
            sl = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": mf[sl] if mf is not None else None,
                "maskmem_pos_enc": [x[sl] for x in mp] if mp is not None else None,
                "pred_masks": current_out["pred_masks"][sl],
                "obj_ptr": current_out["obj_ptr"][sl],
                "object_score_logits": current_out["object_score_logits"][sl],
            }
            obj_od[storage_key][frame_idx] = obj_out

    def _clear_non_cond_mem_around_input(self, state, frame_idx):
        r = self.memory_temporal_stride_for_eval
        lo, hi = frame_idx - r * self.num_maskmem, frame_idx + r * self.num_maskmem
        non_cond = state["output_dict"]["non_cond_frame_outputs"]
        for t in range(lo, hi + 1):
            non_cond.pop(t, None)
            for obj_od in state["output_dict_per_obj"].values():
                obj_od["non_cond_frame_outputs"].pop(t, None)
