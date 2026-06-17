"""
auto_prompting/auto_prompter.py
================================
AutoPrompter: unified interface for automatic bounding-box prompt generation.

Methods
-------
  pet      - PET thresholding only (fast, high recall)
  unet     - Small3DUNet proposal network only
  hybrid   - PET U UNet filtered by 3-D IoU overlap (recommended)

PET thresholding strategies (--pet_method)
------------------------------------------
  base41   - 41 % SUVmax        (scale-invariant, no SUV needed)
  nestle   - alpha * I0.7 + Ibgd (requires raw SUV)
  black    - iterative SUVmean   (requires raw SUV)
  daisne   - iterative contrast  (requires raw SUV)

UPDATED: get_proposals() now accepts ``pet_suv_volume`` — a (D,H,W) float32
array of raw SUV values taken directly from the SwinCross NPZ ``pet`` key.
When provided, the uint8->SUV reconstruction step in pet_proposal.py is
skipped entirely, preserving the full dynamic range.

Path summary
------------
  SwinCross NPZ  ->  pet_suv_volume = npz["pet"].astype(float32)
                    suv_max        = None
                    -> nestle/black/daisne use pet_suv_volume directly

  TemPoRAL NPZ   ->  pet_suv_volume = None
                    suv_max        = meta["pet_suv_max"]  (as before)
                    -> nestle/black/daisne use reconstruct_suv(pet_uint8, suv_max)

  base41         ->  neither needed (scale-invariant on uint8)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from auto_prompting.pet_proposal import get_pet_proposals, mask_to_proposals
from auto_prompting.box_utils import iou_3d, nms_3d

logger = logging.getLogger(__name__)


# ── UNet proposals ─────────────────────────────────────────────────────────────

def _unet_proposals(ct_uint8: np.ndarray,
                    pet_uint8: np.ndarray,
                    model,
                    device: str,
                    threshold: float = 0.25,
                    min_volume: int = 50,
                    slice_pad: int = 1,
                    planar_pad: int = 5) -> list[dict]:
    """Run the Small3DUNet and convert the probability map to proposals."""
    ct  = ct_uint8.astype(np.float32)  / 255.0
    pet = pet_uint8.astype(np.float32) / 255.0

    x = torch.tensor(np.stack([ct, pet], axis=0)).unsqueeze(0).float().to(device)

    _, _, D, H, W = x.shape
    pad_d = (16 - D % 16) % 16
    pad_h = (16 - H % 16) % 16
    pad_w = (16 - W % 16) % 16
    x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))

    model.eval()
    with torch.no_grad():
        prob = model(x)[0, 0, :D, :H, :W].cpu().numpy()

    binary = prob > threshold
    return mask_to_proposals(
        binary_mask = binary,
        min_volume  = min_volume,
        slice_pad   = slice_pad,
        planar_pad  = planar_pad,
    )


# ── Hybrid filtering ───────────────────────────────────────────────────────────

def _hybrid_filter(pet_props: list[dict],
                   unet_props: list[dict],
                   iou_threshold: float = 0.05) -> list[dict]:
    """Keep PET proposals that overlap with at least one UNet proposal."""
    if not unet_props:
        logger.warning("UNet produced no proposals; returning PET proposals unfiltered.")
        return pet_props

    filtered = [
        p for p in pet_props
        if any(iou_3d(p["bbox_3d"], u["bbox_3d"]) >= iou_threshold
               for u in unet_props)
    ]

    if not filtered:
        logger.warning(
            "No PET proposals overlapped with UNet proposals (IoU < %.2f). "
            "Falling back to UNet proposals alone.", iou_threshold
        )
        return unet_props

    return filtered


# ── Label assignment ───────────────────────────────────────────────────────────

def _assign_labels(proposals: list[dict]) -> dict[int, list[dict]]:
    """Largest candidate -> GTVp (1), rest -> GTVn (2).

    Single proposal -> assigned to both (cannot distinguish GTVp-only from
    GTVn-only at auto-prompting time; evaluate_predictions.py excludes absent
    labels via NaN so a spurious empty prediction costs nothing).
    """
    if not proposals:
        return {}
    result: dict[int, list[dict]] = {1: [{**proposals[0], "component_id": 1}]}
    if len(proposals) == 1:
        result[2] = [{**proposals[0], "component_id": 1}]
    else:
        result[2] = [{**p, "component_id": i}
                     for i, p in enumerate(proposals[1:], start=1)]
    return result


# ── AutoPrompter ───────────────────────────────────────────────────────────────

class AutoPrompter:
    """Generate automatic bounding-box prompts from CT + PET volumes.

    Parameters
    ----------
    method          : 'pet' | 'unet' | 'hybrid'
    model_path      : path to Small3DUNet checkpoint (.pt)
    pet_method      : PET thresholding strategy
    device          : 'cuda' | 'cpu'
    prob_threshold  : UNet probability threshold
    iou_threshold   : minimum IoU for hybrid PETxUNet filtering
    min_volume      : minimum component size in voxels
    nms_threshold   : 3-D NMS IoU threshold
    slice_pad       : Z-axis bounding-box padding in slices
    planar_pad      : Y/X bounding-box padding in voxels
    """

    def __init__(self,
                 method: str = "hybrid",
                 model_path: Optional[str] = None,
                 pet_method: str = "base41",
                 device: str = "cpu",
                 prob_threshold: float = 0.25,
                 iou_threshold: float = 0.05,
                 min_volume: int = 50,
                 nms_threshold: float = 0.3,
                 slice_pad: int = 1,
                 planar_pad: int = 5) -> None:

        self.method         = method.lower()
        self.pet_method     = pet_method.lower()
        self.device         = device
        self.prob_threshold = prob_threshold
        self.iou_threshold  = iou_threshold
        self.min_volume     = min_volume
        self.nms_threshold  = nms_threshold
        self.slice_pad      = slice_pad
        self.planar_pad     = planar_pad
        self.model          = None

        if self.method in ("unet", "hybrid"):
            if model_path is None:
                logger.warning(
                    "method='%s' but no model_path provided. "
                    "Falling back to 'pet' mode.", self.method
                )
                self.method = "pet"
            else:
                self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        from auto_prompting.proposal_net import Small3DUNet
        logger.info("Loading proposal network from %s", path)
        self.model = Small3DUNet.load(path, device=self.device)

    # ── Main interface ─────────────────────────────────────────────────────────

    def get_proposals(self,
                      ct_uint8: np.ndarray,
                      pet_uint8: np.ndarray,
                      suv_max: Optional[float] = None,
                      max_proposals: int = 10,
                      pet_suv_volume: Optional[np.ndarray] = None,
                      ) -> dict[int, list[dict]]:
        """Return a label-keyed dict of candidate proposals.

        Parameters
        ----------
        ct_uint8       : (D, H, W) uint8 CT array
        pet_uint8      : (D, H, W) uint8 PET array (normalised to [0, 255])
        suv_max        : per-patient SUV ceiling for TemPoRAL reconstruct_suv().
                         Ignored when ``pet_suv_volume`` is provided.
        max_proposals  : cap on total proposals after NMS
        pet_suv_volume : (D, H, W) float32 raw SUV array from SwinCross NPZ
                         ``pet`` key.  When provided, nestle/black/daisne use
                         this directly instead of reconstructing from uint8.
                         For base41 this parameter is unused.
        """
        pet_props = unet_props = []

        if self.method in ("pet", "hybrid"):
            pet_props = get_pet_proposals(
                pet_uint8   = pet_uint8,
                suv_max     = suv_max,
                method      = self.pet_method,
                min_volume  = self.min_volume,
                slice_pad   = self.slice_pad,
                planar_pad  = self.planar_pad,
                pet_suv     = pet_suv_volume,   # None for TemPoRAL, float32 for SwinCross
            )
            logger.debug("PET proposals (before NMS): %d", len(pet_props))

        if self.method in ("unet", "hybrid"):
            unet_props = _unet_proposals(
                ct_uint8   = ct_uint8,
                pet_uint8  = pet_uint8,
                model      = self.model,
                device     = self.device,
                threshold  = self.prob_threshold,
                min_volume = self.min_volume,
                slice_pad  = self.slice_pad,
                planar_pad = self.planar_pad,
            )
            logger.debug("UNet proposals: %d", len(unet_props))

        if self.method == "pet":
            raw = pet_props
        elif self.method == "unet":
            raw = unet_props
        else:
            raw = _hybrid_filter(pet_props, unet_props, self.iou_threshold)

        proposals = nms_3d(raw, iou_threshold=self.nms_threshold)[:max_proposals]
        for i, p in enumerate(proposals, start=1):
            p["component_id"] = i

        result = _assign_labels(proposals)
        logger.info("Auto-prompts -> GTVp: %d, GTVn: %d",
                    len(result.get(1, [])), len(result.get(2, [])))
        return result

    def __repr__(self) -> str:
        return (f"AutoPrompter(method={self.method!r}, "
                f"pet_method={self.pet_method!r}, "
                f"slice_pad={self.slice_pad}, planar_pad={self.planar_pad}, "
                f"model={'loaded' if self.model else 'none'})")
