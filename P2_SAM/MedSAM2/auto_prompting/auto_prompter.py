"""
auto_prompting/auto_prompter.py
================================
AutoPrompter: unified interface for automatic bounding-box prompt generation.

Methods
-------
  pet      – PET thresholding only (fast, high recall)
  unet     – Small3DUNet proposal network only
  hybrid   – PET ∪ UNet filtered by 3-D IoU overlap (recommended)

Output format (matches infer_hecktor.py expectations)
-------------------------------------------------------
{
    1: [   # GTVp – the largest candidate
        {"component_id": 1, "voxel_count": int, "z_mid": int,
         "bbox_2d": np.ndarray([x0, y0, x1, y1])}
    ],
    2: [   # GTVn – all remaining candidates
        {"component_id": 1, ...},
        ...
    ]
}

Label assignment heuristic:
  The biggest proposal (by voxel_count) → GTVp (label 1)
  All remaining proposals               → GTVn (label 2)
This is a reasonable starting point for H&N; GTVp is typically the
dominant primary tumour, GTVn are smaller nodal metastases.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from auto_prompting.pet_proposals import get_pet_proposals, mask_to_proposals
from auto_prompting.box_utils import iou_3d, nms_3d

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Helper: run proposal net on a full volume
# ──────────────────────────────────────────────────────────────────────────────

def _unet_proposals(ct_uint8: np.ndarray,
                    pet_uint8: np.ndarray,
                    model,
                    device: str,
                    threshold: float = 0.25,
                    min_volume: int = 50,
                    bbox_pad: int = 5) -> list[dict]:
    """Run the Small3DUNet and convert the probability map to proposals."""
    ct  = ct_uint8.astype(np.float32) / 255.0
    pet = pet_uint8.astype(np.float32) / 255.0

    x = torch.tensor(np.stack([ct, pet], axis=0)).unsqueeze(0).float()  # (1, 2, D, H, W)
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        prob = model(x)[0, 0].cpu().numpy()   # (D, H, W) in [0, 1]

    binary = prob > threshold
    return mask_to_proposals(binary, min_volume=min_volume, bbox_pad=bbox_pad)


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid filtering
# ──────────────────────────────────────────────────────────────────────────────

def _hybrid_filter(pet_props: list[dict],
                   unet_props: list[dict],
                   iou_threshold: float = 0.05) -> list[dict]:
    """Keep PET proposals that overlap with at least one UNet proposal.

    Falls back to UNet proposals if no overlap is found (e.g., the PET
    method produced only false positives or the UNet found nothing).
    """
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


# ──────────────────────────────────────────────────────────────────────────────
# Label assignment
# ──────────────────────────────────────────────────────────────────────────────

def _assign_labels(proposals: list[dict]) -> dict[int, list[dict]]:
    """Assign GTVp (1) to the largest candidate, GTVn (2) to the rest."""
    if not proposals:
        return {}

    result: dict[int, list[dict]] = {}

    # Largest → GTVp
    gtvp = {**proposals[0], "component_id": 1}
    result[1] = [gtvp]

    # Remaining → GTVn
    if len(proposals) > 1:
        gtvn = []
        for i, p in enumerate(proposals[1:], start=1):
            gtvn.append({**p, "component_id": i})
        result[2] = gtvn

    return result


# ──────────────────────────────────────────────────────────────────────────────
# AutoPrompter
# ──────────────────────────────────────────────────────────────────────────────

class AutoPrompter:
    """Generate automatic bounding-box prompts from CT + PET volumes.

    Parameters
    ----------
    method          : 'pet' | 'unet' | 'hybrid'
    model_path      : path to Small3DUNet checkpoint (.pt)
                      Required for 'unet' and 'hybrid'.
    pet_method      : PET thresholding strategy ('base41' | 'black' | 'daisne')
    device          : 'cuda' | 'cpu'
    prob_threshold  : UNet probability threshold (default 0.25 – recall-biased)
    iou_threshold   : minimum IoU for hybrid filtering (default 0.05)
    min_volume      : minimum component size in voxels (default 50)
    nms_threshold   : 3-D NMS IoU threshold (default 0.3)
    bbox_pad        : bounding-box padding in voxels (default 5)
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
                 bbox_pad: int = 5) -> None:

        self.method = method.lower()
        self.pet_method = pet_method.lower()
        self.device = device
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.min_volume = min_volume
        self.nms_threshold = nms_threshold
        self.bbox_pad = bbox_pad
        self.model = None

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

    # ──────────────────────────────────────────────────────────────────────
    # Main interface
    # ──────────────────────────────────────────────────────────────────────

    def get_proposals(self,
                      ct_uint8: np.ndarray,
                      pet_uint8: np.ndarray,
                      suv_max: Optional[float] = None,
                      max_proposals: int = 10) -> dict[int, list[dict]]:
        """Return a label-keyed dict of candidate proposals.

        Parameters
        ----------
        ct_uint8     : (D, H, W) uint8 CT array (from NPZ)
        pet_uint8    : (D, H, W) uint8 PET array (from NPZ)
        suv_max      : per-patient SUV scaling factor (from NPZ)
                       Required for 'black' and 'daisne' PET methods.
        max_proposals: cap on total proposals after NMS (default 10)

        Returns
        -------
        {label_id: [component_dict, ...]}
        """
        raw_proposals: list[dict] = []

        if self.method in ("pet", "hybrid"):
            pet_props = get_pet_proposals(
                pet_uint8,
                suv_max=suv_max,
                method=self.pet_method,
                min_volume=self.min_volume,
                bbox_pad=self.bbox_pad,
            )
            logger.info("PET proposals (before NMS): %d", len(pet_props))

        if self.method in ("unet", "hybrid"):
            unet_props = _unet_proposals(
                ct_uint8, pet_uint8,
                model=self.model,
                device=self.device,
                threshold=self.prob_threshold,
                min_volume=self.min_volume,
                bbox_pad=self.bbox_pad,
            )
            logger.info("UNet proposals: %d", len(unet_props))

        if self.method == "pet":
            raw_proposals = pet_props
        elif self.method == "unet":
            raw_proposals = unet_props
        elif self.method == "hybrid":
            raw_proposals = _hybrid_filter(pet_props, unet_props, self.iou_threshold)

        # NMS to remove near-duplicate boxes
        proposals = nms_3d(raw_proposals, iou_threshold=self.nms_threshold)

        # Cap and re-index
        proposals = proposals[:max_proposals]
        for i, p in enumerate(proposals, start=1):
            p["component_id"] = i

        result = _assign_labels(proposals)
        n1 = len(result.get(1, []))
        n2 = len(result.get(2, []))
        logger.info("Final proposals → GTVp: %d, GTVn: %d", n1, n2)

        return result

    def __repr__(self) -> str:
        return (f"AutoPrompter(method={self.method!r}, "
                f"pet_method={self.pet_method!r}, "
                f"model={'loaded' if self.model else 'none'})")
