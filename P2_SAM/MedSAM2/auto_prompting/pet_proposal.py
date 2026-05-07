"""
auto_prompting/pet_proposals.py
================================
PET intensity-based bounding-box proposals.

Three thresholding strategies are implemented:

  base41  – 41 % of SUV_max (fastest, best overall Dice in literature)
  black   – iterative, threshold = 0.307 * SUV_mean + 0.588
  daisne  – iterative, threshold = 31.3 + 77.7 / Contmeas

IMPORTANT – SUV scaling
-----------------------
The Black and Daisne constants were calibrated on raw SUV values (g/mL).
Running them on the uint8-normalised arrays stored in the NPZ files will
produce nonsensical thresholds.  Always call ``reconstruct_suv`` first when
using those methods.  ``base41`` is scale-invariant (41 % of max) and works
directly on uint8 data.

If ``pet_suv_max`` was not saved in the NPZ (old-format files), Black and
Daisne fall back gracefully to ``base41`` with a logged warning.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import label as cc_label

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# SUV reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_suv(pet_uint8: np.ndarray, suv_max: float) -> np.ndarray:
    """Reverse the [0, 255] uint8 normalization to approximate SUV values.

    The normalisation in ``prepare_hecktor_npz.py`` is:
        pet_uint8 = clip(pet_suv / suv_max, 0, 1) * 255

    So the inverse is:
        suv ≈ (pet_uint8 / 255) * suv_max

    Parameters
    ----------
    pet_uint8 : (D, H, W) uint8 array
    suv_max   : float  – per-patient scaling factor saved in the NPZ

    Returns
    -------
    (D, H, W) float32 SUV array
    """
    return (pet_uint8.astype(np.float32) / 255.0) * float(suv_max)


# ──────────────────────────────────────────────────────────────────────────────
# Threshold functions
# ──────────────────────────────────────────────────────────────────────────────

def base41_mask(pet: np.ndarray) -> np.ndarray:
    """41 % of SUV_max thresholding.  Works on both uint8 and raw SUV."""
    thr = 0.41 * float(pet.max())
    return pet > thr


def black_mask(suv: np.ndarray,
               alpha: float = 0.307,
               beta: float = 0.588,
               max_iter: int = 200) -> np.ndarray:
    """Black iterative method.  Requires raw SUV values (g/mL).

    threshold_k+1 = alpha * SUV_mean(mask_k) + beta

    Convergence when |vol_k - vol_{k-1}| <= 1 voxel.
    """
    thr = 0.41 * float(suv.max())
    prev_vol = -1
    mask = suv > thr
    for _ in range(max_iter):
        vol = int(mask.sum())
        if abs(vol - prev_vol) <= 1 or vol == 0:
            break
        prev_vol = vol
        suv_mean = float(suv[mask].mean())
        thr = alpha * suv_mean + beta
        mask = suv > thr
    return mask


def daisne_mask(suv: np.ndarray,
                a: float = 31.3,
                b: float = 77.7,
                max_iter: int = 200) -> np.ndarray:
    """Daisne iterative method.  Requires raw SUV values (g/mL).

    threshold_k+1 = a + b / Contmeas_k
    Contmeas = Maxavg / Bavg
      Maxavg – mean SUV of the top-10 % of voxels in the current mask
      Bavg   – mean SUV of background voxels with SUV > 0.5

    Convergence when |vol_k - vol_{k-1}| <= 1 voxel.
    """
    thr = 0.41 * float(suv.max())
    prev_vol = -1
    mask = suv > thr
    for _ in range(max_iter):
        vol = int(mask.sum())
        if abs(vol - prev_vol) <= 1 or vol == 0:
            break
        prev_vol = vol
        mask_vals = suv[mask]
        maxavg = float(np.percentile(mask_vals, 90))
        bg = (~mask) & (suv > 0.5)
        bavg = float(suv[bg].mean()) if bg.sum() > 0 else 1.0
        contmeas = maxavg / bavg if bavg > 0 else 1.0
        if contmeas <= 0:
            break
        thr = a + b / contmeas
        mask = suv > thr
    return mask


# ──────────────────────────────────────────────────────────────────────────────
# Component filtering
# ──────────────────────────────────────────────────────────────────────────────

def _shape_features(comp_mask: np.ndarray) -> dict | None:
    coords = np.array(np.where(comp_mask))
    if coords.shape[1] == 0:
        return None
    z, y, x = coords
    dz = int(z.max() - z.min()) + 1
    dy = int(y.max() - y.min()) + 1
    dx = int(x.max() - x.min()) + 1
    vol = coords.shape[1]
    bbox_vol = dz * dy * dx
    return {
        "volume": vol,
        "compactness": vol / (bbox_vol + 1e-6),
        "elongation": max(dx, dy, dz) / (min(dx, dy, dz) + 1e-6),
    }


def _passes_filter(feats: dict | None,
                   min_volume: int,
                   max_elongation: float,
                   min_compactness: float) -> bool:
    if feats is None:
        return False
    return (feats["volume"] >= min_volume
            and feats["elongation"] <= max_elongation
            and feats["compactness"] >= min_compactness)


# ──────────────────────────────────────────────────────────────────────────────
# Bounding-box extraction from a binary mask
# ──────────────────────────────────────────────────────────────────────────────

def mask_to_proposals(binary_mask: np.ndarray,
                      min_volume: int = 50,
                      max_elongation: float = 15.0,
                      min_compactness: float = 0.04,
                      bbox_pad: int = 5) -> list[dict]:
    """Convert a 3-D binary mask into a list of proposal dicts.

    Each proposal contains:
        z_mid       – axial slice with the largest cross-section
        bbox_2d     – [x_min, y_min, x_max, y_max] on z_mid (padded)
        bbox_3d     – [z0, z1, y0, y1, x0, x1] (padded)
        voxel_count – number of foreground voxels in the component
        component_id– 1-based index

    Sorted descending by ``voxel_count``.
    """
    D, H, W = binary_mask.shape
    labeled, num = cc_label(binary_mask)
    proposals = []

    for comp_id in range(1, num + 1):
        comp = labeled == comp_id
        feats = _shape_features(comp)
        if not _passes_filter(feats, min_volume, max_elongation, min_compactness):
            continue

        zz, yy, xx = np.where(comp)
        z0 = max(0, int(zz.min()) - bbox_pad)
        z1 = min(D, int(zz.max()) + 1 + bbox_pad)
        y0 = max(0, int(yy.min()) - bbox_pad)
        y1 = min(H, int(yy.max()) + 1 + bbox_pad)
        x0 = max(0, int(xx.min()) - bbox_pad)
        x1 = min(W, int(xx.max()) + 1 + bbox_pad)

        # Z slice with the largest cross-sectional area
        areas = np.array([comp[z].sum() for z in range(z0, z1)])
        z_mid = z0 + int(np.argmax(areas))

        # Tight 2-D bbox on z_mid slice, then expand to padded 3-D bounds
        ys_mid, xs_mid = np.where(comp[z_mid])
        if len(xs_mid) == 0:
            # Fallback: use padded 3D bounds projected onto this slice
            bbox_2d = np.array([x0, y0, x1, y1], dtype=np.int32)
        else:
            bbox_2d = np.array([
                max(0, int(xs_mid.min()) - bbox_pad),
                max(0, int(ys_mid.min()) - bbox_pad),
                min(W, int(xs_mid.max()) + 1 + bbox_pad),
                min(H, int(ys_mid.max()) + 1 + bbox_pad),
            ], dtype=np.int32)

        proposals.append({
            "component_id": comp_id,
            "voxel_count":  int(feats["volume"]),
            "z_mid":        z_mid,
            "bbox_2d":      bbox_2d,
            "bbox_3d":      [z0, z1, y0, y1, x0, x1],
        })

    proposals.sort(key=lambda p: p["voxel_count"], reverse=True)
    return proposals


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def get_pet_proposals(pet_uint8: np.ndarray,
                      suv_max: Optional[float] = None,
                      method: str = "base41",
                      min_volume: int = 50,
                      max_elongation: float = 15.0,
                      min_compactness: float = 0.04,
                      bbox_pad: int = 5) -> list[dict]:
    """Run PET thresholding and return a list of candidate proposals.

    Parameters
    ----------
    pet_uint8      : (D, H, W) uint8 PET array from the NPZ file
    suv_max        : float or None – per-patient scale factor from the NPZ
                     Required for 'black' and 'daisne'.
    method         : 'base41' | 'black' | 'daisne'
    min_volume     : minimum component size in voxels
    max_elongation : discard very elongated components (vessels, etc.)
    min_compactness: discard very sparse components
    bbox_pad       : padding added around each component bounding box

    Returns
    -------
    list of proposal dicts sorted by voxel_count descending
    """
    method = method.lower()

    if method in ("black", "daisne"):
        if suv_max is None or suv_max <= 0:
            warnings.warn(
                f"Method '{method}' requires pet_suv_max but it is not available "
                "(NPZ may have been prepared with an old version of prepare_hecktor_npz.py). "
                "Falling back to 'base41'.",
                UserWarning,
                stacklevel=2,
            )
            method = "base41"
        else:
            suv = reconstruct_suv(pet_uint8, suv_max)

    if method == "base41":
        binary = base41_mask(pet_uint8)
    elif method == "black":
        binary = black_mask(suv)
    elif method == "daisne":
        binary = daisne_mask(suv)
    else:
        raise ValueError(f"Unknown PET method '{method}'. "
                         "Choose 'base41', 'black', or 'daisne'.")

    return mask_to_proposals(binary, min_volume, max_elongation, min_compactness, bbox_pad)
