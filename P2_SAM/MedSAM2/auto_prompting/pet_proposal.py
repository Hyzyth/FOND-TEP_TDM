"""
auto_prompting/pet_proposals.py
================================
PET intensity-based bounding-box proposals.

Four thresholding strategies are implemented:

  base41  – 41 % of SUV_max (fastest, best overall Dice in published benchmarks)
  nestle  – non-iterative: threshold = alpha * I0.7 + Ibgd
  black   – iterative: threshold = 0.307 * SUV_mean + 0.588
  daisne  – iterative: threshold = 31.3 + 77.7 / Contmeas

Published benchmark (semi-automatic segmentation on HECKTOR):

  Method   Recall  Precision  Dice
  base41   0.56    0.16       0.33   ← best Dice
  nestle   0.36    0.50       0.19   ← best precision (fewest FP)
  black    0.64    0.14       0.20
  daisne   0.78    0.06       0.12   ← highest recall (most FP)

IMPORTANT – SUV scaling
-----------------------
Black and Daisne constants are calibrated for raw SUV values (g/mL).
The Nestle constants (alpha=0.15, Ibgd from liver) are likewise SUV-based.
Always call `reconstruct_suv` first when using those methods.

If `pet_suv_max` is not present in the NPZ (old-format files), all
iterative/contrast-based methods fall back to `base41` with a warning.

Component extraction
--------------------
`mask_to_proposals` reuses `slicer.find_all_components_for_label` and
`slicer.get_z_prompt_from_component` to avoid duplicating connected-component
and bounding-box logic that already exists in the codebase.  The only
additions here are the shape filters (elongation, compactness) that the slicer
does not apply, since those are specific to the proposal-generation context.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import label as cc_label

logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# Lazy slicer import (slicer.py lives at repo root, not in this package)
# ---------------------------------------------------------------------------

def _import_slicer():
    """Import slicer utilities, adding the repo root to sys.path if needed."""
    try:
        from slicer import find_all_components_for_label, get_z_prompt_from_component
        return find_all_components_for_label, get_z_prompt_from_component
    except ImportError:
        # Walk up until we find slicer.py
        here = os.path.dirname(os.path.abspath(__file__))
        for candidate in [here, os.path.dirname(here)]:
            if os.path.isfile(os.path.join(candidate, "slicer.py")):
                if candidate not in sys.path:
                    sys.path.insert(0, candidate)
                from slicer import find_all_components_for_label, get_z_prompt_from_component
                return find_all_components_for_label, get_z_prompt_from_component
        raise ImportError(
            "slicer.py not found. Make sure the repo root is on PYTHONPATH."
        )


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


def nestle_mask(suv: np.ndarray,
                alpha: float = 0.15,
                bg_min_suv: float = 0.5) -> np.ndarray:
    """Nestle non-iterative method.  Requires raw SUV values (g/mL).

    threshold = alpha * I0.7 + Ibgd

    I0.7  = mean SUV inside a 70%-SUVmax segmentation
    Ibgd  = mean SUV of background tissue
            (original paper: liver/mediastinum via CNN segmentation)
            (here: mean of all non-mask voxels with SUV > bg_min_suv,
             which avoids air but approximates the background tissue SUV)

    Parameters
    ----------
    suv       : raw SUV float32 array
    alpha     : weight on I0.7 (default 0.15 as in the original paper)
    bg_min_suv: minimum SUV to include in the background estimate (default 0.5)
    """
    suv_max = float(suv.max())
    mask_70 = suv > 0.70 * suv_max
    if mask_70.sum() == 0:
        logger.warning("Nestle: 70%%SUVmax mask is empty — falling back to base41.")
        return base41_mask(suv)

    i07  = float(suv[mask_70].mean())
    bg   = (~mask_70) & (suv > bg_min_suv)
    ibgd = float(suv[bg].mean()) if bg.sum() > 0 else 1.0

    thr = alpha * i07 + ibgd
    return suv > thr


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
        if mask_vals.size == 0:
            break

        # Maxavg: top 10% mean
        k = max(1, int(0.1 * mask_vals.size))
        maxavg = float(np.mean(np.sort(mask_vals)[-k:]))

        # Background: full complement
        bg = ~mask
        bavg = float(suv[bg].mean()) if bg.sum() > 0 else float(suv.mean())

        contmeas = maxavg / (bavg + 1e-6)
        thr = a + b / contmeas
        mask = suv > thr
    return mask

# ---------------------------------------------------------------------------
# Shape filtering
# ---------------------------------------------------------------------------

def _shape_features(voxel_count: int, bbox_voxel: dict) -> dict:
    """Compute elongation and compactness from the component bbox."""
    z0, z1 = bbox_voxel["z"]
    y0, y1 = bbox_voxel["y"]
    x0, x1 = bbox_voxel["x"]
    dz = max(1, z1 - z0)
    dy = max(1, y1 - y0)
    dx = max(1, x1 - x0)
    bbox_vol = dz * dy * dx
    return {
        "compactness": voxel_count / (bbox_vol + 1e-6),
        "elongation":  max(dx, dy, dz) / (min(dx, dy, dz) + 1e-6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Bounding-box extraction from a binary mask
# ──────────────────────────────────────────────────────────────────────────────

def mask_to_proposals(binary_mask: np.ndarray,
                      min_volume: int = 50,
                      max_elongation: float = 15.0,
                      min_compactness: float = 0.04,
                      slice_pad: int = 1,
                      planar_pad: int = 5) -> list[dict]:
    """Convert a 3-D binary mask into a list of proposal dicts.

    Delegates connected-component labelling and bounding-box extraction to
    ``slicer.find_all_components_for_label`` + ``slicer.get_z_prompt_from_component``
    to avoid duplicating that logic.  Additional shape filtering (elongation,
    compactness) is applied on top, since the slicer does not filter.

    Parameters
    ----------
    binary_mask    : (D, H, W) bool/uint8 array  (foreground = 1)
    min_volume     : discard components smaller than this (voxels)
    max_elongation : discard very elongated blobs (vessels, noise)
    min_compactness: discard very sparse blobs
    slice_pad      : padding along the axial (Z) axis in slices
    planar_pad     : padding on the 2-D plane (Y, X) in voxels

    Returns
    -------
    list of dicts, sorted by voxel_count descending.  Each dict:
        component_id, voxel_count, z_mid, bbox_2d, bbox_3d
    """
    find_all, get_z = _import_slicer()

    # Treat binary mask as a single-label integer mask (label value = 1)
    int_mask = binary_mask.astype(np.int16)

    comps = find_all(
        mask        = int_mask,
        label_value = 1,
        slice_pad   = slice_pad,
        planar_pad  = planar_pad,
    )

    proposals = []
    for comp in comps:
        vc = comp["voxel_count"]
        if vc < min_volume:
            continue

        feats = _shape_features(vc, comp["bbox_voxel"])
        if feats["elongation"]  > max_elongation:
            continue
        if feats["compactness"] < min_compactness:
            continue

        # Always derive a Z-axis prompt for the video predictor
        z_mid, bbox_2d = get_z(comp)

        bv = comp["bbox_voxel"]
        proposals.append({
            "component_id": comp["component_id"],
            "voxel_count":  vc,
            "z_mid":        z_mid,
            "bbox_2d":      bbox_2d,
            "bbox_3d": [bv["z"][0], bv["z"][1],
                        bv["y"][0], bv["y"][1],
                        bv["x"][0], bv["x"][1]],
        })

    proposals.sort(key=lambda p: p["voxel_count"], reverse=True)
    return proposals


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

_METHODS_REQUIRING_SUV = ("nestle", "black", "daisne")


def get_pet_proposals(pet_uint8: np.ndarray,
                      suv_max: Optional[float] = None,
                      method: str = "base41",
                      min_volume: int = 50,
                      max_elongation: float = 15.0,
                      min_compactness: float = 0.04,
                      slice_pad: int = 1,
                      planar_pad: int = 5) -> list[dict]:
    """Run PET thresholding and return a list of candidate proposals.

    Parameters
    ----------
    pet_uint8      : (D, H, W) uint8 PET array from the NPZ file
    suv_max        : float or None – per-patient scale factor (from NPZ key
                     ``pet_suv_max``, written by updated prepare_hecktor_npz.py).
                     Required for 'nestle', 'black', and 'daisne'.
    method         : 'base41' | 'nestle' | 'black' | 'daisne'
    min_volume     : minimum component size in voxels
    max_elongation : discard very elongated components (vessels, etc.)
    min_compactness: discard very sparse components
    slice_pad      : axial (Z) bounding-box padding in slices
                     (applied along the viewing/propagation axis)
    planar_pad     : planar (Y, X) bounding-box padding in voxels

    Returns
    -------
    list of proposal dicts sorted by voxel_count descending
    """
    method = method.lower()

    if method in _METHODS_REQUIRING_SUV:
        if suv_max is None or suv_max <= 0:
            warnings.warn(
                f"Method '{method}' requires raw SUV values (pet_suv_max) but "
                "it is not available in the NPZ.  Re-run prepare_hecktor_npz.py "
                "to save pet_suv_max.  Falling back to 'base41'.",
                UserWarning, stacklevel=2,
            )
            method = "base41"
        else:
            suv = reconstruct_suv(pet_uint8, suv_max)

    if method == "base41":
        binary = base41_mask(pet_uint8)
    elif method == "nestle":
        binary = nestle_mask(suv)
    elif method == "black":
        binary = black_mask(suv)
    elif method == "daisne":
        binary = daisne_mask(suv)
    else:
        raise ValueError(
            f"Unknown PET method '{method}'. "
            "Choose 'base41', 'nestle', 'black', or 'daisne'."
        )

    return mask_to_proposals(
        binary_mask    = binary,
        min_volume     = min_volume,
        max_elongation = max_elongation,
        min_compactness= min_compactness,
        slice_pad      = slice_pad,
        planar_pad     = planar_pad,
    )
