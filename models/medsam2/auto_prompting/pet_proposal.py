"""
auto_prompting/pet_proposals.py
================================
PET intensity-based bounding-box proposals.

Four thresholding strategies are implemented:

  base41  - 41 % of SUV_max        (fastest, scale-invariant, works on any input)
  nestle  - alpha * I0.7 + Ibgd    (best precision, requires raw SUV)
  black   - iterative SUVmean       (high recall, requires raw SUV)
  daisne  - iterative contrast      (highest recall, requires raw SUV)

UPDATED: get_pet_proposals() now accepts an optional ``pet_suv`` kwarg that
carries a pre-computed raw-SUV float32 array (directly from the SwinCross NPZ
``pet`` key).  When provided, the reconstruct_suv() step is skipped entirely,
preserving the full dynamic range of SUV values without any clipping artefact
from the uint8 round-trip.

Backward compatibility
----------------------
  TemPoRAL path (old MedSAM2 NPZ): pass ``pet_uint8`` + ``suv_max`` as before.
    reconstruct_suv(pet_uint8, suv_max) is called internally.
  SwinCross path (new):             pass ``pet_suv`` (float32 volume).
    reconstruct_suv is skipped.
  If neither is available:          nestle/black/daisne fall back to base41
    with a warning (same behaviour as before).
"""

from __future__ import annotations

import logging
import sys
import os
import warnings
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _import_slicer():
    """Import slicer utilities, adding the repo root to sys.path if needed."""
    try:
        from slicer import find_all_components_for_label, get_z_prompt_from_component
        return find_all_components_for_label, get_z_prompt_from_component
    except ImportError:
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


# ── SUV reconstruction (TemPoRAL path) ────────────────────────────────────────

def reconstruct_suv(pet_uint8: np.ndarray, suv_max: float) -> np.ndarray:
    """Reverse the uint8 normalisation to recover approximate SUV values.

    Used only for the TemPoRAL NPZ path where PET was stored as uint8.
    For SwinCross NPZ, raw SUV is already available in float16/float32 - pass
    it directly via the ``pet_suv`` argument to ``get_pet_proposals()``.

    The normalisation in ``prepare_temporal_npz.py`` is:
        pet_uint8 = clip(pet_suv / suv_max, 0, 1) * 255

    Inverse:
        suv ≈ (pet_uint8 / 255) * suv_max
    """
    return (pet_uint8.astype(np.float32) / 255.0) * float(suv_max)


# ── Thresholding functions ─────────────────────────────────────────────────────

def base41_mask(pet: np.ndarray, pct: float = 0.41) -> np.ndarray:
    """pct x SUVmax thresholding.  Scale-invariant - works on uint8 or raw SUV."""
    return pet > pct * float(pet.max())


def nestle_mask(suv: np.ndarray,
                alpha: float = 0.15,
                bg_min_suv: float = 0.5) -> np.ndarray:
    """Nestle non-iterative method.  Requires raw SUV values (g/mL)."""
    suv_max = float(suv.max())
    mask_70 = suv > 0.70 * suv_max
    if mask_70.sum() == 0:
        logger.warning("Nestle: 70%%SUVmax mask is empty - falling back to base41.")
        return base41_mask(suv)
    i07  = float(suv[mask_70].mean())
    bg   = (~mask_70) & (suv > bg_min_suv)
    ibgd = float(suv[bg].mean()) if bg.sum() > 0 else 1.0
    return suv > (alpha * i07 + ibgd)


def black_mask(suv: np.ndarray,
               alpha: float = 0.307,
               beta: float = 0.588,
               max_iter: int = 200) -> np.ndarray:
    """Black iterative method.  Requires raw SUV values (g/mL)."""
    thr  = 0.41 * float(suv.max())
    prev = -1
    mask = suv > thr
    for _ in range(max_iter):
        vol = int(mask.sum())
        if abs(vol - prev) <= 1 or vol == 0:
            break
        prev = vol
        thr  = alpha * float(suv[mask].mean()) + beta
        mask = suv > thr
    return mask


def daisne_mask(suv: np.ndarray,
                a: float = 31.3,
                b: float = 77.7,
                max_iter: int = 200) -> np.ndarray:
    """Daisne iterative method.  Requires raw SUV values (g/mL)."""
    thr  = 0.41 * float(suv.max())
    prev = -1
    mask = suv > thr
    for _ in range(max_iter):
        vol = int(mask.sum())
        if abs(vol - prev) <= 1 or vol == 0:
            break
        prev = vol
        vals    = suv[mask]
        maxavg  = float(np.percentile(vals, 90))
        bg      = (~mask) & (suv > 0.5)
        bavg    = float(suv[bg].mean()) if bg.sum() > 0 else 1.0
        contmeas = maxavg / bavg if bavg > 0 else 1.0
        if contmeas <= 0:
            break
        thr  = a + b / contmeas
        mask = suv > thr
    return mask


# ── Shape filtering ────────────────────────────────────────────────────────────

def _shape_features(voxel_count: int, bbox_voxel: dict) -> dict:
    z0, z1 = bbox_voxel["z"]
    y0, y1 = bbox_voxel["y"]
    x0, x1 = bbox_voxel["x"]
    dz = max(1, z1 - z0); dy = max(1, y1 - y0); dx = max(1, x1 - x0)
    bbox_vol = dz * dy * dx
    return {
        "compactness": voxel_count / (bbox_vol + 1e-6),
        "elongation":  max(dx, dy, dz) / (min(dx, dy, dz) + 1e-6),
    }


# ── Proposal extraction from binary mask ──────────────────────────────────────

def mask_to_proposals(binary_mask: np.ndarray,
                      min_volume: int = 50,
                      max_elongation: float = 15.0,
                      min_compactness: float = 0.04,
                      slice_pad: int = 1,
                      planar_pad: int = 5) -> list[dict]:
    """Convert a 3-D binary mask into a list of proposal dicts."""
    find_all, get_z = _import_slicer()
    int_mask = binary_mask.astype(np.int16)
    comps    = find_all(mask=int_mask, label_value=1,
                        slice_pad=slice_pad, planar_pad=planar_pad)
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


# ── Public entry point ────────────────────────────────────────────────────────

_METHODS_REQUIRING_SUV = ("nestle", "black", "daisne")


def get_pet_proposals(pet_uint8: np.ndarray,
                      suv_max: Optional[float] = None,
                      method: str = "base41",
                      min_volume: int = 50,
                      max_elongation: float = 15.0,
                      min_compactness: float = 0.04,
                      slice_pad: int = 1,
                      planar_pad: int = 5,
                      pet_suv: Optional[np.ndarray] = None,
                      **method_kwargs) -> list[dict]:
    """Run PET thresholding and return a list of candidate proposals.

    Parameters
    ----------
    pet_uint8 : (D, H, W) uint8 array
        PET in normalised uint8 form [0, 255].  Used by base41 and as
        fallback for shape when pet_suv is absent.
    suv_max   : float, optional
        Per-patient SUV ceiling used to reconstruct raw SUV from uint8.
        Required for nestle/black/daisne when ``pet_suv`` is not provided
        (TemPoRAL path).
    method    : str
        'base41' | 'nestle' | 'black' | 'daisne'
    pet_suv   : (D, H, W) float32 array, optional
        **Raw SUV values** directly from the SwinCross NPZ ``pet`` key
        (float16 cast to float32).  When provided, the reconstruct_suv()
        step is skipped and this array is passed directly to the
        thresholding function.  Takes precedence over ``suv_max``.

        For SwinCross NPZ:  pass  ``pet_suv=npz["pet"].astype(np.float32)``
        For TemPoRAL NPZ:   pass  ``suv_max=meta["pet_suv_max"]``  (as before)
    """
    method = method.lower()

    if method in _METHODS_REQUIRING_SUV:
        if pet_suv is not None:
            # ── SwinCross path: raw SUV already available ─────────────────
            suv = pet_suv.astype(np.float32)
            logger.debug("Using pre-loaded raw SUV array for method '%s'.", method)
        elif suv_max is not None and suv_max > 0:
            # ── TemPoRAL path: reconstruct from uint8 + ceiling ───────────
            suv = reconstruct_suv(pet_uint8, suv_max)
            logger.debug("Reconstructed SUV from uint8 (suv_max=%.2f) for method '%s'.",
                         suv_max, method)
        else:
            warnings.warn(
                f"Method '{method}' requires raw SUV but neither 'pet_suv' nor "
                "'suv_max' is available.  Falling back to 'base41'.",
                UserWarning, stacklevel=2,
            )
            method = "base41"

    if method == "base41":
        binary = base41_mask(pet_uint8, **method_kwargs)
    elif method == "nestle":
        binary = nestle_mask(suv, **method_kwargs)
    elif method == "black":
        binary = black_mask(suv, **method_kwargs)
    elif method == "daisne":
        binary = daisne_mask(suv, **method_kwargs)
    else:
        raise ValueError(
            f"Unknown PET method '{method}'. "
            "Choose 'base41', 'nestle', 'black', or 'daisne'."
        )

    return mask_to_proposals(
        binary_mask     = binary,
        min_volume      = min_volume,
        max_elongation  = max_elongation,
        min_compactness = min_compactness,
        slice_pad       = slice_pad,
        planar_pad      = planar_pad,
    )
