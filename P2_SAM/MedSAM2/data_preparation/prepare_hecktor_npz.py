"""
prepare_hecktor_npz.py
======================
Converts HECKTOR Task-1 segmentation data (NIfTI format) into the NPZ format
expected by MedSAM2's training and inference pipelines.

Expected HECKTOR data structure
--------------------------------
/data/santiago/HECKTOR_data/Task_1_segmentation/{patient_id}/
    {patient_id}__CT.nii.gz   – CT scan (HU values)
    {patient_id}__PT.nii.gz   – PET scan (SUV values)
    {patient_id}__GTVt.nii.gz – Primary tumour mask  (label 1 = GTVp)
    {patient_id}__GTVn.nii.gz – Nodal tumour mask    (label 1 = GTVn, optional)

Output NPZ format per patient
-------------------------------
    ct_imgs  : (D, H, W) uint8  – CT windowed & normalised to [0, 255]
    pet_imgs : (D, H, W) uint8  – PET normalised to [0, 255]
    gts      : (D, H, W) uint8  – 0 = background, 1 = GTVp, 2 = GTVn
    spacing  : (3,) float64     – voxel spacing in mm  (z, y, x)
    patient_id: str             – patient identifier

Usage
-----
python data_preparation/prepare_hecktor_npz.py \
    --data_dir /data/santiago/HECKTOR_data/Task_1_segmentation \
    --output_dir /data/ethan/MedSAM2/hecktor_npz \
    --val_ratio 0.2 \
    --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Default preprocessing hyperparameters
# ---------------------------------------------------------------------------

# CT soft-tissue window for head-and-neck (HU).
# Increasing the upper bound captures more bone context.
CT_WINDOW_LOW: int = -200    # HU lower bound
CT_WINDOW_HIGH: int = 800    # HU upper bound

# PET SUV clipping – values above this are saturated to 255.
# 99th-percentile-based normalisation is used when PET_SUV_MAX is None.
PET_SUV_MAX: Optional[float] = None   # None → per-patient 99th-percentile

# Minimum number of foreground voxels a slice must contain to be included.
MIN_FOREGROUND_VOXELS: int = 10


# ---------------------------------------------------------------------------
# File-naming patterns (tried in order until a match is found)
# ---------------------------------------------------------------------------

CT_PATTERNS = [
    "{pid}__CT.nii.gz",
    "CT.nii.gz",
    "{pid}_ct.nii.gz",
    "{pid}_CT.nii.gz",
]

PET_PATTERNS = [
    "{pid}__PT.nii.gz",
    "PET.nii.gz",
    "{pid}_pt.nii.gz",
    "{pid}_PET.nii.gz",
    "{pid}__PET.nii.gz",
]

GTVP_PATTERNS = [
    "{pid}__GTVt.nii.gz",
    "{pid}__GTV-T.nii.gz",
    "{pid}__GTV_T.nii.gz",
    "GTVt.nii.gz",
    "GTVp.nii.gz",
    "{pid}_GTVt.nii.gz",
]

GTVN_PATTERNS = [
    "{pid}__GTVn.nii.gz",
    "{pid}__GTV-N.nii.gz",
    "{pid}__GTV_N.nii.gz",
    "GTVn.nii.gz",
    "{pid}_GTVn.nii.gz",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def find_file(directory: Path, patterns: list, patient_id: str) -> Optional[Path]:
    """Return the first existing file matching any of the given patterns.

    Parameters
    ----------
    directory : Path
        Directory to search in.
    patterns : list[str]
        Filename patterns (may contain ``{pid}`` placeholder).
    patient_id : str
        Patient identifier used to fill ``{pid}``.

    Returns
    -------
    Path or None
        Path of the found file, or ``None`` if no pattern matches.
    """
    for pattern in patterns:
        candidate = directory / pattern.format(pid=patient_id)
        if candidate.exists():
            return candidate
    return None


def resample_to_reference(
    moving: sitk.Image,
    reference: sitk.Image,
    interpolator=sitk.sitkLinear,
    default_value: float = 0.0,
) -> sitk.Image:
    """Resample *moving* to the voxel grid defined by *reference*.

    Parameters
    ----------
    moving : sitk.Image
        Image to resample.
    reference : sitk.Image
        Image whose spacing / origin / direction will be matched.
    interpolator :
        SimpleITK interpolation method.
    default_value : float
        Fill value for regions outside *moving*.

    Returns
    -------
    sitk.Image
        Resampled image.
    """
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving)


def window_ct(ct_array: np.ndarray, low: int, high: int) -> np.ndarray:
    """Clip CT HU values and normalise to uint8 [0, 255].

    Parameters
    ----------
    ct_array : np.ndarray
        Raw CT array in HU.
    low, high : int
        Lower and upper HU bounds for the window.

    Returns
    -------
    np.ndarray  (dtype uint8)
    """
    ct_clipped = np.clip(ct_array, low, high).astype(np.float32)
    ct_norm = (ct_clipped - low) / (high - low) * 255.0
    return ct_norm.astype(np.uint8)


def normalise_pet(
    pet_array: np.ndarray,
    suv_max: Optional[float] = None,
) -> np.ndarray:
    """Normalise PET SUV values to uint8 [0, 255].

    Clips the array at ``suv_max`` (or the 99th percentile when *suv_max*
    is ``None``) then linearly maps to [0, 255].

    Parameters
    ----------
    pet_array : np.ndarray
        Raw PET array in SUV units.
    suv_max : float or None
        Upper bound for normalisation.  ``None`` ⇒ 99th percentile.

    Returns
    -------
    np.ndarray  (dtype uint8)
    """
    pet_array = pet_array.astype(np.float32)
    pet_array = np.clip(pet_array, 0.0, None)   # SUV cannot be negative

    upper = suv_max if suv_max is not None else float(np.percentile(pet_array, 99))
    if upper <= 0:
        upper = 1.0   # edge-case guard

    pet_norm = pet_array / upper * 255.0
    pet_norm = np.clip(pet_norm, 0.0, 255.0)
    return pet_norm.astype(np.uint8)


def build_combined_mask(
    gtvp_array: np.ndarray,
    gtvn_array: Optional[np.ndarray],
) -> np.ndarray:
    """Merge GTVp and GTVn into a single label mask.

    Label convention
    ----------------
    0 – background
    1 – GTVp (primary tumour)
    2 – GTVn (nodal tumour)

    Parameters
    ----------
    gtvp_array : np.ndarray
        Binary GTVp mask (any non-zero value is foreground).
    gtvn_array : np.ndarray or None
        Binary GTVn mask.  Pass ``None`` when absent.

    Returns
    -------
    np.ndarray  (dtype uint8)
    """
    combined = np.zeros_like(gtvp_array, dtype=np.uint8)
    combined[gtvp_array > 0] = 1
    if gtvn_array is not None:
        combined[gtvn_array > 0] = 2
    return combined


def crop_to_foreground(
    ct: np.ndarray,
    pet: np.ndarray,
    gts: np.ndarray,
    margin_z: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop the 3-D volume to the axial extent of the annotation plus a margin.

    This removes empty slices far from any tumour, reducing the number of
    "video frames" the model needs to track through.

    Parameters
    ----------
    ct, pet, gts : np.ndarray
        Co-registered arrays of shape (D, H, W).
    margin_z : int
        Number of extra slices to keep above and below the labelled region.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Cropped (ct, pet, gts).
    """
    labelled_slices = np.where(gts.sum(axis=(1, 2)) > 0)[0]
    if len(labelled_slices) == 0:
        return ct, pet, gts

    z_min = max(0, labelled_slices[0] - margin_z)
    z_max = min(ct.shape[0], labelled_slices[-1] + margin_z + 1)
    return ct[z_min:z_max], pet[z_min:z_max], gts[z_min:z_max]


# ---------------------------------------------------------------------------
# Per-patient processing
# ---------------------------------------------------------------------------

def process_patient(
    patient_dir: Path,
    patient_id: str,
    ct_window_low: int = CT_WINDOW_LOW,
    ct_window_high: int = CT_WINDOW_HIGH,
    pet_suv_max: Optional[float] = PET_SUV_MAX,
    crop_z: bool = True,
    crop_z_margin: int = 5,
) -> Optional[dict]:
    """Load, preprocess, and return NPZ-ready data for one patient.

    Parameters
    ----------
    patient_dir : Path
        Directory containing the patient's NIfTI files.
    patient_id : str
        Patient identifier used for filename resolution.
    ct_window_low, ct_window_high : int
        HU bounds for CT windowing.
    pet_suv_max : float or None
        SUV upper bound for PET normalisation.
    crop_z : bool
        Whether to crop axial extent to tumour + margin.
    crop_z_margin : int
        Extra slices above / below tumour when cropping.

    Returns
    -------
    dict or None
        Dictionary with keys ``ct_imgs``, ``pet_imgs``, ``gts``, ``spacing``,
        ``patient_id``; or ``None`` when required files are missing.
    """
    # ── Locate required files ─────────────────────────────────────────────
    ct_path = find_file(patient_dir, CT_PATTERNS, patient_id)
    pet_path = find_file(patient_dir, PET_PATTERNS, patient_id)
    gtvp_path = find_file(patient_dir, GTVP_PATTERNS, patient_id)

    if ct_path is None or pet_path is None or gtvp_path is None:
        print(
            f"  [SKIP] {patient_id}: missing required files "
            f"(CT={ct_path is not None}, PET={pet_path is not None}, "
            f"GTVp={gtvp_path is not None})"
        )
        return None

    gtvn_path = find_file(patient_dir, GTVN_PATTERNS, patient_id)  # optional

    # ── Load images ───────────────────────────────────────────────────────
    ct_sitk = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    pet_sitk = sitk.ReadImage(str(pet_path), sitk.sitkFloat32)
    gtvp_sitk = sitk.ReadImage(str(gtvp_path), sitk.sitkUInt8)
    gtvn_sitk = sitk.ReadImage(str(gtvn_path), sitk.sitkUInt8) if gtvn_path else None

    # ── Resample PET and masks onto the CT grid ───────────────────────────
    # CT is the reference space because its resolution is typically finer.
    pet_sitk = resample_to_reference(pet_sitk, ct_sitk, sitk.sitkLinear, 0.0)
    gtvp_sitk = resample_to_reference(
        gtvp_sitk, ct_sitk, sitk.sitkNearestNeighbor, 0
    )
    if gtvn_sitk is not None:
        gtvn_sitk = resample_to_reference(
            gtvn_sitk, ct_sitk, sitk.sitkNearestNeighbor, 0
        )

    # ── Convert to numpy (SimpleITK returns z, y, x order) ───────────────
    ct_array = sitk.GetArrayFromImage(ct_sitk)           # (D, H, W) float32
    pet_array = sitk.GetArrayFromImage(pet_sitk)         # (D, H, W) float32
    gtvp_array = sitk.GetArrayFromImage(gtvp_sitk)       # (D, H, W) uint8
    gtvn_array = (
        sitk.GetArrayFromImage(gtvn_sitk) if gtvn_sitk is not None else None
    )

    # ── Record spacing (z, y, x in mm) ───────────────────────────────────
    spacing = np.array(ct_sitk.GetSpacing()[::-1], dtype=np.float64)  # (z,y,x)

    # ── Preprocess intensities ────────────────────────────────────────────
    ct_imgs = window_ct(ct_array, ct_window_low, ct_window_high)
    pet_imgs = normalise_pet(pet_array, pet_suv_max)
    gts = build_combined_mask(gtvp_array, gtvn_array)

    # ── Validate ─────────────────────────────────────────────────────────
    if gts.sum() == 0:
        print(f"  [WARN] {patient_id}: empty segmentation mask – skipping.")
        return None

    # ── Optional axial crop ───────────────────────────────────────────────
    if crop_z:
        ct_imgs, pet_imgs, gts = crop_to_foreground(
            ct_imgs, pet_imgs, gts, margin_z=crop_z_margin
        )

    return {
        "ct_imgs": ct_imgs,
        "pet_imgs": pet_imgs,
        "gts": gts,
        "spacing": spacing,
        "patient_id": patient_id,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    """Prepare all patients and write NPZ files to *output_dir*."""

    data_root = Path(args.data_dir)
    output_root = Path(args.output_dir)

    # Collect patient IDs (each sub-directory is one patient).
    patient_ids = sorted(
        [p.name for p in data_root.iterdir() if p.is_dir()]
    )

    if not patient_ids:
        raise RuntimeError(f"No patient directories found under {data_root}")

    print(f"Found {len(patient_ids)} patients.")

    # ── Train / validation split ──────────────────────────────────────────
    random.seed(args.seed)
    random.shuffle(patient_ids)
    n_val = max(1, int(len(patient_ids) * args.val_ratio))
    val_ids = set(patient_ids[:n_val])
    train_ids = set(patient_ids[n_val:])

    split_log = {"train": sorted(train_ids), "val": sorted(val_ids)}
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "data_split.json", "w") as f:
        json.dump(split_log, f, indent=2)

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val")

    # ── Process each patient ──────────────────────────────────────────────
    skipped = []
    for pid in tqdm(patient_ids, desc="Processing patients"):
        patient_dir = data_root / pid
        result = process_patient(
            patient_dir=patient_dir,
            patient_id=pid,
            ct_window_low=args.ct_low,
            ct_window_high=args.ct_high,
            pet_suv_max=args.pet_suv_max if args.pet_suv_max > 0 else None,
            crop_z=not args.no_crop,
            crop_z_margin=args.crop_margin,
        )

        if result is None:
            skipped.append(pid)
            continue

        # Determine split subdirectory.
        subset = "val" if pid in val_ids else "train"
        out_dir = output_root / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pid}.npz"

        np.savez_compressed(
            out_path,
            ct_imgs=result["ct_imgs"],
            pet_imgs=result["pet_imgs"],
            gts=result["gts"],
            spacing=result["spacing"],
        )

    print(f"\nDone.  Skipped {len(skipped)} patients: {skipped}")
    print(f"Output written to: {output_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert HECKTOR NIfTI data to MedSAM2 NPZ format."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/data/santiago/HECKTOR_data/Task_1_segmentation",
        help="Root directory containing one sub-folder per patient.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/ethan/MedSAM2/hecktor_npz",
        help="Output directory for NPZ files (will be created if absent).",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of patients reserved for validation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split.",
    )
    parser.add_argument(
        "--ct_low",
        type=int,
        default=CT_WINDOW_LOW,
        help=f"CT window lower bound in HU (default: {CT_WINDOW_LOW}).",
    )
    parser.add_argument(
        "--ct_high",
        type=int,
        default=CT_WINDOW_HIGH,
        help=f"CT window upper bound in HU (default: {CT_WINDOW_HIGH}).",
    )
    parser.add_argument(
        "--pet_suv_max",
        type=float,
        default=0.0,
        help=(
            "Fixed SUV upper bound for PET normalisation.  "
            "Set to 0 (default) to use per-patient 99th percentile."
        ),
    )
    parser.add_argument(
        "--no_crop",
        action="store_true",
        help="Disable axial cropping to tumour extent.",
    )
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=5,
        help="Extra axial slices around tumour when cropping (default: 5).",
    )

    main(parser.parse_args())
