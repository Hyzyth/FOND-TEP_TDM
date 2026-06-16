"""
prepare_hecktor_npz.py
======================
** DEPRECATED - No longer used for MedSAM2 training or inference. **

As of 2026-06, MedSAM2 reads directly from the SwinCross-format NPZ files
produced by P1_SWIN/npz_version/prepare_hecktor2026_kfold_npz.py.

The SwinCross pipeline generates:
  - ct    (R, A, S) int16   - CT in HU
  - pet   (R, A, S) float16 - PET in SUV
  - label (R, A, S) uint8   - 0=bg, 1=GTVp, 2=GTVn
  - inverse-transform metadata for NIfTI export
  - per-fold JSON split files compatible with all three models

Run this instead:
    cd P1_SWIN
    bash SwinCross_NPZ_Dataset_Building.sh  # BUILD_HECKTOR_2026_KFOLD=true

This file is kept only for historical reference and is NOT called by any
current shell script.

Original docstring preserved below.
----------------------------------------------------------------------

Converts HECKTOR Task-1 segmentation data (NIfTI format) into the NPZ format
expected by MedSAM2's training and inference pipelines.

Expected HECKTOR data structure
--------------------------------
/data/santiago/HECKTOR_data/Task_1_segmentation/{patient_id}/
    {patient_id}_CT.nii.gz   - CT scan (HU values)
    {patient_id}_PT.nii.gz   - PET scan (SUV values)
    {patient_id}.nii.gz      - Combined GT mask:
                                  0 = background
                                  1 = GTVp (primary tumour)
                                  2 = GTVn (nodal tumour, may be absent)

Output NPZ format per patient
-------------------------------
    ct_imgs    : (D, H, W) uint8  - CT windowed & normalised to [0, 255]
    pet_imgs   : (D, H, W) uint8  - PET normalised to [0, 255]
    gts        : (D, H, W) uint8  - 0=background, 1=GTVp, 2=GTVn
    spacing    : (3,) float64     - voxel spacing in mm (z, y, x)
    patient_id : str              - patient identifier
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

CT_WINDOW_LOW: int = -200
CT_WINDOW_HIGH: int = 800
PET_SUV_MAX: Optional[float] = None
MIN_FOREGROUND_VOXELS: int = 10
DEFAULT_CROP_MARGIN: int = 5


# ---------------------------------------------------------------------------
# File-naming patterns
# ---------------------------------------------------------------------------

CT_PATTERNS = [
    "{pid}_CT.nii.gz",
    "{pid}__CT.nii.gz",
    "CT.nii.gz",
    "{pid}_ct.nii.gz",
]

PET_PATTERNS = [
    "{pid}_PT.nii.gz",
    "{pid}__PT.nii.gz",
    "PT.nii.gz",
    "{pid}_pt.nii.gz",
    "{pid}_PET.nii.gz",
]

GT_PATTERNS = [
    "{pid}.nii.gz",
    "{pid}_gt.nii.gz",
    "{pid}_GT.nii.gz",
    "gt.nii.gz",
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def find_file(directory: Path, patterns: list, patient_id: str) -> Optional[Path]:
    for pattern in patterns:
        candidate = directory / pattern.format(pid=patient_id)
        if candidate.exists():
            return candidate
    return None


def resample_to_reference(moving, reference,
                           interpolator=sitk.sitkLinear,
                           default_value=0.0) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetTransform(sitk.Transform())
    return resampler.Execute(moving)


def window_ct(ct_array: np.ndarray, low: int, high: int) -> np.ndarray:
    ct_clipped = np.clip(ct_array, low, high).astype(np.float32)
    ct_norm = (ct_clipped - low) / (high - low) * 255.0
    return ct_norm.astype(np.uint8)


def normalise_pet(
    pet_array: np.ndarray,
    suv_max: Optional[float] = None,
    return_scale: bool = False,
) -> np.ndarray | Tuple[np.ndarray, float]:
    pet_array = pet_array.astype(np.float32)
    pet_array = np.clip(pet_array, 0.0, None)
    upper = suv_max if suv_max is not None else float(np.percentile(pet_array, 99))
    if upper <= 0:
        upper = 1.0
    pet_norm = np.clip(pet_array / upper * 255.0, 0.0, 255.0).astype(np.uint8)
    if return_scale:
        return pet_norm, upper
    return pet_norm


def crop_to_foreground(
    ct: np.ndarray,
    pet: np.ndarray,
    gts: np.ndarray,
    margin_z: int = DEFAULT_CROP_MARGIN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    crop_z_margin: int = DEFAULT_CROP_MARGIN,
) -> Optional[dict]:
    ct_path  = find_file(patient_dir, CT_PATTERNS,  patient_id)
    pet_path = find_file(patient_dir, PET_PATTERNS, patient_id)
    gt_path  = find_file(patient_dir, GT_PATTERNS,  patient_id)

    if ct_path is None or pet_path is None or gt_path is None:
        print(
            f"  [SKIP] {patient_id}: missing files "
            f"(CT={ct_path is not None}, PET={pet_path is not None}, "
            f"GT={gt_path is not None})"
        )
        return None

    ct_sitk  = sitk.ReadImage(str(ct_path),  sitk.sitkFloat32)
    pet_sitk = sitk.ReadImage(str(pet_path), sitk.sitkFloat32)
    gt_sitk  = sitk.ReadImage(str(gt_path),  sitk.sitkUInt8)

    pet_sitk = resample_to_reference(pet_sitk, ct_sitk, sitk.sitkLinear,       0.0)
    gt_sitk  = resample_to_reference(gt_sitk,  ct_sitk, sitk.sitkNearestNeighbor, 0)

    ct_array  = sitk.GetArrayFromImage(ct_sitk)
    pet_array = sitk.GetArrayFromImage(pet_sitk)
    gts       = sitk.GetArrayFromImage(gt_sitk)

    spacing = np.array(ct_sitk.GetSpacing()[::-1], dtype=np.float64)

    unique_labels = np.unique(gts)
    if not np.any(unique_labels > 0):
        print(f"  [WARN] {patient_id}: GT mask is entirely empty - skipping.")
        return None

    unexpected = set(unique_labels.tolist()) - {0, 1, 2}
    if unexpected:
        print(f"  [WARN] {patient_id}: unexpected GT labels {unexpected} - proceeding anyway.")

    print(f"  {patient_id}: GT labels present = {unique_labels.tolist()}")

    ct_imgs  = window_ct(ct_array, ct_window_low, ct_window_high)
    pet_imgs, actual_suv_max = normalise_pet(
        pet_array, suv_max=pet_suv_max, return_scale=True)

    if crop_z:
        ct_imgs, pet_imgs, gts = crop_to_foreground(
            ct_imgs, pet_imgs, gts, margin_z=crop_z_margin)

    return {
        "ct_imgs":     ct_imgs,
        "pet_imgs":    pet_imgs,
        "gts":         gts,
        "spacing":     spacing,
        "patient_id":  patient_id,
        "pet_suv_max": np.float32(actual_suv_max),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    import warnings
    warnings.warn(
        "prepare_hecktor_npz.py is DEPRECATED. "
        "MedSAM2 now reads SwinCross-format NPZ files directly. "
        "Run P1_SWIN/SwinCross_NPZ_Dataset_Building.sh instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    data_root   = Path(args.data_dir)
    output_root = Path(args.output_dir)

    patient_ids = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    if not patient_ids:
        raise RuntimeError(f"No patient directories found under {data_root}")

    print(f"Found {len(patient_ids)} patients.")

    random.seed(args.seed)
    random.shuffle(patient_ids)
    n_val    = max(1, int(len(patient_ids) * args.val_ratio))
    val_ids  = set(patient_ids[:n_val])
    train_ids = set(patient_ids[n_val:])

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "data_split.json", "w") as f:
        json.dump({"train": sorted(train_ids), "val": sorted(val_ids)}, f, indent=2)

    print(f"Split: {len(train_ids)} train / {len(val_ids)} val")

    skipped = []
    for pid in tqdm(patient_ids, desc="preprocessing"):
        patient_dir = data_root / pid
        result = process_patient(
            patient_dir     = patient_dir,
            patient_id      = pid,
            ct_window_low   = args.ct_low,
            ct_window_high  = args.ct_high,
            pet_suv_max     = args.pet_suv_max if args.pet_suv_max > 0 else None,
            crop_z          = not args.no_crop,
            crop_z_margin   = args.crop_margin,
        )
        if result is None:
            skipped.append(pid)
            continue

        subset  = "val" if pid in val_ids else "train"
        out_dir = output_root / subset
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pid}.npz"

        np.savez_compressed(
            out_path,
            ct_imgs     = result["ct_imgs"],
            pet_imgs    = result["pet_imgs"],
            gts         = result["gts"],
            spacing     = result["spacing"],
            pet_suv_max = result["pet_suv_max"],
        )
        print(f"  Saved → {out_path}  shape={result['gts'].shape}  "
              f"suv_max={result['pet_suv_max']:.2f}")

    print(f"\nDone.  Skipped {len(skipped)} patient(s): {skipped}")
    print(f"Output written to: {output_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[DEPRECATED] Convert HECKTOR NIfTI data to MedSAM2 NPZ format. "
                    "Use SwinCross_NPZ_Dataset_Building.sh instead."
    )
    parser.add_argument("--data_dir",    type=str,
        default="/data/santiago/HECKTOR_data/2025/Task_1_segmentation")
    parser.add_argument("--output_dir",  type=str,
        default="/data/ethan/MedSAM2/hecktor_npz")
    parser.add_argument("--val_ratio",   type=float, default=0.2)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--ct_low",      type=int,   default=CT_WINDOW_LOW)
    parser.add_argument("--ct_high",     type=int,   default=CT_WINDOW_HIGH)
    parser.add_argument("--pet_suv_max", type=float, default=0.0)
    parser.add_argument("--no_crop",     action="store_true")
    parser.add_argument("--crop_margin", type=int,   default=DEFAULT_CROP_MARGIN)
    main(parser.parse_args())
