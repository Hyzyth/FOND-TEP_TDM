#!/usr/bin/env python3
"""
prepare_hecktor_npz_swincross.py
=================================
Offline preprocessing for SwinCross: converts HECKTOR NIfTI data to NPZ.

Storage dtype optimisation (vs. float32 for everything):
  ct  : int16   (2 bytes/voxel)  — HU values fit in [-32768, 32767]
  pet : float16 (2 bytes/voxel)  — SUV values need ~3 significant digits, float16 is sufficient
  Combined saving: ~50 % smaller NPZ files, ~50 % faster disk I/O at load time.

Steps per patient:
  1. Load CT, PET (PT), GT   (GT = {0=bg, 1=GTVp, 2=GTVn})
  2. Resample PET/GT to CT grid if sizes differ
  3. Orient everything to RAS
  4. Crop foreground with a margin
  5. Save NPZ (ct int16 + pet float16 + label uint8 + inverse-transform metadata)
  6. Save original-space GT NIfTI for evaluate_predictions.py
  7. Generate MONAI-compatible JSON split

NPZ content
-----------
  ct             : (R, A, S) int16     CT in HU              [MONAI RAS spatial convention]
  pet            : (R, A, S) float16   PET in SUV
  label          : (R, A, S) uint8     0=bg, 1=GTVp, 2=GTVn
  ras_origin     : (3,)      float64   ITK origin of the full RAS volume
  ras_direction  : (9,)      float64   ITK direction cosines (row-major)
  ras_size_itk   : (3,)      int64     ITK size (x, y, z) of full RAS vol
  crop_start     : (3,)      int64     crop start in MONAI (R, A, S) indices
  crop_end       : (3,)      int64     crop end   in MONAI (R, A, S) indices
  orig_spacing   : (3,)      float64   original CT ITK spacing (x,y,z) mm
  orig_origin    : (3,)      float64
  orig_direction : (9,)      float64
  orig_size_itk  : (3,)      int64     original CT ITK size (x,y,z)

JSON entry format
-----------------
  { "npz":     "train/HGJ_001.npz",
    "label":   "labelsTr/HGJ_001_gt.nii.gz",
    "case_id": "HGJ_001" }

Usage
-----
  python npz_version/prepare_hecktor_npz_swincross.py \\
      --data_dir /data/santiago/HECKTOR_data/2025/Task_1_segmentation \\
      --output_dir /data/ethan/PP_hecktor_swincross_npz \\
      --val_split 0.2 --seed 42
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

# ── File-name patterns ────────────────────────────────────────────────────────
CT_PATTERNS  = ["{pid}__CT.nii.gz", "{pid}_CT.nii.gz", "CT.nii.gz"]
PET_PATTERNS = ["{pid}__PT.nii.gz", "{pid}_PT.nii.gz", "PT.nii.gz"]
GT_PATTERNS  = ["{pid}.nii.gz",     "{pid}_gt.nii.gz", "{pid}_GT.nii.gz"]

TARGET_SPACING    = (1.0, 1.0, 1.0)
FOREGROUND_MARGIN = 5


# ── SimpleITK helpers ─────────────────────────────────────────────────────────

def find_file(directory: Path, patterns: list, pid: str) -> Optional[Path]:
    for pat in patterns:
        cand = directory / pat.format(pid=pid)
        if cand.exists():
            return cand
    return None


def resample_to_reference(src: sitk.Image, ref: sitk.Image,
                           is_label: bool) -> sitk.Image:
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(
        sitk.sitkNearestNeighbor if is_label else sitk.sitkBSplineResamplerOrder3)
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    return r.Execute(src)


def orient_to_ras(img: sitk.Image) -> sitk.Image:
    f = sitk.DICOMOrientImageFilter()
    f.SetDesiredCoordinateOrientation("RAS")
    return f.Execute(img)


def sitk_to_monai(img: sitk.Image) -> np.ndarray:
    """sitk (S,A,R) → MONAI (R,A,S) = (x,y,z)."""
    return sitk.GetArrayFromImage(img).transpose(2, 1, 0)


# ── Dtype-optimised save helpers ──────────────────────────────────────────────

def to_ct_int16(arr: np.ndarray) -> np.ndarray:
    """
    Clip to int16 range and cast.
    CT HU values comfortably fit in [-32768, 32767]; clipping is a safety net
    against rare metal-artefact outliers.
    """
    return np.clip(arr, np.iinfo(np.int16).min,
                        np.iinfo(np.int16).max).astype(np.int16)


def to_pet_float16(arr: np.ndarray) -> np.ndarray:
    """
    Cast PET SUV to float16.
    SUV values (0–~30) need ~3 significant digits; float16 provides ~3.3.
    Clamp to float16 max to avoid inf on extreme outliers.
    """
    return np.clip(arr.astype(np.float32), 0.0,
                   np.finfo(np.float16).max).astype(np.float16)


# ── Per-patient processing ────────────────────────────────────────────────────

def process_patient(patient_dir: Path, pid: str) -> Optional[dict]:
    ct_path  = find_file(patient_dir, CT_PATTERNS,  pid)
    pet_path = find_file(patient_dir, PET_PATTERNS, pid)
    gt_path  = find_file(patient_dir, GT_PATTERNS,  pid)

    if not all([ct_path, pet_path, gt_path]):
        print(f"  [SKIP] {pid}: missing "
              f"CT={ct_path is not None} PET={pet_path is not None} GT={gt_path is not None}")
        return None

    ct_orig  = sitk.ReadImage(str(ct_path),  sitk.sitkFloat32)
    pet_orig = sitk.ReadImage(str(pet_path), sitk.sitkFloat32)
    gt_orig  = sitk.ReadImage(str(gt_path),  sitk.sitkUInt8)

    # ── Capture original CT metadata (for inverse transform in test.py) ──
    orig_spacing   = np.array(ct_orig.GetSpacing(),   dtype=np.float64)
    orig_origin    = np.array(ct_orig.GetOrigin(),    dtype=np.float64)
    orig_direction = np.array(ct_orig.GetDirection(), dtype=np.float64)
    orig_size_itk  = np.array(ct_orig.GetSize(),      dtype=np.int64)

    # ── Align PET / GT to CT grid ─────────────────────────────────────────
    if (pet_orig.GetSize() != ct_orig.GetSize()
            or pet_orig.GetSpacing() != ct_orig.GetSpacing()):
        pet_orig = resample_to_reference(pet_orig, ct_orig, is_label=False)
    if gt_orig.GetSize() != ct_orig.GetSize():
        gt_orig = resample_to_reference(gt_orig, ct_orig, is_label=True)

    ct_ras  = orient_to_ras(ct_orig)
    pet_ras = orient_to_ras(pet_orig)
    gt_ras  = orient_to_ras(gt_orig)

    # ── Store RAS metadata (needed to invert in test.py) ─────────────────
    ras_origin    = np.array(ct_ras.GetOrigin(),    dtype=np.float64)
    ras_direction = np.array(ct_ras.GetDirection(), dtype=np.float64)
    ras_size_itk  = np.array(ct_ras.GetSize(),      dtype=np.int64)

    # Convert to MONAI (R, A, S) — still float32 for foreground detection
    ct_arr  = sitk_to_monai(ct_ras).astype(np.float32)
    pet_arr = sitk_to_monai(pet_ras).astype(np.float32)
    gt_arr  = sitk_to_monai(gt_ras).astype(np.uint8)

    # Foreground crop (use float32 for reliable non-zero test)
    fg_mask = (ct_arr != 0) | (pet_arr != 0)
    coords  = np.where(fg_mask)
    if len(coords[0]) == 0:
        print(f"  [WARN] {pid}: empty foreground — skipping")
        return None

    crop_start = np.maximum(
        0,
        np.array([c.min() for c in coords], dtype=np.int64) - FOREGROUND_MARGIN
    )
    crop_end = np.minimum(
        np.array(ct_arr.shape, dtype=np.int64),
        np.array([c.max() + 1 for c in coords], dtype=np.int64) + FOREGROUND_MARGIN
    )
    sl = tuple(slice(int(s), int(e)) for s, e in zip(crop_start, crop_end))

    ct_c  = ct_arr[sl]
    pet_c = pet_arr[sl]
    gt_c  = gt_arr[sl]

    print(f"  ✅ {pid}  labels={np.unique(gt_arr).tolist()}  crop={ct_c.shape}")

    return {
        # Optimised storage dtypes
        "ct":          to_ct_int16(ct_c),       # (R, A, S) int16
        "pet":         to_pet_float16(pet_c),   # (R, A, S) float16
        "label":       gt_c,                    # (R, A, S) uint8
        # Inverse-transform metadata
        "ras_origin":    ras_origin,
        "ras_direction": ras_direction,
        "ras_size_itk":  ras_size_itk,
        "crop_start":    crop_start,
        "crop_end":      crop_end,
        "orig_spacing":  orig_spacing,
        "orig_origin":   orig_origin,
        "orig_direction":orig_direction,
        "orig_size_itk": orig_size_itk,
        "_gt_sitk_orig": gt_orig,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    data_root = Path(args.data_dir)
    out_root  = Path(args.output_dir)

    patient_ids = sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    if not patient_ids:
        raise RuntimeError(f"No patient directories found in {data_root}")
    print(f"Found {len(patient_ids)} patient(s).")

    random.seed(args.seed)
    shuffled = list(patient_ids)
    random.shuffle(shuffled)
    n_val   = max(1, int(len(shuffled) * args.val_split))
    val_set = set(shuffled[:n_val])

    for sub in ["train", "val", "labelsTr", "labelsTs"]:
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    json_data = {
        "description": "HECKTOR — SwinCross NPZ (RAS, foreground-cropped, int16/f16)",
        "labels":      {"0": "background", "1": "GTVp", "2": "GTVn"},
        "training":    [],
        "validation":  [],
    }
    split_record: dict = {"train": [], "val": []}
    skipped: list = []

    for pid in tqdm(patient_ids, desc="preprocessing"):
        result = process_patient(data_root / pid, pid)
        if result is None:
            skipped.append(pid)
            continue

        is_val     = pid in val_set
        subset     = "val" if is_val else "train"
        lbl_subdir = "labelsTs" if is_val else "labelsTr"
        npz_rel    = f"{subset}/{pid}.npz"
        lbl_rel    = f"{lbl_subdir}/{pid}_gt.nii.gz"

        np.savez_compressed(
            str(out_root / npz_rel),
            ct             = result["ct"],            # int16
            pet            = result["pet"],           # float16
            label          = result["label"],         # uint8
            ras_origin     = result["ras_origin"],
            ras_direction  = result["ras_direction"],
            ras_size_itk   = result["ras_size_itk"],
            crop_start     = result["crop_start"],
            crop_end       = result["crop_end"],
            orig_spacing   = result["orig_spacing"],
            orig_origin    = result["orig_origin"],
            orig_direction = result["orig_direction"],
            orig_size_itk  = result["orig_size_itk"],
        )
        sitk.WriteImage(result["_gt_sitk_orig"], str(out_root / lbl_rel))

        entry    = {"npz": npz_rel, "label": lbl_rel, "case_id": pid}
        json_key = "validation" if is_val else "training"
        json_data[json_key].append(entry)
        split_record[subset].append(pid)

    json_path = out_root / args.json_name
    with open(str(json_path), "w") as f:
        json.dump(json_data, f, indent=2)
    with open(str(out_root / "data_split.json"), "w") as f:
        json.dump(split_record, f, indent=2)

    print(f"\nDone. train={len(json_data['training'])}  "
          f"val={len(json_data['validation'])}  skipped={len(skipped)}")
    print(f"JSON → {json_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HECKTOR → SwinCross NPZ (int16 CT / float16 PET)")
    ap.add_argument("--data_dir",   default="/data/santiago/HECKTOR_data/2025/Task_1_segmentation")
    ap.add_argument("--output_dir", default="/data/ethan/PP_hecktor_swincross_npz")
    ap.add_argument("--json_name",  default="dataset_swincross.json")
    ap.add_argument("--val_split",  type=float, default=0.2)
    ap.add_argument("--seed",       type=int,   default=42)
    main(ap.parse_args())
