#!/usr/bin/env python3
"""
prepare_temporal_npz_swincross.py
==================================
Offline preprocessing for SwinCross: converts Database_nifti_TEMPORAL to NPZ.

Storage dtype optimisation: ct → int16, pet → float16  (same as HECKTOR script).
File-discovery logic mirrors dataset_builder_TEMPORAL.py.
All cases land in "validation" (zero-shot inference only).

Usage:
  python npz_version/prepare_temporal_npz_swincross.py
  python npz_version/prepare_temporal_npz_swincross.py --timepoints pre,per
  python npz_version/prepare_temporal_npz_swincross.py --dry_run
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from prepare_hecktor_npz_swincross import (
    FOREGROUND_MARGIN,
    resample_to_reference, orient_to_ras, sitk_to_monai,
    to_ct_int16, to_pet_float16,
)

DEFAULT_INPUT  = "/data/santiago/Database_nifti_TEMPORAL"
DEFAULT_OUTPUT = "/data/ethan/PP_temporal_swincross_npz"
DEFAULT_JSON   = "dataset_swincross_temporal.json"

TIMEPOINT_MAP = {
    "pre":  "pre",  "Pre":  "pre",  "tep_pre":  "pre",  "TEP_pre":  "pre",
    "per":  "per",  "Per":  "per",  "tep_per":  "per",  "TEP_per":  "per",
    "Nouveau_dossier": "per",
    "post": "post", "Post": "post", "tep_post": "post", "TEP_post": "post",
    "TEPpost": "post",
    "20": "20gy", "20_Gy": "20gy", "TEP_20": "20gy",
    "40": "40gy", "tep_40": "40gy", "TEP_40": "40gy", "TEP40": "40gy",
}


def normalize_timepoint(name: str) -> str:
    return TIMEPOINT_MAP.get(name, "")


def is_numeric_patient(name: str) -> bool:
    return bool(re.fullmatch(r"\d+", name))

# ── File-discovery helpers ────────────────────────────────────────────────────────────────

def scan_study_dir(directory: str):
    ct_files, pet_files, rtstruct_dirs = [], [], []
    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return ct_files, pet_files, rtstruct_dirs
    for entry in entries:
        full = os.path.join(directory, entry)
        if os.path.isfile(full) and entry.endswith(".nii.gz"):
            if entry.startswith("SUVbwPT_"):
                pet_files.insert(0, full)
            elif entry.startswith("PT_"):
                pet_files.append(full)
            elif entry.startswith("CT_") and "RTStruct" not in entry:
                ct_files.append(full)
        elif os.path.isdir(full) and entry.lower().startswith("ct_rtstruct"):
            rtstruct_dirs.append(full)
    return ct_files, pet_files, rtstruct_dirs


def _classify_name(name: str) -> str:
    clean = re.sub(r"[\s_\-]", "", name.lower()).replace(".nii.gz", "").replace(".nii", "")
    if clean in {"t", "tumor", "tumour", "gtvt", "primary", "primarytumor"}:
        return "t"
    if clean in {"n", "node", "nodal", "nodule", "gtvn", "lymph", "lymphnode", "lymphnodegtv"}:
        return "n"
    return ""


def find_gt_masks(rtstruct_dir: str):
    t_path = n_path = None
    try:
        entries = sorted(os.listdir(rtstruct_dir))
    except Exception:
        return None, None
    for entry in entries:
        if not entry.endswith(".nii.gz"):
            continue
        kind = _classify_name(entry)
        full = os.path.join(rtstruct_dir, entry)
        if kind == "t" and t_path is None:
            t_path = full
        elif kind == "n" and n_path is None:
            n_path = full
    if t_path is not None or n_path is not None:
        return t_path, n_path
    for entry in entries:
        sub = os.path.join(rtstruct_dir, entry)
        if not os.path.isdir(sub):
            continue
        kind = _classify_name(entry)
        if kind not in ("t", "n"):
            continue
        niftis = sorted(f for f in os.listdir(sub) if f.endswith(".nii.gz"))
        if not niftis:
            continue
        cand = os.path.join(sub, niftis[0])
        if kind == "t" and t_path is None:
            t_path = cand
        elif kind == "n" and n_path is None:
            n_path = cand
    return t_path, n_path


def best_gt(rtstruct_dirs: list):
    best_n_fallback = None
    for rdir in sorted(rtstruct_dirs):
        t, n = find_gt_masks(rdir)
        if t is not None:
            return t, n
        if n is not None and best_n_fallback is None:
            best_n_fallback = n
    return None, best_n_fallback


def _same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(),   b.GetSpacing(),   atol=1e-4)
            and np.allclose(a.GetOrigin(),    b.GetOrigin(),    atol=1e-4)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-4))


def build_gt_sitk(t_path: Optional[str], n_path: Optional[str],
                  ct_orig: sitk.Image) -> sitk.Image:
    """Combine T (label=1) and N (label=2) masks onto the CT grid."""
    mask_arr = np.zeros(ct_orig.GetSize()[::-1], dtype=np.uint8)
    if t_path:
        t_img = sitk.Cast(sitk.ReadImage(t_path), sitk.sitkUInt8)
        if not _same_grid(t_img, ct_orig):
            t_img = resample_to_reference(t_img, ct_orig, is_label=True)
        mask_arr[sitk.GetArrayFromImage(t_img) > 0] = 1
    if n_path:
        n_img = sitk.Cast(sitk.ReadImage(n_path), sitk.sitkUInt8)
        if not _same_grid(n_img, ct_orig):
            n_img = resample_to_reference(n_img, ct_orig, is_label=True)
        mask_arr[(sitk.GetArrayFromImage(n_img) > 0) & (mask_arr == 0)] = 2
    out = sitk.GetImageFromArray(mask_arr)
    out.CopyInformation(ct_orig)
    return out


def process_case(ct_path: str, pet_path: str,
                 t_path, n_path) -> Optional[dict]:
    ct_orig  = sitk.ReadImage(ct_path,  sitk.sitkFloat32)
    pet_orig = sitk.ReadImage(pet_path, sitk.sitkFloat32)

    orig_spacing   = np.array(ct_orig.GetSpacing(),   dtype=np.float64)
    orig_origin    = np.array(ct_orig.GetOrigin(),    dtype=np.float64)
    orig_direction = np.array(ct_orig.GetDirection(), dtype=np.float64)
    orig_size_itk  = np.array(ct_orig.GetSize(),      dtype=np.int64)

    if not _same_grid(pet_orig, ct_orig):
        pet_orig = resample_to_reference(pet_orig, ct_orig, is_label=False)

    gt_orig = build_gt_sitk(t_path, n_path, ct_orig)

    ct_ras  = orient_to_ras(ct_orig)
    pet_ras = orient_to_ras(pet_orig)
    gt_ras  = orient_to_ras(gt_orig)

    ras_origin    = np.array(ct_ras.GetOrigin(),    dtype=np.float64)
    ras_direction = np.array(ct_ras.GetDirection(), dtype=np.float64)
    ras_size_itk  = np.array(ct_ras.GetSize(),      dtype=np.int64)

    ct_arr  = sitk_to_monai(ct_ras).astype(np.float32)
    pet_arr = sitk_to_monai(pet_ras).astype(np.float32)
    gt_arr  = sitk_to_monai(gt_ras).astype(np.uint8)

    fg_mask = (ct_arr != 0) | (pet_arr != 0)
    coords  = np.where(fg_mask)
    if len(coords[0]) == 0:
        crop_start = np.zeros(3, dtype=np.int64)
        crop_end   = np.array(ct_arr.shape, dtype=np.int64)
    else:
        crop_start = np.maximum(
            0,
            np.array([c.min() for c in coords], dtype=np.int64) - FOREGROUND_MARGIN
        )
        crop_end = np.minimum(
            np.array(ct_arr.shape, dtype=np.int64),
            np.array([c.max() + 1 for c in coords], dtype=np.int64) + FOREGROUND_MARGIN
        )

    sl     = tuple(slice(int(s), int(e)) for s, e in zip(crop_start, crop_end))
    ct_c   = ct_arr[sl]
    pet_c  = pet_arr[sl]
    gt_c   = gt_arr[sl]

    return {
        "ct":          to_ct_int16(ct_c),
        "pet":         to_pet_float16(pet_c),
        "label":       gt_c,
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


def main():
    ap = argparse.ArgumentParser(description="TemPoRAL → SwinCross NPZ (int16 CT / float16 PET)")
    ap.add_argument("--input_folder",  default=DEFAULT_INPUT)
    ap.add_argument("--output_folder", default=DEFAULT_OUTPUT)
    ap.add_argument("--json_name",     default=DEFAULT_JSON)
    ap.add_argument("--timepoints",    default="all")
    ap.add_argument("--dry_run",       action="store_true")
    args = ap.parse_args()

    allowed_tp = None if args.timepoints == "all" else set(args.timepoints.split(","))

    if not args.dry_run:
        for sub in ["npz", "labelsTs"]:
            os.makedirs(os.path.join(args.output_folder, sub), exist_ok=True)

    json_data = {
        "description": "Database_nifti_TEMPORAL — SwinCross NPZ zero-shot (int16/f16)",
        "labels":      {"0": "background", "1": "GTVp", "2": "GTVn"},
        "training":    [],
        "validation":  [],
    }

    patient_folders = sorted(
        [e for e in os.listdir(args.input_folder)
         if is_numeric_patient(e)
         and os.path.isdir(os.path.join(args.input_folder, e))],
        key=int,
    )
    print(f"Found {len(patient_folders)} numeric patient folders.\n")

    ok = skipped = 0

    for pat_id in tqdm(patient_folders, desc="patients"):
        pat_path = os.path.join(args.input_folder, pat_id)
        tp_dirs  = sorted(d for d in os.listdir(pat_path)
                          if os.path.isdir(os.path.join(pat_path, d)))

        for tp_raw in tp_dirs:
            tp_norm = normalize_timepoint(tp_raw)
            if not tp_norm:
                continue
            if allowed_tp and tp_norm not in allowed_tp:
                continue

            tp_path    = os.path.join(pat_path, tp_raw)
            study_subs = sorted(os.path.join(tp_path, d) for d in os.listdir(tp_path)
                                if os.path.isdir(os.path.join(tp_path, d)))
            search_dirs = study_subs if study_subs else [tp_path]

            for study_dir in search_dirs:
                ct_files, pet_files, rtstruct_dirs = scan_study_dir(study_dir)
                if not ct_files or not pet_files:
                    continue

                ct_path  = ct_files[0]
                pet_path = pet_files[0]

                if not rtstruct_dirs:
                    t_path = n_path = None
                    gt_reason = "no_rtstruct_dir"
                else:
                    if len(rtstruct_dirs) > 1:
                        print(f"  ℹ  Multiple RTStruct dirs for {pat_id}/{tp_raw}")
                    t_path, n_path = best_gt(rtstruct_dirs)
                    if t_path is None and n_path is None:
                        gt_reason = "no_mask_in_rtstruct"
                    elif t_path is None:
                        gt_reason = "n_only"
                    elif n_path is None:
                        gt_reason = "t_only"
                    else:
                        gt_reason = "ok"

                gt_available = gt_reason in ("ok", "n_only", "t_only")
                study_date   = os.path.basename(study_dir).replace("__Studies", "")
                case_id      = f"pat{pat_id}_{tp_norm}_{study_date}"

                print(f"  {'✅' if gt_available else '⚠ '} {case_id}  gt={gt_reason}")

                npz_rel = f"npz/{case_id}.npz"
                lbl_rel = f"labelsTs/{case_id}_gt.nii.gz"

                if not args.dry_run:
                    try:
                        result = process_case(ct_path, pet_path, t_path, n_path)
                        if result is None:
                            skipped += 1
                            continue
                        np.savez_compressed(
                            os.path.join(args.output_folder, npz_rel),
                            ct             = result["ct"],
                            pet            = result["pet"],
                            label          = result["label"],
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
                        sitk.WriteImage(result["_gt_sitk_orig"],
                                        os.path.join(args.output_folder, lbl_rel))
                    except Exception as e:
                        skipped += 1
                        print(f"  ❌ {case_id}: {e}")
                        continue

                json_data["validation"].append({
                    "npz":          npz_rel,
                    "label":        lbl_rel,
                    "case_id":      case_id,
                    "patient":      pat_id,
                    "timepoint":    tp_norm,
                    "study_date":   study_date,
                    "gt_available": gt_available,
                    "gt_reason":    gt_reason,
                    "has_gtv_t":    t_path is not None,
                    "has_gtv_n":    n_path is not None,
                })
                ok += 1

    if not args.dry_run:
        json_path = os.path.join(args.output_folder, args.json_name)
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nDone. {ok} cases, {skipped} skipped → {json_path}")
    else:
        print(f"\n[dry-run] {ok} cases would be built.")


if __name__ == "__main__":
    main()
