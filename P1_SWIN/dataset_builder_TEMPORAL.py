#!/usr/bin/env python3
"""
dataset_builder_temporal.py
============================
Prepares /data/santiago/Database_nifti_TEMPORAL for zero-shot SwinCross inference.

For each numeric patient folder (1, 2, 3 … 127 etc.), for each timepoint
subfolder that contains:
  - A CT scan         (CT_*.nii.gz at study level)
  - A SUV-normalised PET  (SUVbwPT_*.nii.gz, fallback PT_*.nii.gz)
  - At least one CT_RTStruct_* subfolder with a T mask (tumor)
    and optionally an N mask (lymph node)

The script:
  1. Merges PET + CT → 4-D NIfTI  (channel 0 = PET, channel 1 = CT)
  2. Combines tumor T (=1) and node N (=2) into a single GT mask on the CT grid
  3. Writes everything under --output_folder / imagesTs | labelsTs
  4. Generates a MONAI-compatible JSON (all cases in "validation")

Usage:
    python3 dataset_builder_temporal.py                          # defaults
    python3 dataset_builder_temporal.py --timepoints pre,per    # filter
    python3 dataset_builder_temporal.py --dry_run               # no writes
"""

import os
import json
import re
import argparse
import SimpleITK as sitk
import numpy as np

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "/data/santiago/Database_nifti_TEMPORAL"
DEFAULT_OUTPUT = "/data/ethan/SwinCross/PP_temporal_dataset"
DEFAULT_JSON   = "dataset_swincross_temporal.json"

# ── Timepoint normalisation map ──────────────────────────────────────────────
# All known folder names → canonical label
TIMEPOINT_MAP = {
    # pre-treatment
    "pre":          "pre",  "Pre":          "pre",
    "tep_pre":      "pre",  "TEP_pre":      "pre",
    # during treatment (per-traitement)
    "per":          "per",  "Per":          "per",
    "tep_per":      "per",  "TEP_per":      "per",
    "Nouveau_dossier": "per",               # patient 101 duplicate
    # post-treatment
    "post":         "post", "Post":         "post",
    "tep_post":     "post", "TEP_post":     "post", "TEPpost": "post",
    # intermediate dose scans (20 Gy / 40 Gy)
    "20":           "20gy", "20_Gy":        "20gy",
    "tep_40":       "40gy", "TEP_40":       "40gy", "TEP40":   "40gy",
}


def normalize_timepoint(name: str) -> str:
    """Return a canonical timepoint label, or the lowercased raw name."""
    return TIMEPOINT_MAP.get(name, name.lower().replace(" ", "_"))


def is_numeric_patient(name: str) -> bool:
    """Accept only purely numeric folder names (1, 2, … 127 …)."""
    return bool(re.fullmatch(r"\d+", name))


# ── File discovery ────────────────────────────────────────────────────────────

def scan_study_dir(directory: str):
    """
    Return (ct_files, pet_files, rtstruct_dirs) for a given study folder.
    SUVbwPT files are placed first in pet_files (preferred over raw PT).
    Only CT_RTStruct_* subdirs are considered for GT (not PET_RTStruct_*).
    """
    ct_files, pet_files, rtstruct_dirs = [], [], []

    try:
        entries = sorted(os.listdir(directory))
    except PermissionError:
        return ct_files, pet_files, rtstruct_dirs

    for entry in entries:
        full = os.path.join(directory, entry)

        if os.path.isfile(full) and entry.endswith(".nii.gz"):
            if entry.startswith("SUVbwPT_"):
                pet_files.insert(0, full)          # preferred
            elif entry.startswith("PT_"):
                pet_files.append(full)
            elif entry.startswith("CT_"):          # plain CT (not RTStruct)
                ct_files.append(full)

        elif os.path.isdir(full) and entry.startswith("CT_RTStruct"):
            rtstruct_dirs.append(full)

    return ct_files, pet_files, rtstruct_dirs


def find_gt_masks(rtstruct_dir: str):
    """
    Return (t_path, n_path) from a CT_RTStruct_* directory.
    Handles both 'T.nii.gz' / 'N.nii.gz' and 'GTV T.nii.gz' / 'GTV N.nii.gz'.
    """
    t_path = n_path = None
    try:
        files = os.listdir(rtstruct_dir)
    except Exception:
        return None, None

    for f in files:
        if not f.endswith(".nii.gz"):
            continue
        full = os.path.join(rtstruct_dir, f)
        fl   = f.lower()
        if fl in ("t.nii.gz", "gtv t.nii.gz"):
            t_path = full
        elif fl in ("n.nii.gz", "gtv n.nii.gz"):
            n_path = full

    return t_path, n_path


def best_gt(rtstruct_dirs: list):
    """
    Return (t_path, n_path) from the first RTStruct dir that has at least
    a T mask. Tries all dirs in sorted order.
    """
    for rdir in sorted(rtstruct_dirs):
        t, n = find_gt_masks(rdir)
        if t is not None:
            return t, n
    return None, None


# ── SimpleITK helpers ─────────────────────────────────────────────────────────

def resample_to_reference(image, reference, is_label: bool = False):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(reference)
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label
                      else sitk.sitkBSplineResamplerOrder3)
    r.SetTransform(sitk.Transform())
    r.SetOutputPixelType(sitk.sitkUInt8 if is_label else sitk.sitkFloat32)
    return r.Execute(image)


def merge_pet_ct(pet_path: str, ct_path: str, output_path: str):
    """Write a 4-D [PET, CT] NIfTI (channel 0 = PET, channel 1 = CT)."""

    pet = sitk.ReadImage(pet_path)
    ct  = sitk.ReadImage(ct_path)

    # Resample PET to CT grid if needed
    if pet.GetSize() != ct.GetSize() or pet.GetSpacing() != ct.GetSpacing():
        pet = resample_to_reference(pet, ct, is_label=False)

    # --- Eenforce same pixel type ---
    pet = sitk.Cast(pet, sitk.sitkFloat32)
    ct  = sitk.Cast(ct,  sitk.sitkFloat32)

    assert pet.GetPixelID() == ct.GetPixelID(), "Pixel types still mismatch after casting"
    merged = sitk.JoinSeries([pet, ct])
    sitk.WriteImage(merged, output_path)


def build_gt_mask(t_path, n_path, ct_path: str, output_path: str) -> bool:
    """
    Build a combined GT mask (T=1, N=2) on the CT grid.
    Returns True if the mask is non-empty.
    """
    ct = sitk.ReadImage(ct_path)

    mask_np = np.zeros(ct.GetSize()[::-1], dtype=np.uint8)  # z,y,x

    if t_path:
        t_img = sitk.Cast(sitk.ReadImage(t_path), sitk.sitkUInt8)
        if t_img.GetSize() != ct.GetSize():
            t_img = resample_to_reference(t_img, ct, is_label=True)
        mask_np[sitk.GetArrayFromImage(t_img) > 0] = 1

    if n_path:
        n_img = sitk.Cast(sitk.ReadImage(n_path), sitk.sitkUInt8)
        if n_img.GetSize() != ct.GetSize():
            n_img = resample_to_reference(n_img, ct, is_label=True)
        # N fills only background voxels (T takes priority)
        mask_np[(sitk.GetArrayFromImage(n_img) > 0) & (mask_np == 0)] = 2

    out = sitk.GetImageFromArray(mask_np)
    out.CopyInformation(ct)
    sitk.WriteImage(out, output_path)

    return bool(mask_np.sum() > 0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build zero-shot inference dataset from Database_nifti_TEMPORAL"
    )
    parser.add_argument("--input_folder",  default=DEFAULT_INPUT,
                        help="Root of Database_nifti_TEMPORAL")
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT,
                        help="Where to write the processed dataset")
    parser.add_argument("--json_name",     default=DEFAULT_JSON,
                        help="JSON filename (written inside output_folder)")
    parser.add_argument("--timepoints",    default="all",
                        help="Comma-separated canonical timepoints to include "
                             "(pre, per, post, 40gy, 20gy) — default: all")
    parser.add_argument("--dry_run", action="store_true",
                        help="Discover files but do not write anything")
    args = parser.parse_args()

    allowed_tp = (None if args.timepoints == "all"
                  else set(args.timepoints.split(",")))

    if not args.dry_run:
        for sub in ("imagesTs", "labelsTs"):
            os.makedirs(os.path.join(args.output_folder, sub), exist_ok=True)

    json_data = {
        "description": "Database_nifti_TEMPORAL — zero-shot SwinCross inference",
        "labels": {"0": "background", "1": "tumor (GTV-T)", "2": "node (GTV-N)"},
        "tensorImageSize": "4D",
        "modality": {"0": "SUVbw PET", "1": "CT"},
        "training": [],
        "validation": [],
    }

    patient_folders = sorted(
        [e for e in os.listdir(args.input_folder)
         if is_numeric_patient(e)
         and os.path.isdir(os.path.join(args.input_folder, e))],
        key=int
    )
    print(f"Found {len(patient_folders)} numeric patient folders.\n")

    ok, skipped = 0, 0

    for pat_id in patient_folders:
        pat_path = os.path.join(args.input_folder, pat_id)

        # First-level subdirs of the patient folder = timepoint dirs
        tp_dirs = sorted(
            [d for d in os.listdir(pat_path)
             if os.path.isdir(os.path.join(pat_path, d))]
        )

        for tp_raw in tp_dirs:
            tp_norm = normalize_timepoint(tp_raw)

            if allowed_tp and tp_norm not in allowed_tp:
                continue

            tp_path = os.path.join(pat_path, tp_raw)

            # Second-level: date/study subfolder(s) inside the timepoint.
            # If none exist the files are directly in tp_path (rare edge case).
            study_subdirs = sorted(
                [os.path.join(tp_path, d) for d in os.listdir(tp_path)
                 if os.path.isdir(os.path.join(tp_path, d))]
            )
            search_dirs = study_subdirs if study_subdirs else [tp_path]

            for study_dir in search_dirs:
                ct_files, pet_files, rtstruct_dirs = scan_study_dir(study_dir)

                if not ct_files or not pet_files or not rtstruct_dirs:
                    continue   # incomplete data — skip silently

                ct_path  = ct_files[0]
                pet_path = pet_files[0]  # SUVbwPT first if available

                t_path, n_path = best_gt(rtstruct_dirs)
                if t_path is None:
                    skipped += 1
                    print(f"  ⚠  No tumor GT — {pat_id}/{tp_raw} — skipped")
                    continue

                # Build a unique case identifier
                study_date = os.path.basename(study_dir).replace("__Studies", "")
                case_id    = f"pat{pat_id}_{tp_norm}_{study_date}"

                out_img = os.path.join(
                    args.output_folder, "imagesTs", f"{case_id}_petct.nii.gz"
                )
                out_lbl = os.path.join(
                    args.output_folder, "labelsTs", f"{case_id}_gt.nii.gz"
                )

                pet_type = "SUVbw" if "SUVbw" in os.path.basename(pet_path) else "PT"
                n_flag   = " +N" if n_path else ""
                print(f"  ✅ {case_id}  [{pet_type}{n_flag}]")

                if not args.dry_run:
                    try:
                        merge_pet_ct(pet_path, ct_path, out_img)
                        non_empty = build_gt_mask(t_path, n_path, ct_path, out_lbl)
                        if not non_empty:
                            print(f"     ⚠  GT mask is empty for {case_id}")
                    except Exception as e:
                        skipped += 1
                        print(f"  ❌ Error on {case_id}: {e}")
                        continue

                json_data["validation"].append({
                    "image": f"imagesTs/{case_id}_petct.nii.gz",
                    "label": f"labelsTs/{case_id}_gt.nii.gz",
                    "case_id":   case_id,
                    "patient":   pat_id,
                    "timepoint": tp_norm,
                    "study_date": study_date,
                })
                ok += 1

    if not args.dry_run:
        json_path = os.path.join(args.output_folder, args.json_name)
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=4)
        print(f"\nDone! {ok} cases built, {skipped} skipped.")
        print(f"   Output : {args.output_folder}")
        print(f"   JSON   : {json_path}")
    else:
        print(f"\n[dry-run] {ok} cases would be built, {skipped} skipped.")


if __name__ == "__main__":
    main()