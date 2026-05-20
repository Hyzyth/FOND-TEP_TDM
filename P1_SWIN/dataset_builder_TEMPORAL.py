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
  2. Combines tumor T (=1) and node N (=2) into a single GT mask on the CT grid.
     Cases with only an N mask (no T) are KEPT — the GT will contain only
     label=2 voxels. Cases with neither T nor N are skipped.
  3. Writes everything under --output_folder:
       imagesTs/    — fused 4-D PET+CT  (model input)
       labelsTs/    — combined GT mask
       rawImagesTs/ — raw (non-fused) PET and CT kept separately
  4. Generates a MONAI-compatible JSON (all cases in "validation").
     Each entry carries has_gtv_t and has_gtv_n boolean flags for
     per-class Dice evaluation downstream.

Dataset structure assumed:
  <input_folder>/
    <patient_id>/         ← purely numeric (1 … 127)
      <timepoint_dir>/    ← must be a key in TIMEPOINT_MAP; others are skipped
        <study_date>/
          CT_*.nii.gz
          SUVbwPT_*.nii.gz  (preferred)  or  PT_*.nii.gz
          CT_RTStruct*/
            T.nii.gz  /  GTV T.nii.gz   (or in T/ sub-dir)
            N.nii.gz  /  GTV N.nii.gz   (or in N/ sub-dir)

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
    return TIMEPOINT_MAP.get(name, "")   # "" signals: skip this folder


def is_numeric_patient(name: str) -> bool:
    """Accept only purely numeric folder names (1, 2, … 127 …)."""
    return bool(re.fullmatch(r"\d+", name))


# ── File discovery ────────────────────────────────────────────────────────────

def scan_study_dir(directory: str):
    """
    Return (ct_files, pet_files, ct_rtstruct_dirs) for a given study folder.

    Notes:
      - SUVbwPT files are inserted at index 0 (preferred over raw PT).
      - Only CT_RTStruct* directories are returned; PET_RTStruct* is ignored
        because it has different resolution from the inference output.
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
            elif entry.startswith("CT_") and "RTStruct" not in entry:
                ct_files.append(full)

        elif os.path.isdir(full) and entry.lower().startswith("ct_rtstruct"):
            rtstruct_dirs.append(full)             # PET_RTStruct excluded

    return ct_files, pet_files, rtstruct_dirs


def _classify_name(name: str) -> str:
    """
    Classify a file/folder name (lowercased, stripped) as 't', 'n', or ''.

    Tumor (T) patterns:  t, tumor, gtv_t, gtv-t, gtv t, primary
    Node  (N) patterns:  n, node, nodal, nodule, gtv_n, gtv-n, gtv n, lymph
    """
    clean = re.sub(r"[\s_\-]", "", name.lower())
    clean = clean.replace(".nii.gz", "").replace(".nii", "")

    T_PATTERNS = {"t", "tumor", "tumour", "gtvt", "primary", "primarytumor"}
    N_PATTERNS = {"n", "node", "nodal", "nodule", "gtvn", "lymph", "lymphnode",
                  "lymphnodegtv"}

    if clean in T_PATTERNS:
        return "t"
    if clean in N_PATTERNS:
        return "n"
    return ""


def find_gt_masks(rtstruct_dir: str):
    """
    Return (t_path, n_path) from a CT_RTStruct_* directory.

    Handles two layouts:
      A) Files directly in the rtstruct dir:
           T.nii.gz / GTV T.nii.gz / Tumor.nii.gz / …
           N.nii.gz / GTV N.nii.gz / Nodule.nii.gz / …
      B) One sub-directory per structure, each containing one .nii.gz:
           T/   something.nii.gz
           N/   something.nii.gz
    """
    t_path = n_path = None

    try:
        entries = sorted(os.listdir(rtstruct_dir))
    except Exception:
        return None, None

    # ── Pass A: direct .nii.gz files ──────────────────────────────────────
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
        return t_path, n_path          # found at least one mask → done

    # ── Pass B: sub-directories ────────────────────────────────────────────
    for entry in entries:
        sub_path = os.path.join(rtstruct_dir, entry)
        if not os.path.isdir(sub_path):
            continue
        kind = _classify_name(entry)
        if kind not in ("t", "n"):
            continue
        # Take the first .nii.gz inside the sub-directory
        sub_niftis = sorted(
            f for f in os.listdir(sub_path) if f.endswith(".nii.gz"))
        if not sub_niftis:
            continue
        candidate = os.path.join(sub_path, sub_niftis[0])
        if kind == "t" and t_path is None:
            t_path = candidate
        elif kind == "n" and n_path is None:
            n_path = candidate

    return t_path, n_path


def best_gt(rtstruct_dirs: list):
    """
    Return (t_path, n_path).

    After dataset curation there should be exactly one CT_RTStruct dir per
    study, so this function simply processes the first (and expected only) dir.
    A fallback loop is kept in case the assumption breaks.
    """
    best_n_fallback = None

    for rdir in sorted(rtstruct_dirs):
        t, n = find_gt_masks(rdir)
        if t is not None:
            return t, n          # T found — best case
        if n is not None and best_n_fallback is None:
            best_n_fallback = n  # N-only fallback

    return None, best_n_fallback


# ── SimpleITK helpers ─────────────────────────────────────────────────────────

def _same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    """True only if size, spacing, origin AND direction all match."""
    return (
        a.GetSize()      == b.GetSize()
        and np.allclose(a.GetSpacing(),   b.GetSpacing(),   atol=1e-4)
        and np.allclose(a.GetOrigin(),    b.GetOrigin(),    atol=1e-4)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-4)
    )


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

    # Resample PET to CT grid if any spatial metadata differs (size, spacing,
    # origin, or direction).  Checking size+spacing alone is insufficient —
    # two images can share the same grid dimensions while sitting at different
    # positions/orientations in physical space.
    if not _same_grid(pet, ct):
        pet = resample_to_reference(pet, ct, is_label=False)

    pet = sitk.Cast(pet, sitk.sitkFloat32)
    ct  = sitk.Cast(ct,  sitk.sitkFloat32)

    assert pet.GetPixelID() == ct.GetPixelID(), "Pixel types still mismatch after casting"
    merged = sitk.JoinSeries([pet, ct])
    sitk.WriteImage(merged, output_path)


def build_gt_mask(t_path, n_path, ct_path: str, output_path: str) -> bool:
    """
    Build a combined GT mask (T=1, N=2) on the CT grid.

    t_path may be None for N-only cases — the resulting mask will contain
    only label=2 voxels (no label=1 region).  n_path may also be None for
    T-only cases.  At least one of t_path / n_path must be non-None.

    BUG FIX: masks are ALWAYS resampled to the CT reference regardless of
    whether their voxel-grid size happens to match.  Two images can share the
    same (nx, ny, nz) while having a different origin or direction cosine
    matrix — if we skip resampling in that case the mask array ends up in the
    wrong physical frame, producing the apparent rotation seen in the viewer.

    Returns True if the final mask is non-empty.
    """
    ct = sitk.ReadImage(ct_path)

    mask_np = np.zeros(ct.GetSize()[::-1], dtype=np.uint8)  # (nz, ny, nx)

    if t_path:
        t_img = sitk.Cast(sitk.ReadImage(t_path), sitk.sitkUInt8)
        # Always resample — ensures physical-space alignment even when sizes match
        if not _same_grid(t_img, ct):
            t_img = resample_to_reference(t_img, ct, is_label=True)
        mask_np[sitk.GetArrayFromImage(t_img) > 0] = 1

    if n_path:
        n_img = sitk.Cast(sitk.ReadImage(n_path), sitk.sitkUInt8)
        if not _same_grid(n_img, ct):
            n_img = resample_to_reference(n_img, ct, is_label=True)
        # N fills only background voxels (T takes priority)
        mask_np[(sitk.GetArrayFromImage(n_img) > 0) & (mask_np == 0)] = 2

    out = sitk.GetImageFromArray(mask_np)
    out.CopyInformation(ct)
    sitk.WriteImage(out, output_path)

    return bool(mask_np.sum() > 0)


def copy_raw_images(pet_path: str, ct_path: str,
                    out_pet_path: str, out_ct_path: str):
    """
    Save the raw (non-fused) PET and CT as individual 3-D NIfTIs.
    PET is resampled to the CT grid (same fix as the fused pipeline) so that
    the two files share identical spatial metadata for easy overlay in viewers.
    """
    pet = sitk.ReadImage(pet_path)
    ct  = sitk.ReadImage(ct_path)

    if not _same_grid(pet, ct):
        pet = resample_to_reference(pet, ct, is_label=False)

    sitk.WriteImage(sitk.Cast(pet, sitk.sitkFloat32), out_pet_path)
    sitk.WriteImage(sitk.Cast(ct,  sitk.sitkFloat32), out_ct_path)


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
        for sub in ("imagesTs", "labelsTs", "rawImagesTs"):
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

            # ── Skip unrecognised / unclassed timepoint folders ────────────
            if not tp_norm:
                print(f"  ⏭  Skipping unclassed folder: {pat_id}/{tp_raw}")
                continue

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

                # Hard skip: cannot run inference without both modalities.
                if not ct_files or not pet_files:
                    continue

                ct_path  = ct_files[0]
                pet_path = pet_files[0]    # SUVbwPT first if available
                pet_type = "SUVbw" if "SUVbw" in os.path.basename(pet_path) else "PT"
                
                # ── Determine GT availability ──────────────────────────────
                # Cases without any RTStruct dir, or where no mask file is
                # found inside the RTStruct dirs, are NOT skipped — they still
                # get a fused image built and appear in the JSON.  test.py
                # detects gt_available=False and skips only the Dice step,
                # writing a "no_gt_available" row to the CSV instead.

                if not rtstruct_dirs:
                    t_path = n_path = None
                    gt_reason = "no_rtstruct_dir"
                else:
                    # ── Warn when multiple competing RTStruct dirs are present ──
                    if len(rtstruct_dirs) > 1:
                        print(f"  ℹ  Multiple RTStruct dirs ({len(rtstruct_dirs)}) for "
                              f"{pat_id}/{tp_raw} — using first dir that has a T mask "
                              f"(or first N-only dir). Review manually if needed:")
                        for rd in sorted(rtstruct_dirs):
                            print(f"      {os.path.basename(rd)}")

                    t_path, n_path = best_gt(rtstruct_dirs)

                    if t_path is None and n_path is None:
                        gt_reason = "no_mask_in_rtstruct"
                    elif t_path is None:
                        gt_reason = "n_only"      # N-only: T resolved post-treatment
                    elif n_path is None:
                        gt_reason = "t_only"
                    else:
                        gt_reason = "ok"

                gt_available = gt_reason in ("ok", "n_only", "t_only")

                # ── Log what we found ──────────────────────────────────────
                study_date = os.path.basename(study_dir).replace("__Studies", "")
                case_id    = f"pat{pat_id}_{tp_norm}_{study_date}"

                if gt_reason == "no_rtstruct_dir":
                    print(f"  ⚠  {case_id}  [{pet_type}] — no RTStruct dir "
                          f"(inference only, evaluation skipped)")
                elif gt_reason == "no_mask_in_rtstruct":
                    print(f"  ⚠  {case_id}  [{pet_type}] — RTStruct present but "
                          f"no T/N masks found (inference only, evaluation skipped)")
                elif gt_reason == "n_only":
                    print(f"  ✅ {case_id}  [{pet_type} N] — N-only case "
                          f"(GTV-T absent from mask, label=2 only)")
                elif gt_reason == "t_only":
                    print(f"  ✅ {case_id}  [{pet_type} T] — T-only case "
                          f"(GTV-N absent from mask, label=1 only)")
                else:
                    print(f"  ✅ {case_id}  [{pet_type} T N]")

                # ── Output paths ───────────────────────────────────────────
                out_img = os.path.join(
                    args.output_folder, "imagesTs", f"{case_id}_petct.nii.gz"
                )
                out_lbl = os.path.join(
                    args.output_folder, "labelsTs", f"{case_id}_gt.nii.gz"
                )
                out_raw_pet = os.path.join(
                    args.output_folder, "rawImagesTs", f"{case_id}_pet.nii.gz"
                )
                out_raw_ct = os.path.join(
                    args.output_folder, "rawImagesTs", f"{case_id}_ct.nii.gz"
                )

                if not args.dry_run:
                    try:
                        merge_pet_ct(pet_path, ct_path, out_img)
                        # build_gt_mask handles t_path=None and/or n_path=None:
                        # both None → all-zero label (MONAI can still load it).
                        non_empty = build_gt_mask(t_path, n_path, ct_path, out_lbl)
                        if gt_available and not non_empty:
                            print(f"     ⚠  GT mask written but is empty for {case_id}")
                        copy_raw_images(pet_path, ct_path, out_raw_pet, out_raw_ct)
                    except Exception as e:
                        skipped += 1
                        print(f"  ❌ Error on {case_id}: {e}")
                        continue

                # ── JSON entry ─────────────────────────────────────────────
                # gt_available=False → test.py skips Dice and logs the reason.
                # has_gtv_t / has_gtv_n → controls per-class Dice scoring.
                # gt_reason → human-readable explanation stored for reference.
                json_data["validation"].append({
                    "image":        f"imagesTs/{case_id}_petct.nii.gz",
                    "label":        f"labelsTs/{case_id}_gt.nii.gz",
                    "raw_pet":      f"rawImagesTs/{case_id}_pet.nii.gz",
                    "raw_ct":       f"rawImagesTs/{case_id}_ct.nii.gz",
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
            json.dump(json_data, f, indent=4)
        print(f"\nDone! {ok} cases built, {skipped} skipped.")
        print(f"   Output : {args.output_folder}")
        print(f"   JSON   : {json_path}")
    else:
        print(f"\n[dry-run] {ok} cases would be built, {skipped} skipped.")


if __name__ == "__main__":
    main()
