#!/usr/bin/env python3
"""
prepare_temporal_npz.py
========================
Converts Database_nifti_TEMPORAL to NPZ format for MedSAM2 zero-shot inference.

File-discovery logic mirrors SwinCross's dataset_builder_TEMPORAL.py.
Preprocessing mirrors prepare_hecktor_npz.py (CT windowing, PET normalisation).

Source:
  /data/santiago/Database_nifti_TEMPORAL/
    <patient_id>/         ← purely numeric
      <timepoint_dir>/    ← key in TIMEPOINT_MAP; others skipped
        <study_date>/
          CT_*.nii.gz
          SUVbwPT_*.nii.gz  (preferred)  or  PT_*.nii.gz
          CT_RTStruct*/
            T.nii.gz / GTV T.nii.gz / …
            N.nii.gz / GTV N.nii.gz / …

Output (--output_dir):
  {case_id}.npz     ct_imgs (D,H,W) uint8, pet_imgs (D,H,W) uint8,
                    gts (D,H,W) uint8 {0,1,2}, spacing (3,) float64 (z,y,x),
                    pet_suv_max float32
  manifest.json     per-case metadata consumed by evaluate_npz.py

Usage:
  python3 data_preparation/prepare_temporal_npz.py
  python3 data_preparation/prepare_temporal_npz.py --timepoints pre,per
  python3 data_preparation/prepare_temporal_npz.py --dry_run
"""

import argparse
import json
import os
import re

import numpy as np
import SimpleITK as sitk

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_INPUT  = "/data/santiago/Database_nifti_TEMPORAL"
DEFAULT_OUTPUT = "/data/ethan/MedSAM2/temporal_npz"

CT_WINDOW_LOW  = -200   # HU
CT_WINDOW_HIGH =  800   # HU

# ── Timepoint normalisation map ───────────────────────────────────────────────
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
    "20":           "20gy", "20_Gy":        "20gy", "TEP_20":       "20gy",
    "40":           "40gy", "tep_40":       "40gy", "TEP_40":       "40gy", "TEP40":       "40gy",
}


def normalize_timepoint(name: str) -> str:
    return TIMEPOINT_MAP.get(name, "")


def is_numeric_patient(name: str) -> bool:
    return bool(re.fullmatch(r"\d+", name))


# ── File discovery ───────────────────────────────────────────────────────────

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
        sub_path = os.path.join(rtstruct_dir, entry)
        if not os.path.isdir(sub_path):
            continue
        kind = _classify_name(entry)
        if kind not in ("t", "n"):
            continue
        sub_niftis = sorted(f for f in os.listdir(sub_path) if f.endswith(".nii.gz"))
        if not sub_niftis:
            continue
        candidate = os.path.join(sub_path, sub_niftis[0])
        if kind == "t" and t_path is None:
            t_path = candidate
        elif kind == "n" and n_path is None:
            n_path = candidate
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


# ── SimpleITK helpers ─────────────────────────────────────────────────────────

def _same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(),   b.GetSpacing(),   atol=1e-4)
        and np.allclose(a.GetOrigin(),    b.GetOrigin(),    atol=1e-4)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=1e-4)
    )


def resample_to_reference(image, reference, is_label: bool = False):
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(reference)
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSplineResamplerOrder3)
    r.SetTransform(sitk.Transform())
    r.SetOutputPixelType(sitk.sitkUInt8 if is_label else sitk.sitkFloat32)
    return r.Execute(image)


# ── Intensity preprocessing (from prepare_hecktor_npz.py) ────────────────────

def window_ct(ct_array: np.ndarray, low: int = CT_WINDOW_LOW,
              high: int = CT_WINDOW_HIGH) -> np.ndarray:
    ct_clipped = np.clip(ct_array, low, high).astype(np.float32)
    return ((ct_clipped - low) / (high - low) * 255.0).astype(np.uint8)


def normalise_pet(pet_array: np.ndarray, suv_max=None):
    """Returns (pet_uint8, actual_ceiling)."""
    pet_array = np.clip(pet_array.astype(np.float32), 0.0, None)
    upper = suv_max if (suv_max is not None and suv_max > 0) else float(np.percentile(pet_array, 99))
    if upper <= 0:
        upper = 1.0
    return np.clip(pet_array / upper * 255.0, 0.0, 255.0).astype(np.uint8), upper


def crop_to_foreground(ct, pet, gts, margin_z: int = 5):
    labelled = np.where(gts.sum(axis=(1, 2)) > 0)[0]
    if len(labelled) == 0:
        return ct, pet, gts
    z_min = max(0, labelled[0] - margin_z)
    z_max = min(ct.shape[0], labelled[-1] + margin_z + 1)
    return ct[z_min:z_max], pet[z_min:z_max], gts[z_min:z_max]


# ── Per-case processing ───────────────────────────────────────────────────────

def process_case(ct_path: str, pet_path: str,
                 t_path, n_path,
                 gt_available: bool,
                 args) -> dict | None:
    """Load, preprocess, and pack one case into NPZ-ready arrays."""
    ct_sitk  = sitk.ReadImage(ct_path,  sitk.sitkFloat32)
    pet_sitk = sitk.ReadImage(pet_path, sitk.sitkFloat32)

    if not _same_grid(pet_sitk, ct_sitk):
        pet_sitk = resample_to_reference(pet_sitk, ct_sitk, is_label=False)

    ct_np  = sitk.GetArrayFromImage(ct_sitk)   # (D, H, W) float32
    pet_np = sitk.GetArrayFromImage(pet_sitk)  # (D, H, W) float32

    # Spacing: SimpleITK gives (x, y, z); we store (z, y, x)
    spacing_xyz = ct_sitk.GetSpacing()
    spacing_zyx = np.array([spacing_xyz[2], spacing_xyz[1], spacing_xyz[0]], dtype=np.float64)

    # Build GT mask (0=bg, 1=GTVp/T, 2=GTVn/N)
    mask_np = np.zeros(ct_np.shape, dtype=np.uint8)
    if t_path:
        t_sitk = sitk.Cast(sitk.ReadImage(t_path), sitk.sitkUInt8)
        if not _same_grid(t_sitk, ct_sitk):
            t_sitk = resample_to_reference(t_sitk, ct_sitk, is_label=True)
        mask_np[sitk.GetArrayFromImage(t_sitk) > 0] = 1
    if n_path:
        n_sitk = sitk.Cast(sitk.ReadImage(n_path), sitk.sitkUInt8)
        if not _same_grid(n_sitk, ct_sitk):
            n_sitk = resample_to_reference(n_sitk, ct_sitk, is_label=True)
        n_arr = sitk.GetArrayFromImage(n_sitk)
        mask_np[(n_arr > 0) & (mask_np == 0)] = 2

    ct_imgs  = window_ct(ct_np, args.ct_low, args.ct_high)
    pet_imgs, suv_max = normalise_pet(pet_np)

    if not args.no_crop and gt_available and mask_np.any():
        ct_imgs, pet_imgs, mask_np = crop_to_foreground(ct_imgs, pet_imgs, mask_np,
                                                         margin_z=args.crop_margin)

    return {
        "ct_imgs":     ct_imgs,
        "pet_imgs":    pet_imgs,
        "gts":         mask_np,
        "spacing":     spacing_zyx,
        "pet_suv_max": np.float32(suv_max),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build MedSAM2 NPZ dataset from Database_nifti_TEMPORAL"
    )
    parser.add_argument("--input_folder",  default=DEFAULT_INPUT)
    parser.add_argument("--output_folder", default=DEFAULT_OUTPUT)
    parser.add_argument("--timepoints",    default="all",
                        help="Comma-separated canonical timepoints or 'all'")
    parser.add_argument("--ct_low",        type=int, default=CT_WINDOW_LOW)
    parser.add_argument("--ct_high",       type=int, default=CT_WINDOW_HIGH)
    parser.add_argument("--crop_margin",   type=int, default=5)
    parser.add_argument("--no_crop",       action="store_true")
    parser.add_argument("--dry_run",       action="store_true")
    args = parser.parse_args()

    allowed_tp = None if args.timepoints == "all" else set(args.timepoints.split(","))

    if not args.dry_run:
        os.makedirs(args.output_folder, exist_ok=True)

    manifest = {
        "description": "Database_nifti_TEMPORAL — MedSAM2 zero-shot inference",
        "labels": {"0": "background", "1": "GTVp (T)", "2": "GTVn (N)"},
        "cases": [],
    }

    patient_folders = sorted(
        [e for e in os.listdir(args.input_folder)
         if is_numeric_patient(e)
         and os.path.isdir(os.path.join(args.input_folder, e))],
        key=int,
    )
    print(f"Found {len(patient_folders)} numeric patient folders.\n")

    ok = skipped = 0

    for pat_id in patient_folders:
        pat_path = os.path.join(args.input_folder, pat_id)
        tp_dirs  = sorted(d for d in os.listdir(pat_path)
                          if os.path.isdir(os.path.join(pat_path, d)))

        for tp_raw in tp_dirs:
            tp_norm = normalize_timepoint(tp_raw)
            if not tp_norm:
                print(f"  ⏭  Skipping unclassed: {pat_id}/{tp_raw}")
                continue
            if allowed_tp and tp_norm not in allowed_tp:
                continue

            tp_path = os.path.join(pat_path, tp_raw)
            study_subdirs = sorted(
                os.path.join(tp_path, d) for d in os.listdir(tp_path)
                if os.path.isdir(os.path.join(tp_path, d))
            )
            search_dirs = study_subdirs if study_subdirs else [tp_path]

            for study_dir in search_dirs:
                ct_files, pet_files, rtstruct_dirs = scan_study_dir(study_dir)
                if not ct_files or not pet_files:
                    continue

                ct_path  = ct_files[0]
                pet_path = pet_files[0]
                pet_type = "SUVbw" if "SUVbw" in os.path.basename(pet_path) else "PT"

                # ── GT discovery ───────────────────────────────────────────
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

                study_date = os.path.basename(study_dir).replace("__Studies", "")
                case_id    = f"pat{pat_id}_{tp_norm}_{study_date}"
                out_npz    = os.path.join(args.output_folder, f"{case_id}.npz")

                # ── Log ────────────────────────────────────────────────────
                flag = "✅" if gt_available else "⚠ "
                print(f"  {flag} {case_id}  [{pet_type}]  gt={gt_reason}")

                # ── Process & save ─────────────────────────────────────────
                if not args.dry_run:
                    try:
                        result = process_case(ct_path, pet_path, t_path, n_path,
                                              gt_available, args)
                        if result is None:
                            skipped += 1
                            continue
                        np.savez_compressed(
                            out_npz,
                            ct_imgs     = result["ct_imgs"],
                            pet_imgs    = result["pet_imgs"],
                            gts         = result["gts"],
                            spacing     = result["spacing"],
                            pet_suv_max = result["pet_suv_max"],
                        )
                    except Exception as e:
                        skipped += 1
                        print(f"  ❌ Error on {case_id}: {e}")
                        continue

                manifest["cases"].append({
                    "case_id":      case_id,
                    "npz_file":     f"{case_id}.npz",
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
        manifest_path = os.path.join(args.output_folder, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nDone. {ok} cases built, {skipped} skipped.")
        print(f"  Output  : {args.output_folder}")
        print(f"  Manifest: {manifest_path}")
    else:
        print(f"\n[dry-run] {ok} cases would be built, {skipped} skipped.")


if __name__ == "__main__":
    main()
