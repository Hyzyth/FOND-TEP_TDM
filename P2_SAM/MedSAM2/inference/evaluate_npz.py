#!/usr/bin/env python3
"""
inference/evaluate_npz.py
===========================
Universal rich evaluation for MedSAM2 predictions on HECKTOR 2026 + TemPoRAL.

Supports two case-discovery modes
-----------------------------------
JSON mode (SwinCross JSON)  - --data_dir + --json_list
  Reads "training"/"validation"/"testing" keys.
  GT loaded from the NIfTI path stored in the JSON entry ("label" key) - this
  is the original-space GT NIfTI written by prepare_hecktor2026_kfold_npz.py.
  Predictions are the NIfTI files written by infer_npz.py.

Manifest mode (TemPoRAL)  - --data_dir + --json_list manifest.json
  Reads "cases" list.
  GT loaded from TemPoRAL NPZ "gts" key (uint8, DxHxW).
  Predictions are the NIfTI files written by infer_npz.py.

Legacy directory mode  - --pred_dir + --gt_dir
  Pairs prediction NPZ/NIfTI files with GT NPZ/NIfTI files by case_id.
  Supports both SwinCross GT NPZ ("label" key) and TemPoRAL GT NPZ ("gts" key).

Spacing handling
----------------
SwinCross GT NIfTI  -> spacing read directly from the NIfTI image via SimpleITK.
TemPoRAL GT NPZ     -> "spacing" key is (sz, sy, sx); converted to ITK (sx, sy, sz)
                       for SimpleITK surface-distance computation.
Prediction NIfTI    -> spacing read from the NIfTI image (set correctly by infer_npz.py).

MEAN summary row
----------------
A MEAN row is appended at the end of the CSV, matching SwinCross evaluate_predictions.py.

Metric set
----------
Dice, Jaccard, Hausdorff (mm), MHD/ASSD (mm), Overlap on GT (recall),
Overlap on Pred (precision), volumes (mm³), object counts, volume similarity,
TP/FP/FN/TN volumes (mm³).

Usage - SwinCross JSON (HECKTOR test vault)
-------------------------------------------
    python inference/evaluate_npz.py \\
        --data_dir  /data/ethan/PP_hecktor2026_kfold_npz \\
        --json_list dataset_swincross_2026kfold_test.json \\
        --output_dir /data/ethan/MedSAM2/predictions/hecktor_TEST_vault

Usage - TemPoRAL manifest
--------------------------
    python inference/evaluate_npz.py \\
        --data_dir  /data/ethan/MedSAM2/temporal_npz \\
        --json_list manifest.json \\
        --output_dir /data/ethan/MedSAM2/predictions/temporal_gt_zeroshot
"""

import argparse
import csv
import json
import math
import os
import warnings
from glob import glob
from os.path import basename, join

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

warnings.filterwarnings("ignore")

# ── CSV field list (identical to SwinCross evaluate_predictions.py) ───────────
_CLASS_INFO = {1: ("GTVp", "has_gtv_t"), 2: ("GTVn", "has_gtv_n")}

CSV_FIELDS = [
    "case_id", "patient", "timepoint", "study_date",
    "gt_available", "gt_reason", "has_gtv_t", "has_gtv_n",
    "gt_labels_present", "pred_labels_present",
    # ── Core overlap metrics ───────────────────────────────────────────────
    "GTVp_dice",         "GTVn_dice",         "mean_dice",
    "GTVp_jaccard",      "GTVn_jaccard",
    # ── Surface-distance metrics (mm) ─────────────────────────────────────
    "GTVp_hausdorff_mm", "GTVn_hausdorff_mm",
    "GTVp_mhd_mm",       "GTVn_mhd_mm",
    # ── Directional overlap ratios ────────────────────────────────────────
    "GTVp_overlap_gt",   "GTVn_overlap_gt",   # TP / |GT|  (= sensitivity / recall)
    "GTVp_overlap_pred", "GTVn_overlap_pred",  # TP / |Pred| (= precision / PPV)
    # ── Volumes ───────────────────────────────────────────────────────────
    "gt_vol_GTVp_mm3", "gt_vol_GTVn_mm3", "gt_vol_total_mm3",
    "pred_vol_GTVp_mm3", "pred_vol_GTVn_mm3", "pred_vol_total_mm3",
    # ── Object counts ─────────────────────────────────────────────────────
    "gt_count_GTVp", "pred_count_GTVp",
    "gt_count_GTVn", "pred_count_GTVn",
    # ── Volume similarity ─────────────────────────────────────────────────
    "vol_sim_GTVp", "vol_sim_GTVn",
    # ── Confusion matrix volumes (mm³) ────────────────────────────────────
    "GTVp_TP_mm3", "GTVp_FP_mm3", "GTVp_FN_mm3", "GTVp_TN_mm3",
    "GTVn_TP_mm3", "GTVn_FP_mm3", "GTVn_FN_mm3", "GTVn_TN_mm3",
    "comments",
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _safe(v, d=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(round(float(v), d))


def _count_objects(arr: np.ndarray, val: int) -> int:
    mask = (arr == val)
    if not mask.any():
        return 0
    struct = ndimage.generate_binary_structure(arr.ndim, arr.ndim)
    _, n   = ndimage.label(mask, structure=struct)
    return n


def _overlap(tp, fp, fn, tn):
    dice   = 2*tp / (2*tp+fp+fn) if (2*tp+fp+fn) > 0 else None
    jac    = tp / (tp+fp+fn)     if (tp+fp+fn)   > 0 else None
    ov_gt  = tp / (tp+fn)        if (tp+fn)       > 0 else None
    ov_pr  = tp / (tp+fp)        if (tp+fp)       > 0 else None
    return dice, jac, ov_gt, ov_pr


def _vol_sim(a, b):
    return 1.0 - abs(a-b)/(a+b) if (a+b) > 0 else None


def _hausdorff_mhd(pred_np, gt_np, cls, spacing_xyz):
    """spacing_xyz must be (sx, sy, sz) in mm - SimpleITK SetSpacing convention."""
    pm = (pred_np == cls).astype(np.uint8)
    gm = (gt_np   == cls).astype(np.uint8)
    if not pm.any() or not gm.any():
        return None, None
    ps = sitk.GetImageFromArray(pm)
    ps.SetSpacing([float(s) for s in spacing_xyz])
    gs = sitk.GetImageFromArray(gm)
    gs.SetSpacing([float(s) for s in spacing_xyz])
    try:
        f = sitk.HausdorffDistanceImageFilter()
        f.Execute(gs, ps)
        return f.GetHausdorffDistance(), f.GetAverageHausdorffDistance()
    except Exception as exc:
        print(f"   Hausdorff failed cls={cls}: {exc}")
        return None, None


def _infer_flags(gt_np, meta):
    m = dict(meta)
    for key, cls in (("has_gtv_t", 1), ("has_gtv_n", 2)):
        if m.get(key) in (None, "", "None"):
            m[key] = bool((gt_np == cls).any())
    return m


# ── Core metric computation ────────────────────────────────────────────────────

def compute_rich_metrics(pred_np: np.ndarray, gt_np: np.ndarray,
                          spacing_xyz: tuple, case_meta: dict) -> dict:
    """Compute all metrics for one case.

    Parameters
    ----------
    pred_np      : (Z, Y, X) uint8 prediction  (any consistent 3-D order)
    gt_np        : (Z, Y, X) uint8 ground truth  (same order)
    spacing_xyz  : (sx, sy, sz) voxel size in mm - ITK convention for sitk
    case_meta    : dict with case_id, patient, timepoint, …
    """
    vox_mm3    = float(spacing_xyz[0]) * float(spacing_xyz[1]) * float(spacing_xyz[2])
    case_meta  = _infer_flags(gt_np, case_meta)
    total_vox  = float(pred_np.size)
    gt_avail   = case_meta.get("gt_available", True)

    row = {f: "" for f in CSV_FIELDS}
    row.update({
        "case_id":            case_meta.get("case_id",    ""),
        "patient":            case_meta.get("patient",    ""),
        "timepoint":          case_meta.get("timepoint",  ""),
        "study_date":         case_meta.get("study_date", ""),
        "gt_available":       gt_avail,
        "gt_reason":          case_meta.get("gt_reason",  "ok"),
        "has_gtv_t":          case_meta.get("has_gtv_t",  ""),
        "has_gtv_n":          case_meta.get("has_gtv_n",  ""),
        "gt_labels_present":  ", ".join(n for c,(n,_) in _CLASS_INFO.items()
                                         if (gt_np==c).any()) or "none",
        "pred_labels_present":  ", ".join(n for c,(n,_) in _CLASS_INFO.items()
                                           if (pred_np==c).any()) or "none",
        "gt_vol_total_mm3":   round(float(np.sum(gt_np   > 0)) * vox_mm3, 1),
        "pred_vol_total_mm3": round(float(np.sum(pred_np > 0)) * vox_mm3, 1),
    })

    if not gt_avail:
        row["comments"] = "no_GT_annotation"
        return row

    comments, dice_vals = [], []

    for cls, (cname, has_key) in _CLASS_INFO.items():
        gm = (gt_np   == cls)
        pm = (pred_np == cls)
        gv = float(gm.sum()) * vox_mm3
        pv = float(pm.sum()) * vox_mm3

        row[f"gt_vol_{cname}_mm3"]   = round(gv, 1)
        row[f"pred_vol_{cname}_mm3"] = round(pv, 1)
        row[f"vol_sim_{cname}"]      = _safe(_vol_sim(gv, pv))
        row[f"gt_count_{cname}"]     = _count_objects(gt_np,   cls)
        row[f"pred_count_{cname}"]   = _count_objects(pred_np, cls)

        tp = float(np.logical_and(pm,  gm).sum())
        fp = float(np.logical_and(pm, ~gm).sum())
        fn = float(np.logical_and(~pm,  gm).sum())
        tn = total_vox - tp - fp - fn

        row.update({
            f"{cname}_TP_mm3": round(tp*vox_mm3, 1),
            f"{cname}_FP_mm3": round(fp*vox_mm3, 1),
            f"{cname}_FN_mm3": round(fn*vox_mm3, 1),
            f"{cname}_TN_mm3": round(tn*vox_mm3, 1),
        })

        if case_meta.get(has_key) is False:
            comments.append(f"no_{cname}_gt")
            if pv > 0:
                comments.append(f"pred_{cname}_FP_{round(pv/1000,1)}cm3")
            continue

        if not gm.any() and not pm.any():
            comments.append(f"{cname}_TN"); continue
        if not gm.any(): comments.append(f"no_{cname}_gt_voxels")
        if not pm.any(): comments.append(f"pred_{cname}_empty")

        dice, jac, ov_gt, ov_pr = _overlap(tp, fp, fn, tn)
        if dice is not None:
            dice_vals.append(dice)
        row.update({
            f"{cname}_dice":         _safe(dice),
            f"{cname}_jaccard":      _safe(jac),
            f"{cname}_overlap_gt":   _safe(ov_gt),
            f"{cname}_overlap_pred": _safe(ov_pr),
        })
        if dice is not None and dice < 0.3:
            comments.append(f"{cname}_low_dice_{dice:.2f}")

        hd, mhd = _hausdorff_mhd(pred_np, gt_np, cls, spacing_xyz)
        row[f"{cname}_hausdorff_mm"] = _safe(hd,  2)
        row[f"{cname}_mhd_mm"]       = _safe(mhd, 2)

    row["mean_dice"] = _safe(float(np.mean(dice_vals))) if dice_vals else ""
    row["comments"]  = "; ".join(dict.fromkeys(comments))
    return row


# ── GT + prediction loaders ────────────────────────────────────────────────────

def _load_nifti(path: str):
    """Return (np.ndarray uint8, spacing_xyz tuple)."""
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img).astype(np.uint8), img.GetSpacing()


def _load_gt_swincross_npz(npz_path: str):
    """SwinCross NPZ -> (label uint8, spacing_xyz).

    label    : (R,A,S) uint8  - kept as-is; pred will be compared in same space
    spacing  : orig_spacing (sx,sy,sz) ITK order
    """
    with np.load(npz_path, allow_pickle=False) as npz:
        lbl = npz["label"].astype(np.uint8)
        spacing_xyz = tuple(float(s) for s in npz["orig_spacing"])
    return lbl, spacing_xyz


def _load_gt_temporal_npz(npz_path: str):
    """TemPoRAL NPZ -> (gts uint8, spacing_xyz).

    gts      : (D,H,W) uint8
    spacing  : stored as (sz,sy,sx) -> converted to (sx,sy,sz) for sitk
    """
    with np.load(npz_path, allow_pickle=False) as npz:
        gts = npz["gts"].astype(np.uint8)
        sz, sy, sx = npz["spacing"]
    return gts, (float(sx), float(sy), float(sz))


def _load_gt_auto(path: str):
    """Auto-detect GT source (NIfTI or NPZ, SwinCross or TemPoRAL)."""
    if path.endswith(".nii.gz") or path.endswith(".nii"):
        return _load_nifti(path)
    # NPZ - detect by keys
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
    if "label" in keys:
        return _load_gt_swincross_npz(path)
    if "gts" in keys:
        return _load_gt_temporal_npz(path)
    raise ValueError(f"Cannot determine GT format from {path}. Keys: {keys}")


def _load_pred_nifti(path: str):
    """Load prediction NIfTI -> (uint8, spacing_xyz)."""
    return _load_nifti(path)


def _resample_pred_to_gt(pred_sitk: sitk.Image,
                          gt_sitk:   sitk.Image) -> sitk.Image:
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(gt_sitk)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetTransform(sitk.Transform())
    r.SetOutputPixelType(sitk.sitkUInt8)
    return r.Execute(pred_sitk)


# ── Entry building ─────────────────────────────────────────────────────────────

def _build_entries(args) -> list:
    """Return list of dicts: {case_id, pred_nifti, gt_path, meta}."""
    entries = []

    if args.json_list:
        data_dir  = args.data_dir or "."
        json_path = (join(data_dir, args.json_list)
                     if not os.path.isabs(args.json_list) else args.json_list)
        with open(json_path) as f:
            js = json.load(f)

        # ── TemPoRAL manifest.json ─────────────────────────────────────
        if "cases" in js:
            for e in js["cases"]:
                case_id = e.get("case_id", "")
                entries.append({
                    "case_id":   case_id,
                    "pred_nifti": join(args.output_dir, f"{case_id}_Pred.nii.gz"),
                    "gt_path":   join(data_dir, e.get("npz_file", "")),
                    "meta":      dict(e, gt_available=e.get("gt_available", True)),
                })
            return entries

        # ── SwinCross JSON ─────────────────────────────────────────────
        all_entries = []
        for key in ("validation", "training", "testing"):
            all_entries.extend(js.get(key, []))

        for e in all_entries:
            case_id = (e.get("case_id")
                       or basename(e.get("npz", "unknown")).replace(".npz", ""))
            # GT: prefer the NIfTI path stored in the JSON entry
            gt_path = None
            if e.get("label"):
                gt_path = join(data_dir, e["label"])
            elif e.get("npz"):
                gt_path = join(data_dir, e["npz"])

            entries.append({
                "case_id":    case_id,
                "pred_nifti": join(args.output_dir, f"{case_id}_Pred.nii.gz"),
                "gt_path":    gt_path,
                "meta":       dict(e, case_id=case_id,
                                   gt_available=e.get("gt_available", True)),
            })
        return entries

    # ── Legacy --pred_dir + --gt_dir ──────────────────────────────────
    pred_files = sorted(glob(join(args.pred_dir, "*_Pred.nii.gz")))
    if not pred_files:
        pred_files = sorted(glob(join(args.pred_dir, "*.nii.gz")))

    # Load manifest for temporal metadata if present
    extra_meta: dict = {}
    if args.manifest and os.path.exists(args.manifest):
        with open(args.manifest) as f:
            mf = json.load(f)
        for c in mf.get("cases", []):
            extra_meta[c["case_id"]] = c

    for pf in pred_files:
        case_id = (basename(pf)
                   .replace("_Pred.nii.gz", "")
                   .replace(".nii.gz", ""))
        # Try GT as NIfTI first, then NPZ
        gt_nifti = join(args.gt_dir, f"{case_id}_gt.nii.gz")
        gt_npz   = join(args.gt_dir, f"{case_id}.npz")
        gt_path  = gt_nifti if os.path.exists(gt_nifti) else gt_npz

        meta = {"case_id": case_id, "gt_available": True}
        meta.update(extra_meta.get(case_id, {}))

        entries.append({
            "case_id":    case_id,
            "pred_nifti": pf,
            "gt_path":    gt_path,
            "meta":       meta,
        })
    return entries


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Universal rich evaluation for MedSAM2 - HECKTOR 2026 + TemPoRAL"
    )
    # JSON / manifest mode
    parser.add_argument("--data_dir",   default=None)
    parser.add_argument("--json_list",  default=None,
                        help="SwinCross JSON filename OR 'manifest.json'.")
    # Legacy directory mode
    parser.add_argument("--pred_dir",   default=None)
    parser.add_argument("--gt_dir",     default=None)
    parser.add_argument("--manifest",   default=None,
                        help="[Legacy] Path to manifest.json for temporal metadata.")
    # Output
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = join(args.output_dir, "per_case_evaluation_rich.csv")

    all_entries = _build_entries(args)
    if not all_entries:
        print("No entries found. Check --data_dir / --json_list or --pred_dir.")
        return

    first_write = True
    all_rows    = []

    for idx, entry in enumerate(all_entries):
        case_id = entry["case_id"]
        meta    = entry.get("meta", {"case_id": case_id, "gt_available": True})
        meta.setdefault("case_id",      case_id)
        meta.setdefault("gt_available", True)

        print(f"\n[{idx+1}/{len(all_entries)}] {case_id}")

        # ── Load prediction ───────────────────────────────────────────
        pred_nifti = entry.get("pred_nifti", "")
        if not pred_nifti or not os.path.exists(pred_nifti):
            print("  Prediction NIfTI not found - skipping.")
            continue
        pred_np, pred_spacing = _load_pred_nifti(pred_nifti)

        # ── Load ground truth ─────────────────────────────────────────
        gt_path = entry.get("gt_path", "")
        if not gt_path or not os.path.exists(gt_path):
            print("  Ground truth not found - skipping.")
            continue
        gt_np, gt_spacing = _load_gt_auto(gt_path)

        # Use GT spacing as the reference (prediction was written in GT space)
        spacing_xyz = gt_spacing

        # ── Shape alignment ───────────────────────────────────────────
        if pred_np.shape != gt_np.shape:
            print(f"  ⚠  Shape mismatch pred={pred_np.shape} gt={gt_np.shape} - resampling.")
            p_sitk = sitk.GetImageFromArray(pred_np)
            p_sitk.SetSpacing([float(s) for s in spacing_xyz])
            g_sitk = sitk.GetImageFromArray(gt_np)
            g_sitk.SetSpacing([float(s) for s in spacing_xyz])
            pred_np = sitk.GetArrayFromImage(
                _resample_pred_to_gt(p_sitk, g_sitk)).astype(np.uint8)

        # ── Skip no-GT cases ──────────────────────────────────────────
        if not meta.get("gt_available", True):
            print("  No GT - skipping metrics.")
            row = {f: "" for f in CSV_FIELDS}
            row.update({"case_id": case_id, "comments": "no_GT_annotation"})
        else:
            row = compute_rich_metrics(pred_np, gt_np, spacing_xyz, meta)
            print(
                f"  GTVp  Dice={row.get('GTVp_dice','NA')}  "
                f"Jac={row.get('GTVp_jaccard','NA')}  "
                f"HD={row.get('GTVp_hausdorff_mm','NA')}mm"
            )
            print(
                f"  GTVn  Dice={row.get('GTVn_dice','NA')}  "
                f"Jac={row.get('GTVn_jaccard','NA')}  "
                f"HD={row.get('GTVn_hausdorff_mm','NA')}mm"
            )
            print(f"  Mean Dice={row.get('mean_dice','NA')}")

        all_rows.append(row)

        with open(csv_path, "w" if first_write else "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if first_write:
                w.writeheader()
            w.writerow(row)
        first_write = False

    # ── MEAN row ──────────────────────────────────────────────────────
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        num_cols  = df.select_dtypes(include=["float64", "int64"]).columns
        mean_vals = df[num_cols].mean().round(4).to_dict()
        mean_row  = {f: "" for f in CSV_FIELDS}
        mean_row.update(mean_vals)
        mean_row["case_id"] = "MEAN"
        mean_row["patient"] = "ALL_CASES"
        mean_row["comments"] = f"Average across {len(df)} cases"
        pd.DataFrame([mean_row]).to_csv(csv_path, mode="a", header=False, index=False)
        print(f"\nAppended MEAN row ({len(df)} cases).")
    except ImportError:
        print("\nPandas not installed - skipping MEAN row.")

    mean_dices = [float(r["mean_dice"]) for r in all_rows if r.get("mean_dice")]
    print("\n" + "=" * 60)
    if mean_dices:
        print(f"Overall Mean Dice: {np.mean(mean_dices):.4f}  (n={len(mean_dices)})")
    else:
        print("No valid cases with GT evaluated.")
    print(f"CSV -> {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
