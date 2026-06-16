#!/usr/bin/env python3
"""
evaluate_predictions.py
========================
Enriched evaluation script for SwinCross predictions on HECKTOR or TemPoRAL.

Change vs. original: case_id is now read from the JSON entry's "case_id" field
(produced by the NPZ preprocessing scripts) with a fallback to the old "image"
filename derivation, so both old and new JSON formats are supported.
"""

import argparse
import csv
import glob
import json
import math
import os
import warnings
import numpy as np
import SimpleITK as sitk
from scipy import ndimage

warnings.filterwarnings("ignore")

_CLASS_INFO = {1: ("GTVp", "has_gtv_t"), 2: ("GTVn", "has_gtv_n")}

CSV_FIELDS = [
    "case_id", "patient", "timepoint", "study_date",
    "gt_available", "gt_reason", "has_gtv_t", "has_gtv_n",
    "gt_labels_present", "pred_labels_present",
    "GTVp_dice",         "GTVn_dice",         "mean_dice",
    "GTVp_jaccard",      "GTVn_jaccard",
    "GTVp_hausdorff_mm", "GTVn_hausdorff_mm",
    "GTVp_mhd_mm",       "GTVn_mhd_mm",
    "GTVp_overlap_gt",   "GTVn_overlap_gt",
    "GTVp_overlap_pred", "GTVn_overlap_pred",
    "gt_vol_GTVp_mm3", "gt_vol_GTVn_mm3", "gt_vol_total_mm3",
    "pred_vol_GTVp_mm3", "pred_vol_GTVn_mm3", "pred_vol_total_mm3",
    "gt_count_GTVp", "pred_count_GTVp",
    "gt_count_GTVn", "pred_count_GTVn",
    "vol_sim_GTVp", "vol_sim_GTVn",
    "GTVp_TP_mm3", "GTVp_FP_mm3", "GTVp_FN_mm3", "GTVp_TN_mm3",
    "GTVn_TP_mm3", "GTVn_FP_mm3", "GTVn_FN_mm3", "GTVn_TN_mm3",
    "comments",
]


def _safe(value, decimals=4):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(round(float(value), decimals))


def count_objects(image_np, target_value):
    binary_mask = (image_np == target_value)
    if not binary_mask.any():
        return 0
    structure = ndimage.generate_binary_structure(image_np.ndim, image_np.ndim)
    _, n = ndimage.label(binary_mask, structure=structure)
    return n


def _overlap_metrics(tp, fp, fn, tn):
    dice        = 2.0*tp / (2.0*tp+fp+fn) if (2.0*tp+fp+fn) > 0 else None
    jaccard     = tp / (tp+fp+fn)          if (tp+fp+fn) > 0        else None
    overlap_gt  = tp / (tp+fn)             if (tp+fn) > 0           else None
    overlap_pred= tp / (tp+fp)             if (tp+fp) > 0           else None
    return dice, jaccard, overlap_gt, overlap_pred


def _vol_sim(va, vb):
    return 1.0 - abs(va-vb)/(va+vb) if (va+vb) > 0 else None


def _hausdorff_mhd(pred_np, gt_np, cls, reference_sitk):
    pred_mask = (pred_np == cls).astype(np.uint8)
    gt_mask   = (gt_np   == cls).astype(np.uint8)
    if not pred_mask.any() or not gt_mask.any():
        return None, None
    pred_bin = sitk.GetImageFromArray(pred_mask)
    pred_bin.CopyInformation(reference_sitk)
    gt_bin   = sitk.GetImageFromArray(gt_mask)
    gt_bin.CopyInformation(reference_sitk)
    try:
        f = sitk.HausdorffDistanceImageFilter()
        f.Execute(gt_bin, pred_bin)
        return f.GetHausdorffDistance(), f.GetAverageHausdorffDistance()
    except Exception as exc:
        print(f"   ⚠  Hausdorff failed for class {cls}: {exc}")
        return None, None


def _infer_gtv_flags(gt_np, case_meta):
    meta = dict(case_meta)
    for key, cls in (("has_gtv_t", 1), ("has_gtv_n", 2)):
        if meta.get(key) in (None, "", "None"):
            meta[key] = bool((gt_np == cls).any())
    return meta


def _derive_case_id(entry):
    """
    Return the case_id for a JSON entry.
    Priority:
      1. entry["case_id"]  — set by NPZ preprocessing scripts
      2. Derived from entry["image"] filename — legacy MONAI JSON format
      3. Derived from entry["npz"] filename   — fallback
    """
    if entry.get("case_id"):
        return str(entry["case_id"])
    img_name = os.path.basename(entry.get("image", entry.get("npz", "unknown")))
    return img_name.split(".")[0].replace("_petct", "")


def compute_rich_metrics(pred_path, gt_path, case_meta):
    pred_sitk = sitk.ReadImage(pred_path)
    gt_sitk   = sitk.ReadImage(gt_path)
    sx, sy, sz = gt_sitk.GetSpacing()
    vox_mm3    = sx * sy * sz
    pred_np = sitk.GetArrayFromImage(pred_sitk).astype(np.uint8)
    gt_np   = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)

    if pred_np.shape != gt_np.shape:
        print(f"  ⚠  Shape mismatch: pred {pred_np.shape} vs GT {gt_np.shape}. Resampling.")
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(gt_sitk)
        r.SetInterpolator(sitk.sitkNearestNeighbor)
        r.SetTransform(sitk.Transform())
        r.SetOutputPixelType(sitk.sitkUInt8)
        pred_sitk = r.Execute(pred_sitk)
        pred_np   = sitk.GetArrayFromImage(pred_sitk).astype(np.uint8)

    case_meta  = _infer_gtv_flags(gt_np, case_meta)
    total_vox  = float(pred_np.size)
    gt_available = case_meta.get("gt_available", True)

    row = {f: "" for f in CSV_FIELDS}
    row.update({
        "case_id":            case_meta.get("case_id", _derive_case_id(case_meta)),
        "patient":            case_meta.get("patient",    ""),
        "timepoint":          case_meta.get("timepoint",  ""),
        "study_date":         case_meta.get("study_date", ""),
        "gt_available":       gt_available,
        "gt_reason":          case_meta.get("gt_reason",  "ok"),
        "has_gtv_t":          case_meta.get("has_gtv_t",  ""),
        "has_gtv_n":          case_meta.get("has_gtv_n",  ""),
        "gt_labels_present":  ", ".join(
            [n for c,(n,_) in _CLASS_INFO.items() if (gt_np==c).any()]) or "none",
        "pred_labels_present": ", ".join(
            [n for c,(n,_) in _CLASS_INFO.items() if (pred_np==c).any()]) or "none",
        "gt_vol_total_mm3":   round(float(np.sum(gt_np   > 0)) * vox_mm3, 1),
        "pred_vol_total_mm3": round(float(np.sum(pred_np > 0)) * vox_mm3, 1),
    })

    if not gt_available:
        row["comments"] = "no_GT_annotation"
        return row

    comments, dice_vals = [], []

    for cls, (cls_name, has_key) in _CLASS_INFO.items():
        gt_mask   = (gt_np   == cls)
        pred_mask = (pred_np == cls)
        gt_vol    = float(gt_mask.sum())   * vox_mm3
        pred_vol  = float(pred_mask.sum()) * vox_mm3

        row[f"gt_vol_{cls_name}_mm3"]   = round(gt_vol,   1)
        row[f"pred_vol_{cls_name}_mm3"] = round(pred_vol, 1)
        row[f"vol_sim_{cls_name}"]      = _safe(_vol_sim(gt_vol, pred_vol))
        row[f"gt_count_{cls_name}"]     = count_objects(gt_np,   cls)
        row[f"pred_count_{cls_name}"]   = count_objects(pred_np, cls)

        tp_vox = float(np.logical_and(pred_mask,  gt_mask).sum())
        fp_vox = float(np.logical_and(pred_mask, ~gt_mask).sum())
        fn_vox = float(np.logical_and(~pred_mask,  gt_mask).sum())
        tn_vox = total_vox - tp_vox - fp_vox - fn_vox

        row.update({
            f"{cls_name}_TP_mm3": round(tp_vox*vox_mm3, 1),
            f"{cls_name}_FP_mm3": round(fp_vox*vox_mm3, 1),
            f"{cls_name}_FN_mm3": round(fn_vox*vox_mm3, 1),
            f"{cls_name}_TN_mm3": round(tn_vox*vox_mm3, 1),
        })

        if case_meta.get(has_key) is False:
            comments.append(f"no_{cls_name}_gt")
            if pred_vol > 0:
                comments.append(f"pred_{cls_name}_FP_{round(pred_vol/1000,1)}cm3")
            continue

        if not gt_mask.any() and not pred_mask.any():
            comments.append(f"{cls_name}_TN")
            continue
        if not gt_mask.any():
            comments.append(f"no_{cls_name}_gt_voxels")
        if not pred_mask.any():
            comments.append(f"pred_{cls_name}_empty")

        dice, jaccard, overlap_gt, overlap_pred = _overlap_metrics(
            tp_vox, fp_vox, fn_vox, tn_vox)
        if dice is not None:
            dice_vals.append(dice)
        row.update({
            f"{cls_name}_dice":         _safe(dice),
            f"{cls_name}_jaccard":      _safe(jaccard),
            f"{cls_name}_overlap_gt":   _safe(overlap_gt),
            f"{cls_name}_overlap_pred": _safe(overlap_pred),
        })
        if dice is not None and dice < 0.3:
            comments.append(f"{cls_name}_low_dice_{dice:.2f}")

        hd_mm, mhd_mm = _hausdorff_mhd(pred_np, gt_np, cls, gt_sitk)
        row[f"{cls_name}_hausdorff_mm"] = _safe(hd_mm,  2)
        row[f"{cls_name}_mhd_mm"]       = _safe(mhd_mm, 2)

    row["mean_dice"] = _safe(float(np.mean(dice_vals))) if dice_vals else ""
    row["comments"]  = "; ".join(dict.fromkeys(comments))
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Universal Rich Evaluation for SwinCross")
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--json_list",  required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    csv_path = os.path.join(args.output_dir, "per_case_evaluation_rich.csv")

    with open(os.path.join(args.data_dir, args.json_list)) as f:
        dataset_json = json.load(f)

    all_entries = []
    for split in ("validation", "training", "testing"):
        all_entries.extend(dataset_json.get(split, []))

    first_write = True

    for idx, entry in enumerate(all_entries):
        # ── Derive case_id (supports both old "image" key and new "case_id" key) ──
        case_id  = _derive_case_id(entry)
        gt_rel   = entry.get("label", "")
        gt_path  = os.path.join(args.data_dir, gt_rel) if gt_rel else None

        # Prediction filename: always <case_id>_Pred.nii.gz
        pred_path_candidates = glob.glob(
            os.path.join(args.output_dir, f"{case_id}_Pred.nii.gz"))
        # Legacy fallback: DSC embedded in filename
        if not pred_path_candidates:
            pred_path_candidates = glob.glob(
                os.path.join(args.output_dir, f"{case_id}_dsc*_Pred.nii.gz"))

        print(f"\n[{idx+1}/{len(all_entries)}] Evaluating: {case_id}")

        if not pred_path_candidates:
            print("  ❌ Prediction not found. Run test.py first.")
            continue

        pred_path = pred_path_candidates[0]

        # Build a meta dict that mirrors the entry (so compute_rich_metrics
        # has access to has_gtv_t, gt_available, patient, timepoint, etc.)
        case_meta = dict(entry)
        case_meta["case_id"] = case_id

        if (not entry.get("gt_available", True)
                or not gt_path
                or not os.path.exists(gt_path)):
            print("  ℹ No GT available — skipping metric calculation.")
            row = {f: "" for f in CSV_FIELDS}
            row.update({"case_id": case_id, "comments": "no_GT_annotation"})
        else:
            row = compute_rich_metrics(pred_path, gt_path, case_meta)
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

        with open(csv_path, "w" if first_write else "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if first_write:
                writer.writeheader()
            writer.writerow(row)
        first_write = False

    # Compute the means
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)

        # Only compute for numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        mean_values = df[numeric_cols].mean().round(4).to_dict()

        # Create a new row representing the mean
        mean_row = {field: "" for field in CSV_FIELDS}
        mean_row.update(mean_values)
        mean_row["case_id"] = "MEAN"
        mean_row["patient"] = "ALL_CASES"
        mean_row["comments"] = f"Average across {len(df)} cases"

        # Append the mean row to the CSV
        pd.DataFrame([mean_row]).to_csv(csv_path, mode='a', header=False, index=False)
        print(f"\nAppended mean values to CSV.")
    except ImportError:
        print("\nPandas not installed. Skipping mean calculation.")

    print(f"\nCSV → {csv_path}")


if __name__ == "__main__":
    main()
