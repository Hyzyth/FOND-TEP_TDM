#!/usr/bin/env python3
"""
evaluate_predictions.py
========================
Enriched evaluation script for SwinCross predictions on HECKTOR or TemPoRAL.
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

# ── Constants ─────────────────────────────────────────────────────────────────
_CLASS_INFO = {1: ("GTVp", "has_gtv_t"), 2: ("GTVn", "has_gtv_n")}

CSV_FIELDS = [
    "case_id", "patient", "timepoint", "study_date",
    "gt_available", "gt_reason", "has_gtv_t", "has_gtv_n",
    "gt_labels_present", "pred_labels_present",
    "GTVp_dice", "GTVn_dice", "mean_dice",
    "GTVp_iou", "GTVn_iou",
    "gt_vol_GTVp_mm3", "gt_vol_GTVn_mm3", "gt_vol_total_mm3",
    "pred_vol_GTVp_mm3", "pred_vol_GTVn_mm3", "pred_vol_total_mm3",
    "gt_count_GTVp", "pred_count_GTVp",
    "gt_count_GTVn", "pred_count_GTVn",
    "vol_sim_GTVp", "vol_sim_GTVn",
    "GTVp_TP_mm3", "GTVp_FP_mm3", "GTVp_FN_mm3", "GTVp_TN_mm3",
    "GTVn_TP_mm3", "GTVn_FP_mm3", "GTVn_FN_mm3", "GTVn_TN_mm3",
    "GTVp_sensitivity", "GTVp_precision", "GTVp_specificity",
    "GTVn_sensitivity", "GTVn_precision", "GTVn_specificity",
    "comments"
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _safe(value: float, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)): return ""
    return str(round(value, decimals))


def count_objects(image_np: np.ndarray, target_value: int) -> int:
    binary_mask = (image_np == target_value)
    if not binary_mask.any():
        return 0
    structure = ndimage.generate_binary_structure(image_np.ndim, image_np.ndim)
    _, num_features = ndimage.label(binary_mask, structure=structure)
    return num_features


def _metrics(tp: float, fp: float, fn: float, tn: float):
    dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else None
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else None
    sens = tp / (tp + fn) if (tp + fn) > 0 else None
    prec = tp / (tp + fp) if (tp + fp) > 0 else None
    spec = tn / (tn + fp) if (tn + fp) > 0 else None
    return dice, iou, sens, prec, spec


def _vol_sim(va: float, vb: float) -> float | None:
    """
    Volume Similarity = 1 - |Va - Vb| / (Va + Vb).
    Returns None when both volumes are zero (undefined).
    Ranges 0 (completely different volume) to 1 (identical volume).
    """
    return 1.0 - abs(va - vb) / (va + vb) if (va + vb) > 0 else None


def compute_rich_metrics(pred_path: str, gt_path: str, case_meta: dict) -> dict:
    """
    Load prediction and GT from disk, compute all metrics.

    Both files are expected to be in original CT space (same grid).
    Volumes are computed from the GT image's physical spacing.

    Parameters
    ----------
    pred_path   : path to predicted segmentation NIfTI
    gt_path     : path to ground-truth NIfTI (may be all-zero for no-GT cases)
    case_meta   : dict from the dataset JSON entry for this case
    """
    pred_sitk = sitk.ReadImage(pred_path)
    gt_sitk   = sitk.ReadImage(gt_path)

    # Voxel volume from GT spacing (original CT space)
    sx, sy, sz = gt_sitk.GetSpacing()
    vox_mm3    = sx * sy * sz

    pred_np = sitk.GetArrayFromImage(pred_sitk).astype(np.uint8)
    gt_np   = sitk.GetArrayFromImage(gt_sitk).astype(np.uint8)

    # Sanity check — sizes should match (both built on the same CT grid)
    if pred_np.shape != gt_np.shape:
        print(f"  ⚠  Shape mismatch: pred {pred_np.shape} vs GT {gt_np.shape}. "
              f"Resampling prediction to GT grid.")
        r = sitk.ResampleImageFilter()
        r.SetReferenceImage(gt_sitk)
        r.SetInterpolator(sitk.sitkNearestNeighbor)
        r.SetTransform(sitk.Transform())
        r.SetOutputPixelType(sitk.sitkUInt8)
        pred_sitk = r.Execute(pred_sitk)
        pred_np   = sitk.GetArrayFromImage(pred_sitk).astype(np.uint8)

    total_vox = float(pred_np.size)
    gt_available = case_meta.get("gt_available", True)

    row = {f: "" for f in CSV_FIELDS}
    row.update({
        "case_id": case_meta.get("case_id", os.path.basename(pred_path).replace("_Pred.nii.gz", "")),
        "patient": case_meta.get("patient", ""),
        "timepoint": case_meta.get("timepoint", ""),
        "study_date": case_meta.get("study_date", ""),
        "gt_available": gt_available,
        "gt_reason": case_meta.get("gt_reason", "ok"),
        "has_gtv_t": case_meta.get("has_gtv_t", ""),
        "has_gtv_n": case_meta.get("has_gtv_n", ""),
        "gt_labels_present": ", ".join([n for c, (n, _) in _CLASS_INFO.items() if (gt_np == c).any()]) or "none",
        "pred_labels_present": ", ".join([n for c, (n, _) in _CLASS_INFO.items() if (pred_np == c).any()]) or "none",
        "gt_vol_total_mm3": round(float(np.sum(gt_np > 0)) * vox_mm3, 1),
        "pred_vol_total_mm3": round(float(np.sum(pred_np > 0)) * vox_mm3, 1),
    })

    if not gt_available:
        row["comments"] = "no_GT_annotation"
        return row

    comments, dice_vals = [], []

    for cls, (cls_name, has_key) in _CLASS_INFO.items():
        gt_mask, pred_mask = (gt_np == cls), (pred_np == cls)
        gt_vol, pred_vol = float(gt_mask.sum()) * vox_mm3, float(pred_mask.sum()) * vox_mm3

        row[f"gt_vol_{cls_name}_mm3"] = round(gt_vol, 1)
        row[f"pred_vol_{cls_name}_mm3"] = round(pred_vol, 1)
        row[f"vol_sim_{cls_name}"] = _safe(_vol_sim(gt_vol, pred_vol))
        
        # Connected object counting
        row[f"gt_count_{cls_name}"] = count_objects(gt_np, cls)
        row[f"pred_count_{cls_name}"] = count_objects(pred_np, cls)

        tp_vox = float(np.logical_and(pred_mask, gt_mask).sum())
        fp_vox = float(np.logical_and(pred_mask, ~gt_mask).sum())
        fn_vox = float(np.logical_and(~pred_mask, gt_mask).sum())
        tn_vox = total_vox - tp_vox - fp_vox - fn_vox

        row.update({
            f"{cls_name}_TP_mm3": round(tp_vox * vox_mm3, 1),
            f"{cls_name}_FP_mm3": round(fp_vox * vox_mm3, 1),
            f"{cls_name}_FN_mm3": round(fn_vox * vox_mm3, 1),
            f"{cls_name}_TN_mm3": round(tn_vox * vox_mm3, 1),
        })

        if case_meta.get(has_key, None) is False:
            comments.append(f"no_{cls_name}_gt")
            if pred_vol > 0: comments.append(f"pred_{cls_name}_FP_{round(pred_vol / 1000, 1)}cm3")
            continue

        if not gt_mask.any() and not pred_mask.any():
            comments.append(f"{cls_name}_TN")
            continue

        if not gt_mask.any(): comments.append(f"no_{cls_name}_gt_voxels")
        if not pred_mask.any(): comments.append(f"pred_{cls_name}_empty")

        dice, iou, sens, prec, spec = _metrics(tp_vox, fp_vox, fn_vox, tn_vox)
        if dice is not None: dice_vals.append(dice)

        row.update({
            f"{cls_name}_dice": _safe(dice), f"{cls_name}_iou": _safe(iou),
            f"{cls_name}_sensitivity": _safe(sens), f"{cls_name}_precision": _safe(prec),
            f"{cls_name}_specificity": _safe(spec)
        })

        if dice is not None and dice < 0.3: comments.append(f"{cls_name}_low_dice_{dice:.2f}")

    row["mean_dice"] = _safe(float(np.mean(dice_vals))) if dice_vals else ""
    row["comments"] = "; ".join(dict.fromkeys(comments))
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal Rich Evaluation for SwinCross")
    parser.add_argument("--data_dir", required=True, help="Dataset root directory")
    parser.add_argument("--json_list", required=True, help="Dataset JSON file")
    parser.add_argument("--output_dir", required=True, help="Folder containing *_Pred.nii.gz")
    args = parser.parse_args()

    csv_path = os.path.join(args.output_dir, "per_case_evaluation_rich.csv")
    
    with open(os.path.join(args.data_dir, args.json_list)) as f:
        dataset_json = json.load(f)

    all_entries = []
    for split in ("validation", "training", "testing"):
        all_entries.extend(dataset_json.get(split, []))

    all_rows = []
    first_write = True

    for idx, entry in enumerate(all_entries):
        img_name = os.path.basename(entry.get("image", ""))
        img_prefix_clean = img_name.split(".")[0].replace("_petct", "")
        gt_rel = entry.get("label", "")
        gt_path = os.path.join(args.data_dir, gt_rel) if gt_rel else None
        
        # Notice we simplified the wildcard since test.py no longer adds the DSC to the name
        pred_path_candidates = glob.glob(os.path.join(args.output_dir, f"{img_prefix_clean}_Pred.nii.gz"))
        
        # Fallback for old files generated with DSC in name
        if not pred_path_candidates:
            pred_path_candidates = glob.glob(os.path.join(args.output_dir, f"{img_prefix_clean}_dsc*_Pred.nii.gz"))

        print(f"\n[{idx+1}/{len(all_entries)}] Evaluating: {img_prefix_clean}")

        if not pred_path_candidates:
            print("  ❌ Prediction not found on disk. Run test.py first!")
            continue
            
        pred_path = pred_path_candidates[0]

        if not entry.get("gt_available", True) or not gt_path or not os.path.exists(gt_path):
            print("  ℹ No GT available. Skipping metric calculation.")
            row = {f: "" for f in CSV_FIELDS}
            row.update({"case_id": img_prefix_clean, "comments": "no_GT_annotation"})
        else:
            row = compute_rich_metrics(pred_path, gt_path, entry)
            print(f"  GTVp Dice={row.get('GTVp_dice', 'NA')} | GTVn Dice={row.get('GTVn_dice', 'NA')} | Mean={row.get('mean_dice', 'NA')} | Obj_Counts(GT/Pred) T:[{row.get('gt_count_GTVp')}/{row.get('pred_count_GTVp')}] N:[{row.get('gt_count_GTVn')}/{row.get('pred_count_GTVn')}")

        all_rows.append(row)
        
        with open(csv_path, "w" if first_write else "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if first_write: writer.writeheader()
            writer.writerow(row)
        first_write = False

    mean_dices = [float(r["mean_dice"]) for r in all_rows if r.get("mean_dice")]
    print("\n" + "="*50)
    print(f"Overall Mean Dice: {np.mean(mean_dices):.4f} (n={len(mean_dices)})" if mean_dices else "No valid cases evaluated.")
    print(f"Rich CSV written to: {csv_path}")
    print("="*50)

if __name__ == "__main__":
    main()
