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

def _safe(value: float, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(round(float(value), decimals))


def count_objects(image_np: np.ndarray, target_value: int) -> int:
    binary_mask = (image_np == target_value)
    if not binary_mask.any():
        return 0
    structure = ndimage.generate_binary_structure(image_np.ndim, image_np.ndim)
    _, num_features = ndimage.label(binary_mask, structure=structure)
    return num_features


def _overlap_metrics(tp: float, fp: float, fn: float, tn: float):
    """
    Returns (dice, jaccard, overlap_gt, overlap_pred).

    overlap_gt   = TP / (TP + FN)  [= sensitivity / recall]
    overlap_pred = TP / (TP + FP)  [= precision / PPV]
    jaccard      = TP / (TP + FP + FN)  [= IoU]
    """
    dice         = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else None
    jaccard      = tp / (tp + fp + fn)              if (tp + fp + fn) > 0        else None
    overlap_gt   = tp / (tp + fn)                   if (tp + fn) > 0             else None
    overlap_pred = tp / (tp + fp)                   if (tp + fp) > 0             else None
    return dice, jaccard, overlap_gt, overlap_pred


def _vol_sim(va: float, vb: float) -> float | None:
    """
    Volume Similarity = 1 - |Va - Vb| / (Va + Vb).
    Returns None when both volumes are zero (undefined).
    Ranges 0 (completely different volume) to 1 (identical volume).
    """
    return 1.0 - abs(va - vb) / (va + vb) if (va + vb) > 0 else None


def _hausdorff_mhd(pred_np: np.ndarray, gt_np: np.ndarray,
                   cls: int, reference_sitk: sitk.Image):
    """
    Compute Hausdorff Distance and Mean Hausdorff Distance (ASSD) in mm
    for a single class, using SimpleITK's HausdorffDistanceImageFilter.

    Both `pred_np` and `gt_np` are expected in (Z, Y, X) array order and
    share the same grid as `reference_sitk`.

    Returns (hd_mm, mhd_mm) or (None, None) if either mask is empty.
    """
    pred_mask = (pred_np == cls).astype(np.uint8)
    gt_mask   = (gt_np   == cls).astype(np.uint8)

    if not pred_mask.any() or not gt_mask.any():
        return None, None

    pred_bin = sitk.GetImageFromArray(pred_mask)
    pred_bin.CopyInformation(reference_sitk)
    gt_bin = sitk.GetImageFromArray(gt_mask)
    gt_bin.CopyInformation(reference_sitk)

    try:
        hd_filter = sitk.HausdorffDistanceImageFilter()
        hd_filter.Execute(gt_bin, pred_bin)
        return hd_filter.GetHausdorffDistance(), hd_filter.GetAverageHausdorffDistance()
    except Exception as exc:
        print(f"   ⚠  Hausdorff computation failed for class {cls}: {exc}")
        return None, None


def _infer_gtv_flags(gt_np: np.ndarray, case_meta: dict) -> dict:
    """
    Return a (possibly updated) copy of case_meta with has_gtv_t / has_gtv_n
    set from the GT array when the JSON did not supply them (e.g. HECKTOR).
    """
    meta = dict(case_meta)
    for key, cls in (("has_gtv_t", 1), ("has_gtv_n", 2)):
        if meta.get(key) in (None, "", "None"):
            meta[key] = bool((gt_np == cls).any())
    return meta


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

    # Infer GTV flags from GT when not supplied by JSON (e.g. HECKTOR dataset)
    case_meta = _infer_gtv_flags(gt_np, case_meta)

    total_vox = float(pred_np.size)
    gt_available = case_meta.get("gt_available", True)

    row = {f: "" for f in CSV_FIELDS}
    row.update({
        "case_id":            case_meta.get("case_id",
                              os.path.basename(pred_path).replace("_Pred.nii.gz", "")),
        "patient":            case_meta.get("patient", ""),
        "timepoint":          case_meta.get("timepoint", ""),
        "study_date":         case_meta.get("study_date", ""),
        "gt_available":       gt_available,
        "gt_reason":          case_meta.get("gt_reason", "ok"),
        "has_gtv_t":          case_meta.get("has_gtv_t", ""),
        "has_gtv_n":          case_meta.get("has_gtv_n", ""),
        "gt_labels_present":  ", ".join(
            [n for c, (n, _) in _CLASS_INFO.items() if (gt_np == c).any()]) or "none",
        "pred_labels_present": ", ".join(
            [n for c, (n, _) in _CLASS_INFO.items() if (pred_np == c).any()]) or "none",
        "gt_vol_total_mm3":   round(float(np.sum(gt_np > 0)) * vox_mm3, 1),
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

        row[f"gt_vol_{cls_name}_mm3"]   = round(gt_vol, 1)
        row[f"pred_vol_{cls_name}_mm3"] = round(pred_vol, 1)
        row[f"vol_sim_{cls_name}"]      = _safe(_vol_sim(gt_vol, pred_vol))

        # Connected-object counting
        row[f"gt_count_{cls_name}"]   = count_objects(gt_np, cls)
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

        # ── Skip class if it was never annotated in GT ─────────────────
        if case_meta.get(has_key) is False:
            comments.append(f"no_{cls_name}_gt")
            if pred_vol > 0:
                comments.append(
                    f"pred_{cls_name}_FP_{round(pred_vol / 1000, 1)}cm3")
            continue

        # ── True-Negative case (no GT, no prediction) ─────────────────
        if not gt_mask.any() and not pred_mask.any():
            comments.append(f"{cls_name}_TN")
            continue

        if not gt_mask.any():
            comments.append(f"no_{cls_name}_gt_voxels")
        if not pred_mask.any():
            comments.append(f"pred_{cls_name}_empty")

        # ── Scalar overlap metrics ─────────────────────────────────────
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

        # ── Surface-distance metrics ───────────────────────────────────
        # gt_sitk is in (Z,Y,X) array order; CopyInformation inside helper
        hd_mm, mhd_mm = _hausdorff_mhd(pred_np, gt_np, cls, gt_sitk)
        row[f"{cls_name}_hausdorff_mm"] = _safe(hd_mm, 2)
        row[f"{cls_name}_mhd_mm"]       = _safe(mhd_mm, 2)

    row["mean_dice"] = _safe(float(np.mean(dice_vals))) if dice_vals else ""
    row["comments"]  = "; ".join(dict.fromkeys(comments))
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
        img_name         = os.path.basename(entry.get("image", ""))
        img_prefix_clean = img_name.split(".")[0].replace("_petct", "")
        gt_rel           = entry.get("label", "")
        gt_path          = os.path.join(args.data_dir, gt_rel) if gt_rel else None

        # Primary filename pattern
        pred_path_candidates = glob.glob(
            os.path.join(args.output_dir, f"{img_prefix_clean}_Pred.nii.gz"))
        # Fallback for older files that embedded DSC in the name
        if not pred_path_candidates:
            pred_path_candidates = glob.glob(
                os.path.join(args.output_dir,
                             f"{img_prefix_clean}_dsc*_Pred.nii.gz"))

        print(f"\n[{idx+1}/{len(all_entries)}] Evaluating: {img_prefix_clean}")

        if not pred_path_candidates:
            print("  ❌ Prediction not found on disk. Run test.py first!")
            continue

        pred_path = pred_path_candidates[0]

        if (not entry.get("gt_available", True)
                or not gt_path
                or not os.path.exists(gt_path)):
            print("  ℹ No GT available. Skipping metric calculation.")
            row = {f: "" for f in CSV_FIELDS}
            row.update({"case_id": img_prefix_clean,
                        "comments": "no_GT_annotation"})
        else:
            row = compute_rich_metrics(pred_path, gt_path, entry)
            print(
                f"  GTVp  Dice={row.get('GTVp_dice','NA')}  "
                f"Jac={row.get('GTVp_jaccard','NA')}  "
                f"HD={row.get('GTVp_hausdorff_mm','NA')}mm  "
                f"MHD={row.get('GTVp_mhd_mm','NA')}mm  "
                f"OvGT={row.get('GTVp_overlap_gt','NA')}  "
                f"OvPred={row.get('GTVp_overlap_pred','NA')}"
            )
            print(
                f"  GTVn  Dice={row.get('GTVn_dice','NA')}  "
                f"Jac={row.get('GTVn_jaccard','NA')}  "
                f"HD={row.get('GTVn_hausdorff_mm','NA')}mm  "
                f"MHD={row.get('GTVn_mhd_mm','NA')}mm  "
                f"OvGT={row.get('GTVn_overlap_gt','NA')}  "
                f"OvPred={row.get('GTVn_overlap_pred','NA')}"
            )
            print(f"  Mean Dice={row.get('mean_dice','NA')}  "
                  f"Obj(GT/Pred) T:[{row.get('gt_count_GTVp')}/{row.get('pred_count_GTVp')}]"
                  f" N:[{row.get('gt_count_GTVn')}/{row.get('pred_count_GTVn')}]")

        all_rows.append(row)

        with open(csv_path, "w" if first_write else "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if first_write:
                writer.writeheader()
            writer.writerow(row)
        first_write = False

    mean_dices = [float(r["mean_dice"]) for r in all_rows if r.get("mean_dice")]
    print("\n" + "=" * 60)
    if mean_dices:
        print(f"Overall Mean Dice : {np.mean(mean_dices):.4f}  "
              f"(n={len(mean_dices)})")
    else:
        print("No valid cases evaluated.")
    print(f"Rich CSV written to: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
