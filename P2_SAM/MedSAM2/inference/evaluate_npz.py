#!/usr/bin/env python3
"""
evaluate_npz.py
================
Universal rich evaluation for MedSAM2 NPZ predictions.

Computes the same metrics as SwinCross's evaluate_predictions.py:
  Dice, Jaccard, Hausdorff (mm), MHD/ASSD (mm),
  Overlap on GT (recall), Overlap on Pred (precision),
  volumes (mm³), object counts, volume similarity,
  confusion-matrix volumes (TP/FP/FN/TN in mm³).

Inputs:
  --pred_dir  Directory of predicted NPZ files (key: 'segs').
  --gt_dir    Directory of ground-truth NPZ files (key: 'gts', 'spacing').
  --manifest  Optional path to manifest.json produced by prepare_temporal_npz.py
              or prepare_hecktor_npz.py.  When absent, metadata is inferred
              from GT content (compatible with HECKTOR NPZ dataset).
  --output_dir Directory to write per_case_evaluation_rich.csv.

Usage:
  python inference/evaluate_npz.py \\
      --pred_dir /data/ethan/MedSAM2/predictions/gt_val \\
      --gt_dir   /data/ethan/MedSAM2/hecktor_npz/val \\
      --output_dir /data/ethan/MedSAM2/predictions/gt_val

  python inference/evaluate_npz.py \\
      --pred_dir  /data/ethan/MedSAM2/predictions/temporal_zeroshot \\
      --gt_dir    /data/ethan/MedSAM2/temporal_npz \\
      --manifest  /data/ethan/MedSAM2/temporal_npz/manifest.json \\
      --output_dir /data/ethan/MedSAM2/predictions/temporal_zeroshot
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

# ── Constants (identical to SwinCross evaluate_predictions.py) ────────────────
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


# ── Metric helpers ────────────────────────────────────────────────────────────

def _safe(value, decimals=4):
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


def _overlap_metrics(tp, fp, fn, tn):
    dice         = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else None
    jaccard      = tp / (tp + fp + fn)              if (tp + fp + fn) > 0        else None
    overlap_gt   = tp / (tp + fn)                   if (tp + fn) > 0             else None
    overlap_pred = tp / (tp + fp)                   if (tp + fp) > 0             else None
    return dice, jaccard, overlap_gt, overlap_pred


def _vol_sim(va, vb):
    return 1.0 - abs(va - vb) / (va + vb) if (va + vb) > 0 else None


def _hausdorff_mhd(pred_np, gt_np, cls, spacing_zyx):
    """Compute Hausdorff and MHD in mm. spacing_zyx = (sz, sy, sx) in mm."""
    pred_mask = (pred_np == cls).astype(np.uint8)
    gt_mask   = (gt_np   == cls).astype(np.uint8)
    if not pred_mask.any() or not gt_mask.any():
        return None, None

    # Build SimpleITK images with physical spacing
    # sitk stores spacing as (x, y, z); our arrays are (z, y, x)
    sitk_spacing = (float(spacing_zyx[2]), float(spacing_zyx[1]), float(spacing_zyx[0]))

    pred_bin = sitk.GetImageFromArray(pred_mask)
    pred_bin.SetSpacing(sitk_spacing)
    gt_bin = sitk.GetImageFromArray(gt_mask)
    gt_bin.SetSpacing(sitk_spacing)

    try:
        hd_filter = sitk.HausdorffDistanceImageFilter()
        hd_filter.Execute(gt_bin, pred_bin)
        return hd_filter.GetHausdorffDistance(), hd_filter.GetAverageHausdorffDistance()
    except Exception as exc:
        print(f"   ⚠  Hausdorff failed for class {cls}: {exc}")
        return None, None


def _infer_gtv_flags(gt_np, case_meta):
    meta = dict(case_meta)
    for key, cls in (("has_gtv_t", 1), ("has_gtv_n", 2)):
        if meta.get(key) in (None, "", "None"):
            meta[key] = bool((gt_np == cls).any())
    return meta


def compute_rich_metrics(pred_np, gt_np, spacing_zyx, case_meta):
    """
    Compute all metrics for one case.

    Parameters
    ----------
    pred_np     : (D, H, W) uint8  predicted labels {0,1,2}
    gt_np       : (D, H, W) uint8  ground-truth labels {0,1,2}
    spacing_zyx : (3,) float  voxel size in mm (z, y, x)
    case_meta   : dict  metadata from manifest (or inferred)
    """
    vox_mm3   = float(spacing_zyx[0]) * float(spacing_zyx[1]) * float(spacing_zyx[2])
    case_meta = _infer_gtv_flags(gt_np, case_meta)
    total_vox = float(pred_np.size)
    gt_available = case_meta.get("gt_available", True)

    row = {f: "" for f in CSV_FIELDS}
    row.update({
        "case_id":            case_meta.get("case_id", ""),
        "patient":            case_meta.get("patient", ""),
        "timepoint":          case_meta.get("timepoint", ""),
        "study_date":         case_meta.get("study_date", ""),
        "gt_available":       gt_available,
        "gt_reason":          case_meta.get("gt_reason", "ok"),
        "has_gtv_t":          case_meta.get("has_gtv_t", ""),
        "has_gtv_n":          case_meta.get("has_gtv_n", ""),
        "gt_labels_present":  ", ".join(n for c, (n, _) in _CLASS_INFO.items()
                                         if (gt_np == c).any()) or "none",
        "pred_labels_present": ", ".join(n for c, (n, _) in _CLASS_INFO.items()
                                          if (pred_np == c).any()) or "none",
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

        hd_mm, mhd_mm = _hausdorff_mhd(pred_np, gt_np, cls, spacing_zyx)
        row[f"{cls_name}_hausdorff_mm"] = _safe(hd_mm, 2)
        row[f"{cls_name}_mhd_mm"]       = _safe(mhd_mm, 2)

    row["mean_dice"] = _safe(float(np.mean(dice_vals))) if dice_vals else ""
    row["comments"]  = "; ".join(dict.fromkeys(comments))
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Universal rich evaluation for MedSAM2 NPZ predictions")
    parser.add_argument("--pred_dir",   required=True, help="Directory with predicted NPZ files (key: 'segs')")
    parser.add_argument("--gt_dir",     required=True, help="Directory with GT NPZ files (key: 'gts', 'spacing')")
    parser.add_argument("--manifest",   default=None,  help="Path to manifest.json (optional; inferred when absent)")
    parser.add_argument("--output_dir", required=True, help="Directory to write per_case_evaluation_rich.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = join(args.output_dir, "per_case_evaluation_rich.csv")

    # ── Load manifest ──────────────────────────────────────────────────────────
    manifest_path = args.manifest or join(args.gt_dir, "manifest.json")
    meta_by_id = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        for entry in manifest.get("cases", []):
            meta_by_id[entry["case_id"]] = entry
        print(f"Loaded manifest: {len(meta_by_id)} cases from {manifest_path}")
    else:
        print("No manifest found — metadata will be inferred from GT content.")

    # ── Discover predictions ───────────────────────────────────────────────────
    pred_files = sorted(glob(join(args.pred_dir, "*.npz")))
    if not pred_files:
        raise FileNotFoundError(f"No prediction NPZ files in {args.pred_dir}")

    all_rows  = []
    first_write = True

    for idx, pred_path in enumerate(pred_files):
        case_id  = basename(pred_path).replace(".npz", "")
        gt_path  = join(args.gt_dir, basename(pred_path))
        print(f"\n[{idx+1}/{len(pred_files)}] Evaluating: {case_id}")

        if not os.path.exists(gt_path):
            print("  ❌ GT file not found — skipping.")
            continue

        pred_data = np.load(pred_path, allow_pickle=True)
        gt_data   = np.load(gt_path,   allow_pickle=True)
        pred_np   = pred_data["segs"].astype(np.uint8)
        gt_np     = gt_data["gts"].astype(np.uint8)
        spacing   = gt_data["spacing"]   # (z, y, x) mm

        # ── Build case metadata ────────────────────────────────────────────────
        if case_id in meta_by_id:
            case_meta = dict(meta_by_id[case_id])
        else:
            # No manifest — infer flags from GT content
            case_meta = {
                "case_id":      case_id,
                "patient":      case_id,
                "timepoint":    "",
                "study_date":   "",
                "gt_available": True,
                "gt_reason":    "ok",
                "has_gtv_t":    None,   # will be inferred inside compute_rich_metrics
                "has_gtv_n":    None,
            }

        # ── Handle shape mismatch ──────────────────────────────────────────────
        if pred_np.shape != gt_np.shape:
            print(f"  ⚠  Shape mismatch pred={pred_np.shape} gt={gt_np.shape} — resampling pred.")
            # Use SimpleITK to resample
            pred_sitk = sitk.GetImageFromArray(pred_np)
            gt_sitk   = sitk.GetImageFromArray(gt_np)
            sp = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
            pred_sitk.SetSpacing(sp); gt_sitk.SetSpacing(sp)
            r = sitk.ResampleImageFilter()
            r.SetReferenceImage(gt_sitk)
            r.SetInterpolator(sitk.sitkNearestNeighbor)
            r.SetTransform(sitk.Transform())
            r.SetOutputPixelType(sitk.sitkUInt8)
            pred_np = sitk.GetArrayFromImage(r.Execute(pred_sitk)).astype(np.uint8)

        # ── Skip no-GT cases ───────────────────────────────────────────────────
        if not case_meta.get("gt_available", True):
            print("  ℹ No GT available — skipping metric calculation.")
            row = {f: "" for f in CSV_FIELDS}
            row.update({"case_id": case_id, "comments": "no_GT_annotation"})
        else:
            row = compute_rich_metrics(pred_np, gt_np, spacing, case_meta)
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
                  f"Obj(GT/Pred) T:[{row.get('gt_count_GTVp')}/{row.get('pred_count_GTVp')}] "
                  f"N:[{row.get('gt_count_GTVn')}/{row.get('pred_count_GTVn')}]")

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
        print(f"Overall Mean Dice : {np.mean(mean_dices):.4f}  (n={len(mean_dices)})")
    else:
        print("No valid cases evaluated.")
    print(f"Rich CSV written to: {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
