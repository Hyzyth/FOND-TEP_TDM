#!/usr/bin/env python3
"""
evaluate_predictions.py
========================
Enriched evaluation script for SwinCross predictions on HECKTOR or TemPoRAL.

For each case in the dataset JSON:
  1. If a prediction NIfTI already exists in --output_dir → load it from disk.
  2. If not → run inference (same pipeline as test.py), save, then load.

All metrics are computed from disk (original CT space), giving volumes in
physical mm³ regardless of any preprocessing resampling.

CSV columns
-----------
  case_id          — image prefix (e.g. CHUM-001, pat5_pre_20150312)
  timepoint        — from JSON metadata (TemPoRAL only; blank for HECKTOR)
  gt_available     — False when no annotation existed in source data
  gt_reason        — ok / n_only / t_only / no_rtstruct_dir / no_mask_in_rtstruct
  GTVp_dice        — Dice for class 1 (primary tumour), or blank
  GTVn_dice        — Dice for class 2 (lymph nodes), or blank
  mean_dice        — mean over scored classes only
  gt_vol_GTVp_mm3  — GT volume for class 1 in mm³
  gt_vol_GTVn_mm3  — GT volume for class 2 in mm³
  pred_vol_GTVp_mm3
  pred_vol_GTVn_mm3
  vol_sim_GTVp     — volume similarity for class 1 (0–1, 1=identical volume)
  vol_sim_GTVn
  pred_source      — "loaded_from_disk" or "newly_inferred"
  comments         — human-readable notes (no_GTVt_gt, pred_GTVp_empty, …)

Usage
-----
  # Re-evaluate existing predictions only (no inference):
  python3.12 evaluate_predictions.py \
      --pretrained_dir  ./runs/ethan_hecktor_2000ep_run \
      --pretrained_model_name model_best.pth \
      --output_dir      /data/ethan/SwinCross/.../temporal_zeroshot \
      --data_dir        /data/ethan/SwinCross/PP_temporal_dataset \
      --json_list       dataset_swincross_temporal.json

  # Same command works for HECKTOR — just point data_dir / json_list there.
  # Missing predictions will be inferred automatically.
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
import torch
from collections import OrderedDict
from functools import partial
from scipy import ndimage

from monai.data import MetaTensor, decollate_batch
from monai.inferers.utils import sliding_window_inference
from monai.transforms import Invertd, RemoveSmallObjects

from data_utils import get_loader
from networks.SwinTransModels import CONFIGS as CONFIGS_sw_seg
from networks.SwinTransModels import SwinUNETR_CrossModalityFusion_OutSum_6stageOuts

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
_SMALL_OBJ_THRESHOLD_MM3 = 125.0          # 0.125 cm³
_CLASS_INFO = {1: ("GTVp", "has_gtv_t"), 2: ("GTVn", "has_gtv_n")}

CSV_FIELDS = [
    "case_id", "timepoint", "gt_available", "gt_reason",
    "GTVp_dice",        "GTVn_dice",        "mean_dice",
    "gt_vol_GTVp_mm3",  "gt_vol_GTVn_mm3",
    "pred_vol_GTVp_mm3","pred_vol_GTVn_mm3",
    "vol_sim_GTVp",     "vol_sim_GTVn",
    "pred_source",      "comments",
]


# ── Metric helpers ─────────────────────────────────────────────────────────────

def _vol_similarity(va: float, vb: float) -> float | None:
    """
    Volume Similarity = 1 - |Va - Vb| / (Va + Vb).
    Returns None when both volumes are zero (undefined).
    Ranges 0 (completely different volume) to 1 (identical volume).
    """
    if va + vb == 0:
        return None
    return 1.0 - abs(va - vb) / (va + vb)


def _dice_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    denom = pred_mask.sum() + gt_mask.sum()
    if denom == 0:
        return 1.0          # both empty → perfect (shouldn't reach here)
    return float(2.0 * intersection / denom)


def compute_rich_metrics(pred_path: str,
                         gt_path: str,
                         case_meta: dict,
                         pred_source: str) -> dict:
    """
    Load prediction and GT from disk, compute all metrics.

    Both files are expected to be in original CT space (same grid).
    Volumes are computed from the GT image's physical spacing.

    Parameters
    ----------
    pred_path   : path to predicted segmentation NIfTI
    gt_path     : path to ground-truth NIfTI (may be all-zero for no-GT cases)
    case_meta   : dict from the dataset JSON entry for this case
    pred_source : "loaded_from_disk" | "newly_inferred"
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

    gt_available = case_meta.get("gt_available", True)
    gt_reason    = case_meta.get("gt_reason", "ok")

    dice_vals = []
    comments  = []
    row = {
        "case_id":      case_meta.get("case_id", os.path.basename(pred_path)),
        "timepoint":    case_meta.get("timepoint", ""),
        "gt_available": gt_available,
        "gt_reason":    gt_reason,
        "pred_source":  pred_source,
    }

    for cls, (name, has_key) in _CLASS_INFO.items():
        gt_mask   = gt_np   == cls
        pred_mask = pred_np == cls

        gt_vol   = float(gt_mask.sum())   * vox_mm3
        pred_vol = float(pred_mask.sum()) * vox_mm3

        row[f"gt_vol_{name}_mm3"]   = round(gt_vol,   1)
        row[f"pred_vol_{name}_mm3"] = round(pred_vol, 1)

        # ── Decide whether to score this class ─────────────────────────────
        # has_gtv_t / has_gtv_n flags come from the TemPoRAL JSON.
        # For HECKTOR (no such flag), None → score if GT or pred is non-empty.
        has_gtv_flag = case_meta.get(has_key, None)

        if not gt_available:
            # No annotation at all for this timepoint
            dice_val = vol_sim = None
            comments.append(f"no_GT_annotation")

        elif has_gtv_flag is False:
            # Annotation present but this specific structure was absent
            dice_val = vol_sim = None
            comments.append(f"no_{name}_gt")
            if pred_vol > 0:
                comments.append(f"pred_{name}_FP_{round(pred_vol/1000, 1)}cm3")

        elif not gt_mask.any() and not pred_mask.any():
            # Both empty — true negative, do not penalise
            dice_val = vol_sim = None
            comments.append(f"{name}_TN")

        else:
            dice_val = _dice_score(pred_mask, gt_mask)
            vol_sim  = _vol_similarity(gt_vol, pred_vol)
            dice_vals.append(dice_val)

            if not gt_mask.any():
                comments.append(f"no_{name}_gt_voxels")
            if not pred_mask.any():
                comments.append(f"pred_{name}_empty")

        row[f"{name}_dice"]   = round(dice_val, 4) if dice_val is not None else ""
        row[f"vol_sim_{name}"]= round(vol_sim,  4) if vol_sim  is not None else ""

    row["mean_dice"] = round(float(np.mean(dice_vals)), 4) if dice_vals else ""
    row["comments"]  = "; ".join(dict.fromkeys(comments))  # deduplicated, order-preserving
    return row


# ── Inference helpers ──────────────────────────────────────────────────────────

def _remove_small_objects_physical(pred_np, spacing_mm,
                                   threshold_mm3=_SMALL_OBJ_THRESHOLD_MM3,
                                   foreground_classes=(1, 2)):
    voxel_vol = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    min_size  = max(1, math.ceil(threshold_mm3 / voxel_vol))
    remover   = RemoveSmallObjects(min_size=min_size, connectivity=3)
    out = pred_np.copy()
    for cls in foreground_classes:
        binary_t = torch.from_numpy((pred_np == cls).astype(np.uint8)[None])
        filtered = remover(binary_t).numpy()[0]
        out[(out == cls) & (filtered == 0)] = 0
    removed = int((pred_np > 0).sum()) - int((out > 0).sum())
    print(f"   [RemoveSmallObjects] min={min_size} vox "
          f"({threshold_mm3:.0f}mm³) | removed: {removed} vox")
    return out


def run_inference_and_save(batch, model, val_loader, args,
                           test_output_dir: str) -> str:
    """
    Run sliding-window inference on one batch, apply RemoveSmallObjects,
    invert transforms, save to disk.  Returns the output file path.
    """
    device = next(model.parameters()).device
    val_inputs = batch["image"].to(device)

    if hasattr(val_inputs, 'meta') and 'filename_or_obj' in val_inputs.meta:
        img_name = val_inputs.meta['filename_or_obj'][0].split('/')[-1]
    elif 'image_meta_dict' in batch:
        img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
    else:
        img_name = "unknown.nii.gz"

    img_prefix       = img_name.split('.')[0]
    img_prefix_clean = img_prefix.replace("_petct", "")

    print(f"  → Running inference on {img_name} …")
    with torch.no_grad():
        val_outputs = sliding_window_inference(
            val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap
        )

    val_outputs_tensor = torch.softmax(val_outputs, 1)
    val_outputs_tensor = torch.argmax(val_outputs_tensor, 1, keepdim=True).cpu()

    if hasattr(val_inputs, "meta"):
        val_outputs_tensor = MetaTensor(val_outputs_tensor, meta=val_inputs.meta)

    batch["pred"] = val_outputs_tensor
    invertd = Invertd(
        keys="pred",
        transform=val_loader.dataset.transform,
        orig_keys="image",
        nearest_interp=True,
        to_tensor=True,
    )
    batch_inv = [invertd(item) for item in decollate_batch(batch)]

    saved_path = None
    for item_inv in batch_inv:
        pred_np = item_inv["pred"][0].cpu().numpy().astype(np.uint8)

        img_t = item_inv["image"]
        if hasattr(img_t, "meta") and "filename_or_obj" in img_t.meta:
            src_path = img_t.meta["filename_or_obj"]
        elif "image_meta_dict" in item_inv:
            src_path = item_inv["image_meta_dict"]["filename_or_obj"]
        else:
            raise ValueError("Cannot find source image path in batch metadata.")

        if isinstance(src_path, (list, tuple, np.ndarray)):
            src_path = src_path[0]
        if isinstance(src_path, torch.Tensor) or hasattr(src_path, "item"):
            src_path = str(src_path)

        orig_sitk = sitk.ReadImage(src_path)
        if orig_sitk.GetDimension() == 4:
            sz = list(orig_sitk.GetSize())
            orig_sitk = sitk.Extract(orig_sitk, [sz[0], sz[1], sz[2], 0], [0,0,0,0])

        spacing_mm = orig_sitk.GetSpacing()
        pred_np    = _remove_small_objects_physical(pred_np, spacing_mm)

        pred_sitk = sitk.GetImageFromArray(pred_np.transpose(2, 1, 0))
        pred_sitk.CopyInformation(orig_sitk)

        saved_path = os.path.join(
            test_output_dir, f"{img_prefix_clean}_dscNA_Pred.nii.gz"
        )
        sitk.WriteImage(pred_sitk, saved_path)
        print(f"  ✅ Saved: {saved_path}")

    return saved_path


def find_existing_prediction(output_dir: str, img_prefix_clean: str):
    """
    Return path to the first matching *_Pred.nii.gz for this case, or None.
    """
    matches = glob.glob(
        os.path.join(output_dir, f"{img_prefix_clean}_dsc*_Pred.nii.gz")
    )
    return matches[0] if matches else None


# ── CSV writer ─────────────────────────────────────────────────────────────────

def write_csv(csv_path: str, rows: list, mode: str = "a"):
    write_header = (mode == "w") or not os.path.exists(csv_path)
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ── Argument parser ────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Enriched evaluation for SwinCross predictions")
    p.add_argument("--pretrained_dir",        default="./runs/for_log/",    type=str)
    p.add_argument("--pretrained_model_name", default="model_best.pth",     type=str)
    p.add_argument("--saved_checkpoint",      default="ckpt",               type=str)
    p.add_argument("--output_dir",            default=None,                  type=str,
                   help="Folder containing (or to receive) *_Pred.nii.gz files. "
                        "Also where per_case_dice_enriched.csv is written.")
    p.add_argument("--data_dir",  default="Dataset_Final_SwinCross_SITK",   type=str)
    p.add_argument("--json_list", default="dataset_swincross.json",          type=str)
    p.add_argument("--infer_overlap",  default=0.7,  type=float)
    p.add_argument("--in_channels",    default=2,    type=int)
    p.add_argument("--out_channels",   default=3,    type=int)
    p.add_argument("--space_x",        default=1.0,  type=float)
    p.add_argument("--space_y",        default=1.0,  type=float)
    p.add_argument("--space_z",        default=1.0,  type=float)
    p.add_argument("--roi_x",          default=96,   type=int)
    p.add_argument("--roi_y",          default=96,   type=int)
    p.add_argument("--roi_z",          default=96,   type=int)
    p.add_argument("--workers",        default=4,    type=int)
    p.add_argument("--RandFlipd_prob",            default=0.2, type=float)
    p.add_argument("--RandRotate90d_prob",         default=0.2, type=float)
    p.add_argument("--RandScaleIntensityd_prob",   default=0.1, type=float)
    p.add_argument("--RandShiftIntensityd_prob",   default=0.1, type=float)
    p.add_argument("--distributed", action="store_true")
    p.add_argument("--inference_only", action="store_true",
                   help="Skip evaluation (write all rows as no_gt_available).")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parser.parse_args()
    args.test_mode     = True
    args.inference_only = False   # always False here; we drive evaluation ourselves

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Output directory ───────────────────────────────────────────────────────
    test_output_dir = args.output_dir or os.path.join(args.pretrained_dir, "test")
    os.makedirs(test_output_dir, exist_ok=True)

    csv_path = os.path.join(test_output_dir, "per_case_dice_enriched.csv")

    # ── Load dataset JSON (for metadata lookup) ────────────────────────────────
    json_path = os.path.join(args.data_dir, args.json_list)
    meta_lookup = {}      # basename(image) → JSON entry
    try:
        with open(json_path) as f:
            dataset_json = json.load(f)
        for split in ("training", "validation", "testing"):
            for entry in dataset_json.get(split, []):
                key = os.path.basename(entry.get("image", ""))
                meta_lookup[key] = entry
        print(f"Loaded metadata for {len(meta_lookup)} cases from {args.json_list}")
    except Exception as e:
        print(f"⚠  Could not load JSON from {json_path}: {e}")

    # ── Check which cases still need inference ─────────────────────────────────
    # We need the model only if at least one case is missing a prediction.
    val_loader = get_loader(args)

    # Pre-scan: collect (img_name, img_prefix_clean, case_meta, gt_path, pred_exists)
    need_inference = False
    case_list = []
    for batch in val_loader:
        val_inputs = batch["image"]
        if hasattr(val_inputs, "meta") and "filename_or_obj" in val_inputs.meta:
            img_name = val_inputs.meta["filename_or_obj"][0].split("/")[-1]
        elif "image_meta_dict" in batch:
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
        else:
            img_name = "unknown.nii.gz"

        img_prefix_clean = img_name.split(".")[0].replace("_petct", "")
        case_meta        = meta_lookup.get(img_name, {})
        gt_rel           = case_meta.get("label", "")
        gt_path          = os.path.join(args.data_dir, gt_rel) if gt_rel else None
        pred_path        = find_existing_prediction(test_output_dir, img_prefix_clean)

        case_list.append({
            "batch":             batch,
            "img_name":          img_name,
            "img_prefix_clean":  img_prefix_clean,
            "case_meta":         case_meta,
            "gt_path":           gt_path,
            "pred_path":         pred_path,
        })
        if pred_path is None:
            need_inference = True

    # ── Load model only when inference is actually needed ──────────────────────
    model = None
    if need_inference:
        print("\nSome cases are missing predictions — loading model for inference …")
        pretrained_pth = os.path.join(args.pretrained_dir, args.pretrained_model_name)
        config_sw = CONFIGS_sw_seg["SwinUNETR_CMFF-hecktor-v06"]
        model = SwinUNETR_CrossModalityFusion_OutSum_6stageOuts(config_sw)

        checkpoint = torch.load(pretrained_pth, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint

        new_state = OrderedDict(
            (k.replace("backbone.", ""), v) for k, v in state.items()
        )
        model.load_state_dict(new_state, strict=False)
        model.eval()
        model.to(device)
        print("Model loaded.\n")
    else:
        print("\nAll predictions found on disk — no inference needed.\n")

    # ── Main evaluation loop ───────────────────────────────────────────────────
    first_write = True
    overall_dice = []

    for idx, case in enumerate(case_list):
        batch            = case["batch"]
        img_name         = case["img_name"]
        img_prefix_clean = case["img_prefix_clean"]
        case_meta        = case["case_meta"]
        gt_path          = case["gt_path"]
        pred_path        = case["pred_path"]

        print(f"\n[{idx+1}/{len(case_list)}] {img_prefix_clean}")

        # ── Step 1: Ensure prediction exists ──────────────────────────────────
        if pred_path is not None:
            pred_source = "loaded_from_disk"
            print(f"  Prediction found: {os.path.basename(pred_path)}")
        else:
            pred_source = "newly_inferred"
            pred_path = run_inference_and_save(
                batch, model, val_loader, args, test_output_dir
            )
            if pred_path is None:
                print(f"  ❌ Inference failed for {img_prefix_clean}, skipping.")
                continue

        # ── Step 2: Compute rich metrics from disk ─────────────────────────────
        gt_available = case_meta.get("gt_available", True)

        if not gt_available or gt_path is None or not os.path.exists(gt_path):
            # No GT → write a no-evaluation row
            reason = case_meta.get("gt_reason", "no_gt_path")
            row = {f: "" for f in CSV_FIELDS}
            row.update({
                "case_id":      case_meta.get("case_id", img_prefix_clean),
                "timepoint":    case_meta.get("timepoint", ""),
                "gt_available": False,
                "gt_reason":    reason,
                "pred_source":  pred_source,
                "comments":     f"evaluation_skipped:{reason}",
            })
            print(f"  ℹ  GT not available ({reason}) — skipping evaluation.")
        else:
            try:
                row = compute_rich_metrics(pred_path, gt_path, case_meta, pred_source)
            except Exception as e:
                print(f"  ❌ Metric computation failed: {e}")
                row = {f: "" for f in CSV_FIELDS}
                row.update({
                    "case_id":   img_prefix_clean,
                    "comments":  f"metric_error:{e}",
                    "pred_source": pred_source,
                })

            # Accumulate mean dice for overall summary
            if row.get("mean_dice") not in ("", None):
                overall_dice.append(float(row["mean_dice"]))

            print(
                f"  GTVp Dice={row.get('GTVp_dice', 'NA'):>6}  "
                f"GTVn Dice={row.get('GTVn_dice', 'NA'):>6}  "
                f"Mean={row.get('mean_dice', 'NA'):>6}  "
                f"gtVol T={row.get('gt_vol_GTVp_mm3', 'NA')}mm³  "
                f"N={row.get('gt_vol_GTVn_mm3', 'NA')}mm³  "
                f"predVol T={row.get('pred_vol_GTVp_mm3', 'NA')}mm³  "
                f"N={row.get('pred_vol_GTVn_mm3', 'NA')}mm³"
            )
            if row.get("comments"):
                print(f"  Comments: {row['comments']}")

        write_csv(csv_path, [row], mode="w" if first_write else "a")
        first_write = False

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if overall_dice:
        print(f"Overall Mean Dice (evaluated cases): {np.mean(overall_dice):.4f}  "
              f"(n={len(overall_dice)})")
    else:
        print("Overall Mean Dice: N/A (no evaluable cases)")
    print(f"Enriched CSV written to: {csv_path}")
    print("=" * 60)


parser = build_parser()

if __name__ == "__main__":
    main()
