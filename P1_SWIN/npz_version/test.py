# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# (license header preserved)

"""
test.py  —  SwinCross inference on NPZ-preprocessed data
==========================================================

Entry point: iterate the JSON list, load each NPZ, run sliding-window
inference, invert to original space, save <case_id>_Pred.nii.gz.
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
import warnings
import csv
from scipy import ndimage

import numpy as np
import SimpleITK as sitk
import torch
from monai.inferers.utils import sliding_window_inference
from monai.transforms import RemoveSmallObjects
from skimage.segmentation import clear_border

# Safeguard
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from networks.SwinTransModels import CONFIGS as CONFIGS_sw_seg
from networks.SwinTransModels import SwinUNETR_CrossModalityFusion_OutSum_6stageOuts

warnings.filterwarnings("ignore")

# ── Physical post-processing threshold ───────────────────────────────────────
class_thresholds = {
            1: 100.0,  # GTVp (Tumors)
            2: 50.0    # GTVn (Nodules)
        }  # These thresholds are in mm³ and will be converted to voxel counts dynamically per case


# ── Post-processing (unchanged from original) ─────────────────────────────────

def _remove_border_objects_tracked(pred: np.ndarray, spacing_mm: tuple) -> tuple:
    """
    Removes objects touching the volume border and tracks the volume and count removed per class.
    """
    out = pred.copy() # Safe initialization: preserves background and un-targeted classes
    border_vols = {1: 0.0, 2: 0.0}
    border_counts = {1: 0, 2: 0}
    
    struct = ndimage.generate_binary_structure(3, 3)
    vox_vol = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]

    for cls in (1, 2):
        if not (pred == cls).any():
            continue
            
        mask = (pred == cls)
        cleared = clear_border(mask)

        # Find exactly what was deleted
        deleted_mask = mask & (~cleared)
        
        # Zero out the deleted border objects in the output array
        out[deleted_mask] = 0
        
        border_vols[cls] = float(deleted_mask.sum() * vox_vol)
        _, count = ndimage.label(deleted_mask, structure=struct)
        border_counts[cls] = count

    return out, border_vols[1], border_vols[2], border_counts[1], border_counts[2]


def _remove_small_objects_tracked(pred: np.ndarray, spacing_mm: tuple, thresholds_mm3: dict) -> tuple:
    """
    Removes small connected components using class-specific volume thresholds (in mm³),
    and tracks the volume removed per class and count of objects removed per class.
    
    thresholds_mm3 expected format: {1: 100.0, 2: 50.0}
    """
    vox_vol = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    out = pred.copy()
    removed_vols = {1: 0.0, 2: 0.0}
    removed_counts = {1: 0, 2: 0}

    # 3D connectivity structure for counting objects
    struct = ndimage.generate_binary_structure(3, 3)

    for cls, thresh_mm3 in thresholds_mm3.items():
        if not (pred == cls).any():
            continue
            
        # Calculate minimum voxels dynamically for this specific class
        min_vox = max(1, math.ceil(thresh_mm3 / vox_vol))
        remover = RemoveSmallObjects(min_size=min_vox, connectivity=3)
        
        binary = torch.from_numpy((pred == cls).astype(np.uint8)[None])
        filt = remover(binary).numpy()[0]
        
        # Identify exactly what was deleted
        deleted_mask = (out == cls) & (filt == 0)
        removed_vols[cls] = float(deleted_mask.sum() * vox_vol)

        # Count connected components removed
        _, num_removed = ndimage.label(deleted_mask, structure=struct)
        removed_counts[cls] = num_removed
        
        # Apply the deletion to the output mask
        out[deleted_mask] = 0

    return (
        out, 
        removed_vols.get(1, 0.0), removed_vols.get(2, 0.0), 
        removed_counts.get(1, 0), removed_counts.get(2, 0)
    )


# ── Inverse transform: NPZ crop-space → original CT space ────────────────────

def inverse_transform_to_original_space(pred_crop: np.ndarray,
                                         npz_data, ras_spacing) -> sitk.Image:
    """
    Undo the offline preprocessing to bring the prediction back into the
    original CT physical space, ready for evaluate_predictions.py.

    Steps:
      1. Uncrop: pad prediction into the full 1mm RAS volume.
      2. Wrap in a SimpleITK image with the stored RAS metadata.
      3. Resample (nearest-neighbour) to the original CT grid.

    Coordinates
    -----------
    The prediction arrives in MONAI spatial convention (R, A, S) = (x, y, z).
    SimpleITK expects arrays in (z, y, x) = (S, A, R) order.
    The stored ras_size_itk is in ITK (x, y, z) = (R, A, S) convention.
    """
    # 1. Uncrop into the full RAS volume
    ras_size  = [int(x) for x in npz_data["ras_size_itk"]]   # (nx, ny, nz) = (R, A, S)
    full_pred = np.zeros((ras_size[0], ras_size[1], ras_size[2]), dtype=np.uint8)
    cs, ce    = npz_data["crop_start"], npz_data["crop_end"]
    full_pred[cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]] = pred_crop

    # 2. Convert to ITK array order (S, A, R) and build SimpleITK image
    pred_itk_arr = full_pred.transpose(2, 1, 0).astype(np.uint8)   # (nz, ny, nx)
    pred_sitk    = sitk.GetImageFromArray(pred_itk_arr)

    pred_sitk.SetSpacing(ras_spacing)

    pred_sitk.SetOrigin([float(x) for x in npz_data["ras_origin"]])
    pred_sitk.SetDirection([float(x) for x in npz_data["ras_direction"].flatten()])

    # 3. Resample to the original CT space
    orig_size = [int(x) for x in npz_data["orig_size_itk"]]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([float(x) for x in npz_data["orig_spacing"]])
    r.SetOutputOrigin([float(x) for x in npz_data["orig_origin"]])
    r.SetOutputDirection([float(x) for x in npz_data["orig_direction"].flatten()])

    r.SetSize          (orig_size)
    r.SetInterpolator  (sitk.sitkNearestNeighbor)
    r.SetTransform     (sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetOutputPixelType(sitk.sitkUInt8)
    return r.Execute(pred_sitk)

def get_ras_spacing(npz_data) -> tuple:
    dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
    dummy.SetSpacing([float(x) for x in npz_data["orig_spacing"]])
    dummy.SetDirection([float(x) for x in npz_data["orig_direction"].flatten()])
    return sitk.DICOMOrient(dummy, "RAS").GetSpacing()

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="SwinCross NPZ inference pipeline")
parser.add_argument("--pretrained_dir",        default="./runs/for_log/",       type=str)
parser.add_argument("--pretrained_model_name", default="model_best.pth",         type=str)
parser.add_argument("--output_dir",            default=None,                     type=str)
parser.add_argument("--data_dir",              default=".",                      type=str)
parser.add_argument("--json_list",             default="dataset_swincross.json", type=str)
parser.add_argument("--saved_checkpoint",      default="ckpt",                   type=str,
                    help="'ckpt' or 'torchscript'")
parser.add_argument("--infer_overlap",  default=0.5, type=float,
                    help="Sliding-window overlap ratio. 0.5 is a good balance "
                         "of speed vs. quality; use 0.7 for best quality.")
parser.add_argument("--in_channels",   default=2,   type=int)
parser.add_argument("--out_channels",  default=3,   type=int)
parser.add_argument("--roi_x",         default=96,  type=int)
parser.add_argument("--roi_y",         default=96,  type=int)
parser.add_argument("--roi_z",         default=96,  type=int)
parser.add_argument("--workers",       default=2,   type=int)
parser.add_argument("--sw_batch_size", default=4,   type=int,
                    help="Number of sliding-window patches processed simultaneously.")
parser.add_argument("--distributed",   action="store_true")
parser.add_argument("--skip_existing", action="store_true",
                    help="Skip cases whose prediction already exists on disk.")
parser.add_argument("--compile",       action="store_true",
                    help="Wrap model with torch.compile (reduce-overhead mode) "
                         "for faster repeated inference. Requires PyTorch >= 2.0.")
# Legacy args kept for CLI compatibility (not used with NPZ pipeline)
parser.add_argument("--space_x", default=1.0, type=float)
parser.add_argument("--space_y", default=1.0, type=float)
parser.add_argument("--space_z", default=1.0, type=float)
parser.add_argument("--RandFlipd_prob",            default=0.2, type=float)
parser.add_argument("--RandRotate90d_prob",         default=0.2, type=float)
parser.add_argument("--RandScaleIntensityd_prob",   default=0.1, type=float)
parser.add_argument("--RandShiftIntensityd_prob",   default=0.1, type=float)
parser.add_argument("--dropout_rate",  default=0.0, type=float)
parser.add_argument("--feature_size",  default=36,  type=int)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_output_dir = (args.output_dir if args.output_dir
                       else os.path.join(args.pretrained_dir, "test"))
    os.makedirs(test_output_dir, exist_ok=True)

    # ── Load JSON case list ───────────────────────────────────────────────
    json_path = os.path.join(args.data_dir, args.json_list)
    with open(json_path) as f:
        dataset_json = json.load(f)

    all_entries = (dataset_json.get("validation", [])
                   + dataset_json.get("testing",    [])
                   + dataset_json.get("training",   []))

    if not all_entries:
        print("No entries found in JSON. Exiting.")
        return

    roi_size = (args.roi_x, args.roi_y, args.roi_z)
    model    = None   # lazy-loaded on first case that actually needs inference

    # ── CSV logging for post-processing volumes removed ───────────────────
    csv_log_path = os.path.join(args.output_dir, "postprocessing_logs.csv")
    csv_headers = [
        "case_id", "patient", "timepoint", "study_date", 
        "border_removed_GTVp_mm3", "border_removed_GTVn_mm3",
        "small_obj_removed_GTVp_mm3", "small_obj_removed_GTVn_mm3",
        "border_removed_GTVp_count", "border_removed_GTVn_count",
        "small_obj_removed_GTVp_count", "small_obj_removed_GTVn_count",
        "total_removed_GTVp_count", "total_removed_GTVn_count"
    ]
    with open(csv_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
    
    # ── Inference loop ────────────────────────────────────────────────────
    for idx, entry in enumerate(all_entries):
        torch.cuda.empty_cache()
        gc.collect()

        # ── Resolve paths ─────────────────────────────────────────────────
        npz_rel  = entry.get("npz", "")
        case_id  = entry.get("case_id") or os.path.basename(npz_rel).replace(".npz", "")
        npz_path = os.path.join(args.data_dir, npz_rel)

        output_path = os.path.join(test_output_dir, f"{case_id}_Pred.nii.gz")

        if args.skip_existing and os.path.exists(output_path):
            print(f">>  [{idx+1}/{len(all_entries)}] Skipping {case_id} (already exists)")
            continue

        if not os.path.exists(npz_path):
            print(f" [{idx+1}/{len(all_entries)}] NPZ not found: {npz_path}")
            continue

        # ── Load NPZ ──────────────────────────────────────────────────────
        print(f"→ [{idx+1}/{len(all_entries)}] {case_id}")
        npz_data = np.load(npz_path, allow_pickle=False)
        
        # Extract, cast to float32, and fuse (Channel 0 = PET, Channel 1 = CT)
        pet_arr = npz_data["pet"].astype(np.float32)
        ct_arr = npz_data["ct"].astype(np.float32)
        image_np = np.stack([pet_arr, ct_arr], axis=0)

        # ── Lazy model load ───────────────────────────────────────────────
        if model is None:
            print("  Loading model weights...")
            pretrained_pth = os.path.join(args.pretrained_dir,
                                          args.pretrained_model_name)
            if args.saved_checkpoint == "torchscript":
                model = torch.jit.load(pretrained_pth)
            else:
                config_sw = CONFIGS_sw_seg["SwinUNETR_CMFF-hecktor-v06"]
                model     = SwinUNETR_CrossModalityFusion_OutSum_6stageOuts(config_sw)
                ckpt      = torch.load(pretrained_pth, map_location="cpu")
                state     = (ckpt["state_dict"]
                             if isinstance(ckpt, dict) and "state_dict" in ckpt
                             else ckpt)
                model.load_state_dict(state)
            model.eval()
            model.to(device)
            if args.compile:
                # torch.compile with reduce-overhead: best for repeated identical
                # patch shapes (sliding window uses fixed roi_size patches).
                print("  Compiling model with torch.compile (reduce-overhead)...")
                model = torch.compile(model, mode="reduce-overhead")

        # ── Sliding-window inference — always FP16, independent of --noamp ──
        val_inputs = torch.from_numpy(image_np).unsqueeze(0).to(device)  # (1, 2, R, A, S)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=torch.cuda.is_available()):
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, args.sw_batch_size,
                    model, overlap=args.infer_overlap,
                )

        # ── Argmax → (R, A, S) uint8 ─────────────────────────────────────
        pred_np = (torch.argmax(val_outputs, dim=1)
                   .squeeze(0).cpu().numpy().astype(np.uint8))

        del val_inputs, val_outputs

        # --- CALCULATE TRUE NATIVE RAS SPACING ---
        ras_spacing = get_ras_spacing(npz_data)

        # ── Post-processing ───────────────────────────────────────────────
        # 1. Remove border objects and unpack tracked metrics
        (pred_np, 
         border_vol_1, border_vol_2, 
         border_count_1, border_count_2) = _remove_border_objects_tracked(pred_np, spacing_mm=ras_spacing)
        
        # 2. Remove small objects and unpack tracked metrics
        (pred_np, 
         small_vol_1, small_vol_2, 
         small_count_1, small_count_2) = _remove_small_objects_tracked(pred_np, spacing_mm=ras_spacing, thresholds_mm3=class_thresholds)

        # ── Append Tracking Metrics to CSV ────────────────────────────────
        patient = case_id.split('_')[0] if '_' in case_id else case_id
        
        row_data = [
            case_id, 
            patient,     # patient
            "",          # timepoint
            "",          # study_date
            border_vol_1, border_vol_2,
            small_vol_1, small_vol_2,
            border_count_1, border_count_2,
            small_count_1, small_count_2,
            border_count_1 + small_count_1, # total_removed_GTVp_count
            border_count_2 + small_count_2  # total_removed_GTVn_count
        ]
        
        # Append the row to the CSV file
        with open(csv_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)

        # ── Inverse transform → original CT space ─────────────────────────
        prediction_sitk = inverse_transform_to_original_space(pred_np, npz_data, ras_spacing)

        sitk.WriteImage(prediction_sitk, output_path)
        print(f"  ✅ Saved: {output_path}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
