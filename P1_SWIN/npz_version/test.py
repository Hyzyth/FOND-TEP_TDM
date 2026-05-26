# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# (license header preserved)

"""
test.py  —  SwinCross inference on NPZ-preprocessed data
==========================================================

Key changes vs. original:
  - Loads directly from NPZ (no MONAI Invertd, no MetaTensor tracking needed).
  - Inverse transform (uncrop → un-resample → original CT space) is done
    explicitly with SimpleITK using metadata stored in the NPZ at preprocess time.
  - FP16 autocast is always applied during inference (independent of --noamp).
  - Optional torch.compile for the model (--compile flag).
  - Post-processing (border removal, small-object removal) unchanged.

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
_SMALL_OBJ_THRESHOLD_MM3 = 125.0   # 0.125 cm³


# ── Post-processing (unchanged from original) ─────────────────────────────────

def _remove_border_objects(pred_np: np.ndarray, bg_val: int = 0) -> np.ndarray:
    pred_filtered = np.zeros_like(pred_np)
    for cls in np.unique(pred_np):
        if cls == bg_val:
            continue
        cleared = clear_border((pred_np == cls))
        pred_filtered[cleared] = cls
    n_removed = int((pred_np > 0).sum()) - int((pred_filtered > 0).sum())
    if n_removed > 0:
        print(f"   [ClearBorder] voxels removed: {n_removed}")
    return pred_filtered


def _remove_small_objects_physical(pred_np: np.ndarray,
                                   spacing_mm: tuple,
                                   threshold_mm3: float = _SMALL_OBJ_THRESHOLD_MM3,
                                   foreground_classes: tuple = (1, 2)) -> np.ndarray:
    voxel_vol_mm3 = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    min_size_vox  = max(1, math.ceil(threshold_mm3 / voxel_vol_mm3))
    remover       = RemoveSmallObjects(min_size=min_size_vox, connectivity=3)
    pred_filtered = pred_np.copy()
    for cls in foreground_classes:
        binary_t        = torch.from_numpy((pred_np == cls).astype(np.uint8)[None])
        binary_filtered = remover(binary_t).numpy()[0]
        pred_filtered[(pred_filtered == cls) & (binary_filtered == 0)] = 0
    n_removed = int((pred_np > 0).sum()) - int((pred_filtered > 0).sum())
    print(f"   [RemoveSmallObjects] min={min_size_vox} vox "
          f"({threshold_mm3:.0f} mm³) | removed: {n_removed}")
    return pred_filtered


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

    for idx, entry in enumerate(all_entries):
        torch.cuda.empty_cache()
        gc.collect()

        # ── Resolve paths ─────────────────────────────────────────────────
        npz_rel  = entry.get("npz", "")
        case_id  = entry.get("case_id") or os.path.basename(npz_rel).replace(".npz", "")
        npz_path = os.path.join(args.data_dir, npz_rel)

        output_path = os.path.join(test_output_dir, f"{case_id}_Pred.nii.gz")

        if args.skip_existing and os.path.exists(output_path):
            print(f"⏭  [{idx+1}/{len(all_entries)}] Skipping {case_id} (already exists)")
            continue

        if not os.path.exists(npz_path):
            print(f"❌ [{idx+1}/{len(all_entries)}] NPZ not found: {npz_path}")
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
        # Simulate orienting the original metadata to RAS to get the exact spacing 
        # the array had during the network inference.
        dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
        dummy.SetSpacing([float(x) for x in npz_data["orig_spacing"]])
        dummy.SetDirection([float(x) for x in npz_data["orig_direction"].flatten()])
        ras_spacing = sitk.DICOMOrient(dummy, "RAS").GetSpacing()

        # ── Post-processing ───────────────────────────────────────────────
        # Spacing used for physical-threshold small-object removal is 1mm
        # isotropic because the prediction lives in the pre-processed space.
        pred_np = _remove_border_objects(pred_np)
        pred_np = _remove_small_objects_physical(pred_np, spacing_mm=ras_spacing)

        # ── Inverse transform → original CT space ─────────────────────────
        prediction_sitk = inverse_transform_to_original_space(pred_np, npz_data, ras_spacing)

        sitk.WriteImage(prediction_sitk, output_path)
        print(f"  ✅ Saved: {output_path}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
