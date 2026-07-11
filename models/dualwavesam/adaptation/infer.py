"""
infer.py  —  3-class DualwaveSAM inference on HECKTOR 2026 NPZ
==============================================================

Pipeline per patient:
  1. Load NPZ (already preprocessed to RAS 1mm isotropic, foreground-cropped)
  2. Feed axial slices through the model (batched, FP16)
  3. Reassemble slice predictions -> (R, A, S) volume
  4. Resize back to original crop-space resolution
  5. Inverse transform: uncrop -> RAS full volume -> resample to original CT space
  6. Save <case_id>_Pred.nii.gz  (compatible with evaluate_predictions.py)

Usage:
  python {folder}/infer.py \\
      --data_dir /data/ethan/PP_hecktor2026_kfold_npz \\
      --json_list dataset_swincross_2026kfold_test.json \\
      --checkpoint ./runs/DualwaveSAM3c_classic/model_best.pth \\
      --output_dir /data/ethan/DualwaveSAM3c/classic_run/hecktor_TEST_vault
"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
import csv
from scipy import ndimage
from monai.transforms import RemoveSmallObjects
from skimage.segmentation import clear_border

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # DualwaveSAM root (sam_modeling_wave/)
sys.path.insert(0, str(_HERE))          # This package (dataset, model, etc.)

from dataset import (
    SLICE_AXIS, SLICE_SIZE,
    extract_slices, normalise_ct, normalise_pet,
)
from model import DualwaveSAM3Class

# ── Post-processing thresholds ────────────────────
class_thresholds = {
            1: 100.0,  # GTVp (Tumors)
            2: 50.0    # GTVn (Nodules)
        }  # These thresholds are in mm³ and will be converted to voxel counts dynamically per case


def _remove_border_objects_tracked(pred: np.ndarray, spacing_mm: tuple) -> tuple:
    """
    Removes objects touching the volume border and tracks the volume and count removed per class.
    """
    out = np.zeros_like(pred)
    border_vols = {1: 0.0, 2: 0.0}
    border_counts = {1: 0, 2: 0}
    
    struct = ndimage.generate_binary_structure(3, 3)
    vox_vol = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]

    for cls in (1, 2):
        if not (pred == cls).any():
            continue
            
        mask = (pred == cls)
        cleared = clear_border(mask)
        out[cleared] = cls

        # Find exactly what was deleted
        deleted_mask = mask & (~cleared)
        
        border_vols[cls] = float(deleted_mask.sum() * vox_vol)
        _, count = ndimage.label(deleted_mask, structure=struct)
        border_counts[cls] = count

    return out, border_vols[1], border_vols[2], border_counts[1], border_counts[2]

def _remove_nodule_shell(pred: np.ndarray, spacing_mm: tuple, iterations: int = 2) -> tuple:
    """
    Surgically removes GTVp (tumor) pixels that form a shell around GTVn (nodules).
    Returns the cleaned prediction, the volume of the shell removed (mm³), 
    and the COUNT of disconnected shell fragments removed.
    """
    out = pred.copy()
    nodule_mask = (out == 2)
    tumor_mask = (out == 1)

    if not nodule_mask.any() or not tumor_mask.any():
        return out, 0.0, 0

    # 3D dilation of the nodule using 26-connectivity
    struct = ndimage.generate_binary_structure(3, 3)
    dilated_nodule = ndimage.binary_dilation(nodule_mask, structure=struct, iterations=iterations)

    # Find where the dilated nodule overlaps with the tumor (the shell)
    shell_mask = dilated_nodule & tumor_mask
    
    # Erase the shell from the output
    out[shell_mask] = 0

    # Calculate physical volume
    vox_vol = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    shell_vol_mm3 = float(shell_mask.sum() * vox_vol)

    # Count how many disconnected shell fragments were deleted
    _, shell_count = ndimage.label(shell_mask, structure=struct)

    return out, shell_vol_mm3, shell_count

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


# ── Inverse transform ─────────────────────

def inverse_transform(pred_crop: np.ndarray, npz_data, ras_spacing) -> sitk.Image:
    """
    Undo offline preprocessing: uncrop -> full RAS volume -> original CT space.
    """
    ras_size  = [int(x) for x in npz_data["ras_size_itk"]]
    full_pred = np.zeros((ras_size[0], ras_size[1], ras_size[2]), dtype=np.uint8)
    cs, ce    = npz_data["crop_start"], npz_data["crop_end"]
    full_pred[cs[0]:ce[0], cs[1]:ce[1], cs[2]:ce[2]] = pred_crop

    # MONAI (R,A,S) -> ITK (S,A,R)
    arr_itk  = full_pred.transpose(2, 1, 0).astype(np.uint8)
    pred_sitk = sitk.GetImageFromArray(arr_itk)
    pred_sitk.SetSpacing(ras_spacing)
    pred_sitk.SetOrigin([float(x) for x in npz_data["ras_origin"]])
    pred_sitk.SetDirection([float(x) for x in npz_data["ras_direction"].flatten()])

    orig_size = [int(x) for x in npz_data["orig_size_itk"]]
    r = sitk.ResampleImageFilter()
    r.SetOutputSpacing([float(x) for x in npz_data["orig_spacing"]])
    r.SetOutputOrigin([float(x) for x in npz_data["orig_origin"]])
    r.SetOutputDirection([float(x) for x in npz_data["orig_direction"].flatten()])
    r.SetSize(orig_size)
    r.SetInterpolator(sitk.sitkNearestNeighbor)
    r.SetTransform(sitk.Transform())
    r.SetDefaultPixelValue(0)
    r.SetOutputPixelType(sitk.sitkUInt8)
    return r.Execute(pred_sitk)


def get_ras_spacing(npz_data) -> tuple:
    dummy = sitk.Image(1, 1, 1, sitk.sitkUInt8)
    dummy.SetSpacing([float(x) for x in npz_data["orig_spacing"]])
    dummy.SetDirection([float(x) for x in npz_data["orig_direction"].flatten()])
    return sitk.DICOMOrient(dummy, "RAS").GetSpacing()


# ── Per-patient inference ──────────────────────────────────────────────────────

def infer_patient(
    npz_path:   str,
    model:      DualwaveSAM3Class,
    device:     torch.device,
    img_size:   int = SLICE_SIZE,
    batch_size: int = 32,
) -> tuple:
    """
    Returns (pred_vol, npz_meta):
      pred_vol : (R, A, S) uint8 — predicted labels in NPZ crop space
      npz_meta : dict            — NPZ metadata arrays for inverse transform
    """
    with np.load(npz_path, allow_pickle=False) as npz:
        ct_vol   = npz["ct"].astype(np.float32)
        pet_vol  = npz["pet"].astype(np.float32)
        R, A, S  = ct_vol.shape
        # Load all metadata in the same open — avoids a second file open later
        npz_meta = {k: npz[k].copy() for k in npz.files if k not in ("ct", "pet", "label")}

    ct_vol  = normalise_ct(ct_vol)
    pet_vol = normalise_pet(pet_vol)

    # (R, A, S) -> (S, R, A)
    ct_slices  = np.moveaxis(ct_vol,  SLICE_AXIS, 0)   # (S, R, A)
    pet_slices = np.moveaxis(pet_vol, SLICE_AXIS, 0)

    pred_slices = np.zeros((S, R, A), dtype=np.uint8)

    model.eval()
    with torch.no_grad():
        for start in range(0, S, batch_size):
            end  = min(start + batch_size, S)
            imgs = np.zeros((end - start, img_size, img_size, 2), dtype=np.float32)

            for i, s in enumerate(range(start, end)):
                imgs[i, :, :, 0] = cv2.resize(ct_slices[s],  (img_size, img_size), cv2.INTER_LINEAR)
                imgs[i, :, :, 1] = cv2.resize(pet_slices[s], (img_size, img_size), cv2.INTER_LINEAR)

            imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2).to(device)  # (B,2,H,W)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=torch.cuda.is_available()):
                logits, _ = model(imgs_t)   # (B, 3, H, W)

            preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)  # (B, H, W)

            # Resize predictions back to original (R, A) spatial dims
            for i, s in enumerate(range(start, end)):
                if preds[i].shape != (R, A):
                    pred_slices[s] = cv2.resize(preds[i], (A, R), cv2.INTER_NEAREST)
                else:
                    pred_slices[s] = preds[i]

    # (S, R, A) -> (R, A, S)
    pred_vol = np.moveaxis(pred_slices, 0, SLICE_AXIS)
    return pred_vol, npz_meta


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DualwaveSAM3c inference on HECKTOR 2026 NPZ")
    p.add_argument("--data_dir",    required=True)
    p.add_argument("--json_list",   required=True)
    p.add_argument("--checkpoint",  required=True,
                   help="Path to model_best.pth or model_last.pth")
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--split",       default="validation",
                   help="JSON key to iterate: 'validation', 'training', 'testing'")
    p.add_argument("--img_size",    default=256,  type=int)
    p.add_argument("--n_filters",   default=16,   type=int)
    p.add_argument("--wavelet",     default="haar", type=str)
    p.add_argument("--num_classes", default=3,    type=int)
    p.add_argument("--batch_size",  default=32,   type=int,
                   help="Number of slices processed per GPU forward pass.")
    p.add_argument("--gpu",         default=0,    type=int)
    p.add_argument("--skip_existing", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────
    print("Loading model …")
    model = DualwaveSAM3Class(
        img_size=args.img_size,
        n_filters=args.n_filters,
        wavelet=args.wavelet,
        num_classes=args.num_classes,
        use_aux_head=False,   # aux head not needed at inference
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd   = ckpt.get("state_dict", ckpt)
    model.load_state_dict(
        {k.replace("module.", ""): v for k, v in sd.items()},
        strict=False,
    )
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    # ── Load case list ────────────────────────────────────────────────────
    json_path = os.path.join(args.data_dir, args.json_list) \
        if not os.path.isabs(args.json_list) else args.json_list
    with open(json_path) as f:
        js = json.load(f)

    entries = js.get(args.split, [])
    if not entries:
        # Fallback: try all splits
        for key in ("validation", "training", "testing"):
            entries += js.get(key, [])
    print(f"Cases to infer: {len(entries)} (split='{args.split}')")

    # ── CSV logging for post-processing volumes removed ───────────────────
    csv_log_path = os.path.join(args.output_dir, "postprocessing_logs.csv")
    csv_headers = [
        "case_id", "patient", "timepoint", "study_date", 
        "border_removed_GTVp_mm3", "border_removed_GTVn_mm3",
        "shell_removed_GTVp_mm3", 
        "small_obj_removed_GTVp_mm3", "small_obj_removed_GTVn_mm3",
        "border_removed_GTVp_count", "border_removed_GTVn_count",
        "shell_removed_GTVp_count", 
        "small_obj_removed_GTVp_count", "small_obj_removed_GTVn_count",
        "total_removed_GTVp_count", "total_removed_GTVn_count"
    ]
    with open(csv_log_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # ── Inference loop ────────────────────────────────────────────────────
    for idx, entry in enumerate(entries):
        torch.cuda.empty_cache()
        gc.collect()

        npz_rel  = entry.get("npz", "")
        case_id  = entry.get("case_id") or os.path.basename(npz_rel).replace(".npz", "")
        npz_path = os.path.join(args.data_dir, npz_rel)
        out_path = os.path.join(args.output_dir, f"{case_id}_Pred.nii.gz")

        if args.skip_existing and os.path.exists(out_path):
            print(f">>  [{idx+1}/{len(entries)}] Skip {case_id}")
            continue

        if not os.path.exists(npz_path):
            print(f" [{idx+1}/{len(entries)}] NPZ not found: {npz_path}")
            continue

        print(f"->  [{idx+1}/{len(entries)}] {case_id}")

        # ── 1. Slice-wise inference + metadata in one NPZ open ────────────
        pred_crop, npz_data = infer_patient(npz_path, model, device,
                                            img_size=args.img_size,
                                            batch_size=args.batch_size)

        ras_spacing = get_ras_spacing(npz_data)

        # ── 2. Post-processing ────────────────────────────────────────────
        # A. Remove border-touching objects
        pred_crop, border_vol_p, border_vol_n, border_count_p, border_count_n = _remove_border_objects_tracked(
            pred_crop, spacing_mm=ras_spacing
        )

        # B. Remove shell around nodules (GTVp only)
        pred_crop, shell_vol_p, shell_count_p = _remove_nodule_shell(
            pred_crop, spacing_mm=ras_spacing, iterations=2
        )
        
        # C. Remove small components
        pred_crop, small_vol_p, small_vol_n, small_count_p, small_count_n = _remove_small_objects_tracked(
            pred_crop, spacing_mm=ras_spacing, thresholds_mm3=class_thresholds
        )

        # Calculate Grand Totals
        total_count_p = border_count_p + shell_count_p + small_count_p
        total_count_n = border_count_n + small_count_n

        # D. Log the modifications
        with open(csv_log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                case_id,
                entry.get("patient", ""),
                entry.get("timepoint", ""),
                entry.get("study_date", ""),
                round(border_vol_p, 2), round(border_vol_n, 2),
                round(shell_vol_p, 2),
                round(small_vol_p, 2), round(small_vol_n, 2),
                border_count_p, border_count_n,
                shell_count_p,
                small_count_p, small_count_n,
                total_count_p, total_count_n
            ])

        # ── 3. Inverse transform -> original CT space ─────────────────────
        pred_sitk = inverse_transform(pred_crop, npz_data, ras_spacing)

        sitk.WriteImage(pred_sitk, out_path)
        labels_found = np.unique(
            sitk.GetArrayFromImage(pred_sitk)
        ).tolist()
        print(f"   {out_path}  labels={labels_found}")

    print("\nInference complete.")


if __name__ == "__main__":
    main()
