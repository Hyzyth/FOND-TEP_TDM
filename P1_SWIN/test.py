# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv                                                       # MODIFICATION: for per-class Dice CSV output
import json
import math                                                      # MODIFICATION: needed for ceil in small-object threshold
import os
import warnings
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from monai.data import MetaTensor, decollate_batch
from monai.inferers.utils import sliding_window_inference
from monai.transforms import Invertd, RemoveSmallObjects        # MODIFICATION: added RemoveSmallObjects
from data_utils import get_loader
from networks.SwinTransModels import CONFIGS as CONFIGS_sw_seg
from networks.SwinTransModels import *
from trainer import dice
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# MODIFICATION: small-component filtering helpers
# ---------------------------------------------------------------------------
# Physical threshold: 0.125 cm³ = 125 mm³
_SMALL_OBJ_THRESHOLD_MM3 = 125.0   # 0.125 cm³

def _remove_small_objects_physical(pred_np: np.ndarray,
                                   spacing_mm: tuple,
                                   threshold_mm3: float = _SMALL_OBJ_THRESHOLD_MM3,
                                   foreground_classes: tuple = (1, 2)) -> np.ndarray:
    """
    Remove connected components smaller than `threshold_mm3` from a
    multi-class argmax label map, using MONAI's RemoveSmallObjects.

    Parameters
    ----------
    pred_np : np.ndarray, shape (H, W, D), dtype uint8
        Argmax label map with values in {0, 1, 2, ...}.
    spacing_mm : tuple of float
        Physical voxel spacing in mm, e.g. (1.0, 1.0, 1.0).
        Retrieved from the original SimpleITK image so the threshold is
        always in real-world units regardless of preprocessing resampling.
    threshold_mm3 : float
        Minimum component volume to keep, in mm³ (default 125 mm³ = 0.125 cm³).
    foreground_classes : tuple of int
        Which label values to filter (background 0 is always kept).

    Returns
    -------
    np.ndarray, same shape and dtype as `pred_np`, with small components
    zeroed out.
    """
    voxel_vol_mm3 = spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
    min_size_voxels = max(1, math.ceil(threshold_mm3 / voxel_vol_mm3))

    # MONAI RemoveSmallObjects works on channel-first binary tensors (C, H, W, D)
    # connectivity=3 → 26-connectivity in 3D (full neighbourhood)
    remover = RemoveSmallObjects(min_size=min_size_voxels, connectivity=3)

    pred_filtered = pred_np.copy()
    for cls in foreground_classes:
        binary_np = (pred_np == cls).astype(np.uint8)          # (H, W, D)
        binary_t  = torch.from_numpy(binary_np[None])          # (1, H, W, D)
        binary_filtered = remover(binary_t).numpy()[0]          # (H, W, D)
        # Zero out voxels that were removed by the filter
        pred_filtered[(pred_filtered == cls) & (binary_filtered == 0)] = 0

    n_removed = int((pred_np > 0).sum()) - int((pred_filtered > 0).sum())
    print(f"   [RemoveSmallObjects] threshold={min_size_voxels} vox "
          f"({threshold_mm3:.1f} mm³ @ {voxel_vol_mm3:.3f} mm³/vox) | "
          f"voxels removed: {n_removed}")
    return pred_filtered
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# MODIFICATION: per-class Dice CSV writer
# ---------------------------------------------------------------------------
# Class index → human-readable name (matches SwinCross label convention)
_CLASS_NAMES = {1: "GTVp", 2: "GTVn"}

def _write_dice_csv(csv_path: str,
                    rows: list,
                    mode: str = "a") -> None:
    """
    Append per-case Dice results to a CSV file.

    Each row in `rows` is a dict with keys:
        case_id, GTVp_dice, GTVn_dice, mean_dice
    where a value of None means the class was absent from both GT and prediction
    (not scored), and 0.0 means the class was present/predicted but not matched.

    The file is created with a header on first write (mode="w") and appended
    to on subsequent calls (mode="a").  Callers pass mode="w" for the first
    case written within a run, and "a" for all subsequent ones.
    """
    fieldnames = ["case_id", "GTVp_dice", "GTVn_dice", "mean_dice"]
    write_header = (mode == "w") or not os.path.exists(csv_path)
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./runs/for_log/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--output_dir', default=None, type=str, help='output directory for test results')
parser.add_argument('--data_dir', default='Dataset_Final_SwinCross_SITK', type=str, help='dataset directory')
parser.add_argument('--json_list', default='dataset_swincross.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='model_best.pth', type=str, help='pretrained model name')
parser.add_argument('--saved_checkpoint', default='ckpt', type=str, help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=36, type=int, help='feature size dimention')
parser.add_argument('--infer_overlap', default=0.4, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=2, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--lower', default=0.0, type=float, help='a_min in ScaleIntensityRangePercentilesd')
parser.add_argument('--upper', default=100.0, type=float, help='a_max in ScaleIntensityRangePercentilesd')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRangePercentilesd')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRangePercentilesd')
parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=8, type=int, help='number of workers')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--inference_only', action='store_true', help='Run pure inference without ground truth (no Dice, no metrics)')

def count_objects(image, target_value=1): 
    """
    Description:
        Counts the number of distinct objects in a 3D volume using a connected-components approach.
        An object is defined as a set of connected voxels sharing the same target value
        (e.g., 1 for tumors). The function uses full connectivity to ensure that all adjacent voxels
        are considered part of the same object.

    :param image: numpy.ndarray
        3D volume in which objects are to be counted.
    :param target_value: int
        Value in the volume corresponding to the objects to count
        (e.g., 1 for tumors, 2 for nodules).
    :return: int
        Number of distinct connected objects in the volume matching the target value.
    """
    binary_mask = (image == target_value) 
    structure = ndimage.generate_binary_structure(image.ndim, image.ndim)  # full connectivity 
    _, num_features = ndimage.label(binary_mask, structure=structure) 
    return num_features

def main():
    args = parser.parse_args()
    args.test_mode = True
    args.inference_only = args.inference_only
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    output_dir = args.output_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    if args.saved_checkpoint == 'torchscript':
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == 'ckpt':
        config_sw = CONFIGS_sw_seg['SwinUNETR_CMFF-hecktor-v06']
        model = SwinUNETR_CrossModalityFusion_OutSum_6stageOuts(config_sw)
        model_dict = torch.load(pretrained_pth)['state_dict']
        model.load_state_dict(model_dict)

    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []

        if output_dir != None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print("Created output directory: {}".format(output_dir))

        test_output_dir = output_dir if output_dir != None else os.path.join(pretrained_dir, 'test')

        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)

        # -----------------------------------------------------------
        # MODIFICATION: Build filename -> metadata lookup from JSON
        # -----------------------------------------------------------
        json_path = os.path.join(args.data_dir, args.json_list)
        meta_lookup = {}
        try:
            with open(json_path, 'r') as f:
                dataset_json = json.load(f)
                # Look through standard MONAI splits
                for split in ["training", "validation", "testing"]:
                    if split in dataset_json:
                        for item in dataset_json[split]:
                            img_path = item.get("image", "")
                            basename = os.path.basename(img_path)
                            meta_lookup[basename] = item
            print(f"Loaded metadata for {len(meta_lookup)} cases from {args.json_list}")
        except Exception as e:
            print(f"Warning: Could not load JSON metadata from {json_path}. Error: {e}")
        # -----------------------------------------------------------

        # MODIFICATION: path for the per-class Dice CSV; written alongside predictions
        dice_csv_path = os.path.join(test_output_dir, "per_case_dice.csv")
        first_csv_write = True   # controls header row — written once then appended
        
        for i, batch in enumerate(val_loader):

            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device) if not args.inference_only else None

            if hasattr(val_inputs, 'meta') and 'filename_or_obj' in val_inputs.meta:
                img_name = val_inputs.meta['filename_or_obj'][0].split('/')[-1]
            elif 'image_meta_dict' in batch:
                img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            else:
                img_name = f"test_case_{i:03d}.nii.gz"

            img_prefix = img_name.split('.')[0]

            print("Debug : Before inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs,
                                                   (96, 96, 96),
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            print("Debug : After inference on case {}".format(img_name))

            # --- 1. PREPARATION OF PREDICTIONS (MONAI tensor format) ---
            val_outputs_tensor = torch.softmax(val_outputs, dim=1)
            val_outputs_tensor = torch.argmax(val_outputs_tensor, dim=1, keepdim=True).cpu()

            if hasattr(val_inputs, "meta"):
                val_outputs_tensor = MetaTensor(val_outputs_tensor, meta=val_inputs.meta)
            
            val_outputs_np = val_outputs_tensor[:, 0, ...].numpy().astype(np.uint8)

            # --- 2. COMPUTATION OF DICE SCORE AND VOLUME ---
            
            # MODIFICATION: Fetch per-case GT availability
            case_meta = meta_lookup.get(img_name, {})
            # Default to checking global arg if metadata isn't found
            case_gt_available = case_meta.get("gt_available", not args.inference_only)

            if case_gt_available and not args.inference_only:
                val_labels_np = val_labels.cpu().numpy()[:, 0, :, :, :]
                tumor_volume = np.sum(val_labels_np)
                print("number of tumors", count_objects(val_labels_np[0,:,:,:]))

                # MODIFICATION: compute per-class Dice and track individually,
                # then derive mean only over scored classes.
                # A class is scored when either the GT or the prediction is non-empty
                # for that class.  This correctly handles:
                #   - N-only GT  → class 1 scored only if model also predicts label=1
                #   - T-only GT  → class 2 scored only if model also predicts label=2
                #   - Both absent → class excluded from mean (true negative, no penalty)
                per_class_dice = {}   # int class → float dice (or None if not scored)
                dice_list_sub  = []

                for cls in range(1, args.out_channels):
                    gt_present   = np.sum(val_labels_np[0] == cls) > 0
                    pred_present = np.sum(val_outputs_np[0] == cls) > 0

                    if gt_present or pred_present:
                        cls_dice = dice(val_outputs_np[0] == cls, val_labels_np[0] == cls)
                        per_class_dice[cls] = cls_dice
                        dice_list_sub.append(cls_dice)
                        print(f"   Class {cls} ({_CLASS_NAMES.get(cls, str(cls))}): "
                              f"Dice = {cls_dice:.4f}  "
                              f"[GT={'present' if gt_present else 'absent'}, "
                              f"Pred={'present' if pred_present else 'absent'}]")
                    else:
                        per_class_dice[cls] = None
                        print(f"   Class {cls} ({_CLASS_NAMES.get(cls, str(cls))}): "
                              f"not scored (absent in both GT and prediction)")

                mean_dice = np.mean(dice_list_sub) if dice_list_sub else 0.0
                print("ImageName, Mean Organ Dice, and Tumor Volume: {}, {}, {}".format(
                    img_name, mean_dice, tumor_volume))
                dice_list_case.append(mean_dice)

                # MODIFICATION: write per-class dice row to CSV
                csv_row = {
                    "case_id":   img_prefix,
                    "GTVp_dice": per_class_dice.get(1),   # None = not scored
                    "GTVn_dice": per_class_dice.get(2),   # None = not scored
                    "mean_dice": mean_dice,
                }
                _write_dice_csv(
                    dice_csv_path,
                    [csv_row],
                    mode="w" if first_csv_write else "a"
                )
                first_csv_write = False

            else:
                mean_dice = None
                per_class_dice = {}
                print(f"ImageName: {img_name} (Inference only / No GT available)")
                
                # MODIFICATION: Write skipping row to CSV
                csv_row = {
                    "case_id":   img_prefix,
                    "GTVp_dice": "no_gt_available",
                    "GTVn_dice": "no_gt_available",
                    "mean_dice": "no_gt_available",
                }
                _write_dice_csv(
                    dice_csv_path,
                    [csv_row],
                    mode="w" if first_csv_write else "a"
                )
                first_csv_write = False

            # --- 3. CLEAN SAVING USING MONAI InvertD + SIMPLEITK ---
            batch["pred"] = val_outputs_tensor

            invertd = Invertd(
                keys="pred",
                transform=val_loader.dataset.transform,
                orig_keys="image",
                nearest_interp=True,
                to_tensor=True
            )

            batch_inverted = [invertd(item) for item in decollate_batch(batch)]
            
            for item_inv in batch_inverted:
                pred_inverted = item_inv["pred"]
                
                # 3.A: Extract numpy (H, W, D) in MONAI/RAS axis order
                pred_np_final = pred_inverted[0].cpu().numpy().astype(np.uint8)

                # -------------------------------------------------------
                # 3.A-bis: Recover original image spacing for physical
                #          threshold, BEFORE transposing for SimpleITK.
                # -------------------------------------------------------
                img_tensor = item_inv["image"]

                if hasattr(img_tensor, "meta") and "filename_or_obj" in img_tensor.meta:
                    original_image_path = img_tensor.meta["filename_or_obj"]
                elif "image_meta_dict" in item_inv:
                    original_image_path = item_inv["image_meta_dict"]["filename_or_obj"]
                else:
                    raise ValueError("Impossible de trouver le chemin du fichier source.")

                if isinstance(original_image_path, (list, tuple, np.ndarray)):
                    original_image_path = original_image_path[0]
                if isinstance(original_image_path, torch.Tensor) or hasattr(original_image_path, "item"):
                    original_image_path = str(original_image_path)

                original_sitk = sitk.ReadImage(original_image_path)

                # Handle 4D (PET/CT) vs 3D source
                if original_sitk.GetDimension() == 4:
                    size_4d = list(original_sitk.GetSize())
                    original_sitk_3d = sitk.Extract(
                        original_sitk,
                        [size_4d[0], size_4d[1], size_4d[2], 0],
                        [0, 0, 0, 0]
                    )
                    ref_sitk = original_sitk_3d
                else:
                    ref_sitk = original_sitk

                # MODIFICATION: Apply MONAI RemoveSmallObjects using the
                # physical spacing of the original image so the 0.125 cm³
                # threshold is in real-world units (independent of voxel size).
                spacing_mm = ref_sitk.GetSpacing()   # (sx, sy, sz) in mm
                pred_np_final = _remove_small_objects_physical(
                    pred_np_final,
                    spacing_mm=spacing_mm,
                    threshold_mm3=_SMALL_OBJ_THRESHOLD_MM3,
                    foreground_classes=(1, 2),
                )
                # -------------------------------------------------------

                # 3.B: Transpose Z,Y,X for SimpleITK (ITK axis order)
                pred_np_sitk = pred_np_final.transpose(2, 1, 0)
                
                # 3.C: Build SimpleITK image and copy spatial metadata
                prediction_sitk = sitk.GetImageFromArray(pred_np_sitk)
                prediction_sitk.CopyInformation(ref_sitk)

                # 3.D: Save to disk
                # MODIFICATION: per-class Dice is now appended to the filename
                # (GTVp then GTVn), replacing the single mean_dice field.
                if mean_dice is not None:
                    gtvp_str = (f"{per_class_dice.get(1):.2f}"
                                if per_class_dice.get(1) is not None else "NA")
                    gtvn_str = (f"{per_class_dice.get(2):.2f}"
                                if per_class_dice.get(2) is not None else "NA")
                    dsc_tag = f"T{gtvp_str}_N{gtvn_str}"
                else:
                    dsc_tag = "NA"

                filename_output_path = os.path.join(
                    test_output_dir,
                    f'{img_prefix.replace("_petct", "")}_dsc{dsc_tag}_Pred.nii.gz'
                )
                
                sitk.WriteImage(prediction_sitk, filename_output_path)
                print(f"✅ Saved perfectly with MONAI Invertd + SimpleITK: {filename_output_path}")            
            #####################

        if not args.inference_only:
            if len(dice_list_case) > 0:
                print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))
            else:
                print("Overall Mean Dice: N/A (No valid GT cases evaluated)")
            print(f"Per-case Dice CSV written to: {dice_csv_path}")

if __name__ == '__main__':
    main()
