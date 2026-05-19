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
import json
import math
import gc
import os
import warnings
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from monai.data import MetaTensor, decollate_batch
from monai.inferers.utils import sliding_window_inference
from monai.transforms import Invertd, RemoveSmallObjects
from data_utils import get_loader
from networks.SwinTransModels import CONFIGS as CONFIGS_sw_seg
from networks.SwinTransModels import *

warnings.filterwarnings("ignore")

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
parser.add_argument('--skip_existing', action='store_true', help='Skip inference if prediction NIfTI already exists on disk')


def main():
    args = parser.parse_args()
    args.test_mode = True
    val_loader = get_loader(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_output_dir = args.output_dir if args.output_dir else os.path.join(args.pretrained_dir, 'test')
    os.makedirs(test_output_dir, exist_ok=True)

    # 1. Check which files need inference (if skip_existing is used)
    if args.skip_existing:
        print("Checking for existing predictions to skip...")

    model = None # Lazy load model only if we have work to do

    for i, batch in enumerate(val_loader):
        torch.cuda.empty_cache()
        gc.collect()

        # Extract filename securely
        if hasattr(batch["image"], 'meta') and 'filename_or_obj' in batch["image"].meta:
            img_name = batch["image"].meta['filename_or_obj'][0].split('/')[-1]
        elif 'image_meta_dict' in batch:
            img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
        else:
            img_name = f"test_case_{i:03d}.nii.gz"

        img_prefix_clean = img_name.split('.')[0].replace("_petct", "")
        filename_output_path = os.path.join(test_output_dir, f'{img_prefix_clean}_Pred.nii.gz')

        if args.skip_existing and os.path.exists(filename_output_path):
            print(f"⏭  Skipping {img_prefix_clean} (Prediction already exists)")
            continue

        # Lazy load model
        if model is None:
            print("Loading model weights...")
            pretrained_pth = os.path.join(args.pretrained_dir, args.pretrained_model_name)
            if args.saved_checkpoint == 'torchscript':
                model = torch.jit.load(pretrained_pth)
            else:
                config_sw = CONFIGS_sw_seg['SwinUNETR_CMFF-hecktor-v06']
                model = SwinUNETR_CrossModalityFusion_OutSum_6stageOuts(config_sw)
                checkpoint = torch.load(pretrained_pth)
                model_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
                model.load_state_dict(model_dict)
            model.eval()
            model.to(device)

        print(f"→ Running inference on {img_name}")
        val_inputs = batch["image"].to(device)
        # Using autocast for mixed precision inference to save memory and speed up computation on compatible GPUs
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap)

        # --- 1. PREPARATION OF PREDICTIONS (MONAI tensor format) ---
        val_outputs_tensor = torch.softmax(val_outputs, dim=1)
        val_outputs_tensor = torch.argmax(val_outputs_tensor, dim=1, keepdim=True).cpu()

        if hasattr(val_inputs, "meta"):
            val_outputs_tensor = MetaTensor(val_outputs_tensor, meta=val_inputs.meta)

        # --- 3. CLEAN SAVING USING MONAI InvertD + SIMPLEITK ---
        batch["pred"] = val_outputs_tensor
        invertd = Invertd(keys="pred", transform=val_loader.dataset.transform, orig_keys="image", nearest_interp=True, to_tensor=True)
        batch_inverted = [invertd(item) for item in decollate_batch(batch)]
        
        for item_inv in batch_inverted:
            pred_np_final = item_inv["pred"][0].cpu().numpy().astype(np.uint8)

            # 3.A: Recover original image spacing for physical threshold, BEFORE transposing for SimpleITK.
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
                ref_sitk = sitk.Extract(original_sitk, [size_4d[0], size_4d[1], size_4d[2], 0], [0, 0, 0, 0])
            else:
                ref_sitk = original_sitk

            spacing_mm = ref_sitk.GetSpacing()   # (sx, sy, sz) in mm
            pred_np_final = _remove_small_objects_physical(pred_np_final, spacing_mm=spacing_mm)

            # 3.B: Transpose Z,Y,X for SimpleITK (ITK axis order)
            pred_np_sitk = pred_np_final.transpose(2, 1, 0)
                
            # 3.C: Build SimpleITK image and copy spatial metadata
            prediction_sitk = sitk.GetImageFromArray(pred_np_sitk)
            prediction_sitk.CopyInformation(ref_sitk)
            
            sitk.WriteImage(prediction_sitk, filename_output_path)
            print(f"✅ Saved perfectly with MONAI Invertd + SimpleITK: {filename_output_path}")

    print("Inference loop complete.")

if __name__ == '__main__':
    main()
