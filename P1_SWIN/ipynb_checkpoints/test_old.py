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

import os
import torch
from pathlib import Path
import sys
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage
from monai.inferers.utils import sliding_window_inference # modification : changed import path as per VSCode suggestion monai.inferers -> monai.inferers.utils
import nibabel as nib
import argparse
import warnings

# Safeguard
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from P1_SWIN.nii_version.data_utils import get_loader # Modification: Change the reference to the .py in the root with the code changes. Original version stays in utils folder
from P1_SWIN.nii_version.trainer import dice
from networks.SwinTransModels import CONFIGS as CONFIGS_sw_seg
from networks.SwinTransModels import *

# Recursion limit was set for deprecated connected components counting functions
#import sys
#print(sys.getrecursionlimit())
# sys.setrecursionlimit(99999) 
#print(sys.getrecursionlimit())

warnings.filterwarnings("ignore")
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
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels') #changed default out_channels to infere the 3 labels (background, tumor, nodules)
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
parser.add_argument(
    '--inference_only',
    action='store_true',
    help='Run pure inference without ground truth (no Dice, no metrics)'
) # Sa simple présence sans valeur indique True, son absence indique false

# Deprecated : Inefficient and incorrect way to count connected components.
# def count_objects(image):
#     width, height, depth = image.shape[0], image.shape[1], image.shape[2]
#     objects_members = [(x, y, z) for x in range(width) for y in range(height) for z in range(depth) if image[y][x][z] == 1]
#     objects_count = 0
#     while objects_members != []:
#        remove_object(objects_members, objects_members.pop(0))
#        objects_count += 1
#     return objects_count
# def remove_object(objects_members, start_point):
#     x, y ,z = start_point
#     connex = [(x, y + 1,z+1), (x - 1, y, z), (x, y - 1,z),(x, y + 1,z-1),
#               (x + 1, y, z), (x, y + 1,z)]
#     for point in connex:
#         try:
#             objects_members.remove(point)
#             remove_object(objects_members, point)
#         except ValueError:
#             pass

# NEW MOD > Replace the count_objects and remove_object functions with:
## Mathieu comment : you could use the simpleITK method "connected components" to perform this operation

def count_objects(image, target_value=1):
    """Count connected components using scipy (much faster and correct)"""
    binary_mask = (image == target_value).astype(np.uint8)
    labeled_array, num_features = ndimage.label(binary_mask)
    return num_features


def create_prediction_sitk_with_metadata(prediction_np, original_image_path, spacing=(1.0, 1.0, 1.0), monai_affine=None):
    """
    Create a SimpleITK image from prediction array and resample to original preprocessed data space.
    
    Strategy:
    1. Read the original preprocessed 4D file (created by dataset_builder_simpleITK.py)
    2. Extract CT channel as 3D reference (has original spacing, direction, origin)
    3. Create prediction in MONAI's processed space (RAS, 1mm, with cropped origin)
    4. Resample prediction to match original preprocessed file exactly
    
    This ensures predictions align with the original PET/CT merged files in data_dir.
    
    Args:
        prediction_np: numpy array of predictions (3D, uint8) in (z, y, x) order
        original_image_path: path to original 4D PET/CT preprocessed file
        spacing: tuple of (space_x, space_y, space_z) used during MONAI preprocessing
        monai_affine: 4x4 affine matrix from MONAI metadata (contains cropped origin)
    
    Returns:
        sitk.Image: prediction image in original preprocessed file space
    """
    # Step 1: Create prediction in MONAI's processed space
    prediction_sitk = sitk.GetImageFromArray(prediction_np)
    prediction_sitk.SetSpacing(spacing)
    prediction_sitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))  # RAS
    
    if monai_affine is not None:
        origin = (float(monai_affine[0, 3]), float(monai_affine[1, 3]), float(monai_affine[2, 3]))
        prediction_sitk.SetOrigin(origin)
    else:
        prediction_sitk.SetOrigin((0.0, 0.0, 0.0))
    
    # Step 2: If no original file, return in MONAI space
    if original_image_path is None or not os.path.exists(original_image_path):
        print(f"⚠️ Original file not found, saving in MONAI space")
        print(f"✅ Prediction: {prediction_sitk.GetSize()} @ {prediction_sitk.GetSpacing()}")
        return prediction_sitk
    
    # Step 3: Read original preprocessed file and extract CT as reference
    print(f"📂 Reading original: {original_image_path}")
    original_image = sitk.ReadImage(original_image_path)
    
    if original_image.GetDimension() == 4:
        # Extract CT channel (index 1) - PET is index 0
        size_4d = list(original_image.GetSize())
        size_3d = [size_4d[0], size_4d[1], size_4d[2], 0]  # 0 collapses 4th dim
        index_ct = [0, 0, 0, 1]
        reference_3d = sitk.Extract(original_image, size_3d, index_ct)
    else:
        reference_3d = original_image
    
    print(f"🔍 Reference (original preprocessed CT):")
    print(f"   Size: {reference_3d.GetSize()}")
    print(f"   Spacing: {reference_3d.GetSpacing()}")
    print(f"   Origin: {reference_3d.GetOrigin()}")
    print(f"   Direction: {reference_3d.GetDirection()}")
    
    # Step 4: Resample prediction to match reference
    print(f"📐 Resampling prediction to original space...")
    print(f"   From: {prediction_sitk.GetSize()} @ {prediction_sitk.GetSpacing()} (MONAI RAS)")
    print(f"   To:   {reference_3d.GetSize()} @ {reference_3d.GetSpacing()} (Original)")
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_3d)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Critical for segmentation masks
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    resampler.SetTransform(sitk.Transform())  # Identity - rely on image metadata
    
    prediction_resampled = resampler.Execute(prediction_sitk)
    
    print(f"✅ Final: {prediction_resampled.GetSize()} @ {prediction_resampled.GetSpacing()}")
    print(f"   Origin: {prediction_resampled.GetOrigin()}")
    print(f"   Direction: {prediction_resampled.GetDirection()}")
    
    return prediction_resampled

def main():
    args = parser.parse_args()
    args.test_mode = True
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    output_dir = args.output_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    if args.saved_checkpoint == 'torchscript':
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == 'ckpt':
        # config_sw = CONFIGS_sw_seg['SwinUNETR_CMFF-hecktor-v01']
        # model = SwinUNETR_fusion(config_sw)
        config_sw = CONFIGS_sw_seg['SwinUNETR_CMFF-hecktor-v06']
        model = SwinUNETR_CrossModalityFusion_OutSum_6stageOuts(config_sw)
        model_dict = torch.load(pretrained_pth)['state_dict']
        model.load_state_dict(model_dict)

    model.eval()
    model.to(device)

    with torch.no_grad():
        dice_list_case = []

        # Create output directory if it doesn't exist
        if output_dir != None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                print("Created output directory: {}".format(output_dir))

        # FIX: Use os.path.join for proper path handling
        test_output_dir = output_dir if output_dir != None else os.path.join(pretrained_dir, 'test')

        if not os.path.exists(test_output_dir):
            os.makedirs(test_output_dir)
        
        for i, batch in enumerate(val_loader):

            val_inputs = batch["image"].to(device)
            val_labels = batch["label"].to(device)
            
            #this was here before my changes
            #val_inputs = val_inputs[:,1,:,:,:]
            #val_inputs = torch.unsqueeze(val_inputs,1)

            # OLD (deprecated):
            # img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            
            # NMODIFICATION > NEW (MONAI 1.5+ compatible):
            # Try multiple ways to get the filename for compatibility
            if hasattr(val_inputs, 'meta') and 'filename_or_obj' in val_inputs.meta:
                img_name = val_inputs.meta['filename_or_obj'][0].split('/')[-1]
            elif 'image_meta_dict' in batch:
                img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
            else:
                # Fallback: use index
                img_name = f"test_case_{i:03d}.nii.gz"
            # END MODIFICATION

            img_prefix = img_name.split('.')[0]

            #print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs,
                                                   (96, 96, 96), #remember this line if an error in inference occurs with img size changed
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)
            val_labels = val_labels.cpu().numpy()[:, 0, :, :, :]

            tumor_volume = np.sum(np.sum(np.sum(val_labels)))
            #count number of tumors
            #box, label, count = cv.detect_common_objects(tumor_volume)

            #_, thresh = cv2.threshold(val_labels, 0, 1, cv2.THRESH_BINARY_INV)
            #kernal = np.ones((2, 2), np.uint8)
            #dilation = cv2.dilate(thresh, kernal, iterations=2)
            #contours, hierarchy = cv2.findContours(
            #    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #objects = str(len(contours))
            print("number of tumors",count_objects(val_labels[0,:,:,:]))

            #print("tumor volume: {}".format(tumor_volume))
            dice_list_sub = []
            for i in range(1, 2):
                organ_Dice = dice(val_outputs[0] == i, val_labels[0] == i)
                dice_list_sub.append(organ_Dice)
            mean_dice = np.mean(dice_list_sub)
            print("ImageName, Mean Organ Dice, and Tumor Volume: {}, {}, {}".format(img_name,mean_dice,tumor_volume))
            dice_list_case.append(mean_dice)

            # SAVE NIFTI FILES with SimpleITK
            #####################
            # Convert to proper dtypes for NIfTI compatibility
            val_outputs_np = val_outputs[0, :, :, :].astype(np.uint8)  # Predictions as uint8

            # Try to get MONAI's affine matrix for proper coordinate mapping
            monai_affine = None
            if hasattr(val_inputs, 'meta') and 'affine' in val_inputs.meta:
                monai_affine = val_inputs.meta['affine']
                if hasattr(monai_affine, 'numpy'):
                    monai_affine = monai_affine.numpy()
                # Handle batch dimension if present
                if monai_affine.ndim == 3:
                    monai_affine = monai_affine[0]

            # Get path to original preprocessed file for resampling
            original_image_path = None
            if hasattr(val_inputs, 'meta') and 'filename_or_obj' in val_inputs.meta:
                original_image_path = val_inputs.meta['filename_or_obj'][0]
                print(f"📂 Original file path: {original_image_path}")

            # Create prediction SITK image resampled to original preprocessed file space
            prediction_sitk = create_prediction_sitk_with_metadata(
                prediction_np=val_outputs_np,
                original_image_path=original_image_path,
                spacing=(args.space_x, args.space_y, args.space_z),
                monai_affine=monai_affine
            )

            # Build output filename
            filename_output_path = os.path.join(
                test_output_dir, 
                f'{img_prefix.replace("_petct", "")}_dsc{round(mean_dice, 2)}_Pred.nii.gz'
            )
            # Save prediction
            sitk.WriteImage(prediction_sitk, filename_output_path)
            print(f"✅ Saved: {filename_output_path}")
            
            # Optional: Also save processed CT and PET for verification/visualization
            # Uncomment these to save CT/PET in same MONAI space as predictions
            # val_inputs_PET = val_inputs.cpu().numpy()[0, 0, :, :, :].astype(np.float32)
            # val_inputs_CT = val_inputs.cpu().numpy()[0, 1, :, :, :].astype(np.int16)
            # 
            # ct_sitk = sitk.GetImageFromArray(val_inputs_CT)
            # ct_sitk.SetSpacing((args.space_x, args.space_y, args.space_z))
            # ct_sitk.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
            # if monai_affine is not None:
            #     ct_sitk.SetOrigin((float(monai_affine[0,3]), float(monai_affine[1,3]), float(monai_affine[2,3])))
            # sitk.WriteImage(ct_sitk, os.path.join(test_output_dir, f'{img_prefix}_CT_monai.nii.gz'))
            
            #####################

        print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

if __name__ == '__main__':

    main()