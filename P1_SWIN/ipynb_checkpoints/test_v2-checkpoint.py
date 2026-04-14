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
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from monai.inferers.utils import sliding_window_inference # modification : changed import path as per VSCode suggestion monai.inferers -> monai.inferers.utils
import nibabel as nib
from data_utils import get_loader # Modification: Change the reference to the .py in the root with the code changes. Original version stays in utils folder
from trainer import dice
import argparse
from networks.SwinTransModels import CONFIGS as CONFIGS_sw_seg
from networks.SwinTransModels import *
import warnings

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


def create_prediction_sitk_from_monai_debug(
    prediction_np,
    original_image_path=None,
    monai_affine=None,
    spacing=(1.0, 1.0, 1.0)
):
    """
    =====================================================================================
    CONVERSION MONAI (RAS) → SimpleITK (LPS) AVEC DEBUG COMPLET
    =====================================================================================

    CONTEXTE :
    ----------
    - MONAI travaille généralement en espace RAS (Right-Anterior-Superior)
    - SimpleITK / DICOM utilisent LPS (Left-Posterior-Superior)
    - Une conversion incorrecte produit un masque anatomiquement déplacé

    OBJECTIF :
    ----------
    Transformer une prédiction MONAI (RAS + crop + resampling)
    vers l’espace physique original du patient (LPS).

    ÉTAPES :
    --------
    1) Créer image SITK en espace MONAI (RAS)
    2) Convertir physiquement RAS → LPS
    3) Charger image originale comme référence
    4) Resampler pour alignement voxel parfait

    PARAMÈTRES :
    ------------
    prediction_np : np.ndarray (Z,Y,X)
        Masque prédit en espace MONAI

    original_image_path : str
        Chemin vers l’image originale (LPS)

    monai_affine : np.ndarray (4x4)
        Affine MONAI contenant origine + orientation + crop

    spacing : tuple(float)
        Spacing utilisé lors du preprocessing MONAI

    RETOUR :
    --------
    sitk.Image alignée avec l’image originale
    """

    print("\n" + "=" * 80)
    print("🚀 START — MONAI → SITK LPS CONVERSION")
    print("=" * 80)

    # =========================================================================
    # 0) INFOS SUR LE MASQUE ENTRÉE
    # =========================================================================

    print("\n[STEP 0] INPUT PREDICTION INFO")
    print(f"• Shape (Z,Y,X) : {prediction_np.shape}")
    print(f"• Data type     : {prediction_np.dtype}")
    print(f"• Unique labels : {np.unique(prediction_np)}")

    # =========================================================================
    # 1) CRÉATION IMAGE SITK EN ESPACE MONAI (RAS)
    # =========================================================================

    print("\n[STEP 1] CREATE SITK IMAGE IN MONAI SPACE (RAS)")

    # SimpleITK attend un tableau (Z,Y,X)
    pred_sitk = sitk.GetImageFromArray(prediction_np.astype(np.uint8))

    # Spacing isotropique généralement 1mm après preprocessing
    pred_sitk.SetSpacing(spacing)

    # Direction identité → axes alignés (RAS dans contexte MONAI)
    direction_ras = (
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )
    pred_sitk.SetDirection(direction_ras)

    print("• Spacing set to :", pred_sitk.GetSpacing())
    print("• Direction (RAS):", pred_sitk.GetDirection())

    # --- Origine issue de l’affine MONAI (IMPORTANT)
    if monai_affine is not None:
        origin_ras = (
            float(monai_affine[0, 3]),
            float(monai_affine[1, 3]),
            float(monai_affine[2, 3])
        )
        print("• Origin from MONAI affine (RAS):", origin_ras)
    else:
        origin_ras = (0.0, 0.0, 0.0)
        print("⚠️ No MONAI affine provided — using origin (0,0,0)")

    pred_sitk.SetOrigin(origin_ras)

    print("• Image size :", pred_sitk.GetSize())
    print("• Origin     :", pred_sitk.GetOrigin())

    # =========================================================================
    # 2) CONVERSION PHYSIQUE RAS → LPS
    # =========================================================================

    print("\n[STEP 2] CONVERT RAS → LPS")

    print("👉 Flipping X and Y axes")

    # Flip axes X et Y (RAS → LPS)
    flip_filter = sitk.FlipImageFilter()
    flip_filter.SetFlipAxes([True, True, False])

    pred_lps = flip_filter.Execute(pred_sitk)

    # --- Correction de l’origine après flip
    size = np.array(pred_sitk.GetSize())
    spacing_np = np.array(pred_sitk.GetSpacing())
    origin_np = np.array(origin_ras)

    print("• Original size :", size)
    print("• Original spacing :", spacing_np)

    new_origin = origin_np.copy()

    # Formule physique pour inversion d’axe
    new_origin[0] = -origin_np[0] - spacing_np[0] * (size[0] - 1)
    new_origin[1] = -origin_np[1] - spacing_np[1] * (size[1] - 1)

    pred_lps.SetOrigin(tuple(new_origin))

    print("• New origin (LPS):", pred_lps.GetOrigin())

    # Direction identité → standard LPS
    pred_lps.SetDirection(direction_ras)

    print("• Direction (LPS):", pred_lps.GetDirection())
    print("• Size after flip:", pred_lps.GetSize())

    # =========================================================================
    # 3) SI PAS D’IMAGE DE RÉFÉRENCE → FIN
    # =========================================================================

    if original_image_path is None or not os.path.exists(original_image_path):
        print("\n⚠️ No reference image provided — returning LPS prediction only")
        print("=" * 80)
        return pred_lps

    # =========================================================================
    # 4) CHARGEMENT IMAGE ORIGINALE (LPS)
    # =========================================================================

    print("\n[STEP 3] LOAD ORIGINAL IMAGE AS REFERENCE")
    print("📂 Path:", original_image_path)

    ref_img = sitk.ReadImage(original_image_path)

    print("• Dimension :", ref_img.GetDimension())
    print("• Size      :", ref_img.GetSize())
    print("• Spacing   :", ref_img.GetSpacing())
    print("• Origin    :", ref_img.GetOrigin())
    print("• Direction :", ref_img.GetDirection())

    # --- Gestion PET/CT 4D
    if ref_img.GetDimension() == 4:
        print("\n👉 4D image detected — extracting CT channel")

        size4 = list(ref_img.GetSize())
        size3 = [size4[0], size4[1], size4[2], 0]
        index_ct = [0, 0, 0, 1]

        ref_img = sitk.Extract(ref_img, size3, index_ct)

        print("• Extracted 3D size :", ref_img.GetSize())

    # =========================================================================
    # 5) RESAMPLING VERS L’IMAGE ORIGINALE
    # =========================================================================

    print("\n[STEP 4] RESAMPLE TO ORIGINAL IMAGE SPACE")

    print("👉 Prediction before resampling:")
    print("   Size     :", pred_lps.GetSize())
    print("   Spacing  :", pred_lps.GetSpacing())
    print("   Origin   :", pred_lps.GetOrigin())
    print("   Direction:", pred_lps.GetDirection())

    print("\n👉 Reference image:")
    print("   Size     :", ref_img.GetSize())
    print("   Spacing  :", ref_img.GetSpacing())
    print("   Origin   :", ref_img.GetOrigin())
    print("   Direction:", ref_img.GetDirection())

    resampler = sitk.ResampleImageFilter()

    # Référence = image patient
    resampler.SetReferenceImage(ref_img)

    # Nearest neighbor indispensable pour masques
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    # Valeur par défaut hors champ
    resampler.SetDefaultPixelValue(0)

    resampler.SetOutputPixelType(sitk.sitkUInt8)

    pred_resampled = resampler.Execute(pred_lps)

    print("\n✅ RESAMPLING DONE")

    print("👉 Final prediction:")
    print("   Size     :", pred_resampled.GetSize())
    print("   Spacing  :", pred_resampled.GetSpacing())
    print("   Origin   :", pred_resampled.GetOrigin())
    print("   Direction:", pred_resampled.GetDirection())

    print("\n🏁 END — SUCCESS")
    print("=" * 80)

    return pred_resampled


def main():
    args = parser.parse_args()
    args.test_mode = True
    args.inference_only = args.inference_only # Flag de type d'inférence
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
            val_labels = batch["label"].to(device) if not args.inference_only else None # If inference only, we do not need 'label's, it even doesn't exist...
            
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

            print("Debug : Before inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(val_inputs,
                                                   (96, 96, 96), #remember this line if an error in inference occurs with img size changed
                                                   4,
                                                   model,
                                                   overlap=args.infer_overlap)
            print("Debug : After inference on case {}".format(img_name))
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)


            if not args.inference_only:
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
            else:
                mean_dice = None
                print(f"ImageName: {img_name} (Inference only)")

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
            prediction_sitk = create_prediction_sitk_from_monai_debug(
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
        if not args.inference_only:
            print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

if __name__ == '__main__':

    main()