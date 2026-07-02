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
import SimpleITK as sitk
from scipy import ndimage
from monai.inferers.utils import sliding_window_inference # modification : changed import path as per VSCode suggestion monai.inferers -> monai.inferers.utils
from monai.transforms import Invertd
from monai.data import decollate_batch, MetaTensor
import nibabel as nib
import argparse
import warnings

# Safeguard
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from P1_SWIN.obsolete.nii_version.data_utils import get_loader # Modification: Change the reference to the .py in the root with the code changes. Original version stays in utils folder
from P1_SWIN.obsolete.nii_version.trainer import dice

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

# --- 1. PREPARATION DES PREDICTIONS (Tenseur pour MONAI) ---
            val_outputs_tensor = torch.softmax(val_outputs, dim=1)
            val_outputs_tensor = torch.argmax(val_outputs_tensor, dim=1, keepdim=True).cpu()

            # On recrée un MetaTensor en lui greffant l'historique (meta) de l'image originale !
            if hasattr(val_inputs, "meta"):
                val_outputs_tensor = MetaTensor(val_outputs_tensor, meta=val_inputs.meta)
            
            # Version Numpy pour ton calcul de Dice classique
            val_outputs_np = val_outputs_tensor[:, 0, ...].numpy().astype(np.uint8)

            # --- 2. CALCUL DU DICE ET DU VOLUME ---
            if not args.inference_only:
                val_labels_np = val_labels.cpu().numpy()[:, 0, :, :, :]
                tumor_volume = np.sum(val_labels_np)
                print("number of tumors", count_objects(val_labels_np[0,:,:,:]))

                dice_list_sub = []
                for i in range(1, 3): # Calcule pour Tumeur (1) ET Nodules (2)
                    if np.sum(val_labels_np[0] == i) > 0 or np.sum(val_outputs_np[0] == i) > 0:
                        organ_Dice = dice(val_outputs_np[0] == i, val_labels_np[0] == i)
                        dice_list_sub.append(organ_Dice)
                
                mean_dice = np.mean(dice_list_sub) if len(dice_list_sub) > 0 else 0.0
                print("ImageName, Mean Organ Dice, and Tumor Volume: {}, {}, {}".format(img_name, mean_dice, tumor_volume))
                dice_list_case.append(mean_dice)
            else:
                mean_dice = None
                print(f"ImageName: {img_name} (Inference only)")

            # --- 3. SAUVEGARDE PARFAITE AVEC MONAI INVERTD + SIMPLEITK ---
            # On injecte la prédiction dans le dictionnaire du batch
            batch["pred"] = val_outputs_tensor

            # Configuration de l'inverseur MONAI
            invertd = Invertd(
                keys="pred",
                transform=val_loader.dataset.transform, # Récupère les transfos du loader
                orig_keys="image", # Calque l'inversion sur "image"
                nearest_interp=True, # Pas d'interpolation floue pour les classes
                to_tensor=True
            )

            # Application de l'inversion
            batch_inverted = [invertd(item) for item in decollate_batch(batch)]
            
            for item_inv in batch_inverted:
                pred_inverted = item_inv["pred"]
                
                # 3.A: Extraction en Numpy et transposition Z,Y,X pour SimpleITK
                pred_np_final = pred_inverted[0].cpu().numpy().astype(np.uint8)
                pred_np_sitk = pred_np_final.transpose(2, 1, 0)
                
                # 3.B: Création de l'image SimpleITK
                prediction_sitk = sitk.GetImageFromArray(pred_np_sitk)
                
                # 3.C: Récupération du fichier d'origine exact pour copier les infos LPS
                # On lit les métadonnées directement dans le MetaTensor
                img_tensor = item_inv["image"]
                
                if hasattr(img_tensor, "meta") and "filename_or_obj" in img_tensor.meta:
                    original_image_path = img_tensor.meta["filename_or_obj"]
                elif "image_meta_dict" in item_inv: # Fallback pour les vieilles versions
                    original_image_path = item_inv["image_meta_dict"]["filename_or_obj"]
                else:
                    raise ValueError("Impossible de trouver le chemin du fichier source.")

                if isinstance(original_image_path, (list, tuple, np.ndarray)):
                    original_image_path = original_image_path[0]
                
                # Petit nettoyage si le chemin vient sous forme de tenseur ou de chaîne complexe
                if isinstance(original_image_path, torch.Tensor) or hasattr(original_image_path, "item"):
                    # Si c'est un objet interne de MONAI, on tente de le nettoyer
                    original_image_path = str(original_image_path)
                original_sitk = sitk.ReadImage(original_image_path)
                
                # Gestion des images 4D (ex: PET/CT) vers 3D (le masque)
                if original_sitk.GetDimension() == 4:
                    size_4d = list(original_sitk.GetSize())
                    original_sitk_3d = sitk.Extract(original_sitk, [size_4d[0], size_4d[1], size_4d[2], 0], [0, 0, 0, 0])
                    prediction_sitk.CopyInformation(original_sitk_3d)
                else:
                    prediction_sitk.CopyInformation(original_sitk)

                # 3.D: Sauvegarde sur le disque
                filename_output_path = os.path.join(
                    test_output_dir, 
                    f'{img_prefix.replace("_petct", "")}_dsc{round(mean_dice, 2) if mean_dice is not None else "NA"}_Pred.nii.gz'
                )
                
                sitk.WriteImage(prediction_sitk, filename_output_path)
                print(f"✅ Saved perfectly with MONAI Invertd + SimpleITK: {filename_output_path}")            
            #####################

            
        if not args.inference_only:
            print("Overall Mean Dice: {}".format(np.mean(dice_list_case)))

if __name__ == '__main__':

    main()
