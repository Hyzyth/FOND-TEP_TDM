import SimpleITK as sitk
import numpy as np
import torch
import os
from pathlib import Path
import sys
import copy
from monai import transforms
from monai.transforms import Invertd
from scipy.ndimage import center_of_mass

# Safeguard
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
# Import de tes anciennes fonctions de bricolage
from nii_version.test import create_prediction_sitk_with_metadata
from test_v2 import create_prediction_sitk_from_monai_debug
from test_old import create_prediction_sitk_with_metadata as create_prediction_sitk_with_metadata_old

# ============================================================
# 🔧 PARAMÈTRES — à adapter
# ============================================================

#label_path = "./Dataset_Final_SwinCross_SITK/labelsTs/CHUS-007_gt.nii.gz"
label_path = "./Dataset_Final_SwinCross_SITK/labelsTs/HGJ-013_gt.nii.gz"
#label_path = "./Dataset_Final_SwinCross_SITK/labelsTs/CHUS-007_gt.nii.gz"
output_dir = "./data/outs/masks_tests/"

space_x = 1.0
space_y = 1.0
space_z = 1.0

# ============================================================
# 🔹 CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 🔹 1. MONAI INFER TRANSFORM (LABEL VERSION)
# ============================================================

infer_label_transform = transforms.Compose([
    transforms.LoadImaged(keys=["label"]),
    transforms.EnsureChannelFirstd(keys=["label"]),
    transforms.Orientationd(keys=["label"], axcodes="RAS"),
    transforms.Spacingd(
        keys=["label"],
        pixdim=(space_x, space_y, space_z),
        mode=("nearest",)
    ),
    transforms.CropForegroundd(keys=["label"], source_key="label"),
    
    # CRITIQUE POUR LA V4 : On force le tracking des métadonnées
    transforms.EnsureTyped(keys=["label"], track_meta=True),
])

# ============================================================
# 🔹 2. UTILITAIRES
# ============================================================

def dice_score(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    return 2 * inter / (a.sum() + b.sum() + 1e-8)

def center_of_mass_phys(mask_arr, sitk_img):
    if np.sum(mask_arr > 0) == 0:
        return (0.0, 0.0, 0.0) # Sécurité si le masque est vide
    com_vox = center_of_mass(mask_arr > 0)
    return sitk_img.TransformContinuousIndexToPhysicalPoint(com_vox[::-1])

def print_metrics(name, ref_img, pred_img):
    ref_arr = sitk.GetArrayFromImage(ref_img)
    pred_arr = sitk.GetArrayFromImage(pred_img)

    dice = dice_score(ref_arr, pred_arr)

    vol_ref = np.sum(ref_arr > 0)
    vol_pred = np.sum(pred_arr > 0)

    com_ref = center_of_mass_phys(ref_arr, ref_img)
    com_pred = center_of_mass_phys(pred_arr, pred_img)

    com_dist = np.linalg.norm(np.array(com_ref) - np.array(com_pred))

    print(f"\n================ {name} ================")
    print(f"Dice               : {dice:.6f}")
    print(f"Volume diff voxels : {vol_pred - vol_ref}")
    print(f"COM ref (mm)       : {com_ref}")
    print(f"COM pred (mm)      : {com_pred}")
    print(f"COM distance (mm)  : {com_dist:.6f}")

# ============================================================
# 🔹 3. LOAD ORIGINAL LABEL (REFERENCE)
# ============================================================

ref_img = sitk.ReadImage(label_path)
ref_arr = sitk.GetArrayFromImage(ref_img)

print("Original label loaded")
print("Size:", ref_img.GetSize())
print("Spacing:", ref_img.GetSpacing())
print("Origin:", ref_img.GetOrigin())

# ============================================================
# 🔹 4. APPLY MONAI TRANSFORM → RAS
# ============================================================

data = {"label": label_path}
data = infer_label_transform(data)

label_tensor = data["label"]
meta = data["label"].meta

label_ras = label_tensor.numpy()[0]  # remove channel

print("\nAfter MONAI preprocessing (RAS)")
print("Shape:", label_ras.shape)

# ============================================================
# 🔹 5. LES 4 FONCTIONS CANDIDATES
# ============================================================

def func_v1(pred_ras, meta, original_path):
    return create_prediction_sitk_with_metadata_old(
        prediction_np=pred_ras,
        original_image_path=original_path,
        spacing=(space_x, space_y, space_z),
        monai_affine=meta["affine"]
    )

def func_v2(pred_ras, meta, original_path):
    return create_prediction_sitk_with_metadata(
        prediction_np=pred_ras,
        original_image_path=original_path,
        spacing=(space_x, space_y, space_z),
        monai_affine=meta["affine"]
    )

def func_v3(pred_ras, meta, original_path):
    return create_prediction_sitk_from_monai_debug(
        prediction_np=pred_ras,
        original_image_path=original_path,
        spacing=(space_x, space_y, space_z),
        monai_affine=meta["affine"]
    )

# ⭐ LA NOUVELLE MÉTHODE DE RÉFÉRENCE (MONAI INVERTD + SITK) ⭐
def func_v4(monai_dict_data, ref_sitk_img):
    """
    Cette fonction prend le dictionnaire entier car Invertd 
    a besoin de lire l'historique des transformations.
    """
    # 1. On fait une copie profonde pour ne pas altérer l'objet original
    test_data = copy.deepcopy(monai_dict_data)
    
    # 2. On simule la prédiction du modèle (ici, c'est juste le label recadré)
    test_data["pred"] = test_data["label"].clone()
    
    # 3. Configuration de l'Invertd
    invertd = Invertd(
        keys="pred",
        transform=infer_label_transform, # Le pipeline qu'on veut rembobiner
        orig_keys="label",
        nearest_interp=True,
        to_tensor=True
    )
    
    # 4. Application de l'inversion
    inv_data = invertd(test_data)
    
    # 5. Extraction en Numpy et Transposition SITK (Z,Y,X)
    pred_np_final = inv_data["pred"][0].cpu().numpy().astype(np.uint8)
    pred_np_sitk = pred_np_final.transpose(2, 1, 0)
    
    # 6. Création SITK et copie des métadonnées de la référence
    prediction_sitk = sitk.GetImageFromArray(pred_np_sitk)
    prediction_sitk.CopyInformation(ref_sitk_img)
    
    return prediction_sitk

# ============================================================
# 🔹 6. APPLY FUNCTIONS
# ============================================================

img_v1 = func_v1(label_ras, meta, label_path)
img_v2 = func_v2(label_ras, meta, label_path)
img_v3 = func_v3(label_ras, meta, label_path)
img_v4 = func_v4(data, ref_img) # Note: on passe 'data' et 'ref_img' ici

# ============================================================
# 🔹 7. SAVE OUTPUT IMAGES
# ============================================================

path_v1 = os.path.join(output_dir, "mask_v1_old.nii.gz")
path_v2 = os.path.join(output_dir, "mask_v2_ras.nii.gz")
path_v3 = os.path.join(output_dir, "mask_v3_ident.nii.gz")
path_v4 = os.path.join(output_dir, "mask_v4_invertd.nii.gz")

sitk.WriteImage(img_v1, path_v1)
sitk.WriteImage(img_v2, path_v2)
sitk.WriteImage(img_v3, path_v3)
sitk.WriteImage(img_v4, path_v4)

print("\nImages saved in:", output_dir)
print(" -", path_v1)
print(" -", path_v2)
print(" -", path_v3)
print(" -", path_v4)

# ============================================================
# 🔹 8. METRICS
# ============================================================

print_metrics("Function V1 (The Hack)", ref_img, img_v1)
print_metrics("Function V2 (RAS forced)", ref_img, img_v2)
print_metrics("Function V3 (Identity forced)", ref_img, img_v3)
print_metrics("Function V4 (MONAI Invertd + SITK)", ref_img, img_v4)