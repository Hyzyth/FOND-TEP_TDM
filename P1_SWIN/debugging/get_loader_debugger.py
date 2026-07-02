import os
import torch
import numpy as np
import nibabel as nib
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    CropForegroundd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
    RandScaleIntensityd, RandShiftIntensityd, Lambdad
)

# ================= CONFIGURATION SIMULÉE =================
class MockArgs:
    data_dir = "Dataset_Final_SwinCross_SITK"  # Notre dossier de sortie SITK
    json_list = "dataset_swincross.json"
    
    # Résolution cible pour le réseau (ex: 1.5mm isotrope)
    space_x = 1.5
    space_y = 1.5
    space_z = 1.5
    
    # Taille du patch (Ce que le réseau voit)
    roi_x = 96
    roi_y = 96
    roi_z = 96
    
    # Probas d'augmentation (pour tester si elles déforment trop)
    RandFlipd_prob = 0.5
    RandRotate90d_prob = 0.5
    RandScaleIntensityd_prob = 0.1
    RandShiftIntensityd_prob = 0.1

args = MockArgs()
OUTPUT_DEBUG_DIR = "Debug_Loader_Output"
# =========================================================

def save_nifti_channel(tensor_data, affine, filename):
    """Sauvegarde un canal spécifique du tenseur en NIfTI via Nibabel avec correction de type"""
    
    # 1. Gestion de l'affine (Position dans l'espace)
    if isinstance(affine, torch.Tensor):
        affine = affine.numpy()
    if affine is None:
        affine = np.eye(4)

    # 2. Correction du Type de Données (Le coeur du problème)
    # tensor_data est un tableau Numpy à ce stade.
    
    if tensor_data.dtype == np.int64:
        # C'est un masque (Label) en int64 -> On le passe en int16 (suffisant pour 0,1,2)
        print(f"      [Info] Conversion int64 -> int16 pour {filename}")
        tensor_data = tensor_data.astype(np.int16)
        
    elif tensor_data.dtype == np.float64:
        # C'est une image en float64 -> On la passe en float32 (standard médical)
        print(f"      [Info] Conversion float64 -> float32 pour {filename}")
        tensor_data = tensor_data.astype(np.float32)

    # 3. Création et Sauvegarde
    nifti_img = nib.Nifti1Image(tensor_data, affine)
    nib.save(nifti_img, filename)
    print(f"   -> Sauvegardé : {filename}")


def main():
    os.makedirs(OUTPUT_DEBUG_DIR, exist_ok=True)
    
    # 1. Chargement du JSON pour choper un fichier
    import json
    json_path = os.path.join(args.data_dir, args.json_list)
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    
    # On prend le premier patient du training
    sample_data = data_dict['training'][0]
    # On corrige les chemins relatifs
    sample_data['image'] = os.path.join(args.data_dir, sample_data['image'])
    sample_data['label'] = os.path.join(args.data_dir, sample_data['label'])
    
    print(f"🔵 Test sur le patient : {sample_data['image']}")

    # 2. Définition EXACTE de votre Transform de Train
    train_transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        # Sécurisation des labels [0, 1, 2]
        Lambdad(keys="label", func=lambda x: torch.clamp(x, min=0, max=2).to(x.dtype)),

        Orientationd(keys=["image", "label"], axcodes="RAS"),
        
        # C'est ici que le redimensionnement final se fait
        Spacingd(
            keys=["image", "label"],
            pixdim=(args.space_x, args.space_y, args.space_z),
            mode=("bilinear", "nearest")
        ),
        
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # Création des patches (Cubes)
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1, neg=1, num_samples=2, # On demande 2 crops par patient
            image_key="image", image_threshold=0,
        ),

        # Augmentations
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),

        # Nettoyage final label
        Lambdad(keys="label", func=lambda x: torch.clamp(torch.round(x), min=0, max=2).to(torch.int64)),
    ])

    # 3. Exécution de la transfo
    # MONAI Datasets appliquent les transfos. Simulons-le manuellement.
    print("⏳ Application des transformations (ça peut prendre quelques secondes)...")
    
    # Note: RandCropByPosNegLabeld retourne une LISTE de crops si num_samples > 1
    # Donc output_crops sera une liste de dictionnaires
    output_crops = train_transform(sample_data)

    # 4. Sauvegarde pour vérification visuelle
    print(f"\n🟢 Transformations terminées. {len(output_crops)} patches générés.")
    
    for i, crop in enumerate(output_crops):
        print(f"\n--- Traitement du Patch {i+1} ---")
        img_tensor = crop['image']  # (C, H, W, D) -> C=2 (PET, CT)
        lbl_tensor = crop['label']  # (C, H, W, D) -> C=1
        
        # Récupération de l'affine pour que Slicer comprenne la taille physique
        # (Souvent stocké dans les métadonnées par LoadImage)
        affine = crop['image_meta_dict']['affine'] if 'image_meta_dict' in crop else None

        # --- Sauvegarde Canal 0 : PET ---
        pet_data = img_tensor[0, :, :, :].numpy()
        save_nifti_channel(pet_data, affine, os.path.join(OUTPUT_DEBUG_DIR, f"Patch_{i}_PET.nii.gz"))
        
        # --- Sauvegarde Canal 1 : CT ---
        ct_data = img_tensor[1, :, :, :].numpy()
        save_nifti_channel(ct_data, affine, os.path.join(OUTPUT_DEBUG_DIR, f"Patch_{i}_CT.nii.gz"))
        
        # --- Sauvegarde Label ---
        lbl_data = lbl_tensor[0, :, :, :].numpy()
        save_nifti_channel(lbl_data, affine, os.path.join(OUTPUT_DEBUG_DIR, f"Patch_{i}_GT.nii.gz"))

    print(f"\n✨ Terminé ! Ouvrez le dossier '{OUTPUT_DEBUG_DIR}' dans ITK-SNAP.")
    print("👉 Superposez Patch_0_CT.nii.gz et Patch_0_GT.nii.gz pour vérifier l'alignement.")
    print("👉 Vérifiez que le Patch_0_PET.nii.gz correspond bien anatomiquement.")

if __name__ == "__main__":
    main()
