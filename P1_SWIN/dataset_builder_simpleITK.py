import os
import json
import shutil
import SimpleITK as sitk
import random
import argparse

# =================CONFIGURATION=================
# GITHUB dirs par défaut
INPUT_FOLDER = "/data/santiago/HECKTOR_data/2025"
OUTPUT_FOLDER = "/data/ethan/PP_hecktor_dataset_SwinCross"
JSON_FILENAME = "dataset_swincross_hecktor.json"

VAL_SPLIT = 0.2
SEED = 42
# ===============================================

def resample_image_to_reference(image, reference, is_label=False, pixel_type=sitk.sitkFloat32):
    """
    Fonction utilitaire SimpleITK pour projeter une image sur la grille d'une autre.
    """
    # On prépare le filtre de rééchantillonnage
    resampler = sitk.ResampleImageFilter()
    
    # On configure la grille cible (celle du CT de référence)
    resampler.SetReferenceImage(reference)
    
    # Choix de l'interpolation
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSplineResamplerOrder3) 
    # changed sitk.Linear to sitk.Bspline third order for more precision 
    
    # Pas de transformation (identité)
    resampler.SetTransform(sitk.Transform())

    # Ensure output is 4-byte float for PET and 2-byte unsigned int for labels
    resampler.SetOutputPixelType(pixel_type)
    
    # background value is default put to 0.0 in the resampler class of SimpleITK

    # Exécution
    return resampler.Execute(image)

def merge_pet_ct_sitk(pet_path, ct_path, output_path):
    """
    Charge PET et CT avec SimpleITK.
    Rééchantillonne le PET sur le CT.
    Sauvegarde en 4D.
    """
    # 1. Chargement
    pet_img = sitk.ReadImage(pet_path)
    ct_img = sitk.ReadImage(ct_path)
    
    # 2. Vérification des dimensions et rééchantillonnage
    # SimpleITK compare les tailles (Size) et l'espacement (Spacing)
    if pet_img.GetSize() != ct_img.GetSize() or pet_img.GetSpacing() != ct_img.GetSpacing():
        print(f"🔄 SITK: Redimensionnement du PET ({pet_img.GetSize()} -> {ct_img.GetSize()})...")
        pet_img = resample_image_to_reference(pet_img, ct_img, is_label=False)

    # 3. Fusion en 4D
    # SimpleITK a une fonction magique pour ça : JoinSeries
    # Attention à l'ordre : [PET, CT] -> PET canal 0, CT canal 1
    merged_img = sitk.JoinSeries([pet_img, ct_img])
    
    # 4. Sauvegarde
    sitk.WriteImage(merged_img, output_path)
    return True

def process_gt_sitk(gt_path, ct_path, output_path):
    """
    Gère le redimensionnement du masque (GT) si nécessaire
    """
    gt_img = sitk.ReadImage(gt_path)
    ct_img = sitk.ReadImage(ct_path)
    
    if gt_img.GetSize() != ct_img.GetSize():
        print(f"🔄 SITK: Correction du MASQUE...")
        # Note le is_label=True pour utiliser NearestNeighbor
        gt_img = resample_image_to_reference(gt_img, ct_img, is_label=True, pixel_type=sitk.sitkUInt8)
        sitk.WriteImage(gt_img, output_path)
    else:
        # Si c'est déjà bon, on copie juste le fichier (plus rapide)
        shutil.copy(gt_path, output_path)

def main():
    
    parser = argparse.ArgumentParser(description="Build dataset for SwinCross using SimpleITK.")
    parser.add_argument("--input_folder", type=str, default=INPUT_FOLDER, help="Path to the input folder containing patient data.")
    parser.add_argument("--output_folder", type=str, default=OUTPUT_FOLDER, help="Path to the output folder where processed data will be saved.")
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT, help="Fraction of data to use for validation.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)

    dirs = {
        "train_img": os.path.join(args.output_folder, "imagesTr"),
        "train_lbl": os.path.join(args.output_folder, "labelsTr"),
        "val_img": os.path.join(args.output_folder, "imagesTs"),
        "val_lbl": os.path.join(args.output_folder, "labelsTs")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(args.input_folder):
        print(f"❌ Erreur : Le dossier '{args.input_folder}' n'existe pas.")
        return

    patient_folders = [f for f in os.listdir(args.input_folder) if os.path.isdir(os.path.join(args.input_folder, f))]
    random.shuffle(patient_folders)
    
    num_val = int(len(patient_folders) * args.val_split)
    val_patients = patient_folders[:num_val]
    train_patients = patient_folders[num_val:]
    
    print(f"Total patients: {len(patient_folders)} | Train: {len(train_patients)} | Val: {len(val_patients)}")

    json_data = {
        "description": "Head and Neck Tumor Segmentation",
        "labels": {"0": "background", "1": "tumor"},
        "tensorImageSize": "4D",
        "modality": {"0": "PET", "1": "CT"},
        "training": [],
        "validation": []
    }

    for group_name, patients_list in [("training", train_patients), ("validation", val_patients)]:
        print(f"\nTraitement du groupe : {group_name.upper()}...")
        
        for pat_id in patients_list:
            pat_path = os.path.join(args.input_folder, pat_id)
            
            # Définition des noms de fichiers
            file_gt = f"{pat_id}.nii.gz"
            file_ct = f"{pat_id}__CT.nii.gz"
            file_pt = f"{pat_id}__PT.nii.gz"

            path_gt = os.path.join(pat_path, file_gt)
            path_ct = os.path.join(pat_path, file_ct)
            path_pt = os.path.join(pat_path, file_pt)
            
            if not (os.path.exists(path_gt) and os.path.exists(path_ct) and os.path.exists(path_pt)):
                print(f"⚠️ Fichiers manquants pour {pat_id}, on passe.")
                continue

            out_name_img = f"{pat_id}_petct.nii.gz"
            out_name_lbl = f"{pat_id}_gt.nii.gz"
            
            if group_name == "training":
                dest_img = os.path.join(dirs["train_img"], out_name_img)
                dest_lbl = os.path.join(dirs["train_lbl"], out_name_lbl)
                json_key_img = f"imagesTr/{out_name_img}"
                json_key_lbl = f"labelsTr/{out_name_lbl}"
            else:
                dest_img = os.path.join(dirs["val_img"], out_name_img)
                dest_lbl = os.path.join(dirs["val_lbl"], out_name_lbl)
                json_key_img = f"imagesTs/{out_name_img}"
                json_key_lbl = f"labelsTs/{out_name_lbl}"

            try:
                # 1. Fusionner PET + CT avec SITK
                if merge_pet_ct_sitk(path_pt, path_ct, dest_img):
                    
                    # 2. Gérer le masque avec SITK
                    process_gt_sitk(path_gt, path_ct, dest_lbl)
                    
                    # 3. JSON
                    json_data[group_name].append({
                        "image": json_key_img,
                        "label": json_key_lbl
                    })
                    print(f"✅ {pat_id} OK.")
            except Exception as e:
                print(f"❌ Erreur critique sur {pat_id}: {e}")

    # Sauvegarde JSON
    json_path = os.path.join(args.output_folder, JSON_FILENAME)
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"\n✨ Terminé ! Dataset prêt dans : {args.output_folder}")
if __name__ == "__main__":
    main()
