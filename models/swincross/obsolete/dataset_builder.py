#Construire correctement le dataset, quand besoin

import os
import json
import shutil
import numpy as np
import nibabel as nib
import nibabel.processing as nip  # <--- Nouveau module pour le redimensionnement
import random

# =================CONFIGURATION=================
INPUT_FOLDER = "Task_1_15examples"
OUTPUT_FOLDER = "Dataset_Final_SwinCross"
JSON_FILENAME = "dataset_swincross.json"

VAL_SPLIT = 0.2
SEED = 42
# ===============================================

def merge_pet_ct_resampled(pet_path, ct_path, output_path):
    """
    Charge PET et CT.
    Si le PET n'a pas la même taille que le CT, il est rééchantillonné
    pour coller parfaitement à la géométrie du CT.
    Sauvegarde ensuite le fichier fusionné (4D).
    """
    pet_img = nib.load(pet_path)
    ct_img = nib.load(ct_path)
    
    # Vérification des dimensions
    if pet_img.shape != ct_img.shape:
        print(f"🔄 Redimensionnement du PET en cours pour {os.path.basename(pet_path)}...")
        print(f"   (Avant: {pet_img.shape} -> Après: {ct_img.shape})")
        
        # C'est ici que la magie opère : on projette le PET sur la grille du CT
        # order=1 signifie interpolation linéaire (standard pour les images)
        pet_img = nip.resample_from_to(pet_img, ct_img, order=1)

    pet_data = pet_img.get_fdata()
    ct_data = ct_img.get_fdata()

    # Vérification de sécurité finale
    if pet_data.shape != ct_data.shape:
        print(f"❌ Erreur fatale : Le redimensionnement a échoué.")
        return False

    # Fusion (Canal 0 = PET, Canal 1 = CT)
    merged_data = np.stack([pet_data, ct_data], axis=-1)
    
    # On utilise l'affine du CT car c'est lui notre référence maintenant
    merged_img = nib.Nifti1Image(merged_data, ct_img.affine, ct_img.header)
    nib.save(merged_img, output_path)
    return True

def main():
    random.seed(SEED)
    
    dirs = {
        "train_img": os.path.join(OUTPUT_FOLDER, "imagesTr"),
        "train_lbl": os.path.join(OUTPUT_FOLDER, "labelsTr"),
        "val_img": os.path.join(OUTPUT_FOLDER, "imagesTs"),
        "val_lbl": os.path.join(OUTPUT_FOLDER, "labelsTs")
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Erreur : Le dossier '{INPUT_FOLDER}' n'existe pas.")
        return

    patient_folders = [f for f in os.listdir(INPUT_FOLDER) if os.path.isdir(os.path.join(INPUT_FOLDER, f))]
    random.shuffle(patient_folders)
    
    num_val = int(len(patient_folders) * VAL_SPLIT)
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
            pat_path = os.path.join(INPUT_FOLDER, pat_id)
            
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
                # 1. Fusionner PET + CT (avec redimensionnement si besoin)
                if merge_pet_ct_resampled(path_pt, path_ct, dest_img):
                    
                    # --- VÉRIFICATION IMPORTANTE POUR LE MASQUE ---
                    # Parfois le masque aussi n'a pas la même taille (rare mais possible)
                    # On le vérifie par rapport au CT
                    gt_img = nib.load(path_gt)
                    ct_ref = nib.load(path_ct) # On recharge pour être sûr d'avoir la réf
                    
                    if gt_img.shape != ct_ref.shape:
                        print(f"🔄 Redimensionnement du MASQUE pour {pat_id}...")
                        # order=0 est CRUCIAL pour les masques (Nearest Neighbor)
                        # pour garder des valeurs 0 ou 1 strictes (pas de 0.5)
                        gt_img = nip.resample_from_to(gt_img, ct_ref, order=0)
                        nib.save(gt_img, dest_lbl)
                    else:
                        shutil.copy(path_gt, dest_lbl)
                    
                    # 3. Mettre à jour le JSON
                    json_data[group_name].append({
                        "image": json_key_img,
                        "label": json_key_lbl
                    })
                    print(f"✅ {pat_id} OK.")
            except Exception as e:
                print(f"❌ Erreur critique sur {pat_id}: {e}")

    # Sauvegarde JSON
    json_path = os.path.join(OUTPUT_FOLDER, JSON_FILENAME)
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)

    print(f"\n✨ Terminé ! Dataset prêt dans : {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()