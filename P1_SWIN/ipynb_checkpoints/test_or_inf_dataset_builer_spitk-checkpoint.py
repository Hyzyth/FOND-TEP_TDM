import os
import json
import argparse

# ================= CONFIG =================
DEFAULT_INPUT = "Datast001_HECKTOR_SwinCross"
DEFAULT_JSON_NAME = "dataset_test.json"
# ==========================================


def main():
    parser = argparse.ArgumentParser(description="Build TEST or PURE INFERENCE dataset JSON for SwinCross.")
    parser.add_argument("--input_folder", type=str, default=DEFAULT_INPUT,
                        help="Root folder containing 'images' and 'labels' subfolders.")
    parser.add_argument("--json_name", type=str, default=DEFAULT_JSON_NAME,
                        help="Output JSON filename.")
    parser.add_argument("--inference_only", action="store_true",
                        help="If set, create JSON without labels (pure inference mode).")

    args = parser.parse_args()

    images_dir = os.path.join(args.input_folder, "images")
    labels_dir = os.path.join(args.input_folder, "labels")

    if not os.path.exists(images_dir):
        print("❌ Dossier 'images' introuvable.")
        return

    if not args.inference_only and not os.path.exists(labels_dir):
        print("❌ Dossier 'labels' introuvable (requis pour mode test avec évaluation).")
        return

    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".nii.gz")])

    if len(image_files) == 0:
        print("❌ Aucun fichier image trouvé.")
        return

    print(f"🔎 {len(image_files)} cas trouvés.")

    json_data = {
        "description": "HECKTOR SwinCross TEST dataset",
        "labels": {"0": "background", "1": "tumor"},
        "tensorImageSize": "4D",
        "modality": {"0": "PET", "1": "CT"},
        "validation": []
    }

    for img_file in image_files:
        patient_id = img_file.replace("_petct.nii.gz", "")

        image_path = f"images/{img_file}"

        if args.inference_only:
            # ---- PURE INFERENCE ----
            json_data["validation"].append({
                "image": image_path
            })
        else:
            # ---- TEST WITH LABELS ----
            label_file = f"{patient_id}_gt.nii.gz"
            label_path = os.path.join(labels_dir, label_file)

            if not os.path.exists(label_path):
                print(f"⚠️ Label manquant pour {patient_id}, ignoré.")
                continue

            json_data["validation"].append({
                "image": image_path,
                "label": f"labels/{label_file}"
            })

    json_path = os.path.join(args.input_folder, args.json_name)

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)

    print("\n✨ JSON créé avec succès.")
    print(f"📄 Chemin : {json_path}")

    if args.inference_only:
        print("🧠 Mode : INFÉRENCE PURE (pas de labels)")
    else:
        print("📊 Mode : TEST avec évaluation (labels inclus)")


if __name__ == "__main__":
    main()