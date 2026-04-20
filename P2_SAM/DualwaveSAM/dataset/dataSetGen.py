import os
import numpy as np
import nibabel as nib
import utils


def read_data(path_to_nifti, return_numpy=True):
    """
    Read a NIfTI medical image file.

    Args:
        path_to_nifti (str or Path): Path to the NIfTI file.
        return_numpy (bool): If True, returns image as NumPy array.
                             If False, returns nibabel Nifti1Image object.

    Returns:
        np.ndarray or nibabel.nifti1.Nifti1Image:
            - NumPy array if return_numpy=True
            - Raw NIfTI object otherwise
    """
    if return_numpy:
        return nib.load(str(path_to_nifti)).get_fdata()
    return nib.load(str(path_to_nifti))


# =========================
# Dataset selection
# =========================
dataname = "Hecktor"  # Options: CHEN / Hecktor


# =========================
# Collect patient file paths
# =========================
# Each entry in `paths` is assumed to be:
# [CT_path, PET_path, MASK_path]
if dataname == "CHEN":
    paths = utils.get_paths_to_patient_files(
        "/data/code/med_test_case/CA_Seg_161_crop"
    )
else:
    paths = utils.get_paths_to_patient_files(
        "/data/code/H-process/Hecktor"
    )

print(f"Total cases found: {len(paths)}")


# =========================
# Iterate over patients
# =========================
for i in range(len(paths)):
    print(f"Processing case index: {i}")

    # ---- Load modalities ----
    ct = read_data(paths[i][0])
    print(f"CT shape: {ct.shape}")

    pt = read_data(paths[i][1])
    print(f"PET shape: {pt.shape}")

    mask = read_data(paths[i][2])
    print(f"Mask shape: {mask.shape}")

    # ---- Stack CT and PET into multi-channel input ----
    # Final shape: (H, W, D, 2)
    input_data = np.stack([ct, pt], axis=-1)

    # ---- Extract patient ID from directory structure ----
    # Assumption: parent folder name = patient ID
    patient_id = os.path.basename(os.path.dirname(paths[i][0]))

    # ---- Define output path ----
    save_path = f"{dataname}/{patient_id}.npz"

    # ---- Save processed sample ----
    np.savez(save_path, input=input_data, target=mask)

    print(f"Saved patient {patient_id} data to: {save_path}")
    