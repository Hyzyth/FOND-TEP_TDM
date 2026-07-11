import os
import sys
import numpy as np
import nibabel as nib

# CHANGE: added sys.path insert so `import utils` resolves whether this script
# is run from the project root or from inside the dataset/ subdirectory.
sys.path.insert(0, os.path.dirname(__file__))
import utils


def read_data(path_to_nifti, return_numpy=True):
    """
    Read a NIfTI medical image file.

    Args:
        path_to_nifti: path to the NIfTI file
        return_numpy:  if True returns np.ndarray, else Nifti1Image object
    """
    if return_numpy:
        return nib.load(str(path_to_nifti)).get_fdata()
    return nib.load(str(path_to_nifti))


# =========================
# Dataset selection
# =========================
dataname = "Hecktor"   # Options: CHEN / Hecktor

# =========================
# Collect patient file paths
# =========================
if dataname == "CHEN":
    paths = utils.get_paths_to_patient_files(
        "/data/code/med_test_case/CA_Seg_161_crop"
    )
else:
    paths = utils.get_paths_to_patient_files(
        "/data/code/H-process/Hecktor"
    )

print(f"Total cases found: {len(paths)}")

# CHANGE: create output directory if it does not exist, preventing silent crash
os.makedirs(dataname, exist_ok=True)

# =========================
# Iterate over patients
# =========================
for i in range(len(paths)):
    print(f"Processing case index: {i}")

    ct   = read_data(paths[i][0])
    pt   = read_data(paths[i][1])
    mask = read_data(paths[i][2])

    print(f"CT shape: {ct.shape}  PET shape: {pt.shape}  Mask shape: {mask.shape}")

    # Stack CT and PET into multi-channel input → (H, W, D, 2)
    input_data = np.stack([ct, pt], axis=-1)

    patient_id = os.path.basename(os.path.dirname(paths[i][0]))
    save_path  = os.path.join(dataname, f"{patient_id}.npz")

    np.savez(save_path, input=input_data, target=mask)
    print(f"Saved {patient_id} → {save_path}")
