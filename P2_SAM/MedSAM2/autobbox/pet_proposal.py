"""
pet_proposals.py
================
Generate bounding-box prompts from PET volumes using intensity-based proposals.
Designed to integrate with slicer.py-style outputs.
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import label

# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def load_nifti_pet(path: str):
    img = nib.load(path)
    pet = np.asarray(img.get_fdata(), dtype=np.float32).transpose(2, 1, 0)
    return pet, img.affine


# ---------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------

def normalize_pet(pet: np.ndarray, method="zscore"):
    if method == "zscore":
        mean = np.mean(pet)
        std = np.std(pet) + 1e-6
        return (pet - mean) / std

    elif method == "percentile":
        p_low, p_high = np.percentile(pet, (1, 99))
        pet = np.clip(pet, p_low, p_high)
        return (pet - p_low) / (p_high - p_low + 1e-6)

    else:
        raise ValueError("Unknown normalization method")


# ---------------------------------------------------------------------
# Candidate mask generation
# ---------------------------------------------------------------------

def generate_candidate_mask(
    pet: np.ndarray,
    percentile=99.5,
    min_threshold=None,
):
    """
    High-recall thresholding based on PET intensity.
    """
    thr = np.percentile(pet, percentile)

    if min_threshold is not None:
        thr = max(thr, min_threshold)

    mask = pet > thr
    return mask


# ---------------------------------------------------------------------
# Component filtering
# ---------------------------------------------------------------------

def compute_shape_features(component_mask):
    coords = np.array(np.where(component_mask))
    if coords.shape[1] == 0:
        return None

    z, y, x = coords
    dz = z.max() - z.min() + 1
    dy = y.max() - y.min() + 1
    dx = x.max() - x.min() + 1

    volume = len(z)
    bbox_volume = dz * dy * dx

    compactness = volume / (bbox_volume + 1e-6)

    elongation = max(dx, dy, dz) / (min(dx, dy, dz) + 1e-6)

    return {
        "volume": volume,
        "compactness": compactness,
        "elongation": elongation,
        "bbox": (dz, dy, dx),
    }


def filter_component(features,
                     min_volume=50,
                     max_elongation=10,
                     min_compactness=0.05):
    if features is None:
        return False

    if features["volume"] < min_volume:
        return False

    if features["elongation"] > max_elongation:
        return False

    if features["compactness"] < min_compactness:
        return False

    return True


# ---------------------------------------------------------------------
# Main proposal extraction
# ---------------------------------------------------------------------

def extract_pet_bboxes(
    pet: np.ndarray,
    percentile=99.5,
    min_volume=50,
    padding=2,
):
    """
    Returns list of bounding boxes in same format as slicer.py
    """

    pet_norm = normalize_pet(pet, method="zscore")
    candidate_mask = generate_candidate_mask(pet_norm, percentile=percentile)

    labeled, num = label(candidate_mask)

    D, H, W = pet.shape
    results = []

    for comp_id in range(1, num + 1):
        comp = (labeled == comp_id)

        features = compute_shape_features(comp)
        if not filter_component(features, min_volume=min_volume):
            continue

        z_idx, y_idx, x_idx = np.where(comp)

        z0 = max(0, z_idx.min() - padding)
        z1 = min(D, z_idx.max() + 1 + padding)
        y0 = max(0, y_idx.min() - padding)
        y1 = min(H, y_idx.max() + 1 + padding)
        x0 = max(0, x_idx.min() - padding)
        x1 = min(W, x_idx.max() + 1 + padding)

        # central slice
        areas = np.array([comp[z].sum() for z in range(z0, z1)])
        z_mid = z0 + int(np.argmax(areas))

        ys_mid, xs_mid = np.where(comp[z_mid])

        bbox_2d = np.array([
            xs_mid.min(), ys_mid.min(),
            xs_mid.max(), ys_mid.max()
        ], dtype=np.int32)

        results.append({
            "component_id": comp_id,
            "voxel_count": int(features["volume"]),
            "z_mid": z_mid,
            "bbox_voxel": {"z": (z0, z1), "y": (y0, y1), "x": (x0, x1)},
            "bbox_2d": bbox_2d,
        })

    # sort by size
    results.sort(key=lambda x: x["voxel_count"], reverse=True)

    return results
