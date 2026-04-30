import numpy as np
import nibabel as nib
from scipy.ndimage import label, find_objects
from nibabel.affines import apply_affine

def load_nifti_mask(path):
    img = nib.load(path)
    mask = img.get_fdata(dtype=np.int16)
    return mask, img.affine, img.header

def voxel_to_world(affine, z, y, x):
    return apply_affine(affine, (x, y, z))

def find_largest_component_for_label(mask, label_value, padding=0, affine=None):
    mask_label = (mask == label_value)
    if not np.any(mask_label):
        return None

    labeled, _ = label(mask_label)
    slice_areas = [np.sum(labeled[z] > 0) for z in range(mask.shape[0])]
    best_z = int(np.argmax(slice_areas))

    labels_in_slice = np.unique(labeled[best_z])
    labels_in_slice = labels_in_slice[labels_in_slice > 0]
    if len(labels_in_slice) == 0:
        labels_in_slice = np.unique(labeled)
        labels_in_slice = labels_in_slice[labels_in_slice > 0]

    areas = np.array([np.sum(labeled == l) for l in labels_in_slice])
    best_component = int(labels_in_slice[np.argmax(areas)])

    component_mask = (labeled == best_component)
    z_indices = np.where(component_mask)[0]
    min_z, max_z = int(np.min(z_indices)), int(np.max(z_indices))

    slices = find_objects(component_mask)[0]
    z_slice, y_slice, x_slice = slices
    z0 = max(0, z_slice.start - padding)
    z1 = min(mask.shape[0], z_slice.stop + padding)
    y0 = max(0, y_slice.start - padding)
    y1 = min(mask.shape[1], y_slice.stop + padding)
    x0 = max(0, x_slice.start - padding)
    x1 = min(mask.shape[2], x_slice.stop + padding)

    cropped_mask = mask[z0:z1, y0:y1, x0:x1]

    result = {
        "label": int(label_value),
        "best_z": best_z,
        "min_z": min_z,
        "max_z": max_z,
        "bbox_voxel": (z0, z1, y0, y1, x0, x1),
        "cropped_mask": cropped_mask,
    }

    if affine is not None:
        result["bbox_world"] = {
            "min": tuple(voxel_to_world(affine, z0, y0, x0)),
            "max": tuple(voxel_to_world(affine, z1 - 1, y1 - 1, x1 - 1)),
        }

    return result

def find_components(mask, padding=0, affine=None):
    results = {}
    for label_value in (1, 2):
        info = find_largest_component_for_label(mask, label_value, padding=padding, affine=affine)
        if info is not None:
            results[label_value] = info
    return results