import numpy as np
import nibabel as nib
from scipy.ndimage import label, find_objects
from nibabel.affines import apply_affine


def load_nifti_mask(path):
    """
    Load a NIfTI file and return the mask array, affine, and header.

    Args:
        path (str): Path to the NIfTI file.

    Returns:
        tuple: (mask array as int16, affine matrix, NIfTI header)
    """
    img = nib.load(path)
    mask = img.get_fdata()
    if not np.issubdtype(mask.dtype, np.integer):
        mask = mask.astype(np.int16)
    return mask, img.affine, img.header


def voxel_to_world(affine, x, y, z):
    """
    Convert voxel coordinates to world coordinates using the affine matrix.

    Args:
        affine (ndarray): Affine transformation matrix.
        x, y, z (int): Voxel coordinates.

    Returns:
        ndarray: World coordinates.
    """
    return apply_affine(affine, (x, y, z))


def find_largest_component_for_label(mask, label_value, padding=0, affine=None):
    """
    Find and return the largest connected component for a specific label.

    Args:
        mask (ndarray): 3D label mask.
        label_value (int): Label value to search for.
        padding (int, optional): Number of voxels to pad the bounding box.
        affine (ndarray, optional): Affine matrix for world coordinate conversion.

    Returns:
        dict or None: Information about the largest component, or None if not found.
    """
    # Create a binary mask for the selected label value
    mask_label = (mask == label_value)
    if not np.any(mask_label):
        return None

    # Label connected components in the binary mask
    labeled, _ = label(mask_label)

    # Find the axial slice with the largest area for this label
    slice_areas = [np.sum(labeled[z] > 0) for z in range(mask.shape[0])]
    best_z = int(np.argmax(slice_areas))

    # Determine candidate component labels from the best slice
    labels_in_slice = np.unique(labeled[best_z])
    labels_in_slice = labels_in_slice[labels_in_slice > 0]
    if len(labels_in_slice) == 0:
        # If no component found on the best slice, fallback to all components
        labels_in_slice = np.unique(labeled)
        labels_in_slice = labels_in_slice[labels_in_slice > 0]

    # Select the largest connected component by voxel count
    areas = np.array([np.sum(labeled == l) for l in labels_in_slice])
    best_component = int(labels_in_slice[np.argmax(areas)])

    component_mask = (labeled == best_component)

    # Find z-range of the chosen component
    z_indices = np.where(component_mask)[0]
    min_z, max_z = int(np.min(z_indices)), int(np.max(z_indices))

    # Compute the bounding box for the component
    slices_list = find_objects(labeled == best_component)
    if not slices_list:
        return None

    z_slice, y_slice, x_slice = slices_list[0]
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
        # Convert voxel bounding box corners to world coordinates
        result["bbox_world"] = {
            "min_corner": tuple(voxel_to_world(affine, x0, y0, z0)),
            "max_corner": tuple(voxel_to_world(affine, x1 - 1, y1 - 1, z1 - 1)),
        }

    return result


def find_components(mask, padding=0, affine=None, label_values=(1, 2)):
    """
    Find the largest connected component for each label value in label_values.

    Args:
        mask (ndarray): 3D label mask.
        padding (int, optional): Padding for the bounding box.
        affine (ndarray, optional): Affine matrix for world coordinate conversion.
        label_values (iterable of int, optional): Label values to process.

    Returns:
        dict: Mapping of label values to component info dictionaries.
    """
    results = {}
    for label_value in label_values:
        info = find_largest_component_for_label(mask, label_value, padding=padding, affine=affine)
        if info is not None:
            results[label_value] = info
    return results