"""
slicer.py
=========
Utilities for extracting connected components and bounding-box prompts
from HECKTOR GT masks.

The GT mask is a 3-D array with shape (D, H, W) = (z, y, x) and integer labels:
    0 – background
    1 – GTVp (primary tumour)
    2 – GTVn (nodal tumour; may be absent or multi-focal)

All bounding boxes are returned in (x_min, y_min, x_max, y_max) format
matching the convention used by SAM2's add_new_points_or_box().
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import label, find_objects
from nibabel.affines import apply_affine


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_nifti_mask(path: str):
    """Load a NIfTI file and return (mask_int16, affine, header).

    The mask array is returned in (z, y, x) / (D, H, W) order, which is
    what SimpleITK / numpy give after GetArrayFromImage.

    Parameters
    ----------
    path : str

    Returns
    -------
    mask   : np.ndarray  int16, shape (D, H, W)
    affine : np.ndarray  (4, 4)
    header : nibabel header object
    """
    img = nib.load(path)
    # nibabel returns (x, y, z); transpose to (z, y, x) for consistency
    # with the rest of the pipeline that uses (D, H, W) = (z, y, x).
    mask = np.asarray(img.get_fdata(), dtype=np.int16).transpose(2, 1, 0)
    return mask, img.affine, img.header


def voxel_to_world(affine: np.ndarray, z: int, y: int, x: int) -> np.ndarray:
    """Convert (z, y, x) voxel coordinates to world coordinates.

    Parameters
    ----------
    affine : (4, 4) affine matrix
    z, y, x : int

    Returns
    -------
    np.ndarray  world coordinates (3,)
    """
    # nibabel affine expects (x, y, z)
    return apply_affine(affine, (x, y, z))


# ---------------------------------------------------------------------------
# Per-label connected-component extraction
# ---------------------------------------------------------------------------

def find_all_components_for_label(
    mask: np.ndarray,
    label_value: int,
    padding: int = 0,
    affine: np.ndarray = None,
):
    """Find every connected component for a single label value.

    Parameters
    ----------
    mask        : (D, H, W) int array
    label_value : int  – label to search (1 = GTVp, 2 = GTVn)
    padding     : int  – voxel padding added to the bounding box on each side
    affine      : (4,4) array or None  – when given, world-space bbox is added

    Returns
    -------
    list of dict, sorted by voxel_count descending.  Each dict contains:
        label         : int
        component_id  : int  (1-based)
        voxel_count   : int
        z_mid         : int  – axial slice with the largest cross-section
        bbox_voxel    : dict  {z:(z0,z1), y:(y0,y1), x:(x0,x1)}
        bbox_2d       : dict  {z_mid: np.ndarray [x_min,y_min,x_max,y_max]}
        cropped_mask  : np.ndarray  binary, shape of the padded bbox
        bbox_world    : dict  (only if affine is not None)
    """
    # Create a binary mask for the selected label value
    mask_label = (mask == label_value)
    if not np.any(mask_label):
        return []

    labeled_arr, num_components = label(mask_label)

    results = []
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled_arr == comp_id)

        z_indices, y_indices, x_indices = np.where(comp_mask)
        if len(z_indices) == 0:
            continue

        # Bounding box (with padding, clamped to array bounds)
        D, H, W = mask.shape
        z0 = max(0, int(z_indices.min()) - padding)
        z1 = min(D, int(z_indices.max()) + 1 + padding)
        y0 = max(0, int(y_indices.min()) - padding)
        y1 = min(H, int(y_indices.max()) + 1 + padding)
        x0 = max(0, int(x_indices.min()) - padding)
        x1 = min(W, int(x_indices.max()) + 1 + padding)

        cropped_mask = comp_mask[z0:z1, y0:y1, x0:x1]
        voxel_count  = int(np.sum(comp_mask))

        # Key slice: axial index with the largest 2-D cross-section
        areas = np.array([
            comp_mask[z].sum() for z in range(z0, z1)
        ])
        z_mid = z0 + int(np.argmax(areas))

        # 2-D bounding box on the key slice (SAM2 format: x_min,y_min,x_max,y_max)
        ys_mid, xs_mid = np.where(comp_mask[z_mid])
        bbox_2d = np.array([
            int(xs_mid.min()), int(ys_mid.min()),
            int(xs_mid.max()), int(ys_mid.max()),
        ], dtype=np.int32)

        result = {
            "label":        int(label_value),
            "component_id": int(comp_id),
            "voxel_count":  voxel_count,
            "z_mid":        z_mid,
            "bbox_voxel":   {"z": (z0, z1), "y": (y0, y1), "x": (x0, x1)},
            "bbox_2d":      bbox_2d,   # [x_min, y_min, x_max, y_max] on z_mid
            "cropped_mask": cropped_mask,
        }

        if affine is not None:
            result["bbox_world"] = {
                "min_corner": tuple(voxel_to_world(affine, z0, y0, x0)),
                "max_corner": tuple(voxel_to_world(affine, z1 - 1, y1 - 1, x1 - 1)),
            }

        results.append(result)

    results.sort(key=lambda r: r["voxel_count"], reverse=True)
    return results


def find_components(
    mask: np.ndarray,
    padding: int = 0,
    affine: np.ndarray = None,
    label_values=(1, 2),
) -> dict:
    """Find all connected components for each label in *label_values*.

    Parameters
    ----------
    mask         : (D, H, W) int array  (0=bg, 1=GTVp, 2=GTVn)
    padding      : int  voxel padding for bounding boxes
    affine       : (4,4) or None
    label_values : iterable[int]  labels to process

    Returns
    -------
    dict mapping label_id → list of component dicts (sorted largest-first).
    Only labels actually present in the mask are included.
    """
    present = set(np.unique(mask).tolist()).intersection(set(label_values))
    return {
        lv: find_all_components_for_label(mask, lv, padding=padding, affine=affine)
        for lv in present
        if find_all_components_for_label(mask, lv, padding=padding, affine=affine)
    }


# ---------------------------------------------------------------------------
# Convenience: scale a 2-D bbox from original image space to model space
# ---------------------------------------------------------------------------

def scale_bbox_2d(
    bbox: np.ndarray,
    orig_hw: tuple,
    model_size: int = 512,
) -> np.ndarray:
    """Scale [x_min,y_min,x_max,y_max] from orig_hw to model_size×model_size.

    Parameters
    ----------
    bbox       : (4,) int array  [x_min, y_min, x_max, y_max]
    orig_hw    : (H, W)  original slice dimensions
    model_size : int  target square size (default 512)

    Returns
    -------
    np.ndarray  (4,) float32
    """
    H, W = orig_hw
    sx = model_size / W
    sy = model_size / H
    return np.array([
        bbox[0] * sx,
        bbox[1] * sy,
        bbox[2] * sx,
        bbox[3] * sy,
    ], dtype=np.float32)
