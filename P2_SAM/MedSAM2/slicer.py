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
    slice_pad: int = 1,
    planar_pad: int = 5,
    affine: np.ndarray = None,
):
    """Find every connected component for a single label value.

    Parameters
    ----------
    mask        : (D, H, W) int array
    label_value : int  – label to search (1 = GTVp, 2 = GTVn)
    slice_pad   : int  – padding applied along the primary "viewing" axis
    planar_pad  : int  – padding applied along the two planar axes forming the slice
    affine      : (4,4) array or None  – when given, world-space bbox is added

    Returns
    -------
    list of dict, sorted by voxel_count descending.  Each dict contains:
        label         : int
        component_id  : int  (1-based)
        voxel_count   : int
        primary_axis  : int  (0=Z, 1=Y, 2=X)
        mid_slice     : int  – index along primary_axis with the largest cross-section
        bbox_voxel    : dict  {z:(z0,z1), y:(y0,y1), x:(x0,x1)}
        bbox_2d       : np.ndarray  [col_min, row_min, col_max, row_max]
                        on the mid_slice plane (in primary-axis coordinates)
        cropped_mask  : np.ndarray  binary, shape of the padded bbox
        bbox_world    : dict  (only if affine is not None)
    """
    # Create a binary mask for the selected label value
    mask_label = (mask == label_value)
    if not np.any(mask_label):
        return []

    labeled_arr, num_components = label(mask_label)
    results = []

    D, H, W = mask.shape
    
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled_arr == comp_id)
        indices = np.where(comp_mask)
        
        if len(indices[0]) == 0:
            continue

        # Get tight bounds to optimize the area search
        z_min, z_max = indices[0].min(), indices[0].max()
        y_min, y_max = indices[1].min(), indices[1].max()
        x_min, x_max = indices[2].min(), indices[2].max()

        max_area = -1
        primary_axis = 0
        mid_slice = 0

        # Check Axis 0 (Z-axis / Axial plane)
        for z in range(z_min, z_max + 1):
            area = comp_mask[z, y_min:y_max+1, x_min:x_max+1].sum()
            if area > max_area:
                max_area, primary_axis, mid_slice = area, 0, z

        # Check Axis 1 (Y-axis / Coronal plane)
        for y in range(y_min, y_max + 1):
            area = comp_mask[z_min:z_max+1, y, x_min:x_max+1].sum()
            if area > max_area:
                max_area, primary_axis, mid_slice = area, 1, y

        # Check Axis 2 (X-axis / Sagittal plane)
        for x in range(x_min, x_max + 1):
            area = comp_mask[z_min:z_max+1, y_min:y_max+1, x].sum()
            if area > max_area:
                max_area, primary_axis, mid_slice = area, 2, x

        # Assign padding dynamically based on the optimal viewing axis
        pads = [0, 0, 0]
        pads[primary_axis] = slice_pad
        pads[(primary_axis + 1) % 3] = planar_pad
        pads[(primary_axis + 2) % 3] = planar_pad
        pad_z, pad_y, pad_x = pads

        # 3D Bounding box (with dynamic padding, clamped to array bounds)
        z0 = max(0, int(z_min) - pad_z)
        z1 = min(D, int(z_max) + 1 + pad_z)
        y0 = max(0, int(y_min) - pad_y)
        y1 = min(H, int(y_max) + 1 + pad_y)
        x0 = max(0, int(x_min) - pad_x)
        x1 = min(W, int(x_max) + 1 + pad_x)

        cropped_mask = comp_mask[z0:z1, y0:y1, x0:x1]
        voxel_count  = int(np.sum(comp_mask))

        # 2D Bounding box format depends on which axis we are viewing from.
        # Format is always [col_min, row_min, col_max, row_max] relative to the 2D slice.
        if primary_axis == 0:
            # Viewing Z. Planar image is (Y, X). Row=Y, Col=X.
            bbox_2d = np.array([x0, y0, x1, y1], dtype=np.int32)
        elif primary_axis == 1:
            # Viewing Y. Planar image is (Z, X). Row=Z, Col=X.
            bbox_2d = np.array([x0, z0, x1, z1], dtype=np.int32)
        elif primary_axis == 2:
            # Viewing X. Planar image is (Z, Y). Row=Z, Col=Y.
            bbox_2d = np.array([y0, z0, y1, z1], dtype=np.int32)

        result = {
            "label":        int(label_value),
            "component_id": int(comp_id),
            "voxel_count":  voxel_count,
            "primary_axis": primary_axis,  # 0=Z, 1=Y, 2=X
            "mid_slice":    mid_slice,     # The index along the primary_axis
            "bbox_voxel":   {"z": (z0, z1), "y": (y0, y1), "x": (x0, x1)},
            "bbox_2d":      bbox_2d,
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
    slice_pad: int = 1,
    planar_pad: int = 5,
    affine: np.ndarray = None,
    label_values=(1, 2),
) -> dict:
    """Find all connected components for each label in *label_values*."""
    present = set(np.unique(mask).tolist()).intersection(set(label_values))
    result = {}
    for lv in present:
        comps = find_all_components_for_label(
            mask, lv, slice_pad=slice_pad, planar_pad=planar_pad, affine=affine
        )
        if comps:
            result[lv] = comps
    return result


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


# ---------------------------------------------------------------------------
# NEW: Z-axis prompt extraction for the video predictor
# ---------------------------------------------------------------------------

def get_z_prompt_from_component(comp: dict) -> tuple:
    """Extract (z_mid, bbox_2d) always on the axial (Z) axis.

    MedSAM2's slice-by-slice video predictor propagates along Z, so prompts
    must reference a Z-axis slice regardless of which axis slicer.py chose as
    the primary viewing axis.

    When ``primary_axis == 0`` this is a no-op (mid_slice already IS z_mid
    and bbox_2d is already [x_min, y_min, x_max, y_max] on that Z slice).

    When ``primary_axis`` is 1 (Y/coronal) or 2 (X/sagittal), the function
    finds the Z slice with the largest cross-sectional area using the stored
    ``cropped_mask``, then returns the padded 3-D bounding-box corners
    projected onto that slice as bbox_2d.

    Parameters
    ----------
    comp : dict
        One component dict as returned by :func:`find_all_components_for_label`.

    Returns
    -------
    z_mid  : int             axial slice index (into the full volume)
    bbox_2d: np.ndarray(4,)  [x_min, y_min, x_max, y_max] on z_mid
    """
    z0, z1 = comp["bbox_voxel"]["z"]

    if comp["primary_axis"] == 0:
        # mid_slice is already a Z index; bbox_2d already in (x,y) format.
        return comp["mid_slice"], comp["bbox_2d"]

    # Primary axis is Y or X — derive z_mid from the cropped mask.
    cropped = comp["cropped_mask"]   # shape (z1-z0, y1-y0, x1-x0)
    dz = z1 - z0
    z_areas = np.array([cropped[z].sum() for z in range(dz)])
    rel_z = int(np.argmax(z_areas))
    z_mid = z0 + rel_z

    # Use the padded 3-D bounding-box bounds projected onto the Z plane
    x0, x1 = comp["bbox_voxel"]["x"]
    y0, y1 = comp["bbox_voxel"]["y"]
    bbox_2d = np.array([x0, y0, x1, y1], dtype=np.int32)

    return z_mid, bbox_2d
