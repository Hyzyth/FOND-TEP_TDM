import os
import json
import pathlib

import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import functional as F


def get_paths_to_patient_files(path_to_imgs, append_mask=True):
    """
    Collect file paths for each patient in a dataset directory.

    Each patient is expected to be stored in a separate folder containing:
    - CT image:  <patient_id>__CT.nii.gz
    - PET image: <patient_id>__PT.nii.gz
    - Optional segmentation mask: <patient_id>.nii.gz (or dataset-specific variant)

    Parameters
    ----------
    path_to_imgs : str
        Root directory containing one subfolder per patient.
    append_mask : bool
        If True, also returns the ground-truth segmentation mask path.

    Returns
    -------
    list[tuple[pathlib.Path, ...]]
        List of tuples:
        - (CT_path, PET_path) if append_mask=False
        - (CT_path, PET_path, MASK_path) if append_mask=True
    """
    path_to_imgs = pathlib.Path(path_to_imgs)
    print(path_to_imgs)

    # Identify patient folders only (ignore files)
    patients = [p for p in os.listdir(path_to_imgs) if os.path.isdir(path_to_imgs / p)]
    print(patients)

    paths = []
    for p in patients:
        # Standard naming convention per patient folder
        path_to_ct  = path_to_imgs / p / (p + "__CT.nii.gz")
        path_to_pt  = path_to_imgs / p / (p + "__PT.nii.gz")

        if append_mask:
            # Ground-truth mask naming (dataset-specific override noted)
            # CHEN dataset variant example: _2.5CK.nii.gz
            path_to_mask = path_to_imgs / p / (p + ".nii.gz")
            paths.append((path_to_ct, path_to_pt, path_to_mask))
        else:
            paths.append((path_to_ct, path_to_pt))

    return paths


def get_train_val_paths(all_paths, path_to_train_val_pkl):
    """
    Split dataset paths into training and validation subsets using patient IDs.

    The split is defined by a JSON file containing:
    {
        "train": [...patient_ids...],
        "val": [...patient_ids...]
    }

    Matching is performed by checking CT filename substrings.

    Parameters
    ----------
    all_paths : list
        Output of `get_paths_to_patient_files`.
    path_to_train_val_pkl : str
        Path to JSON file containing train/val split definition.

    Returns
    -------
    tuple[list, list]
        (train_paths, val_paths)
    """
    path_to_train_val_pkl = pathlib.Path(path_to_train_val_pkl)

    with open(path_to_train_val_pkl) as f:
        train_val_split = json.load(f)

    # Assign samples based on CT filename matching patient IDs
    train_paths = [
        path for path in all_paths
        if any(pid + "_ct.nii.gz" in str(path[0]) for pid in train_val_split["train"])
    ]
    val_paths = [
        path for path in all_paths
        if any(pid + "_ct.nii.gz" in str(path[0]) for pid in train_val_split["val"])
    ]

    return train_paths, val_paths


def read_nifti(path):
    """Load a NIfTI file using SimpleITK."""
    return sitk.ReadImage(str(path))


def write_nifti(sitk_img, path):
    """Write a SimpleITK image to disk in NIfTI format."""
    writer = sitk.ImageFileWriter()
    writer.SetImageIO("NiftiImageIO")
    writer.SetFileName(str(path))
    writer.Execute(sitk_img)


def get_attributes(sitk_image):
    """Extract spatial metadata from a SimpleITK image. Includes origin, spacing, direction, size, and pixel type."""
    return {
        "orig_pixelid":  sitk_image.GetPixelIDValue(),
        "orig_origin":   sitk_image.GetOrigin(),
        "orig_direction": sitk_image.GetDirection(),
        "orig_spacing":  np.array(sitk_image.GetSpacing()),
        # FIX: np.int removed in NumPy 1.24 – use np.int64 instead
        "orig_size":     np.array(sitk_image.GetSize(), dtype=np.int64),
    }


def resample_sitk_image(
    sitk_image,
    new_spacing=[1, 1, 1],
    new_size=None,
    attributes=None,
    interpolator=sitk.sitkLinear,
    fill_value=0,
):
    """
    Resample a SimpleITK image to a new spacing and/or resolution.

    Parameters
    ----------
    sitk_image  : sitk.Image
    new_spacing : list[float]   target voxel spacing in mm
    new_size    : list[int] | None
    attributes  : dict | None   override original image metadata
    interpolator: sitk interpolator constant
    fill_value  : padding value
    Notes
    -----
    Implementation based on:
    https://github.com/deepmedic/SimpleITK-examples/blob/master/examples/resample_isotropically.py
    """
    sitk_interpolator = interpolator

    # Extract metadata either from provided attributes or from image itself
    if attributes:
        orig_pixelid  = attributes["orig_pixelid"]
        orig_origin   = attributes["orig_origin"]
        orig_direction = attributes["orig_direction"]
        orig_spacing  = attributes["orig_spacing"]
        orig_size     = attributes["orig_size"]
    else:
        orig_pixelid  = sitk_image.GetPixelIDValue()
        orig_origin   = sitk_image.GetOrigin()
        orig_direction = sitk_image.GetDirection()
        orig_spacing  = np.array(sitk_image.GetSpacing())
        # FIX: np.int removed in NumPy 1.24 – use np.int64 instead
        orig_size     = np.array(sitk_image.GetSize(), dtype=np.int64)

    # Compute new image size if not explicitly provided
    if not new_size:
        new_size = orig_size * (orig_spacing / new_spacing)
        new_size = np.ceil(new_size).astype(np.int64)
        new_size = [int(s) for s in new_size]

    # Execute resampling operation
    resample_filter = sitk.ResampleImageFilter()

    resampled_sitk_image = resample_filter.Execute(
        sitk_image,
        new_size,
        sitk.Transform(),
        sitk_interpolator,
        orig_origin,
        new_spacing,
        orig_direction,
        fill_value,
        orig_pixelid,
    )

    return resampled_sitk_image
