"""
This code makes part of the thesis project: Head and Neck Cancer PET/TDM study for recurrence and survival prediction
======================================
Correction of predicted tumor segmentation orientation of swincross unetr model.

The methodology follows the following steps:
1. Load the original CT image and the predicted segmentation (in NIfTI format).
2. Transpose the predicted segmentation array (2,1,0) to match the expected orientation.
3. Align centers in physical space by adjusting segmentation origin (metadata-only, no transformation). 
        Threshold of 40mm offset is used to determine if centering the images is appropriate (preserves correct anatomical placement if already close).
4. Resample the aligned segmentation to match the CT image space (size, spacing, origin).
5. Apply a 180° rotation around the Z-axis using the CT image center as the pivot.
6. Save the reoriented segmentation as a new NIfTI file.

Author: Santiago Osorio
Date: 2026-02
Copilot assistance used for code development
"""

import SimpleITK as sitk
import numpy as np
import os

def resample_like(moving_image, reference_img, is_label=False):
    """
    Resample 'source_img' to match the geometry of 'target_img'.
    For labels, use nearest-neighbor; for images, BSplineResamplerOrder3.
    """
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(reference_img)
    r.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSplineResamplerOrder3) # changed sitk.Linear to sitk.Bspline third order for more precision 
    r.SetTransform(sitk.Transform())
    r.SetOutputPixelType(sitk.sitkUInt8)  # Ensure output is uint8 for label masks
    return r.Execute(moving_image)

# VARIABLES
# 40mm handles the 8mm case (keeps it) and the 138mm case (fixes it).
ALIGNMENT_THRESHOLD_MM = 40.0 #<----------important line. threshold can be adjusted based on results after testing with more cases. 

# Paths
path = 'Swin-Cross--basing-on-MONAI-code-\\test_debug'
saving_path = 'Swin-Cross--basing-on-MONAI-code-\\reoriented_test_sitk_rotation_all'
os.makedirs(saving_path, exist_ok=True)

# Dictionary mapping patient IDs to their CT image paths
patient_ct_dict = {
    "CHUM-001": "E:\\HECKTOR_2025\\HECKTOR 2025 Training Data Updated\\HECKTOR 2025 Training Data\\Task 1\\CHUM-001\\CHUM-001__CT.nii.gz",
    "CHUM-002": "E:\\HECKTOR_2025\\HECKTOR 2025 Training Data Updated\\HECKTOR 2025 Training Data\\Task 1\\CHUM-002\\CHUM-002__CT.nii.gz",
    "CHUS-006": "E:\\HECKTOR_2025\\HECKTOR 2025 Training Data Updated\\HECKTOR 2025 Training Data\\Task 1\\CHUS-006\\CHUS-006__CT.nii.gz"
}

list_of_files = os.listdir(path)
for file in list_of_files:
    
    print("\n" + "="*80)
    print(f"Processing: {file}")
    print("="*80)

    # Extract patient ID from filename (e.g., "CHUM-001" from "CHUM-001_seg.nii.gz")
    patient_id = file.split('_')[0]
    
    if patient_id not in patient_ct_dict:
        print(f"Warning: No CT path found for patient {patient_id}, skipping...")
        continue
    
    try:
        # Load images with SimpleITK
        seg_img = sitk.ReadImage(os.path.join(path, file))
        ct_img = sitk.ReadImage(patient_ct_dict[patient_id])
    except Exception as e:
        print(f"Error loading images for {file}: {e}")
        continue

    print("Original seg info:")
    print(f"  Size: {seg_img.GetSize()}")
    print(f"  Origin: {seg_img.GetOrigin()}")
    print(f"  Spacing: {seg_img.GetSpacing()}")
    print(f"  Direction: {seg_img.GetDirection()}")

    print("\nTarget CT info:")
    print(f"  Size: {ct_img.GetSize()}")
    print(f"  Origin: {ct_img.GetOrigin()}")
    print(f"  Spacing: {ct_img.GetSpacing()}")

    print("\nApproach: transpose → align centers (metadata) → resample → rotate")

    # Step 1: Transpose
    seg_array = sitk.GetArrayFromImage(seg_img)
    seg_array_tp = np.transpose(seg_array, (2, 1, 0))
    seg_tp_img = sitk.GetImageFromArray(seg_array_tp)
    seg_tp_img.SetOrigin(seg_img.GetOrigin())
    seg_tp_img.SetDirection(seg_img.GetDirection())
    seg_tp_img.SetSpacing(seg_img.GetSpacing())

    print(f"\n1. Transposed segmentation:")
    print(f"   Size: {seg_tp_img.GetSize()}")
    print(f"   Unique values: {np.unique(seg_array_tp)}")

    # Step 1.5: Align centers in physical space by adjusting origin (metadata only)
    # This prevents content from being cut during resampling
    center_seg_physical = seg_tp_img.TransformContinuousIndexToPhysicalPoint([sz/2.0 for sz in seg_tp_img.GetSize()])
    center_ct_physical = ct_img.TransformContinuousIndexToPhysicalPoint([sz/2.0 for sz in ct_img.GetSize()])
    translation_vector = np.array(center_ct_physical) - np.array(center_seg_physical)

    # Calculate distance between centers
    dist_centers = np.linalg.norm(translation_vector)
    
    # Threshold (mm) below which we assume valid anatomical offset.
    if dist_centers < ALIGNMENT_THRESHOLD_MM:
        print(f"\nCenters are physically close ({dist_centers:.2f} mm < {ALIGNMENT_THRESHOLD_MM} mm).")
        print("Skipping forced centering to preserve correct anatomical placement.")
        
    else:
        print(f"\n1.5. Physical space alignment (metadata adjustment):")
        print(f"   Transposed seg center (physical): {center_seg_physical}")
        print(f"   CT center (physical): {center_ct_physical}")
        print(f"   Translation vector: {translation_vector}")

        # Adjust origin directly (no resampling, no cropping)
        # The image content stays the same, but its position in physical space changes
        current_origin = np.array(seg_tp_img.GetOrigin())

        # Test both adding and subtracting to determine which aligns centers better
        new_origin_add = current_origin + translation_vector
        new_origin_sub = current_origin - translation_vector

        # Create temp images to test which origin adjustment works
        temp_img_add = sitk.Image(seg_tp_img)
        temp_img_add.SetOrigin(tuple(new_origin_add))
        center_after_add = temp_img_add.TransformContinuousIndexToPhysicalPoint([sz/2.0 for sz in temp_img_add.GetSize()])

        temp_img_sub = sitk.Image(seg_tp_img)
        temp_img_sub.SetOrigin(tuple(new_origin_sub))
        center_after_sub = temp_img_sub.TransformContinuousIndexToPhysicalPoint([sz/2.0 for sz in temp_img_sub.GetSize()])

        # Calculate distances from each result center to CT center
        dist_add = np.linalg.norm(np.array(center_after_add) - np.array(center_ct_physical))
        dist_sub = np.linalg.norm(np.array(center_after_sub) - np.array(center_ct_physical))

        print(f"\n   Testing origin adjustment directions:")
        print(f"   - Add translation: image center distance to CT center = {dist_add:.2f} mm")
        print(f"   - Subtract translation: image center distance to CT center = {dist_sub:.2f} mm")

        # Choose the operation that minimizes distance between centers
        if dist_add < dist_sub:
            new_origin = new_origin_add
            operation = "ADDED"
        else:
            new_origin = new_origin_sub
            operation = "SUBTRACTED"

        seg_tp_img.SetOrigin(tuple(new_origin))

        print(f"   Old origin: {current_origin}")
        print(f"   New origin: {new_origin}")
        print(f"   -> Translation vector {operation} (distance: {min(dist_add, dist_sub):.2f} mm)")
        print(f"   -> Centers now aligned in physical space, no pixels moved")

    aligned_seg_img = seg_tp_img  # Just a reference, no actual transformation applied

    # Step 2: Resample to CT space (aligned, so content won't be cut)
    inCTspace_img = resample_like(aligned_seg_img, ct_img, True)

    print(f"\n2. Resampled to CT space:")
    print(f"   Size: {inCTspace_img.GetSize()}")
    print(f"   Origin: {inCTspace_img.GetOrigin()}")
    print(f"   Spacing: {inCTspace_img.GetSpacing()}")

    # Step 3: Rotate around CT center (both centers should be identical now)
    center_ct = ct_img.TransformContinuousIndexToPhysicalPoint([sz/2.0 for sz in ct_img.GetSize()])

    print(f"\n3. Rotation center (CT center): {center_ct}")

    theta_x = 0  
    theta_y = 0    
    theta_z = np.pi  # 180° around Z axis

    rot_transform = sitk.Euler3DTransform(center_ct, theta_x, theta_y, theta_z, (0, 0, 0))

    # Apply rotation
    rotated_img = sitk.Resample(inCTspace_img, rot_transform, sitk.sitkNearestNeighbor, 0, seg_img.GetPixelID())

    # Check unique values
    array_rot = sitk.GetArrayFromImage(rotated_img)
    unique_vals = np.unique(array_rot)

    print(f"   Rotation: X={int(np.degrees(theta_x))}°, Y={int(np.degrees(theta_y))}°, Z={int(np.degrees(theta_z))}°")
    print(f"   Unique values after rotation: {unique_vals}")
    print(f"   Size after rotation: {rotated_img.GetSize()}")

    # Step 4: Save
    try:
        output_filename = f'sitk_tp_aligned_resample_rot180z_{file}'
        sitk.WriteImage(rotated_img, os.path.join(saving_path, output_filename))

        print(f"\n4. Saved successfully:")
        print(f"   - Final (rotated): {output_filename}")
    except Exception as e:
        print(f"\nError saving files: {e}")

print("\n" + "="*80)
print("*** Load in Slicer to verify orientation ***")
print("="*80)