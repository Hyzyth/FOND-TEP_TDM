"""
auto_prompter.py
================
Generates bounding box prompts without Ground Truth using:
A) PET-driven thresholding (Base 41%, Nestle, Black, Daisne)
B) Lightweight Proposal Network (UNet)
C) Hybrid (PET Recall + UNet Precision)
"""

import numpy as np
from scipy.ndimage import label
import torch

# =============================================================================
# PART A: PET-Driven Proposals
# =============================================================================

def get_base_41_mask(pet_suv: np.ndarray) -> np.ndarray:
    """Standard 41% of SUVmax thresholding."""
    suv_max = pet_suv.max()
    threshold = 0.41 * suv_max
    return pet_suv > threshold

def get_black_mask(pet_suv: np.ndarray, alpha=0.307, beta=0.588) -> np.ndarray:
    """Black iterative method."""
    suv_max = pet_suv.max()
    threshold = 0.41 * suv_max # Start at 41%
    prev_vol = 0
    
    while True:
        mask = pet_suv > threshold
        vol = mask.sum()
        
        if abs(vol - prev_vol) <= 1 or vol == 0:
            break
            
        prev_vol = vol
        suv_mean = pet_suv[mask].mean()
        threshold = alpha * suv_mean + beta
        
    return mask

def get_daisne_mask(pet_suv: np.ndarray, a=31.3, b=77.7) -> np.ndarray:
    """Daisne iterative method based on local contrast."""
    suv_max = pet_suv.max()
    threshold = 0.41 * suv_max
    prev_vol = 0
    
    while True:
        mask = pet_suv > threshold
        vol = mask.sum()
        
        if abs(vol - prev_vol) <= 1 or vol == 0:
            break
            
        prev_vol = vol
        
        # Approximate: Top 0.5mL (assuming e.g. ~500 voxels depending on spacing)
        # For true Daisne, you must calculate exactly 0.5mL using voxel spacing
        # Here we approximate Maxavg as the mean of the top 10% brightest voxels in mask
        mask_voxels = pet_suv[mask]
        maxavg = np.percentile(mask_voxels, 90) 
        
        # Background: region outside segmentation (mean of non-mask > 0 to avoid air)
        bg_mask = (~mask) & (pet_suv > 0.5) 
        bavg = pet_suv[bg_mask].mean() if bg_mask.sum() > 0 else 1.0
        
        contmeas = maxavg / bavg if bavg > 0 else 1.0
        threshold = a + b * (1 / contmeas)
        
    return mask

# =============================================================================
# PART B: Lightweight Proposal Network
# =============================================================================

class LightweightProposalNet:
    """Stub for a 3D UNet or 2.5D slice-based network."""
    def __init__(self, model_weights_path):
        # Load your PyTorch model here (e.g., standard MONAI UNet)
        # self.model = UNet(...).load_state_dict(...)
        pass
        
    @torch.no_grad()
    def predict_mask(self, ct_tensor, pet_tensor, prob_threshold=0.5):
        """Returns a binary mask of proposed lesions."""
        # out_probs = self.model(torch.cat([ct_tensor, pet_tensor], dim=1))
        # return (out_probs > prob_threshold).cpu().numpy()
        pass

# =============================================================================
# PART C: Bounding Box Extraction & Hybrid Logic
# =============================================================================

def extract_bboxes_from_mask(binary_mask: np.ndarray, min_vol: int = 10, pad: int = 5):
    """Converts a binary mask into a list of 3D bounding box dictionaries."""
    labeled_arr, num_components = label(binary_mask)
    proposals = []
    
    D, H, W = binary_mask.shape
    
    for comp_id in range(1, num_components + 1):
        comp_mask = (labeled_arr == comp_id)
        voxel_count = comp_mask.sum()
        
        if voxel_count < min_vol:
            continue
            
        indices = np.where(comp_mask)
        z_min, z_max = indices[0].min(), indices[0].max()
        y_min, y_max = indices[1].min(), indices[1].max()
        x_min, x_max = indices[2].min(), indices[2].max()
        
        # Calculate key slice (z_mid) based on largest cross-sectional area
        max_area = -1
        z_mid = z_min
        for z in range(z_min, z_max + 1):
            area = comp_mask[z].sum()
            if area > max_area:
                max_area = area
                z_mid = z
        
        # 2D Bbox for the Z-axis key slice (applying padding)
        x0, x1 = max(0, x_min - pad), min(W, x_max + pad)
        y0, y1 = max(0, y_min - pad), min(H, y_max + pad)
        
        proposals.append({
            "voxel_count": int(voxel_count),
            "z_mid": z_mid,
            "bbox_2d": np.array([x0, y0, x1, y1], dtype=np.int32),
            "bbox_3d": [z_min, z_max, y_min, y_max, x_min, x_max]
        })
        
    return proposals

def calculate_3d_iou(boxA, boxB):
    """Calculates Intersection over Union for two 3D boxes: [z0, z1, y0, y1, x0, x1]"""
    zA_min, zA_max, yA_min, yA_max, xA_min, xA_max = boxA
    zB_min, zB_max, yB_min, yB_max, xB_min, xB_max = boxB
    
    z_inter = max(0, min(zA_max, zB_max) - max(zA_min, zB_min))
    y_inter = max(0, min(yA_max, yB_max) - max(yA_min, yB_min))
    x_inter = max(0, min(xA_max, xB_max) - max(xA_min, xB_min))
    
    inter_vol = z_inter * y_inter * x_inter
    volA = (zA_max - zA_min) * (yA_max - yA_min) * (xA_max - xA_min)
    volB = (zB_max - zB_min) * (yB_max - yB_min) * (xB_max - xB_min)
    
    union_vol = volA + volB - inter_vol
    return inter_vol / union_vol if union_vol > 0 else 0

def hybrid_proposals(pet_mask: np.ndarray, unet_mask: np.ndarray, iou_threshold=0.1):
    """
    Method C: Uses PET for high recall, filters/merges using UNet for precision.
    Keeps PET boxes that intersect with UNet boxes.
    """
    pet_boxes = extract_bboxes_from_mask(pet_mask)
    unet_boxes = extract_bboxes_from_mask(unet_mask)
    
    final_proposals = []
    
    for p_box in pet_boxes:
        keep = False
        for u_box in unet_boxes:
            if calculate_3d_iou(p_box["bbox_3d"], u_box["bbox_3d"]) > iou_threshold:
                keep = True
                break
        if keep:
            final_proposals.append(p_box)
            
    # Fallback: if network fails completely, fallback to UNet boxes alone or top PET box
    if not final_proposals and unet_boxes:
        return unet_boxes
        
    return final_proposals
