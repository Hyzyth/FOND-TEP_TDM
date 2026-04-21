import os
import sys
import pickle

import numpy as np
import torch
from tqdm import tqdm
from scipy.ndimage import label

sys.path.append("/home/monetai/Desktop/dillan/allBrats/code/brrr/SAM")
from segment_anything import sam_model_registry, SamPredictor

REFINE_CHANNELS = ['ET','TC','WT']
CONF_THRESHOLD      = 0.50   
MEAN_CONF_CUTOFF    = 0.80  
OVERLAP_RATIO_MIN   = 0.4
OVERLAP_RATIO_MAX   = 0.9
MAX_COMPONENTS      = 200

def process_patient(patient_id, info, predictor, device):
    image = info['image']  
    D, H, W = image.shape
    boxes   = info.get('bounding_box', {})
    
    channel_slices = {}
    for ch in REFINE_CHANNELS:
        b = boxes.get(ch)
        if b is None:
            channel_slices[ch] = []
        else:
            dmin, dmax, *_ = b
            channel_slices[ch] = list(range(max(0,dmin), min(D,dmax)+1))
    
    agree_slices = [d for d in range(D)
                    if sum(d in channel_slices[ch] for ch in REFINE_CHANNELS) >= 2]
    if not agree_slices:
        print(f"{patient_id}: no slices with ≥2-channel overlap → REJECT")
        return False

    out_vol = np.zeros((len(REFINE_CHANNELS), D, H, W), np.float32)
    
    channel_stats = {}
    
    for i, ch in enumerate(REFINE_CHANNELS):
        b = boxes.get(ch)
        if b is None:
            print(f"{patient_id}: channel {ch} had no box → REJECT")
            return False
        
        dmin, dmax, hmin, hmax, wmin, wmax = b
        box = np.array([wmin, hmin, wmax, hmax])[None,:]
        
        confidences = []
        merged_mask = np.zeros((D,H,W), np.uint8)
        
        for d in agree_slices:
            rgb = np.stack([image[d]]*3, -1)
            predictor.set_image(rgb)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box,
                multimask_output=False,
            )
            conf = float(scores[0])
            confidences.append(conf)
            if conf >= CONF_THRESHOLD:
                out_vol[i,d] = masks[0]
                merged_mask[d] = masks[0]
        
        mean_conf = np.mean(confidences) if confidences else 0.0
        box_vol  = (dmax-dmin+1)*(hmax-hmin+1)*(wmax-wmin+1)
        mask_vol = merged_mask.sum()
        overlap_ratio = mask_vol / box_vol if box_vol>0 else 0.0
        labeled, n_comp = label(merged_mask)
        
        channel_stats[ch] = (mean_conf, overlap_ratio, n_comp)
        
        if (mean_conf < MEAN_CONF_CUTOFF
            or overlap_ratio < OVERLAP_RATIO_MIN
            or overlap_ratio > OVERLAP_RATIO_MAX
            or n_comp > MAX_COMPONENTS):
            print(f"{patient_id}/{ch}: FAIL "
                  f"conf={mean_conf:.2f}, ratio={overlap_ratio:.2f}, comps={n_comp}")
            return False
    
    info['sam_pseudo_label'] = out_vol
    info.pop('bounding_box', None)
    print(f"{patient_id}: ACCEPTED (stats={channel_stats})")
    return True


def main_worker(orig_pickle_file, temp_dir,
                sam_checkpoint, model_type, start, end):
    with open(orig_pickle_file, 'rb') as f:
        full_data = pickle.load(f)
    
    device    = torch.device('cuda:0')
    sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device)
    sam_model.eval()
    predictor = SamPredictor(sam_model)
    
    filtered = {}
    for pid in tqdm(list(full_data.keys())[start:end], desc="Patients"):
        info = full_data[pid]
        if process_patient(pid, info, predictor, device):
            filtered[pid] = info
    
    out_pkl = os.path.join(temp_dir, "sam_pseudo_labels.pkl")
    with open(out_pkl, 'wb') as f:
        pickle.dump(filtered, f)
    print(f"Saved {len(filtered)}/{end-start} patients to {out_pkl}")


if __name__ == "__main__":
    orig_pickle_file = "pseudo_labels/pseudo_labels.pkl"
    temp_dir         = os.path.dirname(orig_pickle_file)
    sam_checkpoint   = "/home/monetai/Desktop/dillan/allBrats/code/brrr/SAM/sam_vit_h_4b8939.pth"
    model_type       = "vit_h"
    START, END       = 0, 9999

    main_worker(orig_pickle_file, temp_dir,
                sam_checkpoint, model_type, START, END)
