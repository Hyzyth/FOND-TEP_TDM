"""
inference_proposals.py
=====================
Generate proposals using trained network + PET hybrid
"""

import torch
import numpy as np
from model import Small3DUNet
from pet_proposals import (
    load_nifti,
    normalize_pet,
    normalize_ct,
    pet_threshold_mask,
    hybrid_mask,
    extract_components,
)


def run_inference(pet_path, ct_path, model_path):
    pet, _ = load_nifti(pet_path)
    ct, _ = load_nifti(ct_path)

    pet_n = normalize_pet(pet)
    ct_n = normalize_ct(ct)

    x = np.stack([pet_n, ct_n], axis=0)
    x = torch.tensor(x).unsqueeze(0).float()

    model = Small3DUNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        prob = model(x)[0, 0].numpy()

    net_mask = prob > 0.2
    pet_mask = pet_threshold_mask(pet, percentile=99)

    mask = hybrid_mask(pet_mask, net_mask)

    components = extract_components(mask)

    return components
