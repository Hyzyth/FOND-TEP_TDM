"""
auto_prompting
==============
Automatic / semi-automatic bounding-box proposal generation for MedSAM2.

Three strategies are provided:

A) PET-driven proposals (high recall, noisy)
   Methods: base41, black, daisne
   Requires raw SUV values (pet_suv_max stored in NPZ by prepare_hecktor_npz.py).

B) Lightweight proposal network (balanced recall / precision)
   A small 3-D U-Net trained on HECKTOR NPZ data.
   Train with:  python -m auto_prompting.train_proposal_net --help

C) Hybrid (A ∪ B filtered by IoU)
   PET boxes that overlap with network boxes are kept; the rest are dropped.
   Falls back to network boxes alone if no overlap is found.

Usage in inference:

    from auto_prompting import AutoPrompter

    prompter = AutoPrompter(method='hybrid', model_path='proposal_net.pt')
    components_per_label = prompter.get_proposals(ct_imgs, pet_imgs, suv_max)
    # returns {1: [...], 2: [...]} matching the format expected by infer_hecktor.py
"""

from auto_prompting.auto_prompter import AutoPrompter

__all__ = ["AutoPrompter"]
