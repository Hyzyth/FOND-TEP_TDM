"""
training/dataset/hecktor_dataset.py
=====================================
HECKTOR-specific raw dataset for dual-modality (CT + PET) head-and-neck
tumour segmentation.

UPDATED to ingest SwinCross-format NPZ files produced by
``prepare_hecktor2026_kfold_npz.py`` (shared with SwinCross and DualwaveSAM).
This avoids any redundant data conversion step.

SwinCross NPZ content
----------------------
    ``ct``      - (R, A, S) int16    CT in HU (MONAI RAS convention)
    ``pet``     - (R, A, S) float16  PET in SUV
    ``label``   - (R, A, S) uint8    0=bg, 1=GTVp, 2=GTVn
    + inverse-transform metadata (ras_origin, crop_start, …)

Spatial convention
------------------
The MONAI (R, A, S) axes map to:
    R = right-left (axis 0)
    A = anterior-posterior (axis 1)
    S = superior-inferior = axial (axis 2)

Axial slices are extracted along axis 2, matching DualwaveSAM's SLICE_AXIS=2.
Each slice is (R, A), which corresponds to the (H, W) spatial plane seen by
SAM2's image backbone.

JSON format (SwinCross _classic.json)
--------------------------------------
    {
      "training":   [{"npz": "npz/train/HGJ_001.npz", "label": "...", "case_id": "HGJ_001"}, ...],
      "validation": [{"npz": "npz/train/HGJ_002.npz", ...}, ...]
    }
The ``npz`` path is relative to the dataset root (data_dir).

Intensity normalisation (mirrors DualwaveSAM dataset.py)
---------------------------------------------------------
    CT  -> soft-tissue window [-160, +240 HU] clipped then [0, 1]
    PET -> per-volume 99th-percentile clip then [0, 1]
These ops are done online in ``_fuse_modalities``, not during preprocessing.
"""

import json
import os
from typing import List, Optional

import numpy as np
import torch

from training.dataset.vos_raw_dataset import VOSFrame, VOSVideo, VOSRawDataset
from training.dataset.vos_segment_loader import NPZSegmentLoader


# ── Intensity normalisation (identical to DualwaveSAM) ────────────────────────

# CT soft-tissue window
_CT_WINDOW_LO  = -160.0   # HU  (WC 40, WW 400 -> WC − WW/2)
_CT_WINDOW_HI  =  240.0   # HU  (WC 40 + WW/2)

# Axial axis in MONAI (R, A, S) space
_SLICE_AXIS = 2


def _normalise_ct(ct_arr: np.ndarray) -> np.ndarray:
    """Clip to soft-tissue window and scale to [0, 1] float32."""
    ct = np.clip(ct_arr.astype(np.float32), _CT_WINDOW_LO, _CT_WINDOW_HI)
    return (ct - _CT_WINDOW_LO) / (_CT_WINDOW_HI - _CT_WINDOW_LO)


def _normalise_pet(pet_arr: np.ndarray) -> np.ndarray:
    """Clip to 99th-percentile max and scale to [0, 1] float32."""
    pet = pet_arr.astype(np.float32)
    p99 = float(np.percentile(pet[pet > 0], 99)) if (pet > 0).any() else 1.0
    p99 = max(p99, 1e-6)
    return np.clip(pet / p99, 0.0, 1.0)


# ── Dataset ───────────────────────────────────────────────────────────────────

class HECKTORNPZRawDataset(VOSRawDataset):
    """Raw dataset for MedSAM2 training on HECKTOR 2026 SwinCross-format NPZ.

    Reads the shared SwinCross JSON split (``_classic.json``, ``_test.json``, …)
    and exposes each patient as a sequence of axial (R, A) slices for the SAM2
    video-object-segmentation training pipeline.

    Each frame is a (3, H, W) float32 tensor: [CT_normalised, PET_normalised, PET_normalised].
    Repeating PET in channel 2 fills the 3-channel RGB slot expected by the
    ImageNet-pretrained ViT backbone.

    Parameters
    ----------
    data_dir       : str  Root of the SwinCross NPZ dataset
                          (e.g. /data/ethan/PP_hecktor2026_kfold_npz).
    json_list      : str  JSON filename within data_dir
                          (e.g. dataset_swincross_2026kfold_classic.json).
    split          : str  JSON key to use: "training" or "validation" (default "training").
    sample_rate    : int  Keep every N-th axial slice (default 1 = all).
    truncate_video : int  If > 0, keep only the first N slices per patient.
    """

    def __init__(
        self,
        data_dir: str,
        json_list: str,
        split: str = "training",
        sample_rate: int = 1,
        truncate_video: int = -1,
        file_list_txt: Optional[str] = None,
        excluded_videos_list_txt: Optional[str] = None,
    ) -> None:
        self.data_dir      = data_dir
        self.sample_rate   = sample_rate
        self.truncate_video = truncate_video

        json_path = os.path.join(data_dir, json_list) if not os.path.isabs(json_list) else json_list
        with open(json_path) as f:
            dataset_json = json.load(f)

        # Support both named split and combined fallback
        entries = dataset_json.get(split, [])
        if not entries:
            # If the split key is absent (e.g. a test JSON whose cases sit under
            # "validation" for compatibility with SwinCross test.py convention)
            for key in ("training", "validation", "testing"):
                entries = dataset_json.get(key, [])
                if entries:
                    break

        if not entries:
            raise ValueError(
                f"No entries found under split='{split}' in {json_path}. "
                f"Available keys: {list(dataset_json.keys())}"
            )

        # Build list of (npz_rel_path, case_id) tuples
        all_cases = [
            (e["npz"], e.get("case_id") or os.path.basename(e["npz"]).replace(".npz", ""))
            for e in entries
            if e.get("npz")
        ]

        # Optional inclusion / exclusion lists (mirrors PNGRawDataset interface)
        if file_list_txt is not None:
            with open(file_list_txt) as f:
                allowed = {l.strip() for l in f if l.strip()}
            all_cases = [(p, c) for p, c in all_cases if c in allowed]

        excluded: set = set()
        if excluded_videos_list_txt is not None:
            with open(excluded_videos_list_txt) as f:
                excluded = {l.strip() for l in f if l.strip()}
        all_cases = [(p, c) for p, c in all_cases if c not in excluded]

        # video_names stores tuples (npz_rel, case_id)
        self.video_names = all_cases
        print(
            f"  [{split}] HECKTORNPZRawDataset: {len(self.video_names)} patients "
            f"from {os.path.basename(json_list)}"
        )

    # ── VOSRawDataset interface ────────────────────────────────────────────────

    def get_video(self, idx: int):
        """Load one patient NPZ and return (VOSVideo, NPZSegmentLoader).

        Returns
        -------
        tuple[VOSVideo, NPZSegmentLoader]
        """
        npz_rel, case_id = self.video_names[idx]
        npz_path = os.path.join(self.data_dir, npz_rel)

        with np.load(npz_path, allow_pickle=False) as npz:
            ct_ras  = npz["ct"].astype(np.float32)    # (R, A, S)
            pet_ras = npz["pet"].astype(np.float32)   # (R, A, S)
            lbl_ras = npz["label"].astype(np.uint8)   # (R, A, S)

        # Normalise intensities
        ct_ras  = _normalise_ct(ct_ras)
        pet_ras = _normalise_pet(pet_ras)

        # (R, A, S) -> (S, R, A) so that axis 0 iterates axial slices
        ct_slices  = np.moveaxis(ct_ras,  _SLICE_AXIS, 0)   # (S, R, A)
        pet_slices = np.moveaxis(pet_ras, _SLICE_AXIS, 0)
        lbl_slices = np.moveaxis(lbl_ras, _SLICE_AXIS, 0)

        S = ct_slices.shape[0]

        # Optional truncation then sub-sampling
        if self.truncate_video > 0:
            ct_slices  = ct_slices[:self.truncate_video]
            pet_slices = pet_slices[:self.truncate_video]
            lbl_slices = lbl_slices[:self.truncate_video]

        ct_slices  = ct_slices[::self.sample_rate]
        pet_slices = pet_slices[::self.sample_rate]
        lbl_slices = lbl_slices[::self.sample_rate]

        # Build per-slice 3-channel float tensors [CT, PET, PET]
        vos_frames: List[VOSFrame] = []
        for i in range(ct_slices.shape[0]):
            frame_tensor = self._fuse_modalities(ct_slices[i], pet_slices[i])
            vos_frames.append(VOSFrame(frame_idx=i, image_path=None, data=frame_tensor))

        video          = VOSVideo(video_name=case_id, video_id=idx, frames=vos_frames)
        segment_loader = NPZSegmentLoader(lbl_slices)   # (S', R, A) uint8
        return video, segment_loader

    def __len__(self) -> int:
        return len(self.video_names)

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _fuse_modalities(ct_slice: np.ndarray, pet_slice: np.ndarray) -> torch.Tensor:
        """Fuse a single (R, A) CT and PET slice into a (3, R, A) float32 tensor.

        Channel layout: [CT, PET, PET].
        Values are already in [0, 1] after normalisation.
        """
        fused = np.stack([ct_slice, pet_slice, pet_slice], axis=0).astype(np.float32)
        return torch.from_numpy(fused)
