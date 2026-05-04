"""
hecktor_dataset.py
==================
HECKTOR-specific raw dataset for dual-modality (CT + PET) head-and-neck
tumour segmentation.

The HECKTOR NPZ files produced by ``data_preparation/prepare_hecktor_npz.py``
contain:
    ``ct_imgs``  – (D, H, W) uint8, CT windowed to [0, 255]
    ``pet_imgs`` – (D, H, W) uint8, PET SUV normalised to [0, 255]
    ``gts``      – (D, H, W) uint8, combined GT mask:
                       0 = background
                       1 = GTVp (primary tumour)
                       2 = GTVn (nodal tumour; may be absent)
    ``spacing``  – (3,) float64, voxel spacing in mm (z, y, x)

Each axial slice is fused into a 3-channel tensor [CT, PET, PET] so that
SAM2's ImageNet-pretrained ViT backbone can ingest it.
"""

import os
from typing import List, Optional

import numpy as np
import torch

from training.dataset.vos_raw_dataset import VOSFrame, VOSVideo, VOSRawDataset
from training.dataset.vos_segment_loader import NPZSegmentLoader


class HECKTORNPZRawDataset(VOSRawDataset):
    """Dataset loader for HECKTOR NPZ files (CT + PET dual-modality).

    The loader fuses CT and PET into a 3-channel tensor per slice:
        channel 0 → CT
        channel 1 → PET
        channel 2 → PET   (repeat to fill the 3-channel ImageNet backbone)

    Parameters
    ----------
    folder : str
        Directory containing NPZ files (sub-directories are scanned recursively).
    file_list_txt : str, optional
        Plain-text file listing allowed NPZ basenames (one per line, no extension).
    excluded_videos_list_txt : str, optional
        Plain-text file of NPZ basenames to exclude.
    sample_rate : int
        Keep every *sample_rate*-th axial slice (default 1 = all).
    truncate_video : int
        If > 0, keep only the first *truncate_video* slices per patient.
    """

    def __init__(
        self,
        folder: str,
        file_list_txt: Optional[str] = None,
        excluded_videos_list_txt: Optional[str] = None,
        sample_rate: int = 1,
        truncate_video: int = -1,
    ) -> None:
        self.folder = folder
        self.sample_rate = sample_rate
        self.truncate_video = truncate_video

        # Discover all NPZ files recursively.
        all_npz: List[str] = []
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.endswith(".npz"):
                    rel = os.path.relpath(os.path.join(root, fname), folder)
                    all_npz.append(os.path.splitext(rel)[0])

        # Apply inclusion list.
        if file_list_txt is not None:
            with open(file_list_txt) as f:
                allowed = {l.strip() for l in f if l.strip()}
            all_npz = [v for v in all_npz if v in allowed]

        # Apply exclusion list.
        excluded: set = set()
        if excluded_videos_list_txt is not None:
            with open(excluded_videos_list_txt) as f:
                excluded = {os.path.splitext(l.strip())[0] for l in f if l.strip()}

        self.video_names = sorted(v for v in all_npz if v not in excluded)

    # ------------------------------------------------------------------
    # VOSRawDataset interface
    # ------------------------------------------------------------------

    def get_video(self, idx: int):
        """Load one patient NPZ and return ``(VOSVideo, NPZSegmentLoader)``.

        Parameters
        ----------
        idx : int Index into ``self.video_names``.

        Returns
        -------
        tuple[VOSVideo, NPZSegmentLoader]
        """
        video_name = self.video_names[idx]
        npz_path   = os.path.join(self.folder, f"{video_name}.npz")

        npz_data = np.load(npz_path, allow_pickle=True)

        # Validate expected keys
        for key in ("ct_imgs", "pet_imgs", "gts"):
            if key not in npz_data:
                raise KeyError(
                    f"NPZ file {npz_path!r} is missing required key '{key}'. "
                    "Re-run prepare_hecktor_npz.py to regenerate the NPZ files."
                )

        ct_imgs  = npz_data["ct_imgs"]   # (D, H, W) uint8
        pet_imgs = npz_data["pet_imgs"]  # (D, H, W) uint8
        gts      = npz_data["gts"]       # (D, H, W) uint8   0/1/2

        # Optional slice truncation and sub-sampling
        if self.truncate_video > 0:
            ct_imgs  = ct_imgs[:self.truncate_video]
            pet_imgs = pet_imgs[:self.truncate_video]
            gts      = gts[:self.truncate_video]

        ct_imgs  = ct_imgs[::self.sample_rate]
        pet_imgs = pet_imgs[::self.sample_rate]
        gts      = gts[::self.sample_rate]

        # Build per-slice 3-channel float tensors [CT, PET, PET] in [0, 1]
        vos_frames: List[VOSFrame] = []
        for i in range(ct_imgs.shape[0]):
            frame_tensor = self._fuse_modalities(ct_imgs[i], pet_imgs[i])
            vos_frames.append(
                VOSFrame(frame_idx=i, image_path=None, data=frame_tensor)
            )

        video          = VOSVideo(video_name=video_name, video_id=idx, frames=vos_frames)
        segment_loader = NPZSegmentLoader(gts)
        return video, segment_loader

    def __len__(self) -> int:
        return len(self.video_names)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse_modalities(ct_slice: np.ndarray, pet_slice: np.ndarray) -> torch.Tensor:
        """Fuse a single CT and PET axial slice into a (3, H, W) float tensor.

        Channel layout: [CT, PET, PET] (values in [0, 1]).
        Repeating PET in channel 2 fills the 3-channel RGB slot expected by
        the ImageNet-pretrained ViT backbone used in SAM2.

        Parameters
        ----------
        ct_slice  : (H, W) uint8
        pet_slice : (H, W) uint8

        Returns
        -------
        torch.Tensor  (3, H, W) float32
        """
        ct_f  = ct_slice.astype(np.float32)  / 255.0
        pet_f = pet_slice.astype(np.float32) / 255.0
        fused = np.stack([ct_f, pet_f, pet_f], axis=0)  # (3, H, W)
        return torch.from_numpy(fused)
