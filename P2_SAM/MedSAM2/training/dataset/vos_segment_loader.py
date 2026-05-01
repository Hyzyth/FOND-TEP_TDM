"""
training/dataset/vos_segment_loader.py
========================================
Segment loaders for different annotation formats.

Classes
-------
PalettisedPNGSegmentLoader  – single paletted PNG per frame
MultiplePNGSegmentLoader    – one PNG per object per frame
NPZSegmentLoader            – masks stored in a pre-loaded numpy array (HECKTOR)
JSONSegmentLoader           – COCO-style JSON annotations
LazySegments                – lazy-loading wrapper for SA-1B style data
"""

import glob
import json
import os
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image as PILImage


# ──────────────────────────────────────────────────────────────────────────────

class PalettisedPNGSegmentLoader:
    """Load multi-object masks stored as a single palettised PNG per frame.

    Parameters
    ----------
    video_png_root : str
        Directory containing ``<frame_idx>.png`` files.
    sample_rate : int
        Keep every *sample_rate*-th file.
    """

    def __init__(self, video_png_root: str, sample_rate: int = 1) -> None:
        self.video_png_root = video_png_root
        self.sample_rate = sample_rate
        png_filenames = sorted(glob.glob(os.path.join(video_png_root, "*.png")))
        self.frame_id_to_png_filename: Dict[int, str] = {
            idx: fname
            for idx, fname in enumerate(png_filenames[::sample_rate])
        }

    def load(self, frame_id: int) -> Dict[int, torch.Tensor]:
        """Return ``{object_id: binary_mask_tensor}`` for *frame_id*.

        Parameters
        ----------
        frame_id : int

        Returns
        -------
        dict[int, torch.Tensor]  bool tensors of shape (H, W)
        """
        mask_path = self.frame_id_to_png_filename[frame_id]
        masks = np.array(PILImage.open(mask_path).convert("P"))
        object_ids = np.unique(masks)
        object_ids = object_ids[object_ids != 0]
        return {
            int(oid): torch.from_numpy(masks == oid)
            for oid in object_ids
        }

    def __len__(self) -> int:
        return len(self.frame_id_to_png_filename)


# ──────────────────────────────────────────────────────────────────────────────

class MultiplePNGSegmentLoader:
    """Load masks stored as one PNG per object per frame.

    Parameters
    ----------
    video_png_root : str
        Directory whose sub-directories are named by object ID.
    single_object_mode : bool
        When True the root contains the PNGs directly (one object only).
    """

    def __init__(self, video_png_root: str, single_object_mode: bool = False) -> None:
        self.video_png_root = video_png_root
        self.single_object_mode = single_object_mode

        if single_object_mode:
            tmp = glob.glob(os.path.join(video_png_root, "*.png"))[0]
        else:
            tmp = glob.glob(os.path.join(video_png_root, "*", "*.png"))[0]

        tmp_mask = np.array(PILImage.open(tmp))
        self.H, self.W = tmp_mask.shape[:2]
        self.obj_id = (
            int(video_png_root.split("/")[-1]) + 1 if single_object_mode else None
        )

    def load(self, frame_id: int) -> Dict[int, torch.Tensor]:
        if self.single_object_mode:
            return self._load_single(frame_id)
        return self._load_multiple(frame_id)

    def _load_single(self, frame_id: int) -> Dict[int, torch.Tensor]:
        mask_path = os.path.join(self.video_png_root, f"{frame_id:05d}.png")
        if os.path.exists(mask_path):
            mask = np.array(PILImage.open(mask_path)) > 0
        else:
            mask = np.zeros((self.H, self.W), dtype=bool)
        return {self.obj_id: torch.from_numpy(mask)}

    def _load_multiple(self, frame_id: int) -> Dict[int, torch.Tensor]:
        result = {}
        for obj_folder in sorted(glob.glob(os.path.join(self.video_png_root, "*"))):
            obj_id = int(obj_folder.split("/")[-1]) + 1
            mask_path = os.path.join(obj_folder, f"{frame_id:05d}.png")
            if os.path.exists(mask_path):
                mask = np.array(PILImage.open(mask_path)) > 0
            else:
                mask = np.zeros((self.H, self.W), dtype=bool)
            result[obj_id] = torch.from_numpy(mask)
        return result


# ──────────────────────────────────────────────────────────────────────────────

class NPZSegmentLoader:
    """Load per-frame segmentation masks from a pre-loaded numpy array.

    Used with HECKTOR NPZ files where the ``gts`` array has shape (D, H, W)
    and contains integer labels (0 = background, 1 = GTVp, 2 = GTVn).

    Parameters
    ----------
    masks : np.ndarray
        Integer array of shape (D, H, W).
    """

    def __init__(self, masks: np.ndarray) -> None:
        self.masks = masks   # (D, H, W) uint8

    def load(self, frame_idx: int) -> Dict[int, torch.Tensor]:
        """Return ``{label_id: binary_mask_tensor}`` for *frame_idx*.

        Parameters
        ----------
        frame_idx : int

        Returns
        -------
        dict[int, torch.Tensor]  bool tensors of shape (H, W)
        """
        mask = self.masks[frame_idx]
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]
        return {
            int(oid): torch.from_numpy(mask == oid).bool()
            for oid in object_ids
        }


# ──────────────────────────────────────────────────────────────────────────────

class LazySegments:
    """Lazy container for SA-1B style segment annotations (loaded on demand)."""

    def __init__(self, data: dict) -> None:
        self._data = data

    def keys(self):
        return self._data.keys()

    def __getitem__(self, key):
        return self._data[key]


class JSONSegmentLoader:
    """Load segment masks from a COCO-style JSON annotation file.

    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    """

    def __init__(self, json_path: str) -> None:
        with open(json_path) as f:
            self._data = json.load(f)

    def load(
        self,
        frame_id: int,
        obj_ids: Optional[list] = None,
    ) -> Dict[int, torch.Tensor]:
        """Return masks for *frame_id*, optionally filtered to *obj_ids*.

        Parameters
        ----------
        frame_id : int
        obj_ids : list, optional
            If provided only return masks whose object IDs are in this list.

        Returns
        -------
        dict[int, torch.Tensor]
        """
        frame_data = self._data.get(str(frame_id), {})
        result = {}
        for obj_id_str, rle in frame_data.items():
            obj_id = int(obj_id_str)
            if obj_ids is not None and obj_id not in obj_ids:
                continue
            mask = self._decode_rle(rle)
            result[obj_id] = torch.from_numpy(mask)
        return result

    @staticmethod
    def _decode_rle(rle: dict) -> np.ndarray:
        """Decode an uncompressed RLE dict to a boolean numpy array."""
        h, w = rle["size"]
        mask = np.zeros(h * w, dtype=bool)
        idx, parity = 0, False
        for count in rle["counts"]:
            mask[idx : idx + count] = parity
            idx += count
            parity = not parity
        return mask.reshape(w, h).T
