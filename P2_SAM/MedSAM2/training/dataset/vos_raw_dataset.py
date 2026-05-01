"""
training/dataset/vos_raw_dataset.py
=====================================
Raw dataset classes that handle file I/O before the VOS pipeline applies
transforms and sampling.

Classes
-------
VOSFrame    – dataclass describing a single video frame
VOSVideo    – dataclass describing a video (list of VOSFrames)
VOSRawDataset – abstract base
PNGRawDataset – frames as JPEG/PNG images on disk
NPZRawDataset – generic NPZ loader (grayscale, single-channel gts)

For HECKTOR (dual-modality CT+PET) use
``training.dataset.hecktor_dataset.HECKTORNPZRawDataset`` instead of
``NPZRawDataset``.
"""

import glob
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from training.dataset.vos_segment_loader import (
    MultiplePNGSegmentLoader,
    NPZSegmentLoader,
    PalettisedPNGSegmentLoader,
)


@dataclass
class VOSFrame:
    """Single annotated frame within a video.

    Parameters
    ----------
    frame_idx : int
        Zero-based frame index.
    image_path : str or None
        Path to the RGB image file.  ``None`` when ``data`` is pre-loaded.
    data : torch.Tensor or None
        Pre-loaded (3, H, W) float tensor in [0, 1].  Takes precedence
        over ``image_path`` when not ``None``.
    is_conditioning_only : bool
        Reserved flag for future conditioning-only frames.
    """

    frame_idx: int
    image_path: Optional[str]
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    """Sequence of frames representing one patient / video.

    Parameters
    ----------
    video_name : str
    video_id : int
    frames : list[VOSFrame]
    """

    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self) -> int:
        return len(self.frames)


# ──────────────────────────────────────────────────────────────────────────────

class VOSRawDataset:
    """Abstract base class for VOS raw datasets."""

    def get_video(self, idx: int):
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────

class PNGRawDataset(VOSRawDataset):
    """Dataset that reads video frames from JPEG/PNG files on disk.

    Parameters
    ----------
    img_folder : str
        Root directory containing one sub-folder per video.
    gt_folder : str
        Root directory containing mask annotations (mirrored structure).
    file_list_txt : str, optional
        Text file listing allowed video names (one per line).
    excluded_videos_list_txt : str, optional
        Text file listing video names to exclude.
    sample_rate : int
        Keep every *sample_rate*-th frame (default 1).
    is_palette : bool
        When True use ``PalettisedPNGSegmentLoader``; otherwise
        ``MultiplePNGSegmentLoader``.
    single_object_mode : bool
        Each sub-directory contains masks for one object only.
    truncate_video : int
        If > 0 keep only the first *truncate_video* frames.
    frames_sampling_mult : bool
        Multiply dataset size by the number of frames per video.
    """

    def __init__(
        self,
        img_folder: str,
        gt_folder: str,
        file_list_txt: Optional[str] = None,
        excluded_videos_list_txt: Optional[str] = None,
        sample_rate: int = 1,
        is_palette: bool = True,
        single_object_mode: bool = False,
        truncate_video: int = -1,
        frames_sampling_mult: bool = False,
    ) -> None:
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        subset = (
            [l.strip().split(".")[0] for l in open(file_list_txt)]
            if file_list_txt
            else os.listdir(img_folder)
        )
        excluded = (
            {l.strip().split(".")[0] for l in open(excluded_videos_list_txt)}
            if excluded_videos_list_txt
            else set()
        )

        video_names = sorted(v for v in subset if v not in excluded)

        if single_object_mode:
            video_names = sorted(
                os.path.join(v, obj)
                for v in video_names
                for obj in os.listdir(os.path.join(gt_folder, v))
            )

        if frames_sampling_mult:
            video_names = [
                v
                for v in video_names
                for _ in range(len(os.listdir(os.path.join(img_folder, v))))
            ]

        self.video_names = video_names

    def get_video(self, idx: int):
        video_name = self.video_names[idx]

        frame_root = os.path.join(
            self.img_folder,
            os.path.dirname(video_name) if self.single_object_mode else video_name,
        )
        mask_root = os.path.join(self.gt_folder, video_name)

        segment_loader = (
            PalettisedPNGSegmentLoader(mask_root, sample_rate=self.sample_rate)
            if self.is_palette
            else MultiplePNGSegmentLoader(mask_root, self.single_object_mode)
        )

        all_frames = sorted(glob.glob(os.path.join(frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]

        frames = [
            VOSFrame(frame_idx=i, image_path=path)
            for i, path in enumerate(all_frames[:: self.sample_rate])
        ]
        return VOSVideo(video_name, idx, frames), segment_loader

    def __len__(self) -> int:
        return len(self.video_names)


# ──────────────────────────────────────────────────────────────────────────────

class NPZRawDataset(VOSRawDataset):
    """Generic NPZ dataset for single-channel (grayscale) medical images.

    Each NPZ file must contain:
        ``imgs`` – (D, H, W) uint8 grayscale frames
        ``gts``  – (D, H, W) uint8 integer label mask

    For HECKTOR (CT + PET dual-modality) use
    ``training.dataset.hecktor_dataset.HECKTORNPZRawDataset`` instead,
    which handles ``ct_imgs`` + ``pet_imgs`` and fuses them into 3 channels.

    Parameters
    ----------
    folder : str
        Directory (possibly nested) containing NPZ files.
    file_list_txt : str, optional
    excluded_videos_list_txt : str, optional
    sample_rate : int
    truncate_video : int
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
        subset = []
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.endswith(".npz"):
                    rel = os.path.relpath(os.path.join(root, fname), folder)
                    subset.append(os.path.splitext(rel)[0])

        if file_list_txt:
            with open(file_list_txt) as f:
                allowed = {l.strip() for l in f if l.strip()}
            subset = [v for v in subset if v in allowed]

        excluded = set()
        if excluded_videos_list_txt:
            with open(excluded_videos_list_txt) as f:
                excluded = {os.path.splitext(l.strip())[0] for l in f if l.strip()}

        self.video_names = sorted(v for v in subset if v not in excluded)

    def get_video(self, idx: int):
        video_name = self.video_names[idx]
        npz_path = os.path.join(self.folder, f"{video_name}.npz")

        npz_data = np.load(npz_path)
        frames_arr = npz_data["imgs"]          # (D, H, W) uint8
        gts = npz_data["gts"]                  # (D, H, W) uint8

        if self.truncate_video > 0:
            frames_arr = frames_arr[: self.truncate_video]
            gts = gts[: self.truncate_video]

        frames_arr = frames_arr[:: self.sample_rate]
        gts = gts[:: self.sample_rate]

        # Expand grayscale to 3 channels [0, 1].
        frames_3ch = np.repeat(frames_arr[:, np.newaxis], 3, axis=1).astype(
            np.float32
        ) / 255.0  # (D, 3, H, W)

        vos_frames = [
            VOSFrame(
                frame_idx=i,
                image_path=None,
                data=torch.from_numpy(frames_3ch[i]),
            )
            for i in range(frames_3ch.shape[0])
        ]

        video = VOSVideo(video_name, idx, vos_frames)
        segment_loader = NPZSegmentLoader(gts[:: self.sample_rate])
        return video, segment_loader

    def __len__(self) -> int:
        return len(self.video_names)
