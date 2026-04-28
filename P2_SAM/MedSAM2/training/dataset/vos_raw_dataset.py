# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import logging
import os
from dataclasses import dataclass

from typing import List, Optional

import pandas as pd

import torch
import numpy as np

from iopath.common.file_io import g_pathmgr

from training.dataset.vos_segment_loader import (
    MultiplePNGSegmentLoader,
    PalettisedPNGSegmentLoader,
    NPZSegmentLoader
)


@dataclass
class VOSFrame:
    frame_idx: int
    image_path: str
    data: Optional[torch.Tensor] = None
    is_conditioning_only: Optional[bool] = False


@dataclass
class VOSVideo:
    video_name: str
    video_id: int
    frames: List[VOSFrame]

    def __len__(self):
        return len(self.frames)


class VOSRawDataset:
    def __init__(self):
        pass

    def get_video(self, idx):
        raise NotImplementedError()


class PNGRawDataset(VOSRawDataset):
    def __init__(
        self,
        img_folder,
        gt_folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        is_palette=True,
        single_object_mode=False,
        truncate_video=-1,
        frames_sampling_mult=False,
    ):
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.sample_rate = sample_rate
        self.is_palette = is_palette
        self.single_object_mode = single_object_mode
        self.truncate_video = truncate_video

        # Read the subset defined in file_list_txt
        if file_list_txt is not None:
            with g_pathmgr.open(file_list_txt, "r") as f:
                subset = [os.path.splitext(line.strip())[0] for line in f]
        else:
            subset = os.listdir(self.img_folder)

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with g_pathmgr.open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

        if self.single_object_mode:
            # single object mode
            self.video_names = sorted(
                [
                    os.path.join(video_name, obj)
                    for video_name in self.video_names
                    for obj in os.listdir(os.path.join(self.gt_folder, video_name))
                ]
            )

        if frames_sampling_mult:
            video_names_mult = []
            for video_name in self.video_names:
                num_frames = len(os.listdir(os.path.join(self.img_folder, video_name)))
                video_names_mult.extend([video_name] * num_frames)
            self.video_names = video_names_mult

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]

        if self.single_object_mode:
            video_frame_root = os.path.join(
                self.img_folder, os.path.dirname(video_name)
            )
        else:
            video_frame_root = os.path.join(self.img_folder, video_name)

        video_mask_root = os.path.join(self.gt_folder, video_name)

        if self.is_palette:
            segment_loader = PalettisedPNGSegmentLoader(video_mask_root, sample_rate=self.sample_rate)
        else:
            segment_loader = MultiplePNGSegmentLoader(
                video_mask_root, self.single_object_mode
            )

        all_frames = sorted(glob.glob(os.path.join(video_frame_root, "*.jpg")))
        if self.truncate_video > 0:
            all_frames = all_frames[: self.truncate_video]
        frames = []
        for idx, fpath in enumerate(all_frames[::self.sample_rate]):
            fid = idx # int(os.path.basename(fpath).split(".")[0])
            frames.append(VOSFrame(fid, image_path=fpath))
        video = VOSVideo(video_name, idx, frames)
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)

class NPZRawDataset(VOSRawDataset):
    def __init__(
        self,
        folder,
        file_list_txt=None,
        excluded_videos_list_txt=None,
        sample_rate=1,
        truncate_video=-1,
    ):
        self.folder = folder
        self.sample_rate = sample_rate
        self.truncate_video = truncate_video

        # Read all npz files from folder and its subfolders
        subset = []
        for root, _, files in os.walk(self.folder):
            for file in files:
                if file.endswith('.npz'):
                    # Get the relative path from the root folder
                    rel_path = os.path.relpath(os.path.join(root, file), self.folder)
                    # Remove the .npz extension
                    subset.append(os.path.splitext(rel_path)[0])

        # Read the subset defined in file_list_txt if provided
        if file_list_txt is not None:
            with open(file_list_txt, "r") as f:
                subset = [line.strip() for line in f if line.strip() in subset]

        # Read and process excluded files if provided
        if excluded_videos_list_txt is not None:
            with open(excluded_videos_list_txt, "r") as f:
                excluded_files = [os.path.splitext(line.strip())[0] for line in f]
        else:
            excluded_files = []

        # Check if it's not in excluded_files
        self.video_names = sorted(
            [video_name for video_name in subset if video_name not in excluded_files]
        )

    def get_video(self, idx):
        """
        Given a VOSVideo object, return the mask tensors.
        """
        video_name = self.video_names[idx]
        npz_path = os.path.join(self.folder, f"{video_name}.npz")
        
        # Load NPZ file
        npz_data = np.load(npz_path)
        
        # Extract frames and masks
        frames = npz_data['imgs'] / 255.0
        # Expand the grayscale images to three channels
        frames = np.repeat(frames[:, np.newaxis, :, :], 3, axis=1)  # (img_num, 3, H, W)
        masks = npz_data['gts']
        
        if self.truncate_video > 0:
            frames = frames[:self.truncate_video]
            masks = masks[:self.truncate_video]
        
        # Create VOSFrame objects
        vos_frames = []
        for i, frame in enumerate(frames[::self.sample_rate]):
            frame_idx = i * self.sample_rate
            vos_frames.append(VOSFrame(frame_idx, image_path=None, data=torch.from_numpy(frame)))
        
        # Create VOSVideo object
        video = VOSVideo(video_name, idx, vos_frames)
        
        # Create NPZSegmentLoader
        segment_loader = NPZSegmentLoader(masks[::self.sample_rate])
        
        return video, segment_loader

    def __len__(self):
        return len(self.video_names)

