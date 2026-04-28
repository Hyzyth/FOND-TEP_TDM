# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import json
import os

import numpy as np
import pandas as pd
import torch

from PIL import Image as PILImage


class PalettisedPNGSegmentLoader:
    def __init__(self, video_png_root, sample_rate=1):
        """
        SegmentLoader for datasets with masks stored as palettised PNGs.
        video_png_root: the folder contains all the masks stored in png
        """
        self.video_png_root = video_png_root
        self.sample_rate = sample_rate
        # build a mapping from frame id to their PNG mask path
        # note that in some datasets, the PNG paths could have more
        # than 5 digits, e.g. "00000000.png" instead of "00000.png"
        png_filenames = sorted(glob.glob(os.path.join(self.video_png_root, "*.png"))) # os.listdir(self.video_png_root)
        self.frame_id_to_png_filename = {}
        for idx, filename in enumerate(png_filenames[::self.sample_rate]):
            frame_id = idx # int(os.path.basename(filename).split(".")[0])
            self.frame_id_to_png_filename[frame_id] = filename

    def load(self, frame_id):
        """
        load the single palettised mask from the disk (path: f'{self.video_png_root}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        # check the path
        mask_path = os.path.join(
            self.video_png_root, self.frame_id_to_png_filename[frame_id]
        )

        # load the mask
        masks = PILImage.open(mask_path).convert("P")
        masks = np.array(masks)

        object_id = pd.unique(masks.flatten())
        object_id = object_id[object_id != 0]  # remove background (0)

        # convert into N binary segmentation masks
        binary_segments = {}
        for i in object_id:
            bs = masks == i
            binary_segments[i] = torch.from_numpy(bs)

        return binary_segments

    def __len__(self):
        return


class MultiplePNGSegmentLoader:
    def __init__(self, video_png_root, single_object_mode=False):
        """
        video_png_root: the folder contains all the masks stored in png
        single_object_mode: whether to load only a single object at a time
        """
        self.video_png_root = video_png_root
        self.single_object_mode = single_object_mode
        # read a mask to know the resolution of the video
        if self.single_object_mode:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*.png"))[0]
        else:
            tmp_mask_path = glob.glob(os.path.join(video_png_root, "*", "*.png"))[0]
        tmp_mask = np.array(PILImage.open(tmp_mask_path))
        self.H = tmp_mask.shape[0]
        self.W = tmp_mask.shape[1]
        if self.single_object_mode:
            self.obj_id = (
                int(video_png_root.split("/")[-1]) + 1
            )  # offset by 1 as bg is 0
        else:
            self.obj_id = None

    def load(self, frame_id):
        if self.single_object_mode:
            return self._load_single_png(frame_id)
        else:
            return self._load_multiple_pngs(frame_id)

    def _load_single_png(self, frame_id):
        """
        load single png from the disk (path: f'{self.obj_id}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        mask_path = os.path.join(self.video_png_root, f"{frame_id:05d}.png")
        binary_segments = {}

        if os.path.exists(mask_path):
            mask = np.array(PILImage.open(mask_path))
        else:
            # if png doesn't exist, empty mask
            mask = np.zeros((self.H, self.W), dtype=bool)
        binary_segments[self.obj_id] = torch.from_numpy(mask > 0)
        return binary_segments

    def _load_multiple_pngs(self, frame_id):
        """
        load multiple png masks from the disk (path: f'{obj_id}/{frame_id:05d}.png')
        Args:
            frame_id: int, define the mask path
        Return:
            binary_segments: dict
        """
        # get the path
        all_objects = sorted(glob.glob(os.path.join(self.video_png_root, "*")))
        num_objects = len(all_objects)
        assert num_objects > 0

        # load the masks
        binary_segments = {}
        for obj_folder in all_objects:
            # obj_folder is {video_name}/{obj_id}, obj_id is specified by the name of the folder
            obj_id = int(obj_folder.split("/")[-1])
            obj_id = obj_id + 1  # offset 1 as bg is 0
            mask_path = os.path.join(obj_folder, f"{frame_id:05d}.png")
            if os.path.exists(mask_path):
                mask = np.array(PILImage.open(mask_path))
            else:
                mask = np.zeros((self.H, self.W), dtype=bool)
            binary_segments[obj_id] = torch.from_numpy(mask > 0)

        return binary_segments

    def __len__(self):
        return


class NPZSegmentLoader:
    def __init__(self, masks):
        """
        Initialize the NPZSegmentLoader.
        
        Args:
            masks (numpy.ndarray): Array of masks with shape (img_num, H, W).
        """
        self.masks = masks

    def load(self, frame_idx):
        """
        Load the single mask for the given frame index and convert it to binary segments.

        Args:
            frame_idx (int): Index of the frame to load.

        Returns:
            dict: A dictionary where keys are object IDs and values are binary masks.
        """
        mask = self.masks[frame_idx]

        # Find unique object IDs in the mask, excluding the background (0)
        object_ids = np.unique(mask)
        object_ids = object_ids[object_ids != 0]

        # Convert into binary segmentation masks for each object
        binary_segments = {}
        for obj_id in object_ids:
            binary_mask = (mask == obj_id)
            binary_segments[int(obj_id)] = torch.from_numpy(binary_mask).bool()

        return binary_segments
