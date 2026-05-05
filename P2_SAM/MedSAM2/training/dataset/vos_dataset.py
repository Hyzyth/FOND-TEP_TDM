"""
training/dataset/vos_dataset.py
================================
PyTorch Dataset wrapping a :class:`VOSRawDataset` + :class:`VOSSampler`
pair with optional augmentation transforms.

For HECKTOR NPZ frames, where frame.data is already a (3, H, W) float tensor,
the PIL round-trip is bypassed entirely: the tensor is used directly, avoiding
the lossy uint8 quantisation that would occur via _tensor_to_pil → ToTensorAPI.
"""

import logging
import random
from copy import deepcopy
from typing import List

import numpy as np
import torch
from iopath.common.file_io import g_pathmgr
from PIL import Image as PILImage
from torchvision.datasets.vision import VisionDataset

from training.dataset.vos_raw_dataset import VOSRawDataset
from training.dataset.vos_sampler import VOSSampler
from training.dataset.vos_segment_loader import JSONSegmentLoader
from training.utils.data_utils import Frame, Object, VideoDatapoint

MAX_RETRIES = 100


class VOSDataset(VisionDataset):
    """Video Object Segmentation dataset with transform support.

    Parameters
    ----------
    transforms : list
        Sequence of callable augmentation transforms.
    training : bool
        When True failed loads are retried with a random index.
    video_dataset : VOSRawDataset
        Raw dataset providing ``(VOSVideo, segment_loader)`` pairs.
    sampler : VOSSampler
        Sampling strategy for frames and objects.
    multiplier : float
        Repeat factor applied to every sample (see ``RepeatFactorWrapper``).
    always_target : bool
        Insert zero masks for objects absent in a given frame.
    target_segments_available : bool
        Whether ground-truth masks are expected.
    """

    def __init__(
        self,
        transforms,
        training: bool,
        video_dataset: VOSRawDataset,
        sampler: VOSSampler,
        multiplier: float,
        always_target: bool = True,
        target_segments_available: bool = True,
    ) -> None:
        self._transforms = transforms
        self.training = training
        self.video_dataset = video_dataset
        self.sampler = sampler
        self.repeat_factors = torch.ones(
            len(self.video_dataset), dtype=torch.float32
        ) * multiplier
        self.curr_epoch = 0
        self.always_target = always_target
        self.target_segments_available = target_segments_available

    def _get_datapoint(self, idx: int) -> VideoDatapoint:
        for retry in range(MAX_RETRIES):
            try:
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                video, segment_loader = self.video_dataset.get_video(idx)
                sampled = self.sampler.sample(
                    video, segment_loader, epoch=self.curr_epoch
                )
                break
            except Exception as e:
                if self.training:
                    logging.warning(
                        f"Loading failed (idx={idx}, retry={retry}): {e}"
                    )
                    idx = random.randrange(0, len(self.video_dataset))
                else:
                    raise

        datapoint = self._construct(video, sampled, segment_loader)
        for transform in self._transforms:
            datapoint = transform(datapoint, epoch=self.curr_epoch)
        return datapoint

    def _construct(self, video, sampled, segment_loader) -> VideoDatapoint:
        """Build a :class:`VideoDatapoint` from sampled frames and objects."""
        sampled_frames     = sampled.frames
        sampled_object_ids = sampled.object_ids

        rgb_images = _load_images(sampled_frames)
        images: List[Frame] = []

        for frame_idx, frame in enumerate(sampled_frames):
            img = rgb_images[frame_idx]
            # PIL images have a .size attribute; tensors do not.
            if isinstance(img, PILImage.Image):
                w, h = img.size
            else:
                # Pre-loaded float tensor (C, H, W)
                h, w = img.shape[1], img.shape[2]
            images.append(Frame(data=img, objects=[]))

            if isinstance(segment_loader, JSONSegmentLoader):
                segments = segment_loader.load(
                    frame.frame_idx, obj_ids=sampled_object_ids
                )
            else:
                segments = segment_loader.load(frame.frame_idx)

            for obj_id in sampled_object_ids:
                if obj_id in segments:
                    seg = segments[obj_id].to(torch.uint8)
                elif self.always_target:
                    seg = torch.zeros(h, w, dtype=torch.uint8)
                else:
                    continue

                images[frame_idx].objects.append(
                    Object(
                        object_id=obj_id,
                        frame_index=frame.frame_idx,
                        segment=seg,
                    )
                )

        return VideoDatapoint(frames=images, video_id=video.video_id, size=(h, w))

    def __getitem__(self, idx: int) -> VideoDatapoint:
        return self._get_datapoint(idx)

    def __len__(self) -> int:
        return len(self.video_dataset)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_images(frames) -> List[PILImage.Image]:
    """Load images for a list of frames.

    - If ``frame.data`` is a pre-loaded tensor (HECKTOR NPZ path), return it
      directly — no PIL conversion, no uint8 quantisation.
    - Otherwise open the image from disk as an RGB PIL image.
    """
    images = []
    cache: dict = {}
    for frame in frames:
        if frame.data is not None:
            # Pre-loaded (3, H, W) float tensor.
            # Transforms that require PIL (e.g. ColorJitter) will receive
            # this tensor; those transforms must handle both PIL and tensor
            # inputs, which the torchvision functional API supports.
            images.append(frame.data)
        else:
            path = frame.image_path
            if path in cache:
                images.append(deepcopy(images[cache[path]]))
            else:
                with g_pathmgr.open(path, "rb") as fh:
                    images.append(PILImage.open(fh).convert("RGB"))
                cache[path] = len(images) - 1
    return images


def _tensor_to_pil(data: torch.Tensor) -> PILImage.Image:   #Retained for any callers outside the main training loop that may still need a PIL representation. Not used internally by _load_images anymore.
    """Convert a (3, H, W) float [0,1] tensor to a uint8 PIL RGB image."""
    arr = (data.cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return PILImage.fromarray(arr)
