"""
training/utils/data_utils.py
==============================
Data structures and collate function shared by the training pipeline.

BatchedVideoDatapoint is a TensorClass (from tensordict) that holds a batch
of T×B video clips with associated masks and metadata.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from PIL import Image as PILImage
from tensordict import tensorclass


@tensorclass
class BatchedVideoMetaData:
    """Per-object metadata stored as long tensors.

    Attributes
    ----------
    unique_objects_identifier : LongTensor  shape (T, O, 3)
        Columns: video_id, object_id, frame_index.
    frame_orig_size : LongTensor  shape (T, O, 2)
        Original frame (H, W) before resizing.
    """

    unique_objects_identifier: torch.LongTensor
    frame_orig_size: torch.LongTensor


@tensorclass
class BatchedVideoDatapoint:
    """Batched video data for the training loop.

    Attributes
    ----------
    img_batch : FloatTensor  shape (T, B, C, H, W)
        T = frames per video, B = videos per batch.
    obj_to_frame_idx : IntTensor  shape (T, O, 2)
        Columns: frame_index, video_index.
    masks : BoolTensor  shape (T, O, H, W)
    metadata : BatchedVideoMetaData
    dict_key : str
        Dataset key used by the loss dispatcher.
    """

    img_batch: torch.FloatTensor
    obj_to_frame_idx: torch.IntTensor
    masks: torch.BoolTensor
    metadata: BatchedVideoMetaData
    dict_key: str

    def pin_memory(self, device=None):
        return self.apply(torch.Tensor.pin_memory, device=device)

    @property
    def num_frames(self) -> int:
        """Number of frames per video (T)."""
        return self.batch_size[0]

    @property
    def num_videos(self) -> int:
        """Number of videos in the batch (B)."""
        return self.img_batch.shape[1]

    @property
    def flat_obj_to_img_idx(self) -> torch.IntTensor:
        """Flat indices into the (T*B) image axis for each object."""
        frame_idx, video_idx = self.obj_to_frame_idx.unbind(dim=-1)
        return video_idx * self.num_frames + frame_idx

    @property
    def flat_img_batch(self) -> torch.FloatTensor:
        """Image batch flattened to shape (B*T, C, H, W)."""
        return self.img_batch.transpose(0, 1).flatten(0, 1)


@dataclass
class Object:
    """Single annotated object within a frame.

    Attributes
    ----------
    object_id : int
    frame_index : int
    segment : torch.Tensor or dict
        Binary mask tensor (H, W) or RLE dict.
    """

    object_id: int
    frame_index: int
    segment: Union[torch.Tensor, dict]


@dataclass
class Frame:
    """Single video frame with its objects.

    Attributes
    ----------
    data : torch.Tensor or PIL.Image
    objects : list[Object]
    """

    data: Union[torch.Tensor, PILImage.Image]
    objects: List[Object]


@dataclass
class VideoDatapoint:
    """One video/patient with all its annotated frames.

    Attributes
    ----------
    frames : list[Frame]
    video_id : int
    size : tuple[int, int]  (H, W)
    """

    frames: List[Frame]
    video_id: int
    size: Tuple[int, int]


def collate_fn(
    batch: List[VideoDatapoint],
    dict_key: str,
) -> BatchedVideoDatapoint:
    """Collate a list of VideoDatapoints into a BatchedVideoDatapoint.

    Parameters
    ----------
    batch : list[VideoDatapoint]
    dict_key : str
        Dataset identifier forwarded to the loss function dispatcher.

    Returns
    -------
    BatchedVideoDatapoint
    """
    img_batch = torch.stack(
        [torch.stack([f.data for f in vid.frames], dim=0) for vid in batch],
        dim=0,
    ).permute(1, 0, 2, 3, 4)   # (T, B, C, H, W)

    T = img_batch.shape[0]

    step_masks = [[] for _ in range(T)]
    step_obj_to_frame = [[] for _ in range(T)]
    step_obj_ids = [[] for _ in range(T)]
    step_frame_sizes = [[] for _ in range(T)]

    for video_idx, video in enumerate(batch):
        for t, frame in enumerate(video.frames):
            for obj in frame.objects:
                step_obj_to_frame[t].append(
                    torch.tensor([t, video_idx], dtype=torch.int)
                )
                step_masks[t].append(obj.segment.to(torch.bool))
                step_obj_ids[t].append(
                    torch.tensor([video.video_id, obj.object_id, obj.frame_index])
                )
                step_frame_sizes[t].append(torch.tensor(video.size))

    def _stack(lst):
        return torch.stack([torch.stack(items, 0) for items in lst], 0)

    return BatchedVideoDatapoint(
        img_batch=img_batch,
        obj_to_frame_idx=_stack(step_obj_to_frame),
        masks=_stack(step_masks),
        metadata=BatchedVideoMetaData(
            unique_objects_identifier=_stack(step_obj_ids),
            frame_orig_size=_stack(step_frame_sizes),
        ),
        dict_key=dict_key,
        batch_size=[T],
    )
