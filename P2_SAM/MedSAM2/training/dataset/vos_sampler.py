"""
training/dataset/vos_sampler.py
================================
Frame and object samplers for the Video Object Segmentation (VOS) pipeline.

Classes
-------
VOSSampler          – abstract base
RandomUniformSampler – uniform random clip + random object subset (training)
EvalSampler          – all frames + all objects (evaluation)
"""

import random
from dataclasses import dataclass
from typing import List

from training.dataset.vos_segment_loader import LazySegments

MAX_RETRIES = 1000


@dataclass
class SampledFramesAndObjects:
    """Output of a :class:`VOSSampler` call.

    Attributes
    ----------
    frames : list
        Sampled :class:`~training.dataset.vos_raw_dataset.VOSFrame` objects.
    object_ids : list[int]
        Sampled object IDs visible in the first frame.
    """

    frames: List
    object_ids: List[int]


class VOSSampler:
    """Abstract base class for VOS samplers.

    Parameters
    ----------
    sort_frames : bool
        When True the returned frames are sorted by ascending frame index.
    """

    def __init__(self, sort_frames: bool = True) -> None:
        self.sort_frames = sort_frames

    def sample(self, video, segment_loader, epoch=None) -> SampledFramesAndObjects:
        raise NotImplementedError


class RandomUniformSampler(VOSSampler):
    """Sample a contiguous clip of *num_frames* and up to *max_num_objects*.

    Parameters
    ----------
    num_frames : int
        Number of consecutive frames to include in the clip.
    max_num_objects : int
        Maximum number of objects sampled from the first frame.
    reverse_time_prob : float
        Probability of reversing temporal order of the clip.
    """

    def __init__(
        self,
        num_frames: int,
        max_num_objects: int,
        reverse_time_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.max_num_objects = max_num_objects
        self.reverse_time_prob = reverse_time_prob

    def sample(self, video, segment_loader, epoch=None) -> SampledFramesAndObjects:
        """Sample a random clip and object subset.

        Parameters
        ----------
        video : VOSVideo
        segment_loader : any segment loader
        epoch : int, optional  (unused but kept for API compatibility)

        Returns
        -------
        SampledFramesAndObjects

        Raises
        ------
        Exception
            When no valid starting position is found after MAX_RETRIES.
        """
        if len(video.frames) < self.num_frames:
            raise Exception(
                f"Cannot sample {self.num_frames} frames from '{video.video_name}' "
                f"which has only {len(video.frames)} frames."
            )

        for _ in range(MAX_RETRIES):
            start = random.randrange(0, len(video.frames) - self.num_frames + 1)
            frames = [video.frames[start + s] for s in range(self.num_frames)]

            if random.random() < self.reverse_time_prob:
                frames = frames[::-1]

            # Only accept clips where the first frame has foreground.
            first_segs = segment_loader.load(frames[0].frame_idx)
            if isinstance(first_segs, LazySegments):
                visible_ids = list(first_segs.keys())
            else:
                visible_ids = [
                    oid
                    for oid, seg in first_segs.items()
                    if seg.sum() > 0
                ]

            if visible_ids:
                break
        else:
            raise Exception(
                f"No visible objects found after {MAX_RETRIES} retries for '{video.video_name}'."
            )

        object_ids = random.sample(
            visible_ids, min(len(visible_ids), self.max_num_objects)
        )
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)


class EvalSampler(VOSSampler):
    """Return all frames and all objects (for evaluation / inference)."""

    def sample(self, video, segment_loader, epoch=None) -> SampledFramesAndObjects:
        frames = (
            sorted(video.frames, key=lambda f: f.frame_idx)
            if self.sort_frames
            else video.frames
        )
        object_ids = list(segment_loader.load(frames[0].frame_idx).keys())
        if not object_ids:
            raise Exception(
                f"First frame of '{video.video_name}' has no annotated objects."
            )
        return SampledFramesAndObjects(frames=frames, object_ids=object_ids)
