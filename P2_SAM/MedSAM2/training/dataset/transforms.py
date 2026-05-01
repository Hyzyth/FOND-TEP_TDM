"""
training/dataset/transforms.py
================================
Augmentation transforms for the VOS training pipeline.
All transforms operate on :class:`~training.utils.data_utils.VideoDatapoint`
objects and are designed to be composed with :class:`ComposeAPI`.
"""

import random
from typing import Iterable

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from training.utils.data_utils import VideoDatapoint


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _hflip(datapoint: VideoDatapoint, index: int) -> VideoDatapoint:
    datapoint.frames[index].data = F.hflip(datapoint.frames[index].data)
    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.hflip(obj.segment)
    return datapoint


def _vflip(datapoint: VideoDatapoint, index: int) -> VideoDatapoint:
    datapoint.frames[index].data = F.vflip(datapoint.frames[index].data)
    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.vflip(obj.segment)
    return datapoint


def _get_size_with_aspect_ratio(image_size, size, max_size=None):
    w, h = image_size
    if max_size is not None:
        min_dim = float(min(w, h))
        max_dim = float(max(w, h))
        if max_dim / min_dim * size > max_size:
            size = int(max_size * min_dim / max_dim)
    if (w <= h and w == size) or (h <= w and h == size):
        return h, w
    if w < h:
        return int(round(size * h / w)), int(round(size))
    return int(round(size)), int(round(size * w / h))


def _resize(datapoint: VideoDatapoint, index: int, size, max_size=None,
            square: bool = False) -> VideoDatapoint:
    if square:
        target = (size, size)
    else:
        cur_size = datapoint.frames[index].data.size  # PIL (w, h)
        oh, ow = _get_size_with_aspect_ratio(cur_size, size, max_size)
        target = (oh, ow)
    datapoint.frames[index].data = F.resize(datapoint.frames[index].data, target)
    for obj in datapoint.frames[index].objects:
        if obj.segment is not None:
            obj.segment = F.resize(obj.segment[None, None], target).squeeze()
    datapoint.frames[index].size = target
    return datapoint


# ──────────────────────────────────────────────────────────────────────────────
# Public transform classes
# ──────────────────────────────────────────────────────────────────────────────

class ComposeAPI:
    """Apply a sequence of transforms to a :class:`VideoDatapoint`."""

    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, datapoint: VideoDatapoint, **kwargs) -> VideoDatapoint:
        for t in self.transforms:
            datapoint = t(datapoint, **kwargs)
        return datapoint


class RandomHorizontalFlip:
    """Random horizontal flip applied consistently or per-frame.

    Parameters
    ----------
    consistent_transform : bool  same decision for all frames when True
    p : float  flip probability
    """

    def __init__(self, consistent_transform: bool, p: float = 0.5) -> None:
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint: VideoDatapoint, **kwargs) -> VideoDatapoint:
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.frames)):
                    datapoint = _hflip(datapoint, i)
        else:
            for i in range(len(datapoint.frames)):
                if random.random() < self.p:
                    datapoint = _hflip(datapoint, i)
        return datapoint


class RandomVerticalFlip:
    """Random vertical flip."""

    def __init__(self, consistent_transform: bool, p: float = 0.5) -> None:
        self.p = p
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint: VideoDatapoint, **kwargs) -> VideoDatapoint:
        if self.consistent_transform:
            if random.random() < self.p:
                for i in range(len(datapoint.frames)):
                    datapoint = _vflip(datapoint, i)
        else:
            for i in range(len(datapoint.frames)):
                if random.random() < self.p:
                    datapoint = _vflip(datapoint, i)
        return datapoint


class RandomResizeAPI:
    """Randomly resize to one of the given sizes.

    Parameters
    ----------
    sizes : int or iterable[int]  candidate sizes
    consistent_transform : bool
    max_size : int or None
    square : bool  force square output
    """

    def __init__(self, sizes, consistent_transform: bool, max_size=None, square=False) -> None:
        if isinstance(sizes, int):
            sizes = (sizes,)
        self.sizes = list(sizes)
        self.max_size = max_size
        self.square = square
        self.consistent_transform = consistent_transform

    def __call__(self, datapoint: VideoDatapoint, **kwargs) -> VideoDatapoint:
        if self.consistent_transform:
            size = random.choice(self.sizes)
            for i in range(len(datapoint.frames)):
                datapoint = _resize(datapoint, i, size, self.max_size, self.square)
        else:
            for i in range(len(datapoint.frames)):
                size = random.choice(self.sizes)
                datapoint = _resize(datapoint, i, size, self.max_size, self.square)
        return datapoint


class ToTensorAPI:
    """Convert PIL frame data to float tensor."""

    def __call__(self, datapoint: VideoDatapoint, **kwargs) -> VideoDatapoint:
        for img in datapoint.frames:
            img.data = F.to_tensor(img.data)
        return datapoint


class NormalizeAPI:
    """Normalise frame tensors with ImageNet mean and std."""

    def __init__(self, mean, std) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, datapoint: VideoDatapoint, **kwargs) -> VideoDatapoint:
        for img in datapoint.frames:
            img.data = F.normalize(img.data, mean=self.mean, std=self.std)
        return datapoint


class ColorJitter:
    """Random colour jitter applied consistently or per-frame.

    Parameters
    ----------
    consistent_transform : bool
    brightness, contrast, saturation : float  jitter strength
    hue : float or None
    """

    def __init__(self, consistent_transform: bool, brightness=0, contrast=0,
                 saturation=0, hue=None) -> None:
        self.consistent_transform = consistent_transform

        def _range(v):
            if isinstance(v, list):
                return v
            return [max(0, 1 - v), 1 + v]

        self.brightness = _range(brightness)
        self.contrast = _range(contrast)
        self.saturation = _range(saturation)
        self.hue = hue if isinstance(hue, list) or hue is None else [-hue, hue]

    def __call__(self, datapoint: VideoDatapoint, **kwargs) -> VideoDatapoint:
        if self.consistent_transform:
            params = T.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
        for img in datapoint.frames:
            if not self.consistent_transform:
                params = T.ColorJitter.get_params(self.brightness, self.contrast, self.saturation, self.hue)
            fn_idx, bf, cf, sf, hf = params
            for fn_id in fn_idx:
                if fn_id == 0 and bf is not None:
                    img.data = F.adjust_brightness(img.data, bf)
                elif fn_id == 1 and cf is not None:
                    img.data = F.adjust_contrast(img.data, cf)
                elif fn_id == 2 and sf is not None:
                    img.data = F.adjust_saturation(img.data, sf)
                elif fn_id == 3 and hf is not None:
                    img.data = F.adjust_hue(img.data, hf)
        return datapoint


class RandomGrayscale:
    """Random per-frame or consistent conversion to 3-channel grayscale."""

    def __init__(self, consistent_transform: bool, p: float = 0.05) -> None:
        self.p = p
        self.consistent_transform = consistent_transform
        self.Grayscale = T.Grayscale(num_output_channels=3)

    def __call__(self, datapoint: VideoDatapoint, **kwargs) -> VideoDatapoint:
        if self.consistent_transform:
            if random.random() < self.p:
                for img in datapoint.frames:
                    img.data = self.Grayscale(img.data)
        else:
            for img in datapoint.frames:
                if random.random() < self.p:
                    img.data = self.Grayscale(img.data)
        return datapoint


class RandomAffine:
    """Random affine transform (rotation, shear) applied to frames and masks.

    Parameters
    ----------
    degrees : float or list[float]
    consistent_transform : bool
    shear : float or None
    image_interpolation : str  ``'bilinear'`` or ``'bicubic'``
    p : float  probability of applying the transform
    """

    def __init__(self, degrees, consistent_transform: bool, scale=None,
                 translate=None, shear=None, image_mean=(123, 116, 103),
                 image_interpolation="bicubic", p=1.0) -> None:
        self.degrees = degrees if isinstance(degrees, list) else [-degrees, degrees]
        self.scale = scale
        self.shear = shear if isinstance(shear, list) else ([-shear, shear] if shear else None)
        self.translate = translate
        self.fill_img = image_mean
        self.consistent_transform = consistent_transform
        self.p = p
        self.image_interpolation = (
            InterpolationMode.BICUBIC if image_interpolation == "bicubic"
            else InterpolationMode.BILINEAR
        )

    def __call__(self, datapoint: VideoDatapoint, **kwargs) -> VideoDatapoint:
        if random.random() > self.p:
            return datapoint
        _, H, W = F.get_dimensions(datapoint.frames[0].data)
        if self.consistent_transform:
            affine_params = T.RandomAffine.get_params(
                degrees=self.degrees, translate=self.translate,
                scale_ranges=self.scale, shears=self.shear, img_size=[W, H],
            )
        for img in datapoint.frames:
            if not self.consistent_transform:
                affine_params = T.RandomAffine.get_params(
                    degrees=self.degrees, translate=self.translate,
                    scale_ranges=self.scale, shears=self.shear, img_size=[W, H],
                )
            img.data = F.affine(img.data, *affine_params,
                                interpolation=self.image_interpolation, fill=self.fill_img)
            for obj in img.objects:
                if obj.segment is not None:
                    m = F.affine(obj.segment.unsqueeze(0), *affine_params,
                                 interpolation=InterpolationMode.NEAREST, fill=0.0)
                    obj.segment = m.squeeze()
        return datapoint
