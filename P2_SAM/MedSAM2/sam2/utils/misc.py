"""
sam2/utils/misc.py
===================
Miscellaneous utilities: device detection, connected-components, mask-to-box,
video frame loading, and mask hole-filling.
"""

import os
import warnings
from threading import Thread

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def get_sdpa_settings():
    """Detect GPU capabilities and choose scaled-dot-product attention settings.

    Returns
    -------
    (old_gpu, use_flash_attn, math_kernel_on) : tuple[bool, bool, bool]
    """
    if torch.cuda.is_available():
        old_gpu = torch.cuda.get_device_properties(0).major < 7
        use_flash_attn = torch.cuda.get_device_properties(0).major >= 8
        if not use_flash_attn:
            warnings.warn("Flash Attention requires Ampere (8.0) GPU; disabled.", UserWarning, stacklevel=2)
        pytorch_version = tuple(int(v) for v in torch.__version__.split(".")[:2])
        if pytorch_version < (2, 2):
            warnings.warn(f"PyTorch {torch.__version__} lacks Flash Attention v2.", UserWarning, stacklevel=2)
        math_kernel_on = pytorch_version < (2, 2) or not use_flash_attn
    else:
        old_gpu = True
        use_flash_attn = False
        math_kernel_on = True
    return old_gpu, use_flash_attn, math_kernel_on


def get_connected_components(mask: torch.Tensor):
    """Get 8-connected components of a binary mask using the CUDA kernel.

    Parameters
    ----------
    mask : Tensor  (N, 1, H, W) uint8

    Returns
    -------
    (labels, counts) : tuple[Tensor, Tensor]  same shape as mask
    """
    from sam2 import _C
    return _C.get_connected_componnets(mask.to(torch.uint8).contiguous())


def mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """Compute axis-aligned bounding boxes from binary masks.

    Parameters
    ----------
    masks : Tensor  (B, 1, H, W)

    Returns
    -------
    Tensor  (B, 1, 4)  – (x_min, y_min, x_max, y_max)
    """
    B, _, h, w = masks.shape
    device = masks.device
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    return torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)


def fill_holes_in_mask_scores(mask: torch.Tensor, max_area: int) -> torch.Tensor:
    """Fill small background holes in mask logits (post-processing).

    Parameters
    ----------
    mask : Tensor  any shape with mask scores
    max_area : int  connected background regions ≤ this area are filled

    Returns
    -------
    Tensor  same shape as *mask*
    """
    assert max_area > 0
    input_mask = mask
    try:
        labels, areas = get_connected_components(mask <= 0)
        is_hole = (labels > 0) & (areas <= max_area)
        mask = torch.where(is_hole, 0.1, mask)
    except Exception as e:
        warnings.warn(f"Hole-filling skipped: {e}", UserWarning, stacklevel=2)
        mask = input_mask
    return mask


def concat_points(old_point_inputs, new_points, new_labels):
    """Append new prompt points/labels to previous point inputs."""
    if old_point_inputs is None:
        return {"point_coords": new_points, "point_labels": new_labels}
    return {
        "point_coords": torch.cat([old_point_inputs["point_coords"], new_points], dim=1),
        "point_labels": torch.cat([old_point_inputs["point_labels"], new_labels], dim=1),
    }


def _load_img_as_tensor(img_path: str, image_size: int):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype != np.uint8:
        raise RuntimeError(f"Unexpected image dtype {img_np.dtype} at {img_path}")
    img = torch.from_numpy(img_np / 255.0).permute(2, 0, 1).float()
    video_width, video_height = img_pil.size
    return img, video_height, video_width


class AsyncVideoFrameLoader:
    """Lazy async loader for a list of JPEG frame paths."""

    def __init__(self, img_paths, image_size, offload_video_to_cpu,
                 img_mean, img_std, compute_device) -> None:
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.images = [None] * len(img_paths)
        self.exception = None
        self.video_height = None
        self.video_width = None
        self.compute_device = compute_device
        self.__getitem__(0)  # prime the first frame

        def _load_frames():
            try:
                for n in tqdm(range(len(self.images)), desc="frame loading (JPEG)"):
                    self.__getitem__(n)
            except Exception as e:
                self.exception = e

        self.thread = Thread(target=_load_frames, daemon=True)
        self.thread.start()

    def __getitem__(self, index):
        if self.exception is not None:
            raise RuntimeError("Failure in frame-loading thread") from self.exception
        img = self.images[index]
        if img is not None:
            return img
        img, self.video_height, self.video_width = _load_img_as_tensor(
            self.img_paths[index], self.image_size
        )
        img = (img - self.img_mean) / self.img_std
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device, non_blocking=True)
        self.images[index] = img
        return img

    def __len__(self):
        return len(self.images)


def load_video_frames_from_jpg_images(video_path, image_size, offload_video_to_cpu,
                                       img_mean=(0.485, 0.456, 0.406),
                                       img_std=(0.229, 0.224, 0.225),
                                       async_loading_frames=False,
                                       compute_device=torch.device("cuda")):
    """Load JPEG frames from a directory into a float tensor."""
    if not (isinstance(video_path, str) and os.path.isdir(video_path)):
        raise NotImplementedError("Only JPEG frame directories are supported.")
    frame_names = sorted(
        p for p in os.listdir(video_path)
        if os.path.splitext(p)[-1].lower() in {".jpg", ".jpeg"}
    )
    if not frame_names:
        raise RuntimeError(f"No JPEG frames in {video_path}")
    img_paths = [os.path.join(video_path, f) for f in frame_names]
    img_mean_t = torch.tensor(img_mean)[:, None, None]
    img_std_t  = torch.tensor(img_std)[:, None, None]
    if async_loading_frames:
        lazy = AsyncVideoFrameLoader(img_paths, image_size, offload_video_to_cpu,
                                     img_mean_t, img_std_t, compute_device)
        return lazy, lazy.video_height, lazy.video_width
    n = len(frame_names)
    images = torch.zeros(n, 3, image_size, image_size, dtype=torch.float32)
    for i, p in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[i], video_height, video_width = _load_img_as_tensor(p, image_size)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean_t = img_mean_t.to(compute_device)
        img_std_t  = img_std_t.to(compute_device)
    return (images - img_mean_t) / img_std_t, video_height, video_width
