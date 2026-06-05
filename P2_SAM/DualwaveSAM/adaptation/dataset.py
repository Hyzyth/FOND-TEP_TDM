"""
dataset.py  —  DualwaveSAM HECKTOR 2026 NPZ dataset (2D slice-based)
=====================================================================

Loads SwinCross-format NPZ files (ct int16, pet float16, label uint8 0/1/2)
and exposes axial (S-axis) 2D slices for training.

Key design decisions:
  - Slices along the S (axial / superior-inferior) axis, index 2 of the
    MONAI (R, A, S) spatial convention — most anatomically consistent with
    how DualwaveSAM was originally trained on axial PET/CT slices.
  - 3-class output: 0=background, 1=GTVp, 2=GTVn. The model head is adapted
    to output 3 channels (softmax), replacing the original binary sigmoid.
  - During training, foreground-containing slices are oversampled. A
    configurable `bg_ratio` (default 0.2) controls what fraction of sampled
    slices per epoch may be background-only, preventing class collapse without
    discarding global context.
  - All heavy preprocessing (orient, resample, crop) was already done offline
    by prepare_hecktor2026_kfold_npz.py; only lightweight per-slice ops remain.

JSON format expected by this dataset (example entry):
  { "npz": "npz/train/HGJ_001.npz", "label": "labels/train/HGJ_001_gt.nii.gz",
    "case_id": "HGJ_001", "acronym": "HGJ" }
"""

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm

import cv2
import numpy as np
import torch
import lmdb
import pickle
from torch.utils.data import Dataset
from monai.data import LMDBDataset


# ── Constants ──────────────────────────────────────────────────────────────────

# Slice axis in MONAI (R, A, S) convention: axis 2 = S (axial)
SLICE_AXIS = 2

# Input resolution fed to DualwaveSAM (must match model's expected size)
SLICE_SIZE = 256

# For class-balanced crop: CT window used for display/normalisation
CT_WINDOW_CENTER = 40.0   # HU
CT_WINDOW_WIDTH  = 400.0  # HU


# ── Normalisation helpers ──────────────────────────────────────────────────────

def normalise_ct(ct_arr: np.ndarray) -> np.ndarray:
    """
    Soft-window CT HU values to [0, 1] using a clinical soft-tissue window.
    Equivalent to clipping to [WC - WW/2, WC + WW/2] then min-max scaling.
    """
    lo = CT_WINDOW_CENTER - CT_WINDOW_WIDTH / 2.0   # -160 HU
    hi = CT_WINDOW_CENTER + CT_WINDOW_WIDTH / 2.0   #  240 HU
    ct = np.clip(ct_arr.astype(np.float32), lo, hi)
    return (ct - lo) / (hi - lo)   # → [0, 1]


def normalise_pet(pet_arr: np.ndarray) -> np.ndarray:
    """
    Normalise PET SUV to [0, 1] by clipping to a robust 99th-percentile max.
    Avoids hot-spot outliers dominating the scale.
    """
    pet = pet_arr.astype(np.float32)
    p99 = float(np.percentile(pet[pet > 0], 99)) if (pet > 0).any() else 1.0
    p99 = max(p99, 1e-6)
    return np.clip(pet / p99, 0.0, 1.0)


# ── Per-patient slice extractor ────────────────────────────────────────────────

def extract_slices(npz_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one NPZ and return all axial slices.

    Returns
    -------
    ct_slices  : (S, R, A) float32   normalised [0, 1]
    pet_slices : (S, R, A) float32   normalised [0, 1]
    lbl_slices : (S, R, A) uint8     0/1/2
    """
    with np.load(npz_path, allow_pickle=False) as npz:
        ct_vol  = npz["ct"].astype(np.float32)    # (R, A, S)
        pet_vol = npz["pet"].astype(np.float32)   # (R, A, S)
        lbl_vol = npz["label"].astype(np.uint8)   # (R, A, S)

    ct_vol  = normalise_ct(ct_vol)
    pet_vol = normalise_pet(pet_vol)

    # (R, A, S) → (S, R, A)
    ct_slices  = np.moveaxis(ct_vol,  SLICE_AXIS, 0)
    pet_slices = np.moveaxis(pet_vol, SLICE_AXIS, 0)
    lbl_slices = np.moveaxis(lbl_vol, SLICE_AXIS, 0)

    return ct_slices, pet_slices, lbl_slices


def resize_slice(arr: np.ndarray, size: int,
                 interpolation=cv2.INTER_LINEAR) -> np.ndarray:
    """Resize a 2D slice to (size, size)."""
    return cv2.resize(arr, (size, size), interpolation=interpolation)


# ── Training dataset ───────────────────────────────────────────────────────────

class HECKTORNPZDataset(Dataset):
    """
    Slice-level dataset for DualwaveSAM training on HECKTOR 2026 NPZ data.

    All slices from all patients are collected at __init__ time (fast, since
    we only read and normalise the numpy arrays — no heavy I/O per __getitem__).

    Sampling strategy
    -----------------
    The dataset stores two index lists:
      self._fg_indices : indices of slices containing at least one foreground voxel
      self._bg_indices : indices of background-only slices

    __len__ returns len(fg_indices) + int(len(fg_indices) * bg_ratio), which
    sets a "virtual" epoch length.  __getitem__ samples foreground slices for
    the first len(fg_indices) indices and background slices (randomly drawn)
    for the remainder — giving deterministic foreground coverage per epoch.

    Parameters
    ----------
    data_dir    : root of the NPZ dataset (e.g. /data/ethan/PP_hecktor2026_kfold_npz)
    json_path   : path to the JSON split file (absolute, or relative to data_dir)
    split       : "training" | "validation"
    size        : spatial resolution fed to the model (default 256)
    bg_ratio    : fraction of extra background slices per foreground slice (default 0.15)
    cache_rate  : fraction (0-1) of the heaviest volumes to cache in RAM for faster access
    augment     : enable spatial augmentations (training only)
    """

    def __init__(
        self,
        json_path: str,
        split: str = "training",
        size: int = SLICE_SIZE,
        bg_ratio: float = 0.15,
        augment: bool = False,
    ):
        self.size     = size
        self.bg_ratio = bg_ratio
        self.augment  = augment
        self.split    = split
        self.env      = None  # LMDB environment, lazily loaded on first access

        # Point to the pre-generated database files
        lmdb_dir = Path("/data/ethan/DualwaveSAM3c/lmdb_cache")
        json_stem = Path(json_path).stem
        self.lmdb_path = str(lmdb_dir / f"{json_stem}_{split}.lmdb")

        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"Missing LMDB file: {self.lmdb_path}. Please run build_lmdb_cache.py first!")

        print(f"  [{split}] Reading database index...", flush=True)
        
        # Open temporarily just to grab the pre-calculated FG/BG layout
        env_meta = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with env_meta.begin(write=False) as txn:
            has_fg_list = pickle.loads(txn.get(b"__fg_index__"))
        env_meta.close()

        has_fg_arr = np.array(has_fg_list, dtype=bool)
        self._fg_indices = np.where(has_fg_arr)[0].tolist()
        self._bg_indices = np.where(~has_fg_arr)[0].tolist()
        
        n_fg = len(self._fg_indices)
        n_bg_per_epoch = int(n_fg * bg_ratio)
        
        print(f"  [{split}] Compiled {len(has_fg_list)} total slices | FG={n_fg} | BG pool={len(self._bg_indices)}", flush=True)
        
        self._epoch_bg: List[int] = []
        self._resample_bg(n_bg_per_epoch)
    
    def _init_env(self):
        # Called exactly once per DataLoader worker thread
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

    # ── Sampling helpers ──────────────────────────────────────────────────────

    def _resample_bg(self, n: int):
        """Draw a fresh random sample of background indices for one epoch."""
        if self._bg_indices and n > 0:
            self._epoch_bg = random.choices(self._bg_indices, k=n)
        else:
            self._epoch_bg = []

    def on_epoch_end(self):
        """Call at the end of each epoch to refresh the BG sample."""
        self._resample_bg(int(len(self._fg_indices) * self.bg_ratio))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self._fg_indices) + len(self._epoch_bg)

    def __getitem__(self, idx: int):
        self._init_env() # Ensures thread-safe DB access
        
        if idx < len(self._fg_indices):
            real_idx = self._fg_indices[idx]
        else:
            real_idx = self._epoch_bg[idx - len(self._fg_indices)]
            
        with self.env.begin(write=False) as txn:
            raw_data = txn.get(f"{real_idx}".encode("ascii"))
            sample = pickle.loads(raw_data)
        
        ct_slice = sample["ct"]
        pet_slice = sample["pet"]
        lbl_slice = sample["label"]
        
        return self._build_tensor_payload(ct_slice, pet_slice, lbl_slice)

    def _build_tensor_payload(self, ct_slice, pet_slice, lbl_slice):
        ct_r  = resize_slice(ct_slice,  self.size, cv2.INTER_LINEAR)
        pet_r = resize_slice(pet_slice, self.size, cv2.INTER_LINEAR)
        lbl_r = resize_slice(lbl_slice, self.size, cv2.INTER_NEAREST)

        img = np.stack([ct_r, pet_r], axis=-1).astype(np.float32)
        lbl = lbl_r.astype(np.int64)

        if self.augment:
            img, lbl = self._augment(img, lbl)

        img_t = torch.from_numpy(img).permute(2, 0, 1)
        lbl_t = torch.from_numpy(lbl.copy())

        return {"image": img_t, "label": lbl_t}
    
    # ── Augmentation ──────────────────────────────────────────────────────────

    @staticmethod
    def _augment(img: np.ndarray, lbl: np.ndarray):
        """
        Lightweight spatial augmentations applied jointly to image and label.
        All operations preserve label integer values.
        """
        # Horizontal flip
        if random.random() < 0.5:
            img = np.flip(img, axis=1).copy()
            lbl = np.flip(lbl, axis=1).copy()
        # Vertical flip
        if random.random() < 0.5:
            img = np.flip(img, axis=0).copy()
            lbl = np.flip(lbl, axis=0).copy()
        # 90° rotation (k ∈ {1,2,3})
        if random.random() < 0.3:
            k = random.randint(1, 3)
            img = np.rot90(img, k, axes=(0, 1)).copy()
            lbl = np.rot90(lbl, k, axes=(0, 1)).copy()
        # Mild intensity jitter (image only, not label)
        if random.random() < 0.3:
            scale = random.uniform(0.85, 1.15)
            shift = random.uniform(-0.05, 0.05)
            img   = np.clip(img * scale + shift, 0.0, 1.0)
        return img, lbl


# ── Validation / inference dataset (full volumes, no sampling) ─────────────────

class HECKTORNPZInferenceDataset(Dataset):
    """
    Patient-level dataset for validation and inference.

    Returns one dictionary per patient with the full stack of axial slices
    and all NPZ metadata needed for 3D reconstruction.

    Each __getitem__ returns:
      {
        "images"  : (S, 2, H, W) float32 tensor — all axial slices
        "labels"  : (S, H, W)    int64 tensor
        "case_id" : str
        "npz_path": str          — absolute path to NPZ (for inverse transform)
        "orig_shape": (R, A)     — original spatial shape before resize
      }
    """

    def __init__(
        self,
        data_dir: str,
        json_path: str,
        split: str = "validation",
        size: int = SLICE_SIZE,
    ):
        self.data_dir = Path(data_dir)
        self.size     = size

        if not os.path.isabs(json_path):
            json_path = str(self.data_dir / json_path)
        with open(json_path) as f:
            js = json.load(f)

        entries = js.get(split, [])
        if not entries:
            raise ValueError(
                f"No entries found under '{split}' in {json_path}. "
                f"Keys: {list(js.keys())}"
            )

        self.entries = entries
        print(f"  [inference/{split}] {len(entries)} patients")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry    = self.entries[idx]
        npz_rel  = entry.get("npz", "")
        case_id  = entry.get("case_id") or os.path.basename(npz_rel).replace(".npz", "")
        npz_path = str(self.data_dir / npz_rel)

        ct_slices, pet_slices, lbl_slices = extract_slices(npz_path)
        S, R, A = ct_slices.shape   # (S, R, A)

        imgs = np.zeros((S, self.size, self.size, 2), dtype=np.float32)
        lbls = np.zeros((S, self.size, self.size),    dtype=np.int64)

        for s in range(S):
            imgs[s, :, :, 0] = resize_slice(ct_slices[s],  self.size, cv2.INTER_LINEAR)
            imgs[s, :, :, 1] = resize_slice(pet_slices[s], self.size, cv2.INTER_LINEAR)
            lbls[s]          = resize_slice(lbl_slices[s], self.size, cv2.INTER_NEAREST)

        # (S, H, W, 2) → (S, 2, H, W)
        imgs_t = torch.from_numpy(imgs).permute(0, 3, 1, 2)
        lbls_t = torch.from_numpy(lbls)

        return {
            "images":      imgs_t,         # (S, 2, H, W)
            "labels":      lbls_t,         # (S, H, W)
            "case_id":     case_id,
            "npz_path":    npz_path,
            "orig_shape":  (R, A),
            "n_slices":    S,
        }
