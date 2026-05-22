# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
data_utils.py  —  SwinCross NPZ-based data pipeline
=====================================================

Key changes vs. the original:
  1.  LoadNPZd  — new transform: loads pre-processed NPZ files produced by
                  prepare_hecktor_npz_swincross.py / prepare_temporal_npz_swincross.py.
                  Heavy offline ops (orient, resample, crop) are already baked in;
                  only lightweight augmentations remain in the online transform chain.
  2.  cache_num removed  — the original CacheDataset had cache_num=24 hard-coded,
                  silently capping the cache at 24 patients even when cache_rate=1.0.
                  Now cache_rate alone controls caching.
  3.  persistent_workers=True  — workers now stay alive across batches, eliminating
                  per-epoch process-spawn overhead.
  4.  prefetch_factor=2  — each worker pre-loads the next 2 batches so the GPU
                  never stalls on I/O.

JSON entry expected format (produced by the preprocessing scripts):
  { "npz": "train/HGJ_001.npz", "label": "labelsTr/HGJ_001_gt.nii.gz", "case_id": "HGJ_001" }

The "npz" key is used by data_utils.py (training / online validation).
The "label" key is used by evaluate_predictions.py (offline evaluation).
"""

import math
import os

import numpy as np
import torch
from monai import data, transforms
from monai.data.decathlon_datalist import load_decathlon_datalist
from monai.transforms import Transform


# ── Distributed sampler (unchanged) ──────────────────────────────────────────

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle     = shuffle
        self.make_even   = make_even
        self.dataset     = dataset
        self.num_replicas= num_replicas
        self.rank        = rank
        self.epoch       = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size  = self.num_samples * self.num_replicas
        indices          = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(
                        low=0, high=len(indices),
                        size=self.total_size - len(indices))
                    indices += [indices[i] for i in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# ── NPZ loader ────────────────────────────────────────────────────────────────

class LoadNPZd(Transform):
    """
    Load a pre-processed SwinCross NPZ file into the transform dictionary.

    Reads  data["npz"]  (a path string) and populates:
      data["image"]  : (2, R, A, S) float32  — ch0=PET, ch1=CT
      data["label"]  : (1, R, A, S) int64    — 0=bg, 1=GTVp, 2=GTVn
      data[<meta>]   : numpy arrays for inverse-transform (used by test.py)

    By inheriting from monai.transforms.Transform (a non-Randomizable base),
    CacheDataset correctly identifies this as a deterministic transform and
    caches its output, re-applying only the random augmentations on every access.
    """

    _META_KEYS = (
        "ras_origin", "ras_direction", "ras_size_itk",
        "crop_start", "crop_end",
        "orig_spacing", "orig_origin", "orig_direction", "orig_size_itk",
    )

    def __call__(self, data_dict):
        d        = dict(data_dict)
        npz_path = str(d["npz"])

        npz = np.load(npz_path, allow_pickle=False)

        # Extract, cast back to float32, and fuse: (2, R, A, S) — ch0=PET, ch1=CT
        pet_arr = npz["pet"].astype(np.float32)
        ct_arr  = npz["ct"].astype(np.float32)
        d["image"] = np.stack([pet_arr, ct_arr], axis=0)

        # label: (R, A, S) → (1, R, A, S)  int64   (channel dim expected by MONAI)
        d["label"] = npz["label"].copy().astype(np.int64)[np.newaxis]

        # Inverse-transform metadata (consumed by test.py; ignored during training)
        for k in self._META_KEYS:
            if k in npz:
                d[k] = npz[k].copy()

        d["npz_path"] = npz_path
        return d


# ── Label cleanup helpers ─────────────────────────────────────────────────────

def _clamp_label(x):
    """Clamp and round label to valid integer range [0, 2].
    Works on both numpy arrays and torch Tensors."""
    if isinstance(x, np.ndarray):
        return np.clip(np.round(x), 0, 2).astype(np.int64)
    return torch.clamp(torch.round(x), min=0, max=2).to(torch.int64)


# ── DataLoader builder ────────────────────────────────────────────────────────

def get_loader(args):
    """
    Build and return DataLoaders from a SwinCross NPZ JSON list.

    Returns:
      - test_mode=True  : single DataLoader (validation/test set)
      - test_mode=False : [train_loader, val_loader]
    """
    data_dir      = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    loader_npz    = LoadNPZd()

    # ── Training transform ────────────────────────────────────────────────
    # All heavy ops (orient, resample, crop foreground) were done offline.
    # Only fast augmentations remain here.
    train_transform = transforms.Compose([
        loader_npz,
        # Guard: pad to at least ROI size so RandCropByPosNeg never fails
        # on very small volumes.
        transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            mode="constant",
        ),
        transforms.RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            pos=1,
            neg=1,
            num_samples=2,
            image_key="image",
            image_threshold=0,
        ),
        transforms.RandFlipd(keys=["image", "label"],
                             prob=args.RandFlipd_prob, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"],
                             prob=args.RandFlipd_prob, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"],
                             prob=args.RandFlipd_prob, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"],
                                  prob=args.RandRotate90d_prob, max_k=3),
        transforms.RandScaleIntensityd(keys="image",
                                        factors=0.1,
                                        prob=args.RandScaleIntensityd_prob),
        transforms.RandShiftIntensityd(keys="image",
                                        offsets=0.1,
                                        prob=args.RandShiftIntensityd_prob),
        # Clamp after augmentation to ensure label integrity
        transforms.Lambdad(keys="label", func=_clamp_label),
        transforms.ToTensord(keys=["image", "label"]),
    ])

    # ── Validation transform (full volume, no augmentation) ───────────────
    val_transform = transforms.Compose([
        loader_npz,
        transforms.Lambdad(keys="label", func=_clamp_label),
        transforms.ToTensord(keys=["image", "label"]),
    ])

    # ── DataLoader helper ─────────────────────────────────────────────────
    def _make_loader(ds, batch_size, shuffle, sampler):
        n_workers = args.workers
        return data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=n_workers,
            sampler=sampler,
            pin_memory=True,
            # Keep worker processes alive between batches — eliminates
            # per-epoch spawn overhead (was mistakenly commented out).
            persistent_workers=(n_workers > 0),
            # Each worker pre-loads 2 batches so GPU never idles on I/O.
            prefetch_factor=(2 if n_workers > 0 else None),
        )

    # ── Test / inference mode ─────────────────────────────────────────────
    if args.test_mode:
        test_files = load_decathlon_datalist(
            datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        return _make_loader(test_ds, batch_size=1, shuffle=False,
                            sampler=test_sampler)

    # ── Training mode ─────────────────────────────────────────────────────
    datalist = load_decathlon_datalist(
        datalist_json, True, "training", base_dir=data_dir)

    if args.use_normal_dataset or args.cache_rate == 0.0:
        train_ds = data.Dataset(data=datalist, transform=train_transform)
    else:
        # CacheDataset caches up to the last *deterministic* transform
        # (LoadNPZd + SpatialPadd), then applies random transforms on every
        # access.  cache_num is intentionally NOT set so cache_rate alone
        # governs how many patients are cached.
        train_ds = data.CacheDataset(
            data=datalist,
            transform=train_transform,
            cache_rate=args.cache_rate,
            num_workers=args.workers,
        )

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader  = _make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )

    val_files = load_decathlon_datalist(
        datalist_json, True, "validation", base_dir=data_dir)
    val_ds      = data.Dataset(data=val_files, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader  = _make_loader(val_ds, batch_size=1, shuffle=False,
                               sampler=val_sampler)

    return [train_loader, val_loader]
