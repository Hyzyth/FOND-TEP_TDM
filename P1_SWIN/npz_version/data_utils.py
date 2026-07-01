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

JSON entry expected format (produced by the preprocessing scripts):
  { "npz": "train/HGJ_001.npz", "label": "labelsTr/HGJ_001_gt.nii.gz", "case_id": "HGJ_001" }

The "npz" key is used by data_utils.py (training / online validation).
The "label" key is used by evaluate_predictions.py (offline evaluation).
"""

import math
import os
import numpy as np
import torch
import lmdb
import pickle
from pathlib import Path
from monai import data, transforms
from monai.data.decathlon_datalist import load_decathlon_datalist
from monai.transforms import Transform


# ── Distributed sampler ──────────────────────────────────────────

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

# ── LMDB Loader ──────────────────────────────────────────────────────────────
class LoadLMDBd(Transform):
    """
    Reads directly from the LMDB cache using zero-copy byte extraction.
    """
    def __init__(self, lmdb_dir):
        self.lmdb_dir = lmdb_dir
        self.env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)

    def _init_env(self):
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False
            )

    def __call__(self, data_dict):
        self._init_env()
        d = dict(data_dict)
        case_id = str(d["case_id"])

        with self.env.begin(write=False) as txn:
            meta_bytes = txn.get(f"{case_id}_meta".encode("ascii"))
            if meta_bytes is None:
                raise KeyError(f"Case {case_id} not found in LMDB {self.lmdb_dir}.")
            
            meta = pickle.loads(meta_bytes)
            ct_bytes = txn.get(f"{case_id}_ct".encode("ascii"))
            pet_bytes = txn.get(f"{case_id}_pet".encode("ascii"))
            label_bytes = txn.get(f"{case_id}_label".encode("ascii"))

        # Reconstruct arrays instantly. .copy() is required because frombuffer is read-only
        # and MONAI transforms require writable arrays.
        ct_arr = np.frombuffer(ct_bytes, dtype=np.dtype(meta["ct_dtype"])).reshape(meta["ct_shape"]).copy()
        pet_arr = np.frombuffer(pet_bytes, dtype=np.dtype(meta["pet_dtype"])).reshape(meta["pet_shape"]).copy()
        label_arr = np.frombuffer(label_bytes, dtype=np.dtype(meta["label_dtype"])).reshape(meta["label_shape"]).copy()

        # Format exactly as the model expects
        d["image"] = np.stack([pet_arr.astype(np.float32), ct_arr.astype(np.float32)], axis=0)
        d["label"] = label_arr.astype(np.int64)[np.newaxis]
        
        d["case_id"] = meta["case_id"]
        d["npz_path"] = meta["npz_path"]

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
    
    # Dynamically resolve LMDB paths based on JSON name (e.g. dataset_swincross_2026kfold_classic.json)
    json_stem = Path(args.json_list).stem
    train_lmdb_path = os.path.join(args.lmdb_dir, f"{json_stem}_training.lmdb")
    val_lmdb_path = os.path.join(args.lmdb_dir, f"{json_stem}_validation.lmdb")

    # ── Training transform ────────────────────────────────────────────────
    # All heavy ops (orient, resample, crop foreground) were done offline.
    # Only fast augmentations remain here.
    train_transform = transforms.Compose([
        LoadLMDBd(train_lmdb_path),
        transforms.SpatialPadd(
            keys=["image", "label"],
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            mode="constant",
        ),
        transforms.RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(args.roi_x, args.roi_y, args.roi_z),
            num_classes=3,      # 0 = Background, 1 = Tumor (GTVp), 2 = Nodule (GTVn)
            ratios=[1, 1, 1],   # [Bg, Tumor, Nodule] -> Pick equal patches from each class (if available)
            num_samples=3,      # Extract 3 patches per volume
            image_key="image",
            image_threshold=0,
            warn=False,         # Don't warn if not enough tumor/nodule pixels to fill all patches
        ),
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=args.RandFlipd_prob, spatial_axis=2),
        transforms.RandRotate90d(keys=["image", "label"], prob=args.RandRotate90d_prob, max_k=3),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=args.RandScaleIntensityd_prob),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=args.RandShiftIntensityd_prob),
        transforms.Lambdad(keys="label", func=_clamp_label),
        transforms.ToTensord(keys=["image", "label"]),
    ])

    # ── Validation transform (full volume, no augmentation) ───────────────
    val_transform = transforms.Compose([
        LoadLMDBd(val_lmdb_path),
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
            persistent_workers=False,
            # Each worker pre-loads 2 batches so GPU never idles on I/O.
            prefetch_factor=(2 if n_workers > 0 else None),
        )

    # ── Test / inference mode ─────────────────────────────────────────────
    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        return _make_loader(test_ds, batch_size=1, shuffle=False, sampler=test_sampler)

    # ── Training mode ─────────────────────────────────────────────────────
    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)
    
    # LMDB is incredibly fast, so we don't need MONAI's memory cache
    train_ds = data.Dataset(data=datalist, transform=train_transform)

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader  = _make_loader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )

    val_files = load_decathlon_datalist(datalist_json, True, "validation", base_dir=data_dir)
    val_ds      = data.Dataset(data=val_files, transform=val_transform)
    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader  = _make_loader(val_ds, batch_size=1, shuffle=False, sampler=val_sampler)

    return [train_loader, val_loader]
