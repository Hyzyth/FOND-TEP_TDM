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

import os
import math
import numpy as np
import torch
from monai import transforms, data
from monai.data.thread_buffer import ThreadDataLoader  # MODIFICATION > added for faster data loading
from monai.data.decathlon_datalist import load_decathlon_datalist #Adjusted import path as per VSCode suggestion monai.data -> monai.data.decathlon_datalist
import torch

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
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
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
                    extra_ids = np.random.randint(low=0,high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def clamp_label(x):
    return torch.clamp(x, min=0, max=2).to(x.dtype)

def round_and_clamp_label(x):
    return torch.clamp(torch.round(x), min=0, max=2).to(torch.int64)

def get_loader(args):
    data_dir = args.data_dir
    datalist_json = os.path.join(data_dir, args.json_list)
    train_transform = transforms.Compose(
        [
            # 1. Load the images
            transforms.LoadImaged(keys=["image", "label"]),
            # 2. Harmonize Dimensions (The Modern Way)
            # This replaces both AddChanneld and manual channel shifting
            transforms.EnsureChannelFirstd(keys=["image", "label"]), # MODIFICATION
            # Now both are (C, X, Y, Z) correctly
            #transforms.AsChannelFirstd(keys=["image"]),
            #transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),

            # This ensures labels are contiguous and match model output channels.
            # It maps the original labels [0, 1, 2] to the target [0, 1, 2].
            transforms.Lambdad(
                keys="label",
                func=clamp_label
            ),

            # 3. Orient and Space
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            
            # 4. Spacing
            transforms.Spacingd(keys=["image", "label"],
                                pixdim=(args.space_x, args.space_y, args.space_z),
                                mode=("bilinear", "nearest")),
            # transforms.ScaleIntensityRangePercentilesd(keys=["image"],
            #                                           lower=args.lower,
            #                                           upper=args.upper,
            #                                           b_min=args.b_min,
            #                                           b_max=args.b_max,
            #                                           channel_wise=True),
            #transforms.ScaleIntensityRangePercentilesd(),

            #transforms.RandSpatialCropd(keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_x], random_size=False),
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), # MODIFICATION : x was put twice here, changed the last variable args.roi_x to args.roi_z
                pos=1,
                neg=1,
                num_samples=2, # MODIFICATION : reduced from 4 to 2 because of excessive time consumption for 1 epoch
                image_key="image",
                image_threshold=0,
            ),

            # 5. Augmentations
            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=0),

            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=1),

            transforms.RandFlipd(keys=["image", "label"],
                                 prob=args.RandFlipd_prob,
                                 spatial_axis=2),

            transforms.RandRotate90d(
                keys=["image", "label"],
                prob=args.RandRotate90d_prob,
                max_k=3,
                ),

            transforms.RandScaleIntensityd(keys="image",
                                           factors=0.1,
                                           prob=args.RandScaleIntensityd_prob),
            transforms.RandShiftIntensityd(keys="image",
                                           offsets=0.1,
                                           prob=args.RandShiftIntensityd_prob),

            # MODIFICATION: Final cleanup after all augmentations. 
            # AI proposed to round then clamp image values after data augmentation 
            # to ensure labels are valid integers within the expected range [0, 2].
            transforms.Lambdad(
                keys="label",
                func=round_and_clamp_label
            ),
            
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    # -----------------------------
    # PURE INFERENCE TRANSFORM
    # -----------------------------
    infer_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),

            transforms.Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear",)
            ),

            transforms.CropForegroundd(keys=["image"], source_key="image"),

            transforms.EnsureTyped(keys=["image"], track_meta=True),
        ]
    )

    val_transform = transforms.Compose(
        [
            # 1. Load the images
            transforms.LoadImaged(keys=["image", "label"]),
            
            # 2. Harmonize dimensions (Exact mirror of train_transform)
            # Pulls the 2 modalities from the end to the front for the image
            # This replaces both AddChanneld and manual channel shifting and
            # Adds a dummy channel dimension to the label: (H, W, D) -> (1, H, W, D)
            transforms.EnsureChannelFirstd(keys=["image", "label"]), # MODIFICATION

            # This ensures labels are contiguous and match model output channels.
            transforms.Lambdad(
                keys="label",
                func=clamp_label
            ),
            # 3. Orient and Space
            transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
            
            # Shape verified: image is 4D (C, H, W, D), label is (1, H, W, D)
            
            transforms.Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest")
            ),
            
            # 4. Final Prep
            transforms.CropForegroundd(keys=["image", "label"], source_key="image"),

            # # MODIFICATION: Final cleanup after all augmentations. 
            # AI proposed to round then clamp image values after data augmentation 
            # to ensure labels are valid integers within the expected range [0, 2].
            transforms.Lambdad(
                keys="label",
                func=round_and_clamp_label
            ),
        

            transforms.EnsureTyped(
            keys=["image", "label"],
            track_meta=True
        ),
            # transforms.ScaleIntensityRangePercentilesd(keys=["image"],
            #                                            lower=args.lower,
            #                                            upper=args.upper,
            #                                            b_min=args.b_min,
            #                                            b_max=args.b_max,
            #                                            channel_wise=True),
        ]
    )

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "validation",
                                            base_dir=data_dir)
        
            
        # Choix du transform selon le mode
        transform_used = infer_transform if args.inference_only else val_transform

        # MODIFICATION: Using CacheDataset to replace Dataset, because it doesn't pass metadata otherwise
        test_ds = data.CacheDataset(
            data=test_files,
            transform=transform_used,
            cache_rate=0.0,  # No caching needed for test
            num_workers=args.workers
        )

        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(test_ds,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     sampler=test_sampler,
                                     pin_memory=True,
                                     persistent_workers=(args.workers > 0))
        loader = test_loader
    else:
        datalist = load_decathlon_datalist(datalist_json,
                                           True,
                                           "training",
                                           base_dir=data_dir)
        if args.use_normal_dataset:
            train_ds = data.Dataset(data=datalist, transform=train_transform)
        else:
            train_ds = data.CacheDataset(
                data=datalist,
                transform=train_transform,
                cache_num=24,
                cache_rate=args.cache_rate, # Au lieu de 1.0
                num_workers=args.workers,
            )
        train_sampler = Sampler(train_ds) if args.distributed else None

        # MODIFICATION: Original DataLoader commented out, to try ThreadDataLoader, can be restored if needed
        train_loader = data.DataLoader(train_ds,
                                       batch_size=args.batch_size,
                                       shuffle=(train_sampler is None),
                                       num_workers=args.workers,
                                       sampler=train_sampler,
                                       pin_memory=True,
                                       persistent_workers=(args.workers > 0))

        # MODIFICATION: Using ThreadDataLoader for training to reduce epoch time
        # train_loader = ThreadDataLoader(train_ds,
        #                         batch_size=args.batch_size,
        #                         shuffle=(train_sampler is None),
        #                         num_workers=0,  # ThreadDataLoader uses threads, not processes
        #                         use_thread_workers=True,
        #                         buffer_size=2)
        
        val_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "validation",
                                            base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(val_ds,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     persistent_workers=(args.workers > 0))
        loader = [train_loader, val_loader]

    return loader
