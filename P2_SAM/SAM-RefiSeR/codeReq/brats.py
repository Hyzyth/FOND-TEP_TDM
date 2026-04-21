
import torch
import os
from torch.utils.data.dataset import Dataset, ConcatDataset
from utils.all_utils import pad_or_crop_image, minmax, load_nii, pad_image_and_label, listdir, get_brats_folder
from math import comb
from copy import deepcopy
import numpy as np
import random
import pickle

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

from monai.transforms import (
    Compose,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandAdjustContrastd,
    RandBiasFieldd,
    RandGaussianNoise,
    Lambdad,
)

def partial_invert(x, alpha=0.3):
    return (1.0-alpha)*x + alpha*(1.0-x)

invert_contrast = Lambdad(
    keys=["image"],
    func=lambda x: partial_invert(x, alpha=0.1),
)

monai_3d_augment = Compose([
    RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image","label"], prob=0.5, spatial_axes=(1,2)),


])





def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** (n - i)) * ((1 - t) ** i)

def bezier_curve(points, nTimes = 100000):
    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)
    poly_array = np.array([bernstein_poly(i, nPoints - 1, t) for i in range(nPoints)])
    xvals = np.dot(xPoints, poly_array)
    yvals = np.dot(yPoints, poly_array)
    return xvals, yvals

def apply_domain_curve(img_np):

    c = 1
    if img_np.ndim == 4:
        c = img_np.shape[0]

    v = random.random()
    w = random.random()

    if random.random() < 0.5:
        points = [
            [-1, -1],
            [-v,  v],
            [ v, -v],
            [ 1,  1]
        ]
    else:  # domain_label == 2
        points = [
            [ 1,  1],
            [ w, -w],
            [-w,  w],
            [-1, -1]
        ]

    xvals, yvals = bezier_curve(points, nTimes = 100000)

    idx_sort = np.argsort(xvals)
    xvals = xvals[idx_sort]
    yvals = yvals[idx_sort]

    for ch in range(c):
        if c == 1:
            channel_data = img_np
        else:
            channel_data = img_np[ch]

        cmin, cmax = channel_data.min(), channel_data.max()

        if cmin != cmax:
            channel_norm = 2.0 * (channel_data - cmin) / (cmax - cmin) - 1.0
        else:
            if c > 1:
                img_np[ch] = channel_data
            else:
                img_np = channel_data
            continue

        channel_nonlinear = np.interp(channel_norm, xvals, yvals)

        if c == 1:
            img_np = channel_nonlinear
        else:
            img_np[ch] = channel_nonlinear

    return img_np.astype(np.float32)


class BraTSIndependent(Dataset):
    def __init__(
            self,
            patients_dir,
            patient_ids,
            mode,
            target_size = (128, 128, 128),
            version = 'brats2020',
            modality = 't1',
            seed = None
    ):
        super(BraTSIndependent, self).__init__()

        self.patients_dir = patients_dir
        self.patient_ids = patient_ids
        self.mode = mode
        self.target_size = target_size
        self.version = version
        self.modality = modality.lower()
        self.seed = seed

        self.rng = random.Random(seed) if seed is not None else random

        if self.version in ['brats2020']:
            pattern_map = {
                "t1":    "_t1",
                "t1ce":  "_t1ce",
                "t2":    "_t2",
                "flair": "_flair",
            }
            seg_pattern = "_seg"
        else:
            raise ValueError(
                f"Version {self.version} is not supported"
            )

        self.modality_pattern = pattern_map[self.modality]
        self.seg_pattern = seg_pattern

        self.datas = []

        for patient_id in self.patient_ids:
            image_path = f"{patient_id}{self.modality_pattern}.nii.gz"
            label_path = f"{patient_id}{self.seg_pattern}.nii.gz"

            if self.mode in ['train', 'train_val', 'val', 'test', 'visualize']:
                seg_path = label_path
            else:
                seg_path = None

            self.datas.append(
                dict(
                    id=patient_id,
                    image=image_path,
                    seg=seg_path
                )
            )

    def __getitem__(self, idx):
        patient = self.datas[idx]
        patient_id = patient["id"]

        if self.seed is not None:
            self.rng.seed(self.seed + idx)

        patient_image = torch.tensor(
            load_nii(f"{self.patients_dir}/{patient_id}/{patient['image']}")
        ).float()

        if patient['seg'] is not None:
            patient_label = torch.tensor(
                load_nii(f"{self.patients_dir}/{patient_id}/{patient['seg']}").astype('int8'))
        else:
            patient_label = None

        patient_image = patient_image.unsqueeze(0)

        if patient_label is not None and self.mode in ["train", "train_val", "test", "val", "visualize"]:
            ed_label = 2
            ncr_label = 1
            bg_label = 0

            et_label = 4

            et = (patient_label == et_label)
            tc = torch.logical_or(patient_label == ncr_label, et)
            wt = torch.logical_or(tc, patient_label == ed_label)

            patient_label = torch.stack([et, tc, wt], dim=0).float()

        nonzero_index = torch.nonzero(torch.sum(patient_image, axis = 0) != 0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:, 0], nonzero_index[:, 1], nonzero_index[:, 2]
        zmin, ymin, xmin = [
            max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)
        ]
        zmax, ymax, xmax = [
            int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)
        ]

        patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]

        patient_image[0] = minmax(patient_image[0])

        if patient_label is not None:
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]

        crop_list = []
        pad_list = []

        if self.mode in ["train", "train_val", "test"]:
            patient_image, patient_label, pad_list, crop_list = pad_or_crop_image(
                patient_image, 
                patient_label,
                target_size=self.target_size
            )
        elif self.mode == "test_pad":
            d, h, w = patient_image.shape[1:]
            pad_d = (128 - d) if (128 - d) > 0 else 0
            pad_h = (128 - h) if (128 - h) > 0 else 0
            pad_w = (128 - w) if (128 - w) > 0 else 0
            patient_image, patient_label, pad_list = pad_image_and_label(
                patient_image,
                patient_label,
                target_size=(d + pad_d, h + pad_h, w + pad_w)
            )

        if patient_label is None:
            patient_label = torch.zeros(3, *patient_image.shape[1:], dtype = torch.float32)


        return dict(
            patient_id=patient["id"],
            image=patient_image.to(dtype=torch.float32),
            label=patient_label.to(dtype=torch.float32),
            nonzero_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            box_slice=crop_list,
            pad_list=pad_list
        )
    
    def __len__(self):
        return len(self.datas)
    

    
    
class BraTSPrimeDefaultDataset(BraTSIndependent):
    def __init__(
        self,
        patients_dir,
        patient_ids,
        mode,
        target_size=(128, 128, 128),
        version='brats2020',
        modality='t2',
        domain_label = 0,
        augment=True,
        seed=None,
    ):
        super().__init__(
            patients_dir=patients_dir,
            patient_ids=patient_ids,
            mode=mode,
            target_size=target_size,
            version=version,
            modality=modality,
            seed=seed
        )
        self.augment = augment
        self.seed = seed

        self.monai_3d_augment = monai_3d_augment

        self.domain_label = domain_label

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        image = sample['image']
        label = sample['label']

        if self.seed is not None:
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)
            np.random.seed(self.seed + idx)

        if self.augment:
            data_dict = {"image": image, "label": label}
            
            data_dict = self.monai_3d_augment(data_dict)
            
            image = data_dict["image"]
            label = data_dict["label"]

        sample["image"] = image
        sample["label"] = label
        sample["domain_label"] = torch.tensor(self.domain_label, dtype=torch.long)
        return sample


def get_datasets_prime(
    dataset_folder,
    mode,
    target_size = (128, 128, 128),
    version = 'brats2020',
    modality = 't2',
    seedAh = 42
):
    dataset_folder = get_brats_folder(dataset_folder, mode, version = version)
    assert os.path.exists(dataset_folder), f"Dataset folder {dataset_folder} does not exist."

    patient_ids = [x for x in listdir(dataset_folder)]

    if mode == 'train':
        patient_ids = patient_ids[:400]

    if mode == 'test':
        augmentAhh = False
    else:
        augmentAhh = True

    dsPrime = BraTSPrimeDefaultDataset(
        patients_dir = dataset_folder,
        patient_ids = patient_ids,
        mode = mode,
        target_size = target_size,
        version = version,
        modality = modality,
        domain_label = 0,
        augment = augmentAhh,
        seed = seedAh
    )

    return dsPrime




#################################################


class BraTSIndependentDANN(Dataset):
    def __init__(
            self,
            patients_dir,
            patient_ids,
            mode,
            target_size = (128, 128, 128),
            version = 'brats2020',
            modality = 't1',
            seed = None,
            use_seg = True,
    ):
        super(BraTSIndependentDANN, self).__init__()

        self.patients_dir = patients_dir
        self.patient_ids = patient_ids
        self.mode = mode
        self.target_size = target_size
        self.version = version
        self.modality = modality.lower()
        self.seed = seed
        self.use_seg = use_seg

        self.rng = random.Random(seed) if seed is not None else random

        if self.version in ['brats2020']:
            pattern_map = {
                "t1":    "_t1",
                "t1ce":  "_t1ce",
                "t2":    "_t2",
                "flair": "_flair",
            }
            seg_pattern = "_seg"
        else:
            raise ValueError(
                f"Version {self.version} is not supported"
            )

        self.modality_pattern = pattern_map[self.modality]
        self.seg_pattern = seg_pattern

        self.datas = []

        for patient_id in self.patient_ids:
            image_path = f"{patient_id}{self.modality_pattern}.nii.gz"
            label_path = f"{patient_id}{self.seg_pattern}.nii.gz"

            if self.use_seg and self.mode in ['train', 'train_val', 'val', 'test', 'visualize']:
                seg_path = label_path
            else:
                seg_path = None

            self.datas.append(
                dict(
                    id=patient_id,
                    image=image_path,
                    seg=seg_path
                )
            )

    def __getitem__(self, idx):
        patient = self.datas[idx]
        patient_id = patient["id"]

        if self.seed is not None:
            self.rng.seed(self.seed + idx)

        patient_image = torch.tensor(
            load_nii(f"{self.patients_dir}/{patient_id}/{patient['image']}")
        ).float()

        if patient['seg'] is not None:
            patient_label = torch.tensor(
                load_nii(f"{self.patients_dir}/{patient_id}/{patient['seg']}").astype('int8'))
        else:
            patient_label = None

        patient_image = patient_image.unsqueeze(0)

        if patient_label is not None and self.mode in ["train", "train_val", "test", "val", "visualize"]:
            ed_label = 2
            ncr_label = 1
            bg_label = 0

            et_label = 4

            et = (patient_label == et_label)
            tc = torch.logical_or(patient_label == ncr_label, et)
            wt = torch.logical_or(tc, patient_label == ed_label)

            patient_label = torch.stack([et, tc, wt], dim=0).float()

        nonzero_index = torch.nonzero(torch.sum(patient_image, axis = 0) != 0)
        z_indexes, y_indexes, x_indexes = nonzero_index[:, 0], nonzero_index[:, 1], nonzero_index[:, 2]
        zmin, ymin, xmin = [
            max(0, int(torch.min(arr) - 1)) for arr in (z_indexes, y_indexes, x_indexes)
        ]
        zmax, ymax, xmax = [
            int(torch.max(arr) + 1) for arr in (z_indexes, y_indexes, x_indexes)
        ]

        patient_image = patient_image[:, zmin:zmax, ymin:ymax, xmin:xmax]

        patient_image[0] = minmax(patient_image[0])

        if patient_label is not None:
            patient_label = patient_label[:, zmin:zmax, ymin:ymax, xmin:xmax]

        crop_list = []
        pad_list = []

        if self.mode in ["train", "train_val", "test"]:
            patient_image, patient_label, pad_list, crop_list = pad_or_crop_image(
                patient_image, 
                patient_label,
                target_size=self.target_size
            )
        elif self.mode == "test_pad":
            d, h, w = patient_image.shape[1:]
            pad_d = (128 - d) if (128 - d) > 0 else 0
            pad_h = (128 - h) if (128 - h) > 0 else 0
            pad_w = (128 - w) if (128 - w) > 0 else 0
            patient_image, patient_label, pad_list = pad_image_and_label(
                patient_image,
                patient_label,
                target_size=(d + pad_d, h + pad_h, w + pad_w)
            )

        if patient_label is None:
            patient_label = torch.zeros(3, *patient_image.shape[1:], dtype = torch.float32)


        return dict(
            patient_id=patient["id"],
            image=patient_image.to(dtype=torch.float32),
            label=patient_label.to(dtype=torch.float32),
            nonzero_indexes=((zmin, zmax), (ymin, ymax), (xmin, xmax)),
            box_slice=crop_list,
            pad_list=pad_list
        )
    
    def __len__(self):
        return len(self.datas)
    

class BraTSPrimeDANNDataset(BraTSIndependentDANN):
    def __init__(
        self,
        patients_dir,
        patient_ids,
        mode,
        target_size=(128, 128, 128),
        version='brats2020',
        modality='t2',
        domain_label=0,
        augment=True,
        seed=None,
        use_seg=True,
    ):
        super().__init__(
            patients_dir=patients_dir,
            patient_ids=patient_ids,
            mode=mode,
            target_size=target_size,
            version=version,
            modality=modality,
            seed=seed,
            use_seg=use_seg
        )
        self.augment = augment
        self.seed = seed
        self.monai_3d_augment = monai_3d_augment
        self.domain_label = domain_label

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        image = sample['image']
        label = sample['label']

        if self.seed is not None:
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)
            np.random.seed(self.seed + idx)

        if self.augment:
            data_dict = {"image": image, "label": label}
            data_dict = self.monai_3d_augment(data_dict)
            image = data_dict["image"]
            label = data_dict["label"]


        sample["image"] = image
        sample["label"] = label
        sample["domain_label"] = torch.tensor(self.domain_label, dtype=torch.long)
        return sample
    

def get_domain_adaptation_datasets_alwayst2_singletarget(
    dataset_folder,
    mode,
    target_size = (128, 128, 128),
    version = 'brats2020',
    seedAh = 42,
    chosen_modality = 't1'
):

    if mode == 'train':
        train_folder = get_brats_folder(dataset_folder, mode, version = version)
        assert os.path.exists(train_folder), f"Dataset folder {train_folder} does not exist."
        all_patients = listdir(train_folder)
        all_patients.sort()
        random.seed(seedAh)
        random.shuffle(all_patients)

        N = len(all_patients)
        half = N // 2
        source_ids = all_patients[:half]
        target_ids = all_patients[half:]

        ds_source = BraTSPrimeDANNDataset(
            patients_dir=train_folder,
            patient_ids=source_ids,
            mode='train',
            target_size=target_size,
            version=version,
            modality='t2',
            domain_label=0,
            augment=True,
            seed=seedAh,
            use_seg=True
        )

        ds_target = BraTSPrimeDANNDataset(
            patients_dir=train_folder,
            patient_ids=target_ids,
            mode='train',
            target_size=target_size,
            version=version,
            modality=chosen_modality,
            domain_label=1,
            augment=True,
            seed=seedAh,
            use_seg=False
        )

        return ds_source, ds_target
    else:
        dataset_folder = get_brats_folder(dataset_folder, mode, version = version)
        assert os.path.exists(dataset_folder), f"Dataset folder {dataset_folder} does not exist."

        patient_ids = [x for x in listdir(dataset_folder)]

        ds_target = BraTSPrimeDANNDataset(
            patients_dir=dataset_folder,
            patient_ids=patient_ids,
            mode=mode,
            target_size=target_size,
            version=version,
            modality=chosen_modality,
            domain_label=1,
            augment=False,
            seed=seedAh,
            use_seg=True
        )

        return ds_target
    

#################################################


# source_ids_selective_400 = ['BraTS2021_00289', 'BraTS2021_00621', 'BraTS2021_01102', 'BraTS2021_00488', 'BraTS2021_01058', 'BraTS2021_01064', 'BraTS2021_00068', 'BraTS2021_00291', 'BraTS2021_00819', 'BraTS2021_00729', 'BraTS2021_00516', 'BraTS2021_01274', 'BraTS2021_00789', 'BraTS2021_01344', 'BraTS2021_00612', 'BraTS2021_01188', 'BraTS2021_01210', 'BraTS2021_01169', 'BraTS2021_00682', 'BraTS2021_01085', 'BraTS2021_00392', 'BraTS2021_00293', 'BraTS2021_00044', 'BraTS2021_01316', 'BraTS2021_00824', 'BraTS2021_01101', 'BraTS2021_00650', 'BraTS2021_01492', 'BraTS2021_01319', 'BraTS2021_00593', 'BraTS2021_01564', 'BraTS2021_01501', 'BraTS2021_00530', 'BraTS2021_00172', 'BraTS2021_00395', 'BraTS2021_00626', 'BraTS2021_00216', 'BraTS2021_00716', 'BraTS2021_00775', 'BraTS2021_01063', 'BraTS2021_01322', 'BraTS2021_00207', 'BraTS2021_00053', 'BraTS2021_00101', 'BraTS2021_01635', 'BraTS2021_01641', 'BraTS2021_00519', 'BraTS2021_00512', 'BraTS2021_01140', 'BraTS2021_01570', 'BraTS2021_01195', 'BraTS2021_00800', 'BraTS2021_01350', 'BraTS2021_01060', 'BraTS2021_00379', 'BraTS2021_01479', 'BraTS2021_00655', 'BraTS2021_00744', 'BraTS2021_01517', 'BraTS2021_00165', 'BraTS2021_01445', 'BraTS2021_00390', 'BraTS2021_00397', 'BraTS2021_00768', 'BraTS2021_01227', 'BraTS2021_00061', 'BraTS2021_00095', 'BraTS2021_00758', 'BraTS2021_01035', 'BraTS2021_00596', 'BraTS2021_01454', 'BraTS2021_01045', 'BraTS2021_01423', 'BraTS2021_00544', 'BraTS2021_01462', 'BraTS2021_01436', 'BraTS2021_01547', 'BraTS2021_01073', 'BraTS2021_01001', 'BraTS2021_01198', 'BraTS2021_01439', 'BraTS2021_00058', 'BraTS2021_00687', 'BraTS2021_01543', 'BraTS2021_01391', 'BraTS2021_01307', 'BraTS2021_01192', 'BraTS2021_01443', 'BraTS2021_00823', 'BraTS2021_00147', 'BraTS2021_00742', 'BraTS2021_01160', 'BraTS2021_00332', 'BraTS2021_01463', 'BraTS2021_00728', 'BraTS2021_01121', 'BraTS2021_00253', 'BraTS2021_01544', 'BraTS2021_01310', 'BraTS2021_01202', 'BraTS2021_00336', 'BraTS2021_01474', 'BraTS2021_01053', 'BraTS2021_01272', 'BraTS2021_01377', 'BraTS2021_01135', 'BraTS2021_00750', 'BraTS2021_01082', 'BraTS2021_01539', 'BraTS2021_00017', 'BraTS2021_00615', 'BraTS2021_00559', 'BraTS2021_01515', 'BraTS2021_01478', 'BraTS2021_01589', 'BraTS2021_01068', 'BraTS2021_00359', 'BraTS2021_01318', 'BraTS2021_01626', 'BraTS2021_00469', 'BraTS2021_00587', 'BraTS2021_00459', 'BraTS2021_01375', 'BraTS2021_01412', 'BraTS2021_01241', 'BraTS2021_00500', 'BraTS2021_00811', 'BraTS2021_01186', 'BraTS2021_01432', 'BraTS2021_01330', 'BraTS2021_00102', 'BraTS2021_01651', 'BraTS2021_01296', 'BraTS2021_01256', 'BraTS2021_00639', 'BraTS2021_01632', 'BraTS2021_01155', 'BraTS2021_00081', 'BraTS2021_01027', 'BraTS2021_01278', 'BraTS2021_01613', 'BraTS2021_00677', 'BraTS2021_01392', 'BraTS2021_01122', 'BraTS2021_00054', 'BraTS2021_01021', 'BraTS2021_01496', 'BraTS2021_00120', 'BraTS2021_00495', 'BraTS2021_00448', 'BraTS2021_01232', 'BraTS2021_01600', 'BraTS2021_01468', 'BraTS2021_00019', 'BraTS2021_00028', 'BraTS2021_01434', 'BraTS2021_01087', 'BraTS2021_00022', 'BraTS2021_00480', 'BraTS2021_01144', 'BraTS2021_01516', 'BraTS2021_01382', 'BraTS2021_00126', 'BraTS2021_01275', 'BraTS2021_01567', 'BraTS2021_00674', 'BraTS2021_00646', 'BraTS2021_01019', 'BraTS2021_00809', 'BraTS2021_01163', 'BraTS2021_01260', 'BraTS2021_01206', 'BraTS2021_01476', 'BraTS2021_01465', 'BraTS2021_01565', 'BraTS2021_00234', 'BraTS2021_00062', 'BraTS2021_01591', 'BraTS2021_01165', 'BraTS2021_01054', 'BraTS2021_01025', 'BraTS2021_01041', 'BraTS2021_01131', 'BraTS2021_01511', 'BraTS2021_01430', 'BraTS2021_01616', 'BraTS2021_00839', 'BraTS2021_01604', 'BraTS2021_01320', 'BraTS2021_00479', 'BraTS2021_00085', 'BraTS2021_00360', 'BraTS2021_01554', 'BraTS2021_00334', 'BraTS2021_00616', 'BraTS2021_00624', 'BraTS2021_01161', 'BraTS2021_01327', 'BraTS2021_00399', 'BraTS2021_01620', 'BraTS2021_01110', 'BraTS2021_01404', 'BraTS2021_01552', 'BraTS2021_00109', 'BraTS2021_01189', 'BraTS2021_01273', 'BraTS2021_01368', 'BraTS2021_00370', 'BraTS2021_00375', 'BraTS2021_00267', 'BraTS2021_00554', 'BraTS2021_01271', 'BraTS2021_00378', 'BraTS2021_01427', 'BraTS2021_00571', 'BraTS2021_01345', 'BraTS2021_01441', 'BraTS2021_00730', 'BraTS2021_00288', 'BraTS2021_01028', 'BraTS2021_01311', 'BraTS2021_00011', 'BraTS2021_00348', 'BraTS2021_00290', 'BraTS2021_00732', 'BraTS2021_00586', 'BraTS2021_01541', 'BraTS2021_00425', 'BraTS2021_01033', 'BraTS2021_01264', 'BraTS2021_01090', 'BraTS2021_00235', 'BraTS2021_01378', 'BraTS2021_00158', 'BraTS2021_01145', 'BraTS2021_00210', 'BraTS2021_00026', 'BraTS2021_01656', 'BraTS2021_00350', 'BraTS2021_00074', 'BraTS2021_00346', 'BraTS2021_00676', 'BraTS2021_00836', 'BraTS2021_01069', 'BraTS2021_00706', 'BraTS2021_01048', 'BraTS2021_00269', 'BraTS2021_00570', 'BraTS2021_00549', 'BraTS2021_00159', 'BraTS2021_01216', 'BraTS2021_01255', 'BraTS2021_01534', 'BraTS2021_01415', 'BraTS2021_00757', 'BraTS2021_01329', 'BraTS2021_00271', 'BraTS2021_01213', 'BraTS2021_01300', 'BraTS2021_01106', 'BraTS2021_00247', 'BraTS2021_00259', 'BraTS2021_01342', 'BraTS2021_01246', 'BraTS2021_00837', 'BraTS2021_01654', 'BraTS2021_01074', 'BraTS2021_01112', 'BraTS2021_01384', 'BraTS2021_00313', 'BraTS2021_01520', 'BraTS2021_01143', 'BraTS2021_00589', 'BraTS2021_01237', 'BraTS2021_00840', 'BraTS2021_01663', 'BraTS2021_00491', 'BraTS2021_00550', 'BraTS2021_01017', 'BraTS2021_01461', 'BraTS2021_01493', 'BraTS2021_01139', 'BraTS2021_01250', 'BraTS2021_00667', 'BraTS2021_00242', 'BraTS2021_01173', 'BraTS2021_01343', 'BraTS2021_01280', 'BraTS2021_01295', 'BraTS2021_00423', 'BraTS2021_01563', 'BraTS2021_00070', 'BraTS2021_01315', 'BraTS2021_00243', 'BraTS2021_00043', 'BraTS2021_01553', 'BraTS2021_01475', 'BraTS2021_00201', 'BraTS2021_00228', 'BraTS2021_00688', 'BraTS2021_00236', 'BraTS2021_01530', 'BraTS2021_00331', 'BraTS2021_00548', 'BraTS2021_00285', 'BraTS2021_00088', 'BraTS2021_00160', 'BraTS2021_01262', 'BraTS2021_00736', 'BraTS2021_00214', 'BraTS2021_00636', 'BraTS2021_01079', 'BraTS2021_01555', 'BraTS2021_00317', 'BraTS2021_01075', 'BraTS2021_01559', 'BraTS2021_01340', 'BraTS2021_01363', 'BraTS2021_01526', 'BraTS2021_00300', 'BraTS2021_01592', 'BraTS2021_00574', 'BraTS2021_00406', 'BraTS2021_00299', 'BraTS2021_01238', 'BraTS2021_00645', 'BraTS2021_01646', 'BraTS2021_01581', 'BraTS2021_00430', 'BraTS2021_01583', 'BraTS2021_01150', 'BraTS2021_01231', 'BraTS2021_00292', 'BraTS2021_00405', 'BraTS2021_01524', 'BraTS2021_01200', 'BraTS2021_01610', 'BraTS2021_00538', 'BraTS2021_01389', 'BraTS2021_00123', 'BraTS2021_01336', 'BraTS2021_01545', 'BraTS2021_01649', 'BraTS2021_00777', 'BraTS2021_00693', 'BraTS2021_01023', 'BraTS2021_01056', 'BraTS2021_01586', 'BraTS2021_01267', 'BraTS2021_00703', 'BraTS2021_00567', 'BraTS2021_00551', 'BraTS2021_01395', 'BraTS2021_01293', 'BraTS2021_01107', 'BraTS2021_00254', 'BraTS2021_00186', 'BraTS2021_00602', 'BraTS2021_00387', 'BraTS2021_01548', 'BraTS2021_00543', 'BraTS2021_00631', 'BraTS2021_01638', 'BraTS2021_00309', 'BraTS2021_00344', 'BraTS2021_01268', 'BraTS2021_01409', 'BraTS2021_01354', 'BraTS2021_00828', 'BraTS2021_01113', 'BraTS2021_00610', 'BraTS2021_01011', 'BraTS2021_00260', 'BraTS2021_01258', 'BraTS2021_00024', 'BraTS2021_01366', 'BraTS2021_01648', 'BraTS2021_01645', 'BraTS2021_00801', 'BraTS2021_00539', 'BraTS2021_00675', 'BraTS2021_00814', 'BraTS2021_00799', 'BraTS2021_00066', 'BraTS2021_01481', 'BraTS2021_01253', 'BraTS2021_00401', 'BraTS2021_00187', 'BraTS2021_00690', 'BraTS2021_01284', 'BraTS2021_01386', 'BraTS2021_00230', 'BraTS2021_01401', 'BraTS2021_00349', 'BraTS2021_01411', 'BraTS2021_01185', 'BraTS2021_00221', 'BraTS2021_01527', 'BraTS2021_01388', 'BraTS2021_00383', 'BraTS2021_01289', 'BraTS2021_00735', 'BraTS2021_01558', 'BraTS2021_00266', 'BraTS2021_01062', 'BraTS2021_00778', 'BraTS2021_00698', 'BraTS2021_00416', 'BraTS2021_00831', 'BraTS2021_01091', 'BraTS2021_00781', 'BraTS2021_00171', 'BraTS2021_00021', 'BraTS2021_00421', 'BraTS2021_00167', 'BraTS2021_01008', 'BraTS2021_01442', 'BraTS2021_00237', 'BraTS2021_01133', 'BraTS2021_00795', 'BraTS2021_00649', 'BraTS2021_00410', 'BraTS2021_00582', 'BraTS2021_01249', 'BraTS2021_00351', 'BraTS2021_00105', 'BraTS2021_01164', 'BraTS2021_01584', 'BraTS2021_00715', 'BraTS2021_01123', 'BraTS2021_00353', 'BraTS2021_00033', 'BraTS2021_01002', 'BraTS2021_00364', 'BraTS2021_00303', 'BraTS2021_01578', 'BraTS2021_01341']
# target_ids_selective_400 = ['BraTS2021_01603', 'BraTS2021_01184', 'BraTS2021_01362', 'BraTS2021_00611', 'BraTS2021_00371', 'BraTS2021_01039', 'BraTS2021_00030', 'BraTS2021_00572', 'BraTS2021_00144', 'BraTS2021_00727', 'BraTS2021_00071', 'BraTS2021_01510', 'BraTS2021_01157', 'BraTS2021_00584', 'BraTS2021_01182', 'BraTS2021_00737', 'BraTS2021_00576', 'BraTS2021_00457', 'BraTS2021_01546', 'BraTS2021_01221', 'BraTS2021_01193', 'BraTS2021_00668', 'BraTS2021_00128', 'BraTS2021_01446', 'BraTS2021_01261', 'BraTS2021_00002', 'BraTS2021_00036', 'BraTS2021_01628', 'BraTS2021_01032', 'BraTS2021_01360', 'BraTS2021_01290', 'BraTS2021_00098', 'BraTS2021_00051', 'BraTS2021_01245', 'BraTS2021_01512', 'BraTS2021_01097', 'BraTS2021_01370', 'BraTS2021_01174', 'BraTS2021_00537', 'BraTS2021_01142', 'BraTS2021_01055', 'BraTS2021_00663', 'BraTS2021_00723', 'BraTS2021_01598', 'BraTS2021_01615', 'BraTS2021_01593', 'BraTS2021_00211', 'BraTS2021_01491', 'BraTS2021_00282', 'BraTS2021_00270', 'BraTS2021_00149', 'BraTS2021_01154', 'BraTS2021_00788', 'BraTS2021_00107', 'BraTS2021_01080', 'BraTS2021_01618', 'BraTS2021_01407', 'BraTS2021_00131', 'BraTS2021_01072', 'BraTS2021_01119', 'BraTS2021_00106', 'BraTS2021_00692', 'BraTS2021_01114', 'BraTS2021_00298', 'BraTS2021_00578', 'BraTS2021_01022', 'BraTS2021_00697', 'BraTS2021_00009', 'BraTS2021_01590', 'BraTS2021_00684', 'BraTS2021_01371', 'BraTS2021_01464', 'BraTS2021_01156', 'BraTS2021_00641', 'BraTS2021_00630', 'BraTS2021_01348', 'BraTS2021_01265', 'BraTS2021_00590', 'BraTS2021_00782', 'BraTS2021_01525', 'BraTS2021_01217', 'BraTS2021_00731', 'BraTS2021_01406', 'BraTS2021_01196', 'BraTS2021_01177', 'BraTS2021_01257', 'BraTS2021_00143', 'BraTS2021_00203', 'BraTS2021_01588', 'BraTS2021_00018', 'BraTS2021_00142', 'BraTS2021_01205', 'BraTS2021_01103', 'BraTS2021_00625', 'BraTS2021_00262', 'BraTS2021_01490', 'BraTS2021_01248', 'BraTS2021_00505', 'BraTS2021_01235', 'BraTS2021_00413', 'BraTS2021_00552', 'BraTS2021_01506', 'BraTS2021_01199', 'BraTS2021_01623', 'BraTS2021_00157', 'BraTS2021_00078', 'BraTS2021_01480', 'BraTS2021_00239', 'BraTS2021_01016', 'BraTS2021_00498', 'BraTS2021_01574', 'BraTS2021_01372', 'BraTS2021_00373', 'BraTS2021_00156', 'BraTS2021_01180', 'BraTS2021_01624', 'BraTS2021_01426', 'BraTS2021_01396', 'BraTS2021_01171', 'BraTS2021_01134', 'BraTS2021_01266', 'BraTS2021_00113', 'BraTS2021_00657', 'BraTS2021_01286', 'BraTS2021_01228', 'BraTS2021_01381', 'BraTS2021_00449', 'BraTS2021_00620', 'BraTS2021_01279', 'BraTS2021_00304', 'BraTS2021_01149', 'BraTS2021_01292', 'BraTS2021_00591', 'BraTS2021_01100', 'BraTS2021_01364', 'BraTS2021_00830', 'BraTS2021_00003', 'BraTS2021_01521', 'BraTS2021_00325', 'BraTS2021_01325', 'BraTS2021_01661', 'BraTS2021_00442', 'BraTS2021_01141', 'BraTS2021_01312', 'BraTS2021_01298', 'BraTS2021_00329', 'BraTS2021_00501', 'BraTS2021_01640', 'BraTS2021_00025', 'BraTS2021_01226', 'BraTS2021_00103', 'BraTS2021_01287', 'BraTS2021_01240', 'BraTS2021_01078', 'BraTS2021_00146', 'BraTS2021_00529', 'BraTS2021_01522', 'BraTS2021_01313', 'BraTS2021_00366', 'BraTS2021_01452', 'BraTS2021_00280', 'BraTS2021_01621', 'BraTS2021_01433', 'BraTS2021_01634', 'BraTS2021_01569', 'BraTS2021_00528', 'BraTS2021_00217', 'BraTS2021_00709', 'BraTS2021_01012', 'BraTS2021_00121', 'BraTS2021_00193', 'BraTS2021_00108', 'BraTS2021_01115', 'BraTS2021_01024', 'BraTS2021_01489', 'BraTS2021_00739', 'BraTS2021_01061', 'BraTS2021_01594', 'BraTS2021_00185', 'BraTS2021_00380', 'BraTS2021_01020', 'BraTS2021_01302', 'BraTS2021_00513', 'BraTS2021_01125', 'BraTS2021_01269', 'BraTS2021_01263', 'BraTS2021_00263', 'BraTS2021_01431', 'BraTS2021_00444', 'BraTS2021_01373', 'BraTS2021_00580', 'BraTS2021_00502', 'BraTS2021_01187', 'BraTS2021_01326', 'BraTS2021_00691', 'BraTS2021_01507', 'BraTS2021_00132', 'BraTS2021_01508', 'BraTS2021_00117', 'BraTS2021_00137', 'BraTS2021_00014', 'BraTS2021_01622', 'BraTS2021_01457', 'BraTS2021_00485', 'BraTS2021_01244', 'BraTS2021_01281', 'BraTS2021_00035', 'BraTS2021_00708', 'BraTS2021_01568', 'BraTS2021_00470', 'BraTS2021_00525', 'BraTS2021_00241', 'BraTS2021_01095', 'BraTS2021_01191', 'BraTS2021_01071', 'BraTS2021_01399', 'BraTS2021_01472', 'BraTS2021_00651', 'BraTS2021_00402', 'BraTS2021_01397', 'BraTS2021_01499', 'BraTS2021_01051', 'BraTS2021_01304', 'BraTS2021_01207', 'BraTS2021_01484', 'BraTS2021_00328', 'BraTS2021_01239', 'BraTS2021_01132', 'BraTS2021_00258', 'BraTS2021_00250', 'BraTS2021_01214', 'BraTS2021_00133', 'BraTS2021_01147', 'BraTS2021_00162', 'BraTS2021_01582', 'BraTS2021_01498', 'BraTS2021_00510', 'BraTS2021_00115', 'BraTS2021_00507', 'BraTS2021_00642', 'BraTS2021_00760', 'BraTS2021_01429', 'BraTS2021_00031', 'BraTS2021_01153', 'BraTS2021_00679', 'BraTS2021_01294', 'BraTS2021_00000', 'BraTS2021_01211', 'BraTS2021_01224', 'BraTS2021_00320', 'BraTS2021_00792', 'BraTS2021_01460', 'BraTS2021_01629', 'BraTS2021_01533', 'BraTS2021_01437', 'BraTS2021_01666', 'BraTS2021_00623', 'BraTS2021_01331', 'BraTS2021_00212', 'BraTS2021_01179', 'BraTS2021_00352', 'BraTS2021_01167', 'BraTS2021_01477', 'BraTS2021_00006', 'BraTS2021_01551', 'BraTS2021_00312', 'BraTS2021_01067', 'BraTS2021_01579', 'BraTS2021_01664', 'BraTS2021_01607', 'BraTS2021_01359', 'BraTS2021_01542', 'BraTS2021_01585', 'BraTS2021_01418', 'BraTS2021_00231', 'BraTS2021_01453', 'BraTS2021_00020', 'BraTS2021_01243', 'BraTS2021_00533', 'BraTS2021_01208', 'BraTS2021_01118', 'BraTS2021_01451', 'BraTS2021_00808', 'BraTS2021_00816', 'BraTS2021_00127', 'BraTS2021_01291', 'BraTS2021_01609', 'BraTS2021_01413', 'BraTS2021_01612', 'BraTS2021_01337', 'BraTS2021_00301', 'BraTS2021_00218', 'BraTS2021_00100', 'BraTS2021_00176', 'BraTS2021_01158', 'BraTS2021_01181', 'BraTS2021_00274', 'BraTS2021_01514', 'BraTS2021_01572', 'BraTS2021_01005', 'BraTS2021_01276', 'BraTS2021_01057', 'BraTS2021_01277', 'BraTS2021_00555', 'BraTS2021_01222', 'BraTS2021_01631', 'BraTS2021_00523', 'BraTS2021_01653', 'BraTS2021_00558', 'BraTS2021_00286', 'BraTS2021_01104', 'BraTS2021_01361', 'BraTS2021_01000', 'BraTS2021_01166', 'BraTS2021_01376', 'BraTS2021_00433', 'BraTS2021_00658', 'BraTS2021_01471', 'BraTS2021_01259', 'BraTS2021_00431', 'BraTS2021_00130', 'BraTS2021_00565', 'BraTS2021_01010', 'BraTS2021_00661', 'BraTS2021_01643', 'BraTS2021_00483', 'BraTS2021_00111', 'BraTS2021_00680', 'BraTS2021_01424', 'BraTS2021_00453', 'BraTS2021_01247', 'BraTS2021_01428', 'BraTS2021_01358', 'BraTS2021_00568', 'BraTS2021_00806', 'BraTS2021_01108', 'BraTS2021_00322', 'BraTS2021_01655', 'BraTS2021_01485', 'BraTS2021_01215', 'BraTS2021_00339', 'BraTS2021_01562', 'BraTS2021_01309', 'BraTS2021_00138', 'BraTS2021_01367', 'BraTS2021_01419', 'BraTS2021_01625', 'BraTS2021_00561', 'BraTS2021_01398', 'BraTS2021_00426', 'BraTS2021_00740', 'BraTS2021_00787', 'BraTS2021_00321', 'BraTS2021_00773', 'BraTS2021_01347', 'BraTS2021_01099', 'BraTS2021_01606', 'BraTS2021_01596', 'BraTS2021_00204', 'BraTS2021_01637', 'BraTS2021_00154', 'BraTS2021_00608', 'BraTS2021_01550', 'BraTS2021_00481', 'BraTS2021_01385', 'BraTS2021_00096', 'BraTS2021_00136', 'BraTS2021_01448', 'BraTS2021_00386', 'BraTS2021_01270', 'BraTS2021_00759', 'BraTS2021_01321', 'BraTS2021_01339', 'BraTS2021_00613', 'BraTS2021_01242', 'BraTS2021_00151', 'BraTS2021_00803', 'BraTS2021_00249', 'BraTS2021_01220', 'BraTS2021_01105', 'BraTS2021_01487', 'BraTS2021_00094', 'BraTS2021_00556', 'BraTS2021_01299', 'BraTS2021_00724', 'BraTS2021_00751', 'BraTS2021_00188', 'BraTS2021_00807', 'BraTS2021_00177', 'BraTS2021_01650', 'BraTS2021_00705', 'BraTS2021_01537', 'BraTS2021_00446', 'BraTS2021_00305', 'BraTS2021_01619', 'BraTS2021_00714', 'BraTS2021_01644', 'BraTS2021_01440', 'BraTS2021_00316', 'BraTS2021_01597', 'BraTS2021_01529', 'BraTS2021_00012', 'BraTS2021_01601', 'BraTS2021_00579', 'BraTS2021_01282', 'BraTS2021_01089', 'BraTS2021_00455', 'BraTS2021_01043', 'BraTS2021_01647', 'BraTS2021_01444', 'BraTS2021_01369', 'BraTS2021_01467', 'BraTS2021_00400', 'BraTS2021_01251', 'BraTS2021_00052', 'BraTS2021_01297', 'BraTS2021_01176', 'BraTS2021_00493', 'BraTS2021_00451', 'BraTS2021_00184', 'BraTS2021_00056', 'BraTS2021_00060', 'BraTS2021_01049', 'BraTS2021_01283', 'BraTS2021_00166', 'BraTS2021_01233', 'BraTS2021_01502', 'BraTS2021_01408', 'BraTS2021_00206', 'BraTS2021_01497', 'BraTS2021_00275', 'BraTS2021_00464', 'BraTS2021_00517', 'BraTS2021_00575', 'BraTS2021_01503', 'BraTS2021_00049', 'BraTS2021_00227', 'BraTS2021_01357']
# val = ['BraTS2021_00032', 'BraTS2021_00059', 'BraTS2021_00077', 'BraTS2021_00084', 'BraTS2021_00099', 'BraTS2021_00110', 'BraTS2021_00116', 'BraTS2021_00122', 'BraTS2021_00124', 'BraTS2021_00134', 'BraTS2021_00148', 'BraTS2021_00170', 'BraTS2021_00178', 'BraTS2021_00183', 'BraTS2021_00192', 'BraTS2021_00199', 'BraTS2021_00238', 'BraTS2021_00251', 'BraTS2021_00261', 'BraTS2021_00284', 'BraTS2021_00294', 'BraTS2021_00296', 'BraTS2021_00297', 'BraTS2021_00324', 'BraTS2021_00340', 'BraTS2021_00343', 'BraTS2021_00347', 'BraTS2021_00356', 'BraTS2021_00369', 'BraTS2021_00376', 'BraTS2021_00382', 'BraTS2021_00388', 'BraTS2021_00389', 'BraTS2021_00391', 'BraTS2021_00403', 'BraTS2021_00404', 'BraTS2021_00412', 'BraTS2021_00417', 'BraTS2021_00418', 'BraTS2021_00429', 'BraTS2021_00432', 'BraTS2021_00441', 'BraTS2021_00443', 'BraTS2021_00452', 'BraTS2021_00456', 'BraTS2021_00466', 'BraTS2021_00468', 'BraTS2021_00477', 'BraTS2021_00478', 'BraTS2021_00496', 'BraTS2021_00504', 'BraTS2021_00506', 'BraTS2021_00518', 'BraTS2021_00520', 'BraTS2021_00524', 'BraTS2021_00540', 'BraTS2021_00542', 'BraTS2021_00545', 'BraTS2021_00557', 'BraTS2021_00563', 'BraTS2021_00569', 'BraTS2021_00577', 'BraTS2021_00581', 'BraTS2021_00583', 'BraTS2021_00588', 'BraTS2021_00594', 'BraTS2021_00598', 'BraTS2021_00618', 'BraTS2021_00619', 'BraTS2021_00628', 'BraTS2021_00638', 'BraTS2021_00640', 'BraTS2021_00654', 'BraTS2021_00656', 'BraTS2021_00683', 'BraTS2021_00689', 'BraTS2021_00694', 'BraTS2021_00704', 'BraTS2021_00707', 'BraTS2021_00718', 'BraTS2021_00725', 'BraTS2021_00733', 'BraTS2021_00734', 'BraTS2021_00747', 'BraTS2021_00756', 'BraTS2021_00764', 'BraTS2021_00767', 'BraTS2021_00780', 'BraTS2021_00784', 'BraTS2021_00793', 'BraTS2021_00802', 'BraTS2021_00804', 'BraTS2021_00810', 'BraTS2021_00818', 'BraTS2021_00820', 'BraTS2021_00834', 'BraTS2021_01003', 'BraTS2021_01004', 'BraTS2021_01014', 'BraTS2021_01030', 'BraTS2021_01031', 'BraTS2021_01036', 'BraTS2021_01038', 'BraTS2021_01047', 'BraTS2021_01050', 'BraTS2021_01059', 'BraTS2021_01065', 'BraTS2021_01076', 'BraTS2021_01077', 'BraTS2021_01092', 'BraTS2021_01093', 'BraTS2021_01096', 'BraTS2021_01111', 'BraTS2021_01116', 'BraTS2021_01117', 'BraTS2021_01127', 'BraTS2021_01129', 'BraTS2021_01130', 'BraTS2021_01136', 'BraTS2021_01138', 'BraTS2021_01151', 'BraTS2021_01152', 'BraTS2021_01159', 'BraTS2021_01168', 'BraTS2021_01170', 'BraTS2021_01172', 'BraTS2021_01183', 'BraTS2021_01190', 'BraTS2021_01197', 'BraTS2021_01201', 'BraTS2021_01204', 'BraTS2021_01219', 'BraTS2021_01225', 'BraTS2021_01230', 'BraTS2021_01236', 'BraTS2021_01285', 'BraTS2021_01288', 'BraTS2021_01303', 'BraTS2021_01306', 'BraTS2021_01308', 'BraTS2021_01323', 'BraTS2021_01324', 'BraTS2021_01328', 'BraTS2021_01332', 'BraTS2021_01335', 'BraTS2021_01338', 'BraTS2021_01346', 'BraTS2021_01351', 'BraTS2021_01353', 'BraTS2021_01356', 'BraTS2021_01379', 'BraTS2021_01394', 'BraTS2021_01402', 'BraTS2021_01403', 'BraTS2021_01405', 'BraTS2021_01410', 'BraTS2021_01414', 'BraTS2021_01416', 'BraTS2021_01417', 'BraTS2021_01425', 'BraTS2021_01447', 'BraTS2021_01449', 'BraTS2021_01455', 'BraTS2021_01458', 'BraTS2021_01466', 'BraTS2021_01469', 'BraTS2021_01483', 'BraTS2021_01488', 'BraTS2021_01495', 'BraTS2021_01505', 'BraTS2021_01509', 'BraTS2021_01518', 'BraTS2021_01523', 'BraTS2021_01528', 'BraTS2021_01531', 'BraTS2021_01535', 'BraTS2021_01536', 'BraTS2021_01549', 'BraTS2021_01561', 'BraTS2021_01571', 'BraTS2021_01576', 'BraTS2021_01595', 'BraTS2021_01611', 'BraTS2021_01630', 'BraTS2021_01642', 'BraTS2021_01658', 'BraTS2021_01659', 'BraTS2021_01665']

def get_domain_adaptation_datasets_train_selective(
    dataset_folder,
    mode,
    target_size=(128, 128, 128),
    version='brats2020',
    seedAh=42,
    chosen_modality_for_target='t1ce',
    selected='t2',
    fraction: float = None,    
    max_samples: int = None  
):

    train_folder = get_brats_folder(dataset_folder, mode, version=version)
    assert os.path.exists(train_folder), f"Dataset folder {train_folder} does not exist."

    src_ids = list(source_ids_selective_400)
    tgt_ids = list(target_ids_selective_400)

    def subsample(ids):
        n = len(ids)
        if fraction is not None:
            k = max(1, int(n * fraction))
        elif max_samples is not None:
            k = min(n, max_samples)
        else:
            return ids
        random.seed(seedAh)
        return random.sample(ids, k)

    src_ids = subsample(src_ids)
    tgt_ids = subsample(tgt_ids)

    ds_source = BraTSPrimeDANNDataset(
        patients_dir=train_folder,
        patient_ids=src_ids,
        mode='train',
        target_size=target_size,
        version=version,
        modality=selected,
        domain_label=0,
        augment=True,
        seed=seedAh,
        use_seg=True
    )

    ds_target = BraTSPrimeDANNDataset(
        patients_dir=train_folder,
        patient_ids=tgt_ids,
        mode='train',
        target_size=target_size,
        version=version,
        modality=chosen_modality_for_target,
        domain_label=1,
        augment=True,
        seed=seedAh,
        use_seg=True
    )

    return ds_source, ds_target

    

def get_domain_adaptation_datasets_train_selective_one(
    dataset_folder,
    mode,
    target_size = (128, 128, 128),
    version = 'brats2020',
    seedAh = 42,
    chosen_modality = 't1',
    selected_patient = ['BraTS2021_01603']
):

    train_folder = get_brats_folder(dataset_folder, mode, version = version)
    assert os.path.exists(train_folder), f"Dataset folder {train_folder} does not exist."

    ds_source = BraTSPrimeDANNDataset(
        patients_dir=train_folder,
        patient_ids=selected_patient,
        mode='train',
        target_size=target_size,
        version=version,
        modality=chosen_modality,
        domain_label=1,
        augment=False,
        seed=seedAh,
        use_seg=True
    )

    return ds_source


######################################

class PseudoTargetDataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        self.patient_ids = list(self.data.keys())
    def __len__(self):
        return len(self.patient_ids)
    def __getitem__(self, idx):
        pid = self.patient_ids[idx]
        sample = self.data[pid]

        image = torch.tensor(sample['image'], dtype = torch.float32).unsqueeze(0)
        pseudo_label = torch.tensor(sample['pseudo_label'], dtype = torch.float32)
        domain_label = torch.tensor(sample['domain_label'], dtype = torch.long)
        pad_list = sample['pad_list']
        crop_list = sample['crop_list']

        return {
            'patient_id': pid,
            'image': image,
            'label': pseudo_label,
            'domain_label': domain_label,
            'pad_list': pad_list,
            'box_slice': crop_list
        }    

###############################

def get_dataset_visualizing(
    dataset_folder,
    mode,
    target_size=(128, 128, 128),
    version='brats2020',
    modality='t2',
    augment=False,
    seed=42
):
    


    actual_folder = get_brats_folder(dataset_folder, mode, version=version)
    assert os.path.exists(actual_folder), f"Dataset folder {actual_folder} does not exist."
    
    patient_ids = [x for x in listdir(actual_folder)]
    
    dataset = BraTSPrimeDefaultDataset(
        patients_dir=actual_folder,
        patient_ids=patient_ids,
        mode=mode,
        target_size=target_size,
        version=version,
        modality=modality,
        domain_label=0,
        augment=augment,
        seed=seed
    )
    return dataset




##########################

import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from utils.all_utils import load_nii, minmax
from monai.transforms import Compose, RandFlipd, RandRotate90d


def resample_volume(
    image: torch.Tensor,   # (1, S, S, S)
    label: torch.Tensor,   # (C, S, S, S)
    target_size: tuple     # (Td, Th, Tw)
):
    img = image.unsqueeze(0)   # (1,1,S,S,S)
    lbl = label.unsqueeze(0)   # (1,C,S,S,S)
    img_rs = F.interpolate(img, size=target_size, mode='trilinear', align_corners=False)
    lbl_rs = F.interpolate(lbl, size=target_size, mode='nearest')
    return img_rs.squeeze(0), lbl_rs.squeeze(0)


class BraTSPrimeResampleDANNDataset(Dataset):
    def __init__(
        self,
        patients_dir,
        patient_ids,
        mode,
        target_size=(128,128,128),
        version='brats2020',
        modality='t2',
        domain_label=0,
        augment=True,
        seed=None,
        use_seg=True
    ):
        super().__init__()
        self.patients_dir = patients_dir
        self.patient_ids  = patient_ids
        self.mode         = mode
        self.target_size  = target_size
        self.version      = version
        self.modality     = modality.lower()
        self.domain_label = domain_label
        self.augment      = Compose(
            [
                RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
                RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),
                RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),
                RandRotate90d(keys=["image","label"], prob=0.5, spatial_axes=(1,2)),

                RandScaleIntensityd(keys=["image"], factors=0.5, prob=0.5),  
                RandShiftIntensityd(keys=["image"], offsets=0.5, prob=0.5),  
                RandAdjustContrastd(keys=["image"], gamma=(0.2,5.0), prob=0.5),
            ]
        )


        self.seed         = seed
        self.use_seg      = use_seg
        self.rng          = random.Random(seed) if seed is not None else random

        if version=='brats2020':
            mapping = {"t1":"_t1","t1ce":"_t1ce","t2":"_t2","flair":"_flair"}
            seg_pat = "_seg"
        else:
            raise ValueError(f"Unsupported version {version}")
        self.modality_pattern = mapping[self.modality]
        self.seg_pattern      = seg_pat

        self.datas = []
        for pid in patient_ids:
            img_f = f"{pid}{self.modality_pattern}.nii.gz"
            seg_f = f"{pid}{self.seg_pattern}.nii.gz" if use_seg and mode in ['train','val','test','visualize'] else None
            self.datas.append(dict(id=pid, image=img_f, seg=seg_f))

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if self.seed is not None:
            random.seed(self.seed + idx)
            torch.manual_seed(self.seed + idx)
            np.random.seed(self.seed + idx)

        rec = self.datas[idx]
        pid = rec['id']

        img_np = load_nii(os.path.join(self.patients_dir, pid, rec['image']))
        image  = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)  

        if rec['seg'] is not None:
            seg_np = load_nii(os.path.join(self.patients_dir, pid, rec['seg'])).astype('int8')
            et = (seg_np == 4)
            tc = np.logical_or(seg_np == 1, et)
            wt = np.logical_or(tc, seg_np == 2)
            label = torch.stack([
                torch.from_numpy(et),
                torch.from_numpy(tc),
                torch.from_numpy(wt)
            ], dim=0).float()  
        else:
            label = torch.zeros(3, *image.shape[1:], dtype=torch.float32)

        image[0] = minmax(image[0])

        D, H, W = image.shape[1:]


        eps = 0.8
        thr = image[0].max() * eps
        mask = (image[0] > thr)

        nz = torch.nonzero(mask)
        z0, z1 = int(nz[:,0].min()), int(nz[:,0].max()) + 1
        y0, y1 = int(nz[:,1].min()), int(nz[:,1].max()) + 1
        x0, x1 = int(nz[:,2].min()), int(nz[:,2].max()) + 1
        image_fg = image[:, z0:z1, y0:y1, x0:x1]
        label_fg = label[:, z0:z1, y0:y1, x0:x1]

        d, h, w = image_fg.shape[1:]


        _, d, h, w = image_fg.shape
        S = max(d, h, w)
        pz, ph, pw = S - d, S - h, S - w
        pz0, pz1 = pz//2, pz - pz//2
        ph0, ph1 = ph//2, ph - ph//2
        pw0, pw1 = pw//2, pw - pw//2

        mean_val = float(image_fg.mean())
        pad_tuple = (pw0, pw1, ph0, ph1, pz0, pz1)
        image_cube = F.pad(image_fg, pad_tuple, mode='constant', value=mean_val)
        label_cube = F.pad(label_fg, pad_tuple, mode='constant', value=0)

        image_rs, label_rs = resample_volume(image_cube, label_cube, self.target_size)

        if self.augment:
            dct = {'image': image_rs, 'label': label_rs}
            dct = monai_3d_augment(dct)
            image_rs, label_rs = dct['image'], dct['label']

        return {
            'patient_id'  : pid,
            'image'       : image_rs,        
            'label'       : label_rs,       
            'domain_label': torch.tensor(self.domain_label, dtype=torch.long),
        }


def get_source_dataset_packitup(
    packitup_folder: str,
    modality: str = 't1ce',
    mode: str = 'train',
    target_size: tuple = (128, 128, 128),
    version: str = 'brats2020',
    seed: int = 42,
    fraction: float = None,
    max_samples: int = None,
    augment: bool = True,
    domain_label: int = 0,
    use_seg: bool = True,
) -> BraTSPrimeResampleDANNDataset:
    img_dir = os.path.join(packitup_folder, 'img')
    seg_dir = os.path.join(packitup_folder, 'seg')
    assert os.path.isdir(img_dir), f"{img_dir} not found"
    assert os.path.isdir(seg_dir), f"{seg_dir} not found"

    all_imgs = sorted([fn for fn in os.listdir(img_dir) if fn.endswith(f"_{modality}.nii.gz")])
    patient_ids = [fn.replace(f"_{modality}.nii.gz", "") for fn in all_imgs]

    def subsample(ids):
        if fraction is not None:
            k = max(1, int(len(ids) * fraction))
        elif max_samples is not None:
            k = min(len(ids), max_samples)
        else:
            return ids
        random.seed(seed)
        return random.sample(ids, k)

    patient_ids = subsample(patient_ids)

    ds = BraTSPrimeResampleDANNDataset(
        patients_dir=packitup_folder,
        patient_ids=[''] * len(patient_ids),
        mode=mode,
        target_size=target_size,
        version=version,
        modality=modality,
        domain_label=domain_label,
        augment=augment,
        seed=seed,
        use_seg=use_seg
    )

    ds.datas = []
    for pid in patient_ids:
        ds.datas.append({
            'id'   : '',
            'image': os.path.join('img', f"{pid}_{modality}.nii.gz"),
            'seg'  : os.path.join('seg', f"{pid}_seg.nii.gz") if use_seg else None
        })

    return ds
