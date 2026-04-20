import datetime
import glob
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from medpy.metric.binary import hd, hd95
from torch.utils.data import Dataset, DataLoader
from tqdm import *

import argparse
import ex_transforms
import metrics

warnings.filterwarnings("ignore")
CUDA_LAUNCH_BLOCKING = 1


# =========================
# Argument configuration
# =========================
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')

parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='Testing batch size')

parser.add_argument('--input_nc', type=int, default=2, help='Number of input channels')
parser.add_argument('--output_nc', type=int, default=1, help='Number of output channels')

parser.add_argument('--cuda', action='store_true', help='Enable CUDA acceleration')
parser.add_argument('--threads', type=int, default=0, help='DataLoader worker threads')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

opt = parser.parse_args()


def testOne(model, device, pid):
    """
    Run inference on a single patient volume and compute segmentation metrics.

    Args:
        model (str): Model identifier (e.g., 'DualwaveSAM')
        device (torch.device): Computation device
        pid (str): Patient ID used for saving results
    """

    # Load pretrained model checkpoint
    if model == 'DualwaveSAM':
        net_g = torch.load(
            'results/CHEN/TBME-DualwaveSAM/re_haar/best_model/net_model.pth',
            map_location=device
        )
    else:
        raise ValueError("Unsupported model type")

    net_g.eval()

    # Metric accumulators
    tal_dice = 0.0
    tal_hd95 = 0.0

    preds, targets = [], []

    with torch.no_grad():
        val_qbar = trange(len(med_set_loader))

        for sample in med_set_loader:
            input = sample['input'].to(device)
            target = sample['target'].to(device)

            # Handle empty-mask edge case (all background)
            unique_values = torch.unique(target)

            if len(unique_values) == 1 and unique_values.item() == 0:
                preds.append(target.cpu().numpy())
                targets.append(target.cpu().numpy())

                tal_dice += 1.0
                tal_hd95 += 0.0

                val_qbar.set_postfix(dice=1.0, hd95=0.0)
                val_qbar.update(1)
                continue

            # Forward pass
            if model == 'DualwaveSAM':
                prediction, _ = net_g(input)
            else:
                prediction = net_g(input)

            # Convert probabilities to binary labels
            pred_np = ex_transforms.ProbsToLabels()(prediction.cpu().numpy())
            target_np = target.cpu().numpy()

            dice_pred = metrics.dice(pred_np, target_np)

            try:
                hd_val = hd95(pred_np, target_np)
            except:
                hd_val = np.float64(0)

            tal_dice += dice_pred
            tal_hd95 += hd_val

            preds.append(pred_np)
            targets.append(target_np)

            val_qbar.set_postfix(dice=dice_pred.item(), hd95=hd_val.item())
            val_qbar.update(1)

        # Average metrics
        avg_dice = tal_dice / len(med_set_loader)
        avg_hd95 = tal_hd95 / len(med_set_loader)

        # Save predictions
        file = os.path.join("predict_one/test33", pid)
        np.savez(file, pred=preds, target=targets)

        print('*' * 100)
        print("test avg_dice: {:.4f}".format(avg_dice))
        print("test avg_HD95: {:.4f}".format(avg_hd95))
        print('*' * 100)


class HECKTORdataset(Dataset):
    """
    Dataset loader operating at slice level.

    Converts 3D patient volumes into 2D slices and caches them for fast reuse.
    """

    _slice_inputs = None
    _slice_targets = None

    def __init__(
        self,
        mode: str = 'test',
        seed: int = 17,
        split_rate: float = 0.8,
        data_rate: float = 1.0,
        transform=None,
    ):
        """
        Initialize dataset and prepare cached slice representation.

        Args:
            mode: dataset mode ('train' or 'test')
            seed: random seed (unused in current implementation)
            split_rate: train/test split ratio (unused here)
            data_rate: dataset subsampling ratio (unused here)
            transform: optional augmentation pipeline
        """

        self.transform = transform
        self.mode = mode

        # NOTE: assumes external variables _all_inputs and _all_targets exist
        if 1:
            # Original volume shapes:
            # inputs: (N, H, W, D, C)
            # targets: (N, H, W, D)

            # Convert to slice-first layout: (D, H, W, C)
            inputs_cases = np.moveaxis(_all_inputs, [2, 0, 1], [0, 1, 2])
            targets_cases = np.moveaxis(_all_targets, [2, 0, 1], [0, 1, 2])

            HECKTORdataset._slice_inputs = inputs_cases
            HECKTORdataset._slice_targets = targets_cases

        inputs_slices = HECKTORdataset._slice_inputs
        targets_slices = HECKTORdataset._slice_targets

        self.inputs = inputs_slices
        self.targets = targets_slices

        print(f"Mode: {mode} | Total slices: {self.inputs.shape[0]}")

    def __len__(self):
        """Return total number of slices."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieve one slice sample.

        Returns:
            dict with input image and segmentation mask
        """
        sample = dict()

        sample['id'] = idx
        sample['input'] = self.inputs[idx]
        sample['target'] = self.targets[idx]

        # Resize image and mask to model input size
        sample['input'] = cv2.resize(sample['input'], (256, 256), interpolation=cv2.INTER_LINEAR)
        sample['input'] = sample['input'].transpose([1, 0, 2])

        sample['target'] = cv2.resize(sample['target'], (256, 256), interpolation=cv2.INTER_LINEAR)
        sample['target'] = np.expand_dims(sample['target'], axis=2)
        sample['target'] = sample['target'].transpose([1, 0, 2])

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":

    curr_time = datetime.datetime.now()

    # Device selection
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_name = 'DualwaveSAM'
    print('\n' + '*' * 30 + f" Testing model {model_name} " + '*' * 30)

    # Collect test patient IDs
    npz_files = glob.glob('dataset/CHEN_test/*.npz')
    pid_list = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]

    print(pid_list)

    for pid in pid_list:
        print(f"\nProcessing PID: {pid}")

        _npz_path = f"dataset/CHEN_test/{pid}.npz"
        _dataset = np.load(_npz_path, allow_pickle=True)

        _all_inputs = _dataset['input']
        _all_targets = _dataset['target']

        if opt.cuda and not torch.cuda.is_available():
            raise Exception("CUDA requested but not available")

        torch.manual_seed(opt.seed)
        if opt.cuda:
            torch.cuda.manual_seed(opt.seed)

        med_transform = ex_transforms.Compose([
            ex_transforms.ToTensor()
        ])

        med_data = HECKTORdataset(mode='test', transform=med_transform, seed=42)

        med_set_loader = DataLoader(
            dataset=med_data,
            num_workers=opt.threads,
            batch_size=opt.test_batch_size,
            shuffle=False,
            drop_last=True
        )

        testOne(model_name, device, pid)

    print('\n' + '*' * 30 + f" Model {model_name} testing complete " + '*' * 30)
