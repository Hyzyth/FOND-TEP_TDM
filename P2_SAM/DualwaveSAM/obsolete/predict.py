import datetime
import glob
import os
import warnings

import cv2
import numpy as np
import torch
# CHANGE: removed unused torch.nn, torch.optim imports; kept functional ones
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

import argparse
import P2_SAM.DualwaveSAM.obsolete.ex_transforms as ex_transforms
import P2_SAM.DualwaveSAM.obsolete.metrics as metrics

# CHANGE: medpy hd95 is the only metric used here; import directly
from medpy.metric.binary import hd95

warnings.filterwarnings("ignore")
CUDA_LAUNCH_BLOCKING = 1


# =========================
# Argument configuration
# =========================
parser = argparse.ArgumentParser(description='DualwaveSAM inference')

parser.add_argument('--batch_size',      type=int, default=1)
parser.add_argument('--test_batch_size', type=int, default=1)
parser.add_argument('--input_nc',        type=int, default=2)
parser.add_argument('--output_nc',       type=int, default=1)
parser.add_argument('--cuda',            action='store_true')
parser.add_argument('--threads',         type=int, default=0)
parser.add_argument('--seed',            type=int, default=42)

# CHANGE: added --data_dir and --dataset flags so paths are not hardcoded
parser.add_argument('--data_dir',  type=str, default='dataset/CHEN_test',
                    help='Directory containing per-patient .npz test files')
parser.add_argument('--dataset',   type=str, default='CHEN',
                    choices=['CHEN', 'Hecktor'],
                    help='Dataset name used for checkpoint selection')
parser.add_argument('--save_dir',  type=str, default='predict_one/test33',
                    help='Directory to write per-patient prediction .npz files')

opt = parser.parse_args()


# ============================================================
# Dataset
# ============================================================
class HECKTORdataset(Dataset):
    """
    Slice-level dataset for a single patient volume.

    CHANGE: eliminated the global variable leak (_all_inputs / _all_targets
    were read from module-level globals inside __init__).  The arrays are now
    passed explicitly as constructor arguments.
    """

    def __init__(self, all_inputs: np.ndarray, all_targets: np.ndarray,
                 transform=None):
        """
        Args:
            all_inputs:  (H, W, D, C) float array for one patient
            all_targets: (H, W, D)    float array for one patient
            transform:   optional augmentation pipeline
        """
        self.transform = transform

        # Convert volume layouts to slice-first: (D, H, W, C) / (D, H, W)
        inputs_slices  = np.moveaxis(all_inputs,  [2, 0, 1], [0, 1, 2])
        targets_slices = np.moveaxis(all_targets, [2, 0, 1], [0, 1, 2])

        self.inputs  = inputs_slices
        self.targets = targets_slices

        print(f"Total slices loaded: {self.inputs.shape[0]}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        sample = {
            'id':     idx,
            'input':  self.inputs[idx],
            'target': self.targets[idx],
        }

        sample['input'] = cv2.resize(
            sample['input'], (256, 256), interpolation=cv2.INTER_LINEAR
        )
        sample['input'] = sample['input'].transpose([1, 0, 2])

        sample['target'] = cv2.resize(
            sample['target'], (256, 256), interpolation=cv2.INTER_LINEAR
        )
        sample['target'] = np.expand_dims(sample['target'], axis=2)
        sample['target'] = sample['target'].transpose([1, 0, 2])

        if self.transform:
            sample = self.transform(sample)

        return sample


# ============================================================
# Per-patient inference
# ============================================================
def testOne(net_g, device, loader, pid, save_dir):
    """
    Run inference on one patient and compute per-patient metrics.

    CHANGE: model is now received as an already-loaded nn.Module instead of
    being reloaded from disk on every patient call.

    Returns:
        dict with avg_dice and avg_hd95 for this patient
    """
    net_g.eval()

    tal_dice = 0.0
    tal_hd95 = 0.0
    preds, targets = [], []

    with torch.no_grad():
        val_qbar = trange(len(loader), desc=pid)

        for sample in loader:
            inp    = sample['input'].to(device)
            target = sample['target'].to(device)

            unique_values = torch.unique(target)
            if len(unique_values) == 1 and unique_values.item() == 0:
                preds.append(target.cpu().numpy())
                targets.append(target.cpu().numpy())
                tal_dice += 1.0
                tal_hd95 += 0.0
                val_qbar.set_postfix(dice=1.0, hd95=0.0)
                val_qbar.update(1)
                continue

            prediction, _ = net_g(inp)

            pred_np   = ex_transforms.ProbsToLabels()(prediction.cpu().numpy())
            target_np = target.cpu().numpy()

            dice_val = metrics.dice(pred_np, target_np)

            try:
                hd_val = hd95(pred_np, target_np)
            except Exception:
                hd_val = np.float64(0)

            tal_dice += dice_val
            tal_hd95 += hd_val

            preds.append(pred_np)
            targets.append(target_np)

            val_qbar.set_postfix(dice=float(dice_val), hd95=float(hd_val))
            val_qbar.update(1)

    n          = len(loader)
    avg_dice   = tal_dice / n
    avg_hd95   = tal_hd95 / n

    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, pid), pred=preds, target=targets)

    print('*' * 60)
    print(f"[{pid}]  avg_dice={avg_dice:.4f}   avg_HD95={avg_hd95:.4f}")
    print('*' * 60)

    return {'avg_dice': avg_dice, 'avg_hd95': avg_hd95}


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":

    curr_time = datetime.datetime.now()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("CUDA requested but not available")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    model_name = 'DualwaveSAM'
    print('\n' + '*' * 30 + f" Testing {model_name} " + '*' * 30)

    # CHANGE: load model once outside the patient loop instead of reloading
    # per patient, which was wasteful and only worked for CHEN anyway.
    ckpt_path = f'results/{opt.dataset}/TBME-DualwaveSAM/re_haar/best_model/net_model.pth'
    net_g = torch.load(ckpt_path, map_location=device)
    net_g.eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    med_transform = ex_transforms.Compose([ex_transforms.ToTensor()])

    # CHANGE: use opt.data_dir instead of the hardcoded 'dataset/CHEN_test' path
    npz_files = glob.glob(os.path.join(opt.data_dir, '*.npz'))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {opt.data_dir}")

    pid_list = [os.path.splitext(os.path.basename(f))[0] for f in npz_files]
    print(f"Patients to evaluate: {pid_list}\n")

    # CHANGE: accumulate metrics across all patients for a global summary
    all_dice, all_hd95 = [], []

    for pid in pid_list:
        print(f"\n--- Processing: {pid} ---")

        dataset_np = np.load(
            os.path.join(opt.data_dir, f"{pid}.npz"), allow_pickle=True
        )
        all_inputs  = dataset_np['input']
        all_targets = dataset_np['target']

        med_data = HECKTORdataset(
            all_inputs=all_inputs,
            all_targets=all_targets,
            transform=med_transform,
        )
        loader = DataLoader(
            dataset=med_data,
            num_workers=opt.threads,
            batch_size=opt.test_batch_size,
            shuffle=False,
            drop_last=True,
        )

        result = testOne(net_g, device, loader, pid, opt.save_dir)
        all_dice.append(result['avg_dice'])
        all_hd95.append(result['avg_hd95'])

    # CHANGE: print global aggregate metrics across all patients
    print('\n' + '=' * 60)
    print(f"GLOBAL RESULTS  ({len(pid_list)} patients)")
    print(f"  Mean Dice : {np.mean(all_dice):.4f}  ±  {np.std(all_dice):.4f}")
    print(f"  Mean HD95 : {np.mean(all_hd95):.4f}  ±  {np.std(all_hd95):.4f}")
    print('=' * 60)

    elapsed = datetime.datetime.now() - curr_time
    print(f"\nTotal runtime: {elapsed}")
