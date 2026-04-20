# =========================
# Standard library imports
# =========================
import os
import glob
import argparse
import datetime
import warnings

# =========================
# Third-party imports
# =========================
import cv2
import numpy as np
import pandas as pd
from tqdm import *
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from medpy.metric.binary import hd, hd95

# =========================
# Local module imports
# =========================
import ex_transforms
import losses
import metrics

warnings.filterwarnings("ignore")


def get_scheduler(optimizer, opt):
    """
    Create a learning rate scheduler based on the selected policy.

    Supported policies:
    - lambda: linear decay after a fixed number of epochs
    - step: step decay at fixed intervals
    - plateau: reduce LR when a metric has stopped improving
    - cosine: cosine annealing schedule

    Args:
        optimizer: optimizer instance
        opt: configuration object containing scheduler parameters

    Returns:
        scheduler instance
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=1.1)

    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5
        )

    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)

    else:
        raise NotImplementedError(f"Learning rate policy '{opt.lr_policy}' is not implemented")

    return scheduler


def update_learning_rate(scheduler, optimizer):
    """
    Update learning rate at the end of each epoch.

    Args:
        scheduler: learning rate scheduler
        optimizer: optimizer whose LR will be updated
    """
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


# =========================
# Training configuration
# =========================
parser = argparse.ArgumentParser(description='DualwaveSAM training')

parser.add_argument('--batch_size',      type=int,   default=12,      help='Training batch size')
parser.add_argument('--test_batch_size', type=int,   default=12,      help='Testing batch size')

parser.add_argument('--input_nc',  type=int, default=2, help='Number of input channels')
parser.add_argument('--output_nc', type=int, default=1, help='Number of output channels')

parser.add_argument('--epoch_count',  type=int,   default=1,  help='Starting epoch index')
parser.add_argument('--niter',        type=int,   default=10, help='Epochs at initial learning rate')
parser.add_argument('--niter_decay',  type=int,   default=40, help='Epochs for LR decay')

parser.add_argument('--lr',         type=float, default=0.0001, help='Initial learning rate for Adam')
parser.add_argument('--lr_policy',  type=str,   default='lambda',
                    help='Learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=1,
                    help='Step interval for step LR policy')
parser.add_argument('--beta1', type=float, default=0.5, help='Beta1 for Adam optimizer')

parser.add_argument('--cuda',    action='store_true', help='Use CUDA acceleration')
parser.add_argument('--threads', type=int,   default=0,  help='Data loader threads')
parser.add_argument('--seed',    type=int,   default=42, help='Random seed')

# FIX: these were incorrectly typed as int, silently truncating 0.01 → 0
parser.add_argument('--lamb1', type=float, default=0.01,    help='Weight for L1 loss')
parser.add_argument('--lamb2', type=float, default=0.1,     help='Weight for L2 loss')
parser.add_argument('--glr',   type=float, default=0.000128, help='SGD learning rate')

parser.add_argument('--trc',  type=int,   default=1,  help='Training repetition count')
parser.add_argument('--lamb', type=float, default=10, help='Additional L1 weight')

opt = parser.parse_args()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    """
    Save model checkpoint including model weights and optimizer state.

    Args:
        model: trained model
        optimizer: optimizer instance
        filename: output file path
    """
    logger.info("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def testOne(model, device):
    """
    Train and evaluate a single model.

    Includes:
    - training loop
    - validation loop
    - metric computation
    - checkpoint saving

    Args:
        model: model name (string)
        device: torch device (CPU/GPU)
    """
    save_model = os.path.join(save_result_folder, mname + "-" + model)

    if not os.path.exists(save_model):
        os.mkdir(save_model)

    if model == 'DualwaveSAM':
        fold = 'haar'   # wavelet options: haar/db2/sym4/bior4.4
    else:
        fold = 0

    save_folder = os.path.join(save_model, 're_' + str(fold))
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    logger.add(f"log/{CHOOSE_DATA}/{mname}-{model}-{fold}.log")

    # Model initialisation
    if model == 'DualwaveSAM':
        from sam_wave import DualwaveSAM
        net_g = DualwaveSAM().to(device)
    else:
        raise ValueError(f"Invalid model: {model}")

    # Loss functions
    criterionL1  = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    # Optimizer
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler = get_scheduler(optimizer_g, opt)

    best_dice  = 0
    best_epoch = 0
    Dice, TDice, Loss = [], [], []

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        # =========================
        # Training phase
        # =========================
        net_g.train()
        train_dice = 0
        train_loss = 0

        if model == 'DualwaveSAM':
            net_g.model.set_epoch(epoch)

        for i in range(opt.trc):
            qbar = trange(len(train_set_loader))

            for iteration, batch in enumerate(train_set_loader, 1):
                optimizer_g.zero_grad()

                input  = batch['input'].to(device)
                target = batch['target'].to(device)

                # Forward pass
                if model == 'DualwaveSAM':
                    prediction, aux_pseudo_logits = net_g(input)
                else:
                    prediction = net_g(input)

                # Loss computation
                loss_g_l1 = criterionL1(prediction,  target) * opt.lamb1
                loss_g_l2 = criterionMSE(prediction, target) * opt.lamb2

                if model == 'DualwaveSAM':
                    if aux_pseudo_logits is not None:
                        loss_g = (
                            0.8 * losses.Dice_and_FocalLoss()(prediction, target) +
                            0.2 * losses.Dice_and_FocalLoss()(aux_pseudo_logits, target) +
                            loss_g_l1 + loss_g_l2
                        )
                    else:
                        loss_g = (
                            losses.Dice_and_FocalLoss()(prediction, target) +
                            loss_g_l1 + loss_g_l2
                        )
                else:
                    loss_g = (
                        losses.Dice_and_FocalLoss()(prediction, target) +
                        loss_g_l1 + loss_g_l2
                    )

                # Backpropagation
                loss_g.backward()
                optimizer_g.step()

                # Metrics
                dice = metrics.dice(
                    ex_transforms.ProbsToLabels()(prediction.detach().cpu().numpy()),
                    target.detach().cpu().numpy()
                )

                train_dice += dice
                train_loss += loss_g.item()

                qbar.set_postfix(epoch=epoch, loss=loss_g.item(), dice=dice.item())
                qbar.update(1)

        # Update learning rate
        update_learning_rate(net_g_scheduler, optimizer_g)

        # =========================
        # Validation phase
        # =========================
        net_g.eval()

        tal_dice = tal_hd95 = tal_voe = tal_rvd = tal_iou = tal_f1 = 0.0

        inputs, targets, preds = [], [], []

        pred_save = os.path.join(save_folder, 'pred')
        if not os.path.exists(pred_save):
            os.mkdir(pred_save)

        pred_save_folder = os.path.join(pred_save, str(epoch))
        if not os.path.exists(pred_save_folder):
            os.mkdir(pred_save_folder)

        with torch.no_grad():
            val_qbar = trange(len(val_set_loader))

            for sample in val_set_loader:
                input  = sample['input'].to(device)
                target = sample['target'].to(device)

                if model == 'DualwaveSAM':
                    prediction, _ = net_g(input)
                else:
                    prediction = net_g(input)

                inputs.append(input.cpu().numpy())
                preds.append(prediction.cpu().numpy())
                targets.append(target.cpu().numpy())

                # Compute metrics
                pred_np   = ex_transforms.ProbsToLabels()(prediction.cpu().numpy())
                target_np = target.cpu().numpy()

                dice_pred = metrics.dice(pred_np, target_np)

                try:
                    hd_val = hd95(pred_np, target_np)
                except Exception:
                    hd_val = np.float64(0)

                tal_dice += dice_pred
                tal_hd95 += hd_val
                tal_iou  += metrics.iou(pred_np, target_np)
                tal_voe  += metrics.voe(pred_np, target_np)
                tal_rvd  += metrics.rvd(pred_np, target_np)

                # Compute lesion-wise F1 score
                tal_f1s = []
                for i in range(pred_np.shape[0]):
                    _, _, f1 = metrics.compute_lesion_metrics(pred_np[i], target_np[i])
                    tal_f1s.append(f1)
                tal_f1 += np.mean(tal_f1s)

                val_qbar.set_postfix(epoch=epoch, dice=dice_pred.item(), hd95=hd_val.item())
                val_qbar.update(1)

        n_val = len(val_set_loader)
        avg_dice = tal_dice / n_val
        avg_hd95 = tal_hd95 / n_val
        avg_voe  = tal_voe  / n_val
        avg_rvd  = tal_rvd  / n_val
        avg_iou  = tal_iou  / n_val
        avg_f1   = tal_f1   / n_val

        Dice.append(avg_dice)
        Loss.append(train_loss / (len(train_set_loader) * opt.trc))
        TDice.append(train_dice / (len(train_set_loader) * opt.trc))

        # Logging
        logger.info('*' * 100)
        logger.info(f"Epoch {epoch} metrics:")
        logger.info("train avg_loss: {:.4f}".format(Loss[-1]))
        logger.info("train avg_dice: {:.4f}".format(TDice[-1]))
        logger.info("test  avg_dice: {:.4f}".format(avg_dice))
        logger.info("test  avg_HD95: {:.4f}".format(avg_hd95))
        logger.info("test  avg_IoU:  {:.4f}".format(avg_iou))
        logger.info("test  avg_VOE:  {:.4f}".format(avg_voe))
        logger.info("test  avg_RVD:  {:.4f}".format(avg_rvd))
        logger.info("test  avg_F1:   {:.4f}".format(avg_f1))

        # Save best model
        if best_dice < avg_dice:
            best_dice  = avg_dice
            best_epoch = epoch

            model_save = os.path.join(save_folder, 'best_model')
            if not os.path.exists(model_save):
                os.mkdir(model_save)

            torch.save(net_g, os.path.join(model_save, 'net_model.pth'))
            save_checkpoint(
                net_g, optimizer_g,
                filename=os.path.join(model_save, "generator.tar")
            )

            pd.DataFrame({'best_dice': [best_dice], 'best_epoch': [best_epoch]}).to_csv(
                os.path.join(save_folder, 'summary.csv'), index=False, sep=';'
            )

            np.savez(
                os.path.join(save_folder, 'res'),
                input=inputs, pred=preds, target=targets
            )

        logger.info(f"best dice: {best_dice:.4f} \t best epoch: {best_epoch}")
        logger.info('*' * 100)

    curr_time_1 = datetime.datetime.now()
    logger.info(f"Total runtime: {curr_time_1 - curr_time}")


class HECKTORdataset(Dataset):
    """
    Dataset for loading HECKTOR data at slice level.

    Features:
    - Loads pre-split train/test .npz files
    - Retains only slices that contain at least one foreground label (slices containing lesions)
    - Class-level cache avoids re-loading across dataset instantiations
    
    Attributes:
        _cache: class-level cache to store processed data for train/test
    """

    _cache: dict = {'train': None, 'test': None}

    def __init__(self, base_dir: str, mode: str = 'train', transform=None):
        """
        Initialize dataset.

        Args:
            base_dir: root path (folder suffix _train / _test is appended)
            mode: 'train' or 'test'
            transform: optional augmentation pipeline
        """
        self.transform = transform
        self.mode = mode

        if HECKTORdataset._cache[mode] is None:
            folder_path = base_dir + '_' + mode
            npz_files   = glob.glob(os.path.join(folder_path, "*.npz"))

            if len(npz_files) == 0:
                raise FileNotFoundError(f"No .npz files found in {folder_path}")

            all_inputs_list  = []
            all_targets_list = []

            for fpath in npz_files:
                data = np.load(fpath, allow_pickle=True)

                patient_input  = data['input']   # (H, W, D, C)
                patient_target = data['target']  # (H, W, D)

                # Remove extra batch dimension if present
                if patient_input.ndim == 5 and patient_input.shape[0] == 1:
                    patient_input  = patient_input[0]
                    patient_target = patient_target[0]

                # Convert to slice-first format
                # → (D, H, W, C) and (D, H, W)
                patient_input  = np.transpose(patient_input,  (2, 0, 1, 3))
                patient_target = np.transpose(patient_target, (2, 0, 1))

                # Identify slices containing lesions
                has_any  = patient_target.sum(axis=(1, 2)) > 0
                has_one  = np.any(patient_target == 1, axis=(1, 2))
                valid    = has_any & has_one

                all_inputs_list.append(patient_input[valid])
                all_targets_list.append(patient_target[valid])

            # Cache processed slices
            HECKTORdataset._cache[mode] = {
                'inputs':  np.concatenate(all_inputs_list,  axis=0),
                'targets': np.concatenate(all_targets_list, axis=0),
            }

        self.inputs  = HECKTORdataset._cache[mode]['inputs']
        self.targets = HECKTORdataset._cache[mode]['targets']

        print(f"Mode: {mode} | Total slices: {len(self.inputs)}")

    def __len__(self):
        """Return number of slices."""
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Retrieve one sample.

        Returns:
            dict with:
            - input: image tensor
            - target: segmentation mask
        """
        sample = {
            'id':     idx,
            'input':  self.inputs[idx],
            'target': self.targets[idx],
        }

        # Resize input and target
        sample['input'] = cv2.resize(
            sample['input'], (256, 256), interpolation=cv2.INTER_LINEAR
        )
        sample['input'] = sample['input'].transpose([1, 0, 2])

        sample['target'] = cv2.resize(
            sample['target'], (256, 256), interpolation=cv2.INTER_NEAREST
        )
        sample['target'] = np.expand_dims(sample['target'], axis=2)
        sample['target'] = sample['target'].transpose([1, 0, 2])

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    curr_time = datetime.datetime.now()

    # GPU assignment
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    CHOOSE_DATA       = "Hecktor"   # CHEN / Hecktor
    save_result_folder = f'results/{CHOOSE_DATA}'
    base_data_dir     = f"/data/code/MMCA-Net/dataset/{CHOOSE_DATA}"
    mname             = 'TBME'

    os.makedirs(save_result_folder, exist_ok=True)
    os.makedirs(f"log/{CHOOSE_DATA}", exist_ok=True)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    mr_transform = ex_transforms.Compose([
        ex_transforms.Horizontal_Mirroring(p=0.5),
        ex_transforms.Vertical_Mirroring(p=0.5),
        ex_transforms.ToTensor(),
    ])

    val_transform = ex_transforms.Compose([
        ex_transforms.ToTensor(),
    ])

    train_data = HECKTORdataset(base_dir=base_data_dir, mode='train', transform=mr_transform)
    val_data   = HECKTORdataset(base_dir=base_data_dir, mode='test',  transform=val_transform)

    print(f"\nTraining slices:   {len(train_data)}")
    print(f"Validation slices: {len(val_data)}\n")

    train_set_loader = DataLoader(
        dataset=train_data,
        num_workers=opt.threads,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_set_loader = DataLoader(
        dataset=val_data,
        num_workers=opt.threads,
        batch_size=opt.test_batch_size,
        shuffle=False,
        drop_last=True,
    )

    logger.info(f"train batches: {len(train_set_loader)}, val batches: {len(val_set_loader)}")
    logger.info('===> Building models')

    for model_name in ['DualwaveSAM']:
        print('\n' + '*' * 30 + f" Training model {model_name} " + '*' * 30)
        testOne(model_name, device)
        print('\n' + '*' * 30 + f" Finished model {model_name} " + '*' * 30)
