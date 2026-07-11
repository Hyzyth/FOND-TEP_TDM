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
from tqdm import trange
from loguru import logger

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

# CHANGE: removed unused torchvision.transforms import
from medpy.metric.binary import hd95

# =========================
# Local module imports
# =========================
import P2_SAM.DualwaveSAM.obsolete.ex_transforms as ex_transforms
import P2_SAM.DualwaveSAM.obsolete.losses as losses
import P2_SAM.DualwaveSAM.obsolete.metrics as metrics

warnings.filterwarnings("ignore")


def get_scheduler(optimizer, opt):
    """
    Create a learning rate scheduler based on the selected policy.

    Supported: lambda | step | plateau | cosine
    """
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            return 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
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
        raise NotImplementedError(f"LR policy '{opt.lr_policy}' not implemented")

    return scheduler


def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    print('learning rate = %.7f' % optimizer.param_groups[0]['lr'])


# =========================
# Training configuration
# =========================
parser = argparse.ArgumentParser(description='DualwaveSAM training')

parser.add_argument('--batch_size',      type=int,   default=12)
parser.add_argument('--test_batch_size', type=int,   default=12)
parser.add_argument('--input_nc',        type=int,   default=2)
parser.add_argument('--output_nc',       type=int,   default=1)
parser.add_argument('--epoch_count',     type=int,   default=1)
parser.add_argument('--niter',           type=int,   default=10)
parser.add_argument('--niter_decay',     type=int,   default=40)
parser.add_argument('--lr',              type=float, default=0.0001)
parser.add_argument('--lr_policy',       type=str,   default='lambda',
                    help='lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters',  type=int,   default=1)
parser.add_argument('--beta1',           type=float, default=0.5)
parser.add_argument('--cuda',            action='store_true')
parser.add_argument('--threads',         type=int,   default=0)
parser.add_argument('--seed',            type=int,   default=42)
parser.add_argument('--lamb1',           type=float, default=0.01)
parser.add_argument('--lamb2',           type=float, default=0.1)
parser.add_argument('--glr',             type=float, default=0.000128)
parser.add_argument('--trc',             type=int,   default=1)
parser.add_argument('--lamb',            type=float, default=10)

# CHANGE: added --data_dir, --dataset, --save_dir so paths are not hardcoded
parser.add_argument('--data_dir',  type=str, default='/data/code/MMCA-Net/dataset/Hecktor',
                    help='Base path; _train / _test suffixes are appended automatically')
parser.add_argument('--dataset',   type=str, default='Hecktor', choices=['CHEN', 'Hecktor'])
parser.add_argument('--save_dir',  type=str, default='results',
                    help='Root folder for saving checkpoints and logs')

opt = parser.parse_args()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    logger.info("=> Saving checkpoint")
    torch.save({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, filename)


def testOne(model_name, device):
    """Train and evaluate a single model configuration."""

    save_model = os.path.join(save_result_folder, mname + "-" + model_name)
    os.makedirs(save_model, exist_ok=True)

    fold = 'haar' if model_name == 'DualwaveSAM' else 0

    save_folder = os.path.join(save_model, 're_' + str(fold))
    os.makedirs(save_folder, exist_ok=True)

    logger.add(f"log/{opt.dataset}/{mname}-{model_name}-{fold}.log")

    # Model
    if model_name == 'DualwaveSAM':
        from P2_SAM.DualwaveSAM.sam_wave import DualwaveSAM
        net_g = DualwaveSAM().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Loss functions
    criterionL1  = nn.L1Loss().to(device)
    criterionMSE = nn.MSELoss().to(device)

    # CHANGE: only pass parameters that require gradients to the optimizer.
    # Previously all parameters (including frozen prompt_encoder and
    # mask_decoder) were passed, wasting optimizer state memory.
    trainable_params = [p for p in net_g.parameters() if p.requires_grad]
    logger.info(f"Trainable parameter tensors: {len(trainable_params)}")
    optimizer_g      = optim.Adam(trainable_params, lr=opt.lr, betas=(opt.beta1, 0.999))
    net_g_scheduler  = get_scheduler(optimizer_g, opt)

    best_dice  = 0
    best_epoch = 0
    Dice, TDice, Loss = [], [], []

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        # =========================
        # Training
        # =========================
        net_g.train()
        train_dice = 0.0
        train_loss = 0.0

        if model_name == 'DualwaveSAM':
            net_g.model.set_epoch(epoch)

        for _ in range(opt.trc):
            qbar = trange(len(train_set_loader))

            for iteration, batch in enumerate(train_set_loader, 1):
                optimizer_g.zero_grad()

                inp    = batch['input'].to(device)
                target = batch['target'].to(device)

                prediction, aux_pseudo_logits = net_g(inp)

                loss_g_l1 = criterionL1(prediction,  target) * opt.lamb1
                loss_g_l2 = criterionMSE(prediction, target) * opt.lamb2

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

                loss_g.backward()
                optimizer_g.step()

                dice = metrics.dice(
                    ex_transforms.ProbsToLabels()(prediction.detach().cpu().numpy()),
                    target.detach().cpu().numpy()
                )

                train_dice += dice
                train_loss += loss_g.item()

                qbar.set_postfix(epoch=epoch, loss=loss_g.item(), dice=float(dice))
                qbar.update(1)

        update_learning_rate(net_g_scheduler, optimizer_g)

        # =========================
        # Validation
        # =========================
        net_g.eval()

        tal_dice = tal_hd95 = tal_voe = tal_rvd = tal_iou = tal_f1 = 0.0
        inputs, targets, preds = [], [], []

        pred_save_folder = os.path.join(save_folder, 'pred', str(epoch))
        os.makedirs(pred_save_folder, exist_ok=True)

        with torch.no_grad():
            val_qbar = trange(len(val_set_loader))

            for sample in val_set_loader:
                inp    = sample['input'].to(device)
                target = sample['target'].to(device)

                prediction, _ = net_g(inp)

                inputs.append(inp.cpu().numpy())
                preds.append(prediction.cpu().numpy())
                targets.append(target.cpu().numpy())

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

                tal_f1s = [
                    metrics.compute_lesion_metrics(pred_np[i], target_np[i])[2]
                    for i in range(pred_np.shape[0])
                ]
                tal_f1 += np.mean(tal_f1s)

                val_qbar.set_postfix(epoch=epoch, dice=float(dice_pred), hd95=float(hd_val))
                val_qbar.update(1)

        n_val    = len(val_set_loader)
        avg_dice = tal_dice / n_val
        avg_hd95 = tal_hd95 / n_val
        avg_voe  = tal_voe  / n_val
        avg_rvd  = tal_rvd  / n_val
        avg_iou  = tal_iou  / n_val
        avg_f1   = tal_f1   / n_val

        n_train  = len(train_set_loader) * opt.trc
        Dice.append(avg_dice)
        Loss.append(train_loss / n_train)
        TDice.append(train_dice / n_train)

        logger.info('*' * 100)
        logger.info(f"Epoch {epoch}")
        logger.info("train avg_loss: {:.4f}".format(Loss[-1]))
        logger.info("train avg_dice: {:.4f}".format(TDice[-1]))
        logger.info("test  avg_dice: {:.4f}".format(avg_dice))
        logger.info("test  avg_HD95: {:.4f}".format(avg_hd95))
        logger.info("test  avg_IoU:  {:.4f}".format(avg_iou))
        logger.info("test  avg_VOE:  {:.4f}".format(avg_voe))
        logger.info("test  avg_RVD:  {:.4f}".format(avg_rvd))
        logger.info("test  avg_F1:   {:.4f}".format(avg_f1))

        if best_dice < avg_dice:
            best_dice  = avg_dice
            best_epoch = epoch

            model_save = os.path.join(save_folder, 'best_model')
            os.makedirs(model_save, exist_ok=True)

            torch.save(net_g, os.path.join(model_save, 'net_model.pth'))
            save_checkpoint(net_g, optimizer_g,
                            filename=os.path.join(model_save, "generator.tar"))

            pd.DataFrame({'best_dice': [best_dice], 'best_epoch': [best_epoch]}).to_csv(
                os.path.join(save_folder, 'summary.csv'), index=False, sep=';'
            )
            np.savez(os.path.join(save_folder, 'res'),
                     input=inputs, pred=preds, target=targets)

        logger.info(f"best dice: {best_dice:.4f}  best epoch: {best_epoch}")
        logger.info('*' * 100)

    elapsed = datetime.datetime.now() - curr_time
    logger.info(f"Total runtime: {elapsed}")


# ============================================================
# Dataset
# ============================================================
class HECKTORdataset(Dataset):
    """
    Slice-level dataset.  Loads all .npz files from base_dir_train / _test,
    keeps only lesion-containing slices, and caches results at class level.
    """

    _cache: dict = {'train': None, 'test': None}

    def __init__(self, base_dir: str, mode: str = 'train', transform=None):
        self.transform = transform
        self.mode      = mode

        if HECKTORdataset._cache[mode] is None:
            folder_path = base_dir + '_' + mode
            npz_files   = glob.glob(os.path.join(folder_path, "*.npz"))

            if not npz_files:
                raise FileNotFoundError(f"No .npz files in {folder_path}")

            all_inputs_list, all_targets_list = [], []

            for fpath in npz_files:
                data           = np.load(fpath, allow_pickle=True)
                patient_input  = data['input']   # (H, W, D, C)
                patient_target = data['target']  # (H, W, D)

                if patient_input.ndim == 5 and patient_input.shape[0] == 1:
                    patient_input  = patient_input[0]
                    patient_target = patient_target[0]

                patient_input  = np.transpose(patient_input,  (2, 0, 1, 3))
                patient_target = np.transpose(patient_target, (2, 0, 1))

                valid = (patient_target.sum(axis=(1, 2)) > 0) & \
                        np.any(patient_target == 1, axis=(1, 2))

                all_inputs_list.append(patient_input[valid])
                all_targets_list.append(patient_target[valid])

            HECKTORdataset._cache[mode] = {
                'inputs':  np.concatenate(all_inputs_list,  axis=0),
                'targets': np.concatenate(all_targets_list, axis=0),
            }

        self.inputs  = HECKTORdataset._cache[mode]['inputs']
        self.targets = HECKTORdataset._cache[mode]['targets']
        print(f"Mode: {mode} | slices: {len(self.inputs)}")

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
        ).transpose([1, 0, 2])

        sample['target'] = cv2.resize(
            sample['target'], (256, 256), interpolation=cv2.INTER_NEAREST
        )
        sample['target'] = np.expand_dims(sample['target'], axis=2).transpose([1, 0, 2])

        if self.transform:
            sample = self.transform(sample)
        return sample


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    curr_time = datetime.datetime.now()

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # CHANGE: use CLI args instead of hardcoded strings
    save_result_folder = os.path.join(opt.save_dir, opt.dataset)
    base_data_dir      = opt.data_dir
    mname              = 'TBME'

    os.makedirs(save_result_folder, exist_ok=True)
    os.makedirs(f"log/{opt.dataset}", exist_ok=True)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, run without --cuda")

    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    mr_transform = ex_transforms.Compose([
        ex_transforms.Horizontal_Mirroring(p=0.5),
        ex_transforms.Vertical_Mirroring(p=0.5),
        ex_transforms.ToTensor(),
    ])
    val_transform = ex_transforms.Compose([ex_transforms.ToTensor()])

    train_data = HECKTORdataset(base_dir=base_data_dir, mode='train', transform=mr_transform)
    val_data   = HECKTORdataset(base_dir=base_data_dir, mode='test',  transform=val_transform)

    print(f"Training slices:   {len(train_data)}")
    print(f"Validation slices: {len(val_data)}\n")

    train_set_loader = DataLoader(train_data, num_workers=opt.threads,
                                  batch_size=opt.batch_size,      shuffle=True,  drop_last=True)
    val_set_loader   = DataLoader(val_data,   num_workers=opt.threads,
                                  batch_size=opt.test_batch_size, shuffle=False, drop_last=True)

    logger.info(f"train batches: {len(train_set_loader)}, val batches: {len(val_set_loader)}")
    logger.info('===> Building models')

    for model_name in ['DualwaveSAM']:
        print('\n' + '*' * 30 + f" Training {model_name} " + '*' * 30)
        testOne(model_name, device)
        print('\n' + '*' * 30 + f" Finished {model_name} " + '*' * 30)
