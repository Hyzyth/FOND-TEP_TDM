# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0

import argparse
import os
import sys
from pathlib import Path
import warnings
from collections import OrderedDict
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data.distributed
from monai.inferers.utils import sliding_window_inference
from monai.losses.dice import DiceFocalLoss
from monai.losses.tversky import TverskyLoss
from monai.metrics.meandice import DiceMetric
from monai.transforms.compose import Compose
from monai.transforms.post.array import Activations, AsDiscrete
from monai.utils.enums import MetricReduction

# Safeguard
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_utils import get_loader
from networks.SwinTransModels import CONFIGS as CONFIGS_sw_seg
from networks.SwinTransModels import SwinUNETR_CrossModalityFusion_OutSum_6stageOuts
from networks.unetr import UNETR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training

warnings.filterwarnings("ignore")

# ── Custom Loss: Focal Tversky Loss ───────────────────────────────────────
class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        gamma=0.75,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    ):
        super().__init__()

        self.gamma = gamma

        self.tversky = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            alpha=alpha,
            beta=beta,
            reduction="none",   # IMPORTANT
        )

    def forward(self, y_pred, y_true):

        # MONAI TverskyLoss already returns (1 - TI)
        loss = self.tversky(y_pred, y_true)

        # Focal modulation
        loss = torch.clamp(loss, min=1e-7) ** self.gamma

        return loss.mean()

parser = argparse.ArgumentParser(description="SwinCross training pipeline")
parser.add_argument("--checkpoint",        default=None,         help="Resume from checkpoint")
parser.add_argument("--logdir",            default="for_log",    type=str)
parser.add_argument("--pretrained_dir",    default=None,         type=str)
parser.add_argument("--data_dir",          default=".",          type=str)
parser.add_argument("--json_list",         default="dataset_swincross.json", type=str)
parser.add_argument("--pretrained_model_name", default=None,     type=str)
parser.add_argument("--save_checkpoint",   action="store_true")
parser.add_argument("--max_epochs",        default=2000, type=int)
parser.add_argument("--batch_size",        default=2,    type=int)
parser.add_argument("--sw_batch_size",     default=1,    type=int)
parser.add_argument("--optim_lr",          default=1e-4, type=float)
parser.add_argument("--optim_name",        default="adamw", type=str)
parser.add_argument("--reg_weight",        default=1e-5, type=float)
parser.add_argument("--momentum",          default=0.99, type=float)
parser.add_argument("--noamp",             action="store_true",
                    help="Disable AMP for training backward pass. "
                         "Validation inference always uses FP16 regardless.")
parser.add_argument("--val_every",         default=20,   type=int)
parser.add_argument("--distributed",       action="store_true")
parser.add_argument("--world_size",        default=1,    type=int)
parser.add_argument("--rank",              default=0,    type=int)
parser.add_argument("--dist-url",          default="tcp://127.0.0.1:23456", type=str)
parser.add_argument("--dist-backend",      default="nccl", type=str)
parser.add_argument("--workers",           default=4,    type=int)
parser.add_argument("--model_name",        default="unetr", type=str)
parser.add_argument("--pos_embed",         default="perception", type=str)
parser.add_argument("--norm_name",         default="instance", type=str)
parser.add_argument("--num_heads",         default=12,   type=int)
parser.add_argument("--mlp_dim",           default=3072, type=int)
parser.add_argument("--hidden_size",       default=768,  type=int)
parser.add_argument("--feature_size",      default=16,   type=int)
parser.add_argument("--in_channels",       default=2,    type=int)
parser.add_argument("--out_channels",      default=3,    type=int)
parser.add_argument("--res_block",         action="store_true")
parser.add_argument("--conv_block",        action="store_true")
parser.add_argument("--use_normal_dataset",action="store_true")
parser.add_argument("--space_x",           default=1.0,  type=float)
parser.add_argument("--space_y",           default=1.0,  type=float)
parser.add_argument("--space_z",           default=1.0,  type=float)
parser.add_argument("--roi_x",             default=96,   type=int)
parser.add_argument("--roi_y",             default=96,   type=int)
parser.add_argument("--roi_z",             default=96,   type=int)
parser.add_argument("--dropout_rate",      default=0.0,  type=float)
parser.add_argument("--RandFlipd_prob",           default=0.5,  type=float)
parser.add_argument("--RandRotate90d_prob",        default=0.5,  type=float)
parser.add_argument("--RandScaleIntensityd_prob",  default=0.2,  type=float)
parser.add_argument("--RandShiftIntensityd_prob",  default=0.2,  type=float)
parser.add_argument("--infer_overlap",     default=0.5,  type=float,
                    help="Sliding-window overlap for validation. 0.5 recommended.")
parser.add_argument("--lrschedule",        default="warmup_cosine", type=str)
parser.add_argument("--warmup_epochs",     default=50,   type=int)
parser.add_argument("--resume_ckpt",       action="store_true")
parser.add_argument("--smooth_dr",         default=1e-5, type=float)
parser.add_argument("--smooth_nr",         default=1e-5, type=float)
parser.add_argument("--gamma",             default=0.75,  type=float)
parser.add_argument("--alpha",             default=0.3,  type=float) # FP Penalty
parser.add_argument("--beta",              default=0.7,  type=float) # FN Penalty
parser.add_argument("--cache_rate",        default=1.0,  type=float,
                    help="Fraction of training data to cache in RAM. "
                         "0.0 = no cache (always fast with NPZ).")
        
def main():
    args = parser.parse_args()
    args.amp    = not args.noamp
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total GPUs:", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.device)
        torch.backends.cudnn.benchmark = True
        print(f"✅ GPU detected: cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
        print("⚠️  No GPU — running on CPU (AMP disabled)")
        args.distributed = False
        args.amp         = False

    args.test_mode = False
    loader = get_loader(args)

    print(f"\nTrain batches: {len(loader[0])}  |  Val batches: {len(loader[1])}\n")
    print(f"Batch size: {args.batch_size}  |  Max epochs: {args.max_epochs}")
    print(f"GPU: {args.gpu}  |  AMP (training): {args.amp}")

    # ── ROI size for sliding-window inference ─────────────────────────────
    # BUG FIX: was [roi_x, roi_y, roi_x] — last dim is now roi_z.
    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    # ── Model ─────────────────────────────────────────────────────────────
    config_sw = CONFIGS_sw_seg["SwinUNETR_CMFF-hecktor-v06"]
    model     = SwinUNETR_CrossModalityFusion_OutSum_6stageOuts(config_sw)

    if args.resume_ckpt and args.pretrained_dir and args.pretrained_model_name:
        ckpt_path  = os.path.join(args.pretrained_dir, args.pretrained_model_name)
        model_dict = torch.load(ckpt_path, map_location="cpu")
        try:
            model.load_state_dict(model_dict)
        except RuntimeError:
            new_sd = OrderedDict()
            for k, v in model_dict.items():
                new_sd[k.replace("backbone.", "")] = v
            model.load_state_dict(new_sd, strict=False)
        print("Loaded pretrained weights from", ckpt_path)

    # ── Loss ──────────────────────────────────────────────────────────────
    focal_tversky_loss = FocalTverskyLoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        smooth_nr=args.smooth_nr,
        smooth_dr=args.smooth_dr,
    )

    # ── Metrics ───────────────────────────────────────────────────────────
    post_label = AsDiscrete(to_onehot=args.out_channels)
    post_pred  = AsDiscrete(argmax=True, to_onehot=args.out_channels)
    dice_acc   = DiceMetric(
        include_background=False,
        reduction=MetricReduction.MEAN,
        get_not_nans=False,
    )

    def amp_predictor(x):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            return model(x)
    
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=amp_predictor,
        overlap=args.infer_overlap,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    best_acc    = 0.0
    start_epoch = 0
    checkpoint  = None

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        new_sd     = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_sd[k.replace("backbone.", "")] = v
        model.load_state_dict(new_sd, strict=False)
        if "epoch"    in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc    = checkpoint["best_acc"]
        print(f"Resumed from checkpoint: epoch={start_epoch}  best_acc={best_acc:.4f}")

    model.to(args.device)

    if args.distributed and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu,
            find_unused_parameters=True)

    # ── Optimizer ─────────────────────────────────────────────────────────
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum,
            nesterov=True, weight_decay=args.reg_weight)
    else:
        raise ValueError(f"Unknown optimizer: {args.optim_name}")

    if checkpoint is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = None
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.max_epochs,
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs)

    if checkpoint is not None and scheduler is not None:
        print(f"Re-initialising scheduler for max_epochs={args.max_epochs}")
        scheduler.step(start_epoch)
        for pg, lr in zip(optimizer.param_groups, scheduler.get_last_lr()):
            pg["lr"] = lr

    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=focal_tversky_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        start_best_acc=best_acc,
        scaler_state=(checkpoint.get("scaler") if checkpoint else None),
        post_label=post_label,
        post_pred=post_pred,
    )
    return accuracy


if __name__ == "__main__":
    main()
