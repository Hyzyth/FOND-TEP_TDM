"""
train.py  —  3-class DualwaveSAM training on HECKTOR 2026 NPZ
=============================================================

Supports:
  - Classic training  (single train/val split, fold 0)
  - K-Fold training   (one process per fold, called k times by the shell script)
  - Resume from checkpoint

Usage (classic):
  python {folder}/train.py \\
      --data_dir /data/ethan/PP_hecktor2026_kfold_npz \\
      --json_list dataset_swincross_2026kfold_classic.json \\
      --logdir DualwaveSAM3c_classic \\
      --max_epochs 300

Usage (one fold — called by shell script loop):
  python {folder}/train.py \\
      --data_dir /data/ethan/PP_hecktor2026_kfold_npz \\
      --json_list dataset_swincross_2026kfold_fold2.json \\
      --logdir DualwaveSAM3c_kfold_fold2 \\
      --max_epochs 60
"""

import argparse
import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add DualwaveSAM root (wave_encoder etc.) and this package to path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))   # → DualwaveSAM root (sam_modeling_wave/)
sys.path.insert(0, str(_HERE))          # → this package (dataset, model, losses, trainer, etc.)

from dataset      import HECKTORNPZDataset
from losses       import CombinedLoss
from lr_scheduler import LinearWarmupCosineAnnealingLR
from model        import DualwaveSAM3Class
from trainer      import run_training


# ── CLI ────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="3-class DualwaveSAM training on HECKTOR 2026")

# Data
parser.add_argument("--data_dir",  default="/data/ethan/PP_hecktor2026_kfold_npz", type=str)
parser.add_argument("--json_list", default="dataset_swincross_2026kfold_classic.json", type=str)

# Output
parser.add_argument("--logdir",    default="DualwaveSAM3c_classic", type=str,
                    help="Subdirectory under runs/ where checkpoints/logs are saved.")

# Resume
parser.add_argument("--checkpoint", default=None, type=str,
                    help="Path to checkpoint to resume from (model_last.pth).")

# Model
parser.add_argument("--img_size",   default=256,  type=int)
parser.add_argument("--n_filters",  default=16,   type=int,
                    help="WaveEncoder base filter count (default 16 → 256-ch output).")
parser.add_argument("--wavelet",    default="haar", type=str)
parser.add_argument("--num_classes",default=3,    type=int)
parser.add_argument("--no_aux",     action="store_true",
                    help="Disable auxiliary pseudo-mask head.")

# Training
parser.add_argument("--max_epochs",   default=300,  type=int)
parser.add_argument("--batch_size",   default=16,   type=int)
parser.add_argument("--val_every",    default=10,   type=int)
parser.add_argument("--workers",      default=4,    type=int)
parser.add_argument("--optim_lr",     default=1e-4, type=float)
parser.add_argument("--weight_decay", default=1e-5, type=float)
parser.add_argument("--warmup_epochs",default=20,   type=int)
parser.add_argument("--lrschedule",   default="warmup_cosine", type=str,
                    choices=["warmup_cosine", "cosine", "constant"])
parser.add_argument("--noamp",        action="store_true",
                    help="Disable AMP for training backward pass.")

# Dataset sampling
parser.add_argument("--bg_ratio",   default=0.15,  type=float,
                    help="Fraction of background slices per foreground slice per epoch.")

# Loss
parser.add_argument("--alpha",    default=0.3,  type=float, help="Tversky FP weight")
parser.add_argument("--beta",     default=0.7,  type=float, help="Tversky FN weight")
parser.add_argument("--gamma",    default=0.75, type=float, help="Focal exponent")
parser.add_argument("--weight_primary", default=0.8, type=float)
parser.add_argument("--weight_aux",     default=0.2, type=float)

# GPU
parser.add_argument("--gpu", default=0, type=int)

parser.add_argument("--save_checkpoint", action="store_true",
                    help="Passed implicitly; checkpointing is always on.")


def main():
    args = parser.parse_args()
    args.amp = not args.noamp

    # ── Device ────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.device)
        torch.backends.cudnn.benchmark = True
        print(f"✅ GPU: cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
        args.amp    = False
        print("⚠  No GPU — CPU mode")

    # ── Logdir ────────────────────────────────────────────────────────────
    args.logdir = os.path.join("./runs", args.logdir)
    os.makedirs(args.logdir, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\nBuilding datasets …")
    train_ds = HECKTORNPZDataset(
        json_path=args.json_list,
        split="training",
        size=args.img_size,
        bg_ratio=args.bg_ratio,
        augment=True,
    )
    val_ds = HECKTORNPZDataset(
        json_path=args.json_list,
        split="validation",
        size=args.img_size,
        bg_ratio=0.0,     # use all val slices, no down-sampling
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=(2 if args.workers > 0 else None),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=(args.workers > 0),
        prefetch_factor=(2 if args.workers > 0 else None),
    )

    print(f"\nTrain batches: {len(train_loader)} | Val batches: {len(val_loader)}")
    print(f"Batch size: {args.batch_size} | Max epochs: {args.max_epochs}")

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nBuilding model …")
    model = DualwaveSAM3Class(
        img_size=args.img_size,
        n_filters=args.n_filters,
        wavelet=args.wavelet,
        num_classes=args.num_classes,
        use_aux_head=not args.no_aux,
    ).to(args.device)

    # ── Loss ──────────────────────────────────────────────────────────────
    loss_func = CombinedLoss(
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma,
        num_classes=args.num_classes,
        weight_primary=args.weight_primary,
        weight_aux=args.weight_aux,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.optim_lr, weight_decay=args.weight_decay)

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = None
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.max_epochs,
        )
    elif args.lrschedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs
        )

    # ── Resume ────────────────────────────────────────────────────────────
    start_epoch    = 0
    start_best_acc = 0.0
    scaler_state   = None
    checkpoint     = None

    if args.checkpoint and os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        new_sd = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_sd[k.replace("module.", "")] = v
        model.load_state_dict(new_sd, strict=False)
        start_epoch    = checkpoint.get("epoch", 0)
        start_best_acc = checkpoint.get("best_acc", 0.0)
        scaler_state   = checkpoint.get("scaler", None)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler and "scheduler" in checkpoint and checkpoint["scheduler"]:
            scheduler.load_state_dict(checkpoint["scheduler"])
        print(f"Resumed from {args.checkpoint} | epoch={start_epoch} | best={start_best_acc:.4f}")

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\nLogging to: {args.logdir}\n")
    args.rank = 0   # single-GPU, no DDP

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
        start_best_acc=start_best_acc,
        scaler_state=scaler_state,
    )


if __name__ == "__main__":
    main()
