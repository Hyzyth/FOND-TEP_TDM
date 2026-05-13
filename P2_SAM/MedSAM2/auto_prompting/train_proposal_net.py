"""
auto_prompting/train_proposal_net.py
=====================================
Train the Small3DUNet proposal network on HECKTOR NPZ data.

Checkpoints are saved under /data/ethan/MedSAM2/proposal_net/checkpoints/ by
default, matching the rest of the project's data layout.

Usage
-----
python -m auto_prompting.train_proposal_net \\
    --train_dir /data/ethan/MedSAM2/hecktor_npz/train \\
    --val_dir   /data/ethan/MedSAM2/hecktor_npz/val \\
    --num_epochs 40

The best checkpoint (highest val recall) is saved as
    /data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt
Training metrics are written to
    /data/ethan/MedSAM2/proposal_net/training_log.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Ensure the repo root (containing training/) is importable
_HERE     = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse AverageMeter from the project — avoids duplication
from training.utils.train_utils import AverageMeter

from auto_prompting.proposal_net import Small3DUNet, count_parameters

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HECKTORProposalDataset(Dataset):
    """Loads HECKTOR NPZ files and provides random 3-D crops for training.

    Note: This dataset is intentionally separate from HECKTORNPZRawDataset
    (training/dataset/hecktor_dataset.py).  That class produces VOS-format
    VideoDatapoints for the full SAM2 training pipeline.  This class returns
    simple (input_tensor, target_tensor) pairs for the lightweight U-Net.

    Augmentation operates directly on numpy arrays, unlike the PIL/tensor
    transforms in training/dataset/transforms.py which expect VideoDatapoints.

    Input  : (2, D, H, W) float32 [0, 1]  — channels [CT, PET]
    Target : (1, D, H, W) float32 {0, 1}  — binary tumour mask (gts > 0)
    """

    def __init__(self,
                 npz_dir: str,
                 crop_size: tuple[int, int, int] | None = (64, 128, 128),
                 augment: bool = True) -> None:
        self.files = sorted(glob.glob(
            os.path.join(npz_dir, "**", "*.npz"), recursive=True
        ))
        if not self.files:
            raise FileNotFoundError(f"No NPZ files found under {npz_dir}")
        self.crop_size = crop_size
        self.augment   = augment
        logger.info("Dataset: %d patients in %s", len(self.files), npz_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=True)
        ct  = data["ct_imgs"].astype(np.float32)  / 255.0
        pet = data["pet_imgs"].astype(np.float32) / 255.0
        gts = (data["gts"] > 0).astype(np.float32)

        x = np.stack([ct, pet], axis=0)  # (2, D, H, W)
        y = gts[np.newaxis]              # (1, D, H, W)

        if self.crop_size is not None:
            x, y = self._crop(x, y)
        if self.augment:
            x, y = self._augment(x, y)

        return torch.from_numpy(x), torch.from_numpy(y)

    def _crop(self, x, y):
        cd, ch, cw = self.crop_size
        _, D, H, W = x.shape
        # Pad if necessary
        pd = max(0, cd - D); ph = max(0, ch - H); pw = max(0, cw - W)
        if pd > 0 or ph > 0 or pw > 0:
            x = np.pad(x, [(0,0),(0,pd),(0,ph),(0,pw)])
            y = np.pad(y, [(0,0),(0,pd),(0,ph),(0,pw)])
        _, D, H, W = x.shape

        # Bias crop towards foreground
        if y.sum() > 0:
            zs, ys, xs = np.where(y[0] > 0)
            def jittered_centre(mean_v, size, limit):
                jitter = random.randint(-size // 4, size // 4)
                return int(np.clip(mean_v + jitter, 0, limit))
            cz = jittered_centre(zs.mean(), cd, D)
            cy = jittered_centre(ys.mean(), ch, H)
            cx = jittered_centre(xs.mean(), cw, W)
            z0 = int(np.clip(cz - cd // 2, 0, D - cd))
            y0 = int(np.clip(cy - ch // 2, 0, H - ch))
            x0 = int(np.clip(cx - cw // 2, 0, W - cw))
        else:
            z0 = random.randint(0, max(0, D - cd))
            y0 = random.randint(0, max(0, H - ch))
            x0 = random.randint(0, max(0, W - cw))

        return (x[:, z0:z0+cd, y0:y0+ch, x0:x0+cw],
                y[:, z0:z0+cd, y0:y0+ch, x0:x0+cw])

    def _augment(self, x, y):
        for axis in [1, 2, 3]:
            if random.random() < 0.5:
                x = np.flip(x, axis=axis).copy()
                y = np.flip(y, axis=axis).copy()
        for ch in range(x.shape[0]):
            x[ch] = np.clip(x[ch] + np.random.normal(0, 0.02), 0, 1)
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-sample Tversky loss, averaged over the batch.

    Tversky index = TP / (TP + alpha*FP + beta*FN)

    With beta=0.7 alpha=0.3: FN penalised more than FP (recall-biased).
    Gradient at pred~0.02 for a tumour voxel:
        d(TL)/d(pred_i) ~ -beta / (tp + alpha*fp + beta*fn)
    This is large and negative regardless of how many background voxels exist.
    No saddle point.
    """
    B = pred.shape[0]
    p = pred.view(B, -1)
    t = target.view(B, -1)
    tp = (p * t).sum(dim=1)
    fp = (p * (1.0 - t)).sum(dim=1)
    fn = ((1.0 - p) * t).sum(dim=1)
    tversky_idx = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - tversky_idx).mean()


def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Alpha-balanced focal loss.

    (1 - pt)^gamma downweights easy voxels (background already near 0 after
    prior-bias init), focusing gradient on uncertain boundary voxels.
    alpha=0.75 upweights the tumour class.
    """
    bce = -(
        target         * (pred + eps).log() +
        (1.0 - target) * (1.0 - pred + eps).log()
    )
    pt = torch.where(target == 1, pred, 1.0 - pred)
    at = torch.where(
        target == 1,
        torch.full_like(pred, alpha),
        torch.full_like(pred, 1.0 - alpha),
    )
    return (at * (1.0 - pt).pow(gamma) * bce).mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    focal_weight: float = 0.3,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
) -> torch.Tensor:
    """Focal (focal_weight) + Tversky (1-focal_weight).

    The --focal_weight CLI argument replaces --bce_weight with the same
    default (0.3) and the same role (auxiliary loss share).
    """
    fl = focal_loss(pred, target)
    tl = tversky_loss(pred, target, alpha=tversky_alpha, beta=tversky_beta)
    return focal_weight * fl + (1.0 - focal_weight) * tl


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    threshold: float = 0.25,
) -> dict[str, float]:
    """Recall, precision, Dice, and mean_pred at the given threshold.

    mean_pred diagnostics:
      Collapse : mean_pred > 0.3  (output too high everywhere)
      Healthy  : mean_pred ~ prior_prob (~ 0.02 for HECKTOR after bias init)
    """
    recalls, precisions, dices, mean_preds = [], [], [], []
    model.eval()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prob = model(x)
        mean_preds.append(prob.mean().item())

        pred = (prob > threshold).float()
        tp = (pred * y).sum().item()
        fp = (pred * (1 - y)).sum().item()
        fn = ((1 - pred) * y).sum().item()
        recalls.append(tp / (tp + fn + 1e-6))
        precisions.append(tp / (tp + fp + 1e-6))
        dices.append(2 * tp / (2 * tp + fp + fn + 1e-6))

    return {
        "recall":    float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "dice":      float(np.mean(dices)),
        "mean_pred": float(np.mean(mean_preds)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    log_csv = os.path.join(args.output_dir, "training_log.csv")

    logger.info("=" * 60)
    logger.info("  Small3DUNet Proposal Network — Training")
    logger.info("=" * 60)
    logger.info("  Device         : %s", args.device)
    logger.info("  Train dir      : %s", args.train_dir)
    logger.info("  Val dir        : %s", args.val_dir or args.train_dir)
    logger.info("  Output dir     : %s", args.output_dir)
    logger.info("  Crop size      : %s", args.crop_size or "full volume")
    logger.info("  Epochs         : %d", args.num_epochs)
    logger.info("  Batch size     : %d", args.batch_size)
    logger.info("  LR             : %.2e  (cosine -> %.2e)", args.lr, args.lr / 20)
    logger.info("  Loss           : Focal(%.2f) + Tversky(%.2f)  [a=%.2f b=%.2f]",
                args.focal_weight, 1.0 - args.focal_weight,
                args.tversky_alpha, args.tversky_beta)
    logger.info("  Prior prob     : %.3f  (output bias init)", args.prior_prob)
    logger.info("  Val every      : %d epochs", args.val_every)
    logger.info("  Threshold      : %.2f (metrics)", args.threshold)
    logger.info("  Recall guard   : >= %.2f to save best", args.min_recall_for_save)
    logger.info("=" * 60)

    crop_size = (
        tuple(int(v) for v in args.crop_size.split(","))
        if args.crop_size else None
    )

    # ── Datasets ─────────────────────────────────────────────────────────
    train_ds = HECKTORProposalDataset(args.train_dir, crop_size, augment=True)
    val_dir  = args.val_dir or args.train_dir
    val_ds   = HECKTORProposalDataset(val_dir,  crop_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=args.num_workers)

    # ── Model ─────────────────────────────────────────────────────────────
    device = args.device
    model  = Small3DUNet(in_channels=2,
                         base=args.base_features,
                         dropout=args.dropout,
                         prior_prob=args.prior_prob).to(device)
    
    logger.info("Small3DUNet  params: %d  (%.2fM)",
                count_parameters(model), count_parameters(model) / 1e6)
    logger.info("  output_bias init: %.3f  (mean_pred at epoch-0 ~ %.3f)",
                model.output_bias.item(), args.prior_prob)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr / 20
    )

    best_dice = 0.0
    best_path   = os.path.join(args.output_dir, "proposal_net_best.pt")
    last_path   = os.path.join(args.output_dir, "proposal_net_last.pt")

    # ── CSV header ────────────────────────────────────────────────────────
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss",
            "val_recall", "val_precision", "val_dice", "val_mean_pred",
            "lr", "duration_s", "is_best",
        ])

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        t0   = time.time()
        loss_meter = AverageMeter("Loss", torch.device(device), ":.4f")

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = combined_loss(
                pred, y,
                focal_weight  = args.focal_weight,
                tversky_alpha = args.tversky_alpha,
                tversky_beta  = args.tversky_beta,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_meter.update(loss.item(), x.size(0))

        scheduler.step()
        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]

        # ── Validation ───────────────────────────────────────────────────
        is_val_epoch = (epoch % args.val_every == 0) or (epoch == args.num_epochs)
        metrics  = compute_metrics(model, val_loader, device, args.threshold) \
                   if is_val_epoch else {}
        
        # Save best checkpoint when Dice improves AND recall stays above the
        # minimum guard.  This prevents saving a model that achieves high Dice
        # by being overly conservative (low recall).
        is_best = (
            is_val_epoch
            and metrics.get("dice", 0) > best_dice
            and metrics.get("recall", 0) >= args.min_recall_for_save
        )

        if is_val_epoch:
            above_guard = metrics["recall"] >= args.min_recall_for_save
            # mean_pred diagnostic: values near 0.3–0.5 indicate collapse
            mp = metrics["mean_pred"]
            collapse_flag = " [COLLAPSE? mean_pred too high]" if mp > 0.3 else ""
            logger.info(
                "Ep %3d/%d  loss=%.4f  rec=%.3f%s  prec=%.3f  dice=%.3f"
                "  mean_pred=%.3f%s  lr=%.2e  %.1fs%s",
                epoch, args.num_epochs, loss_meter.avg,
                metrics["recall"],
                "" if above_guard else f" [< guard {args.min_recall_for_save:.2f}]",
                metrics["precision"], metrics["dice"],
                mp, collapse_flag,
                lr_now, elapsed, "  ★ best" if is_best else "",
            )
        else:
            logger.info("Ep %3d/%d  loss=%.4f  lr=%.2e  %.1fs",
                        epoch, args.num_epochs, loss_meter.avg, lr_now, elapsed)

        if is_best:
            best_dice = metrics["dice"]
            _save_ckpt(model, best_path, epoch, metrics)
            logger.info("  Saved best → %s  (dice=%.3f  rec=%.3f  mean_pred=%.3f)",
                        best_path, best_dice,
                        metrics["recall"], metrics["mean_pred"])

        # ── CSV row ───────────────────────────────────────────────────────
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{loss_meter.avg:.6f}",
                f"{metrics.get('recall',    ''):.4f}" if metrics else "",
                f"{metrics.get('precision', ''):.4f}" if metrics else "",
                f"{metrics.get('dice',      ''):.4f}" if metrics else "",
                f"{metrics.get('mean_pred', ''):.4f}" if metrics else "",
                f"{lr_now:.2e}",
                f"{elapsed:.1f}",
                int(is_best),
            ])

    _save_ckpt(model, last_path, args.num_epochs, {})
    logger.info("Training complete.")
    logger.info("  Best dice   : %.3f  (recall guard: ≥ %.2f)", 
                best_dice, args.min_recall_for_save)
    logger.info("  Best ckpt   : %s",   best_path)
    logger.info("  Last ckpt   : %s",   last_path)
    logger.info("  Training log: %s",   log_csv)


def _save_ckpt(model: Small3DUNet, path: str, epoch: int, metrics: dict) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "epoch":       epoch,
        "metrics":     metrics,
        "config": {
            "in_channels": model.in_channels,
            "base":        model.base,
            "dropout":     model.dropout,
            "prior_prob":  model.prior_prob,
        },
    }, path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Small3DUNet tumour proposal network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--train_dir", required=True,
                   help="Directory with training NPZ files.")
    p.add_argument("--val_dir", default=None,
                   help="Directory with val NPZ files (defaults to train_dir).")
    p.add_argument("--output_dir",
                   default="/data/ethan/MedSAM2/proposal_net/checkpoints",
                   help="Where to save checkpoints and training_log.csv.")

    # Architecture
    p.add_argument("--base_features", type=int, default=16)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--prior_prob",    type=float, default=0.02,
                   help="Expected foreground fraction — initialises output bias "
                        "so mean_pred ~ prior_prob at epoch 0, avoiding p=0.5 "
                        "saddle point.")

    # Loss
    p.add_argument("--focal_weight",  type=float, default=0.3,
                   help="Focal loss share; Tversky = 1 - focal_weight.")
    p.add_argument("--tversky_alpha", type=float, default=0.3,
                   help="Tversky FP weight (lower = more recall-biased).")
    p.add_argument("--tversky_beta",  type=float, default=0.7,
                   help="Tversky FN weight (higher = more recall-biased).")

    # Training
    p.add_argument("--num_epochs",    type=int,   default=40)
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--crop_size",     type=str,   default="64,128,128",
                   help="D,H,W crop. Empty string = full volume.")
    p.add_argument("--threshold",     type=float, default=0.25)
    p.add_argument("--val_every",     type=int,   default=5)
    p.add_argument("--min_recall_for_save", type=float, default=0.50,
                   help="Minimum recall to save a 'best' checkpoint. "
                        "Lowered from 0.80 because model starts at recall~0 "
                        "after bias init. Raise once training looks healthy.")

    # Misc
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int,   default=42)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
