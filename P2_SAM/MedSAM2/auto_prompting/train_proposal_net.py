"""
auto_prompting/train_proposal_net.py
=====================================
Train the Small3DUNet proposal network on HECKTOR NPZ data.

Objective: high-recall coarse tumour mask.  The network is intentionally small
to avoid overfitting on the limited HECKTOR dataset (~500 patients).

Usage
-----
python -m auto_prompting.train_proposal_net \\
    --train_dir /data/ethan/MedSAM2/hecktor_npz/train \\
    --val_dir   /data/ethan/MedSAM2/hecktor_npz/val \\
    --output_dir ./auto_prompting/checkpoints \\
    --num_epochs 40 \\
    --lr 1e-3 \\
    --crop_size 64,128,128

The best checkpoint (highest val recall) is saved as ``proposal_net_best.pt``
alongside a ``proposal_net_last.pt`` at the end of training.
"""

from __future__ import annotations

import argparse
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

# Make auto_prompting importable when run as __main__
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from auto_prompting.proposal_net import Small3DUNet, count_parameters

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class HECKTORProposalDataset(Dataset):
    """Loads HECKTOR NPZ files and provides random 3-D crops for training.

    Input tensor  : (2, D, H, W) float [0, 1] – channels [CT, PET]
    Target tensor : (1, D, H, W) float [0, 1] – binary tumour mask (gts > 0)
    """

    def __init__(self,
                 npz_dir: str,
                 crop_size: tuple[int, int, int] | None = (64, 128, 128),
                 augment: bool = True,
                 require_foreground: bool = True) -> None:
        self.files = sorted(glob.glob(os.path.join(npz_dir, "**", "*.npz"),
                                      recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No NPZ files found under {npz_dir}")
        self.crop_size = crop_size
        self.augment = augment
        self.require_foreground = require_foreground
        logger.info("Dataset: %d patients in %s", len(self.files), npz_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=True)
        ct  = data["ct_imgs"].astype(np.float32) / 255.0   # (D, H, W)
        pet = data["pet_imgs"].astype(np.float32) / 255.0  # (D, H, W)
        gts = (data["gts"] > 0).astype(np.float32)         # (D, H, W) binary

        x = np.stack([ct, pet], axis=0)   # (2, D, H, W)
        y = gts[np.newaxis]               # (1, D, H, W)

        if self.crop_size is not None:
            x, y = self._crop(x, y)

        if self.augment:
            x, y = self._augment(x, y)

        return torch.from_numpy(x), torch.from_numpy(y)

    # ── Cropping ─────────────────────────────────────────────────────────────

    def _crop(self, x: np.ndarray, y: np.ndarray):
        cd, ch, cw = self.crop_size
        _, D, H, W = x.shape

        # Pad if volume is smaller than crop in any dimension
        pad = [(0, 0),
               (0, max(0, cd - D)),
               (0, max(0, ch - H)),
               (0, max(0, cw - W))]
        if any(p[1] > 0 for p in pad):
            x = np.pad(x, pad, mode="constant")
            y = np.pad(y, pad, mode="constant")
        _, D, H, W = x.shape

        if self.require_foreground and y.sum() > 0:
            # Centre crop on the foreground ± random jitter
            zs, ys, xs = np.where(y[0] > 0)
            cz = int(np.clip(zs.mean() + np.random.randint(-cd // 4, cd // 4 + 1), 0, D))
            cy = int(np.clip(ys.mean() + np.random.randint(-ch // 4, ch // 4 + 1), 0, H))
            cx = int(np.clip(xs.mean() + np.random.randint(-cw // 4, cw // 4 + 1), 0, W))
            z0 = int(np.clip(cz - cd // 2, 0, D - cd))
            y0 = int(np.clip(cy - ch // 2, 0, H - ch))
            x0 = int(np.clip(cx - cw // 2, 0, W - cw))
        else:
            z0 = random.randint(0, max(0, D - cd))
            y0 = random.randint(0, max(0, H - ch))
            x0 = random.randint(0, max(0, W - cw))

        return (x[:, z0:z0+cd, y0:y0+ch, x0:x0+cw],
                y[:, z0:z0+cd, y0:y0+ch, x0:x0+cw])

    # ── Augmentation ─────────────────────────────────────────────────────────

    def _augment(self, x: np.ndarray, y: np.ndarray):
        # Random flips along each axis
        for axis in [1, 2, 3]:
            if random.random() < 0.5:
                x = np.flip(x, axis=axis).copy()
                y = np.flip(y, axis=axis).copy()
        # Mild intensity jitter on CT and PET independently
        for ch in range(x.shape[0]):
            x[ch] = np.clip(x[ch] + np.random.normal(0, 0.02), 0, 1)
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# Loss functions
# ──────────────────────────────────────────────────────────────────────────────

def recall_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 – Recall = FN / (TP + FN).  Encourages high sensitivity."""
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return 1.0 - tp / (tp + fn + eps)


def combined_loss(pred: torch.Tensor,
                  target: torch.Tensor,
                  bce_weight: float = 0.3,
                  recall_weight: float = 0.7) -> torch.Tensor:
    """Recall-biased loss: recall_weight * recall + bce_weight * BCE."""
    bce = F.binary_cross_entropy(pred, target)
    rec = recall_loss(pred, target)
    return bce_weight * bce + recall_weight * rec


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(model: nn.Module,
                    loader: DataLoader,
                    device: str,
                    threshold: float = 0.25) -> dict[str, float]:
    """Compute mean recall, precision, and Dice on a validation set."""
    recalls, precisions, dices = [], [], []
    model.eval()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        prob = model(x)
        pred = (prob > threshold).float()
        target = y

        tp = (pred * target).sum().item()
        fp = (pred * (1 - target)).sum().item()
        fn = ((1 - pred) * target).sum().item()

        rec  = tp / (tp + fn + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-6)

        recalls.append(rec)
        precisions.append(prec)
        dices.append(dice)

    return {
        "recall":    float(np.mean(recalls)),
        "precision": float(np.mean(precisions)),
        "dice":      float(np.mean(dices)),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Soft targets (optional distance-transform weighting)
# ──────────────────────────────────────────────────────────────────────────────

def make_soft_target(gt_np: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gaussian-blurred distance target to handle class imbalance."""
    from scipy.ndimage import distance_transform_edt
    tumor = (gt_np > 0).astype(np.float32)
    dist  = distance_transform_edt(1 - tumor)
    return np.exp(-(dist ** 2) / (2 * sigma ** 2))


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    crop_size = tuple(int(v) for v in args.crop_size.split(",")) if args.crop_size else None

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = HECKTORProposalDataset(
        args.train_dir, crop_size=crop_size, augment=True
    )
    val_dir  = args.val_dir or args.train_dir   # fallback: re-use train for quick sanity
    val_ds   = HECKTORProposalDataset(
        val_dir, crop_size=crop_size, augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    device = args.device
    model = Small3DUNet(
        in_channels=2,
        base=args.base_features,
        dropout=args.dropout,
    ).to(device)
    logger.info("Model parameters: %d (%.2f M)",
                count_parameters(model), count_parameters(model) / 1e6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr / 20
    )

    best_recall = 0.0
    best_path   = os.path.join(args.output_dir, "proposal_net_best.pt")
    last_path   = os.path.join(args.output_dir, "proposal_net_last.pt")

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        t0 = time.time()
        losses = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = combined_loss(pred, y,
                                 bce_weight=args.bce_weight,
                                 recall_weight=1.0 - args.bce_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        avg_loss = float(np.mean(losses))
        elapsed  = time.time() - t0

        # ── Validation ───────────────────────────────────────────────────────
        if epoch % args.val_every == 0 or epoch == args.num_epochs:
            metrics = compute_metrics(model, val_loader, device,
                                      threshold=args.threshold)
            logger.info(
                "Epoch %3d/%d  loss=%.4f  rec=%.3f  prec=%.3f  dice=%.3f  "
                "lr=%.2e  [%.1fs]",
                epoch, args.num_epochs, avg_loss,
                metrics["recall"], metrics["precision"], metrics["dice"],
                scheduler.get_last_lr()[0], elapsed,
            )

            if metrics["recall"] > best_recall:
                best_recall = metrics["recall"]
                _save(model, best_path, epoch, metrics)
                logger.info("  → New best recall %.3f — saved to %s",
                            best_recall, best_path)
        else:
            logger.info("Epoch %3d/%d  loss=%.4f  [%.1fs]",
                        epoch, args.num_epochs, avg_loss, elapsed)

    _save(model, last_path, args.num_epochs, {})
    logger.info("Training complete.  Best recall: %.3f", best_recall)
    logger.info("Best checkpoint : %s", best_path)
    logger.info("Last checkpoint : %s", last_path)


def _save(model: Small3DUNet, path: str, epoch: int, metrics: dict) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": {
                "in_channels": model._in_channels,
                "base": model._base,
                "dropout": model._dropout,
            },
        },
        path,
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Small3DUNet proposal network on HECKTOR NPZ data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--train_dir", required=True,
                   help="Directory with training NPZ files.")
    p.add_argument("--val_dir", default=None,
                   help="Directory with validation NPZ files. "
                        "Defaults to --train_dir if not set.")
    p.add_argument("--output_dir", default="./auto_prompting/checkpoints",
                   help="Where to save checkpoints.")

    # Architecture
    p.add_argument("--base_features", type=int, default=16,
                   help="Base feature-map count for the U-Net.")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Dropout3d probability.")

    # Training
    p.add_argument("--num_epochs", type=int, default=40,
                   help="Number of training epochs.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Batch size (full volumes; reduce if OOM).")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Initial learning rate (cosine decay).")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--bce_weight", type=float, default=0.3,
                   help="Weight on BCE loss; recall weight = 1 - bce_weight.")
    p.add_argument("--crop_size", type=str, default="64,128,128",
                   help="D,H,W random crop size. Set to '' for full-volume.")
    p.add_argument("--threshold", type=float, default=0.25,
                   help="Probability threshold for metric computation.")

    # Misc
    p.add_argument("--val_every", type=int, default=5,
                   help="Run validation every N epochs.")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
