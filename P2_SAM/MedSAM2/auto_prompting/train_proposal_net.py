"""
auto_prompting/train_proposal_net.py
=====================================
Train the Small3DUNet proposal network on HECKTOR 2026 data.

UPDATED to:
  1. Read SwinCross-format NPZ files (``ct`` int16, ``pet`` float16,
     ``label`` uint8, RAS order) instead of the old MedSAM2 NPZ format
     (``ct_imgs`` / ``pet_imgs`` / ``gts`` uint8).
  2. Accept --data_dir + --json_list (SwinCross classic JSON) for the
     train/val split instead of --train_dir / --val_dir plain directories.
     The classic JSON's ``training`` key drives training, ``validation`` key
     drives evaluation - matching the SwinCross and DualwaveSAM convention.
  3. Apply the same CT/PET normalisation as HECKTORNPZRawDataset:
       CT  -> soft-tissue window [-160, +240 HU] -> [0, 1]
       PET -> 99th-percentile clip -> [0, 1]

Usage
-----
python -m auto_prompting.train_proposal_net \\
    --data_dir /data/ethan/PP_hecktor2026_kfold_npz \\
    --json_list dataset_swincross_2026kfold_classic.json \\
    --num_epochs 100

The best checkpoint (highest val recall above --min_recall_for_save) is saved as
    /data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
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

_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from training.utils.train_utils import AverageMeter
from auto_prompting.proposal_net import Small3DUNet, count_parameters

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── Intensity normalisation (mirrors HECKTORNPZRawDataset) ───────────────────

_CT_LO, _CT_HI = -160.0, 240.0


def _normalise_ct(ct_arr: np.ndarray) -> np.ndarray:
    ct = np.clip(ct_arr.astype(np.float32), _CT_LO, _CT_HI)
    return (ct - _CT_LO) / (_CT_HI - _CT_LO)


def _normalise_pet(pet_arr: np.ndarray) -> np.ndarray:
    pet = pet_arr.astype(np.float32)
    p99 = float(np.percentile(pet[pet > 0], 99)) if (pet > 0).any() else 1.0
    p99 = max(p99, 1e-6)
    return np.clip(pet / p99, 0.0, 1.0)


# ── Dataset ───────────────────────────────────────────────────────────────────

class HECKTORProposalDataset(Dataset):
    """Dataset for Small3DUNet proposal network - reads SwinCross NPZ format.

    Each NPZ provides:
      ct    (R, A, S) int16   -> normalised float32 [0, 1]
      pet   (R, A, S) float16 -> normalised float32 [0, 1]
      label (R, A, S) uint8   -> binary mask (label > 0)

    Input  : (2, D, H, W) float32 in [0, 1]  - [CT, PET]
    Target : (1, D, H, W) float32 {0, 1}     - binary foreground mask
    """

    def __init__(
        self,
        data_dir: str,
        json_list: str,
        split: str = "training",
        crop_size: tuple | None = (64, 128, 128),
        augment: bool = True,
    ) -> None:
        self.data_dir  = data_dir
        self.crop_size = crop_size
        self.augment   = augment

        json_path = os.path.join(data_dir, json_list) \
                    if not os.path.isabs(json_list) else json_list
        with open(json_path) as f:
            js = json.load(f)

        entries = js.get(split, [])
        if not entries:
            for key in ("training", "validation"):
                entries = js.get(key, [])
                if entries:
                    break

        self.npz_paths = []
        for e in entries:
            npz_rel = e.get("npz", "")
            if npz_rel:
                abs_path = os.path.join(data_dir, npz_rel)
                if os.path.exists(abs_path):
                    self.npz_paths.append(abs_path)

        if not self.npz_paths:
            raise FileNotFoundError(
                f"No NPZ files found for split='{split}' in {json_path}"
            )
        logger.info("[%s] HECKTORProposalDataset: %d patients", split, len(self.npz_paths))

    def __len__(self) -> int:
        return len(self.npz_paths)

    def __getitem__(self, idx: int):
        with np.load(self.npz_paths[idx], allow_pickle=False) as npz:
            ct  = _normalise_ct(npz["ct"])    # (R, A, S) float32
            pet = _normalise_pet(npz["pet"])   # (R, A, S) float32
            lbl = (npz["label"] > 0).astype(np.float32)  # (R, A, S) binary

        # Stack channels: (2, R, A, S) - treat RAS volume as (C, D, H, W)
        x = np.stack([ct, pet], axis=0)
        y = lbl[np.newaxis]               # (1, R, A, S)

        if self.crop_size is not None:
            x, y = self._crop(x, y)
        if self.augment:
            x, y = self._augment(x, y)

        return torch.from_numpy(x), torch.from_numpy(y)

    # ── Crop ──────────────────────────────────────────────────────────────────

    def _crop(self, x: np.ndarray, y: np.ndarray):
        cd, ch, cw = self.crop_size
        _, D, H, W = x.shape

        # Pad if volume is smaller than crop
        pd = max(0, cd - D); ph = max(0, ch - H); pw = max(0, cw - W)
        if pd > 0 or ph > 0 or pw > 0:
            x = np.pad(x, [(0, 0), (0, pd), (0, ph), (0, pw)])
            y = np.pad(y, [(0, 0), (0, pd), (0, ph), (0, pw)])
        _, D, H, W = x.shape

        # Bias crop towards foreground voxels
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

    # ── Augmentation ──────────────────────────────────────────────────────────

    @staticmethod
    def _augment(x: np.ndarray, y: np.ndarray):
        for axis in [1, 2, 3]:
            if random.random() < 0.5:
                x = np.flip(x, axis=axis).copy()
                y = np.flip(y, axis=axis).copy()
        # Mild intensity jitter (image only)
        for ch in range(x.shape[0]):
            x[ch] = np.clip(x[ch] + np.random.normal(0, 0.02), 0, 1)
        return x, y


# ── Loss functions ────────────────────────────────────────────────────────────

def tversky_loss(pred, target, alpha=0.3, beta=0.7, eps=1e-6):
    B = pred.shape[0]
    p = pred.view(B, -1)
    t = target.view(B, -1)
    tp = (p * t).sum(dim=1)
    fp = (p * (1.0 - t)).sum(dim=1)
    fn = ((1.0 - p) * t).sum(dim=1)
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - ti).mean()


def focal_loss(pred, target, gamma=2.0, alpha=0.75, eps=1e-7):
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


def combined_loss(pred, target, focal_weight=0.3, tversky_alpha=0.3, tversky_beta=0.7):
    fl = focal_loss(pred, target)
    tl = tversky_loss(pred, target, alpha=tversky_alpha, beta=tversky_beta)
    return focal_weight * fl + (1.0 - focal_weight) * tl


# ── Metrics ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(model, loader, device, threshold=0.25):
    recalls, precisions, dices, mean_preds = [], [], [], []
    model.eval()
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Pad to multiple of 16 (same fix as auto_prompter.py)
        _, _, D, H, W = x.shape
        pad_d = (16 - D % 16) % 16
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        xp = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        prob = model(xp)[0, 0, :D, :H, :W].unsqueeze(0).unsqueeze(0)

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


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    log_csv = os.path.join(args.output_dir, "training_log.csv")

    logger.info("=" * 60)
    logger.info("  Small3DUNet Proposal Network - Training")
    logger.info("=" * 60)
    logger.info("  Device        : %s", args.device)
    logger.info("  Data dir      : %s", args.data_dir)
    logger.info("  JSON list     : %s", args.json_list)
    logger.info("  Output dir    : %s", args.output_dir)
    logger.info("  Crop size     : %s", args.crop_size or "full volume")
    logger.info("  Epochs        : %d", args.num_epochs)
    logger.info("  Batch size    : %d", args.batch_size)
    logger.info("  LR            : %.2e  (cosine -> %.2e)", args.lr, args.lr / 20)
    logger.info("  Loss          : Focal(%.2f) + Tversky(%.2f)  [a=%.2f b=%.2f]",
                args.focal_weight, 1.0 - args.focal_weight,
                args.tversky_alpha, args.tversky_beta)
    logger.info("  Prior prob    : %.3f  (output bias init)", args.prior_prob)
    logger.info("  Recall guard  : >= %.2f to save best", args.min_recall_for_save)
    logger.info("=" * 60)

    crop_size = (
        tuple(int(v) for v in args.crop_size.split(","))
        if args.crop_size else None
    )

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = HECKTORProposalDataset(
        args.data_dir, args.json_list, split="training",
        crop_size=crop_size, augment=True,
    )
    val_ds = HECKTORProposalDataset(
        args.data_dir, args.json_list, split="validation",
        crop_size=crop_size, augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    device = args.device
    model  = Small3DUNet(
        in_channels=2,
        base=args.base_features,
        dropout=args.dropout,
        prior_prob=args.prior_prob,
    ).to(device)

    logger.info("Small3DUNet params: %d  (%.2fM)",
                count_parameters(model), count_parameters(model) / 1e6)
    logger.info("  output_bias init: %.3f  (mean_pred at epoch-0 ~ %.3f)",
                model.output_bias.item(), args.prior_prob)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=args.lr / 20)

    best_dice = 0.0
    best_path = os.path.join(args.output_dir, "proposal_net_best.pt")
    last_path = os.path.join(args.output_dir, "proposal_net_last.pt")

    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss",
            "val_recall", "val_precision", "val_dice", "val_mean_pred",
            "lr", "duration_s", "is_best",
        ])

    # ── Epoch loop ────────────────────────────────────────────────────────
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        t0         = time.time()
        loss_meter = AverageMeter("Loss", torch.device(device), ":.4f")

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # Pad to multiple of 16 for UNet strides
            _, _, D, H, W = x.shape
            pad_d = (16 - D % 16) % 16
            pad_h = (16 - H % 16) % 16
            pad_w = (16 - W % 16) % 16
            xp = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
            yp = F.pad(y, (0, pad_w, 0, pad_h, 0, pad_d))

            optimizer.zero_grad()
            pred = model(xp)
            # Crop pred back to original crop size before computing loss
            pred = pred[:, :, :D, :H, :W]
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

        is_val_epoch = (epoch % args.val_every == 0) or (epoch == args.num_epochs)
        metrics  = compute_metrics(model, val_loader, device, args.threshold) \
                   if is_val_epoch else {}

        is_best = (
            is_val_epoch
            and metrics.get("dice",   0) > best_dice
            and metrics.get("recall", 0) >= args.min_recall_for_save
        )

        if is_val_epoch:
            above_guard = metrics["recall"] >= args.min_recall_for_save
            mp          = metrics["mean_pred"]
            collapse    = " [COLLAPSE? mean_pred too high]" if mp > 0.3 else ""
            logger.info(
                "Ep %3d/%d  loss=%.4f  rec=%.3f%s  prec=%.3f  dice=%.3f"
                "  mean_pred=%.3f%s  lr=%.2e  %.1fs%s",
                epoch, args.num_epochs, loss_meter.avg,
                metrics["recall"],
                "" if above_guard else f" [< guard {args.min_recall_for_save:.2f}]",
                metrics["precision"], metrics["dice"],
                mp, collapse, lr_now, elapsed,
                "  ★ best" if is_best else "",
            )
        else:
            logger.info("Ep %3d/%d  loss=%.4f  lr=%.2e  %.1fs",
                        epoch, args.num_epochs, loss_meter.avg, lr_now, elapsed)

        if is_best:
            best_dice = metrics["dice"]
            _save_ckpt(model, best_path, epoch, metrics)
            logger.info("  Saved best -> %s  (dice=%.3f  rec=%.3f)",
                        best_path, best_dice, metrics["recall"])

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch, f"{loss_meter.avg:.6f}",
                f"{metrics.get('recall',    ''):.4f}" if metrics else "",
                f"{metrics.get('precision', ''):.4f}" if metrics else "",
                f"{metrics.get('dice',      ''):.4f}" if metrics else "",
                f"{metrics.get('mean_pred', ''):.4f}" if metrics else "",
                f"{lr_now:.2e}", f"{elapsed:.1f}", int(is_best),
            ])

    _save_ckpt(model, last_path, args.num_epochs, {})
    logger.info("Training complete.")
    logger.info("  Best dice   : %.3f  (recall guard: ≥ %.2f)",
                best_dice, args.min_recall_for_save)
    logger.info("  Best ckpt   : %s", best_path)
    logger.info("  Last ckpt   : %s", last_path)
    logger.info("  Training log: %s", log_csv)


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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the Small3DUNet tumour proposal network on SwinCross NPZ data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data (new - JSON-driven)
    p.add_argument("--data_dir",  required=True,
                   help="SwinCross NPZ root (e.g. /data/ethan/PP_hecktor2026_kfold_npz).")
    p.add_argument("--json_list", default="dataset_swincross_2026kfold_classic.json",
                   help="SwinCross JSON filename within data_dir.")
    # Legacy directory overrides (kept for backward compatibility)
    p.add_argument("--train_dir", default=None,
                   help="[Legacy] Ignored when --json_list is provided.")
    p.add_argument("--val_dir",   default=None,
                   help="[Legacy] Ignored when --json_list is provided.")

    p.add_argument("--output_dir",
                   default="/data/ethan/MedSAM2/proposal_net/checkpoints")
    # Architecture
    p.add_argument("--base_features", type=int,   default=16)
    p.add_argument("--dropout",       type=float, default=0.1)
    p.add_argument("--prior_prob",    type=float, default=0.02)
    # Loss
    p.add_argument("--focal_weight",  type=float, default=0.3)
    p.add_argument("--tversky_alpha", type=float, default=0.3)
    p.add_argument("--tversky_beta",  type=float, default=0.7)
    # Training
    p.add_argument("--num_epochs",    type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=1)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--crop_size",     type=str,   default="64,128,128")
    p.add_argument("--threshold",     type=float, default=0.25)
    p.add_argument("--val_every",     type=int,   default=5)
    p.add_argument("--min_recall_for_save", type=float, default=0.50)
    # Misc
    p.add_argument("--num_workers",   type=int,   default=2)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
