"""
trainer.py  -  Training + validation loop for 3-class DualwaveSAM
=================================================================

Structure mirrors SwinCross trainer.py:
  - Timestamps and ETA on every epoch / validation pass
  - TensorBoard logging (train/loss, val/loss, val/dice)
  - model_best.pth + model_last.pth checkpointing
  - AMP (bfloat16) for training; float16 for inference
  - Initial zero-shot validation before epoch 1

Validation metric: mean Dice over foreground classes (GTVp, GTVn)
  computed slice-wise then averaged over all validation slices.
"""

import gc
import os
import time
from datetime import datetime

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from losses import CombinedLoss


# ── Time formatting ────────────────────────────────────────────────────────────

def _fmt_time(seconds: float) -> str:
    if seconds < 0:
        return "0s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0:
        return f"{d}d {h:02d}h {m:02d}m"
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


# ── Slice-level Dice metric ────────────────────────────────────────────────────

def slice_dice_per_class(
    logits:  torch.Tensor,   # (B, C, H, W)
    targets: torch.Tensor,   # (B, H, W)
    num_classes: int = 3,
    smooth: float = 1e-5,
) -> np.ndarray:
    """
    Returns per-class Dice (numpy, shape (num_classes,)).
    Background (class 0) included but ignored when averaging.
    """
    preds = torch.argmax(logits, dim=1)   # (B, H, W)
    dice  = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        pred_c = (preds == c).float()
        gt_c   = (targets == c).float()
        tp = (pred_c * gt_c).sum().item()
        fp = (pred_c * (1 - gt_c)).sum().item()
        fn = ((1 - pred_c) * gt_c).sum().item()
        dice[c] = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    return dice


# ── Main training loop ─────────────────────────────────────────────────────────

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func: CombinedLoss,
    args,
    scheduler=None,
    start_epoch: int = 0,
    start_best_acc: float = 0.0,
    scaler_state=None,
):
    """
    Run the full training + validation loop.

    Parameters
    ----------
    model           : DualwaveSAM3Class
    train_loader    : DataLoader  (yields {"image": (B,2,H,W), "label": (B,H,W)})
    val_loader      : DataLoader  (same format)
    optimizer       : torch.optim
    loss_func       : CombinedLoss
    args            : argparse.Namespace
    scheduler       : LR scheduler or None
    start_epoch     : int (resume)
    start_best_acc  : float (resume)
    scaler_state    : GradScaler state dict (resume)
    """

    torch.manual_seed(42)
    np.random.seed(42)

    device  = getattr(args, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    num_cls = getattr(args, "num_classes", 3)

    # ── AMP scaler ────────────────────────────────────────────────────────
    scaler = GradScaler(enabled=args.amp)
    if scaler_state and args.amp:
        scaler.load_state_dict(scaler_state)

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = None
    if getattr(args, "rank", 0) == 0:
        os.makedirs(args.logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.logdir)

    best_acc   = float(start_best_acc)
    best_epoch = -1

    # Time tracking
    global_start     = time.time()
    total_train_time = 0.0
    train_epochs_done = 0
    total_val_time   = 0.0
    val_passes_done  = 0

    # ── Helper: run one validation pass ───────────────────────────────────
    def _validate(epoch_label: int) -> tuple:
        """Returns (mean_fg_dice, val_loss)."""
        model.eval()
        val_loss  = 0.0
        val_steps = 0
        all_dice  = []

        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch["image"].to(device)
                labels = batch["label"].to(device)

                with torch.autocast(device_type="cuda",
                                    dtype=torch.float16,
                                    enabled=torch.cuda.is_available()):
                    logits, aux = model(imgs)

                val_loss += loss_func(logits.float(), labels, aux.float() if aux is not None else None).item()
                val_steps += 1

                per_cls = slice_dice_per_class(logits, labels, num_cls)
                # Append the full array instead of just the mean
                all_dice.append(per_cls)

                del imgs, labels, logits, aux

        # Calculate mean for each class across all batches
        if all_dice:
            mean_per_class = np.mean(all_dice, axis=0) # shape: (num_classes,)
            mean_dice = float(mean_per_class[1:].mean()) # mean of foreground (Tumor & Nodule)
        else:
            mean_per_class = np.zeros(num_cls)
            mean_dice = 0.0

        val_loss  = val_loss / max(val_steps, 1)
        return mean_dice, val_loss, mean_per_class

    # ── Initial zero-shot validation ──────────────────────────────────────
    if start_epoch == 0:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] Initial validation (random weights).", flush=True)

        init_dice, init_loss, init_per_class = _validate(0) 
        bg_d, tumor_d, nodule_d = init_per_class[0], init_per_class[1], init_per_class[2]
        
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{ts}] Init | Dice={init_dice:.4f} | Loss={init_loss:.4f} | BG={bg_d:.4f} | Tumor={tumor_d:.4f} | Nodule={nodule_d:.4f}\n", flush=True)
        if writer:
            writer.add_scalar("val/dice", init_dice, 0)
            writer.add_scalar("val/loss", init_loss, 0)
            sd = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({"state_dict": sd}, os.path.join(args.logdir, "model_init.pth"))

    # ════════════════════════════════════════════════════════════════════════
    # Epoch loop
    # ════════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.max_epochs):

        model.train()
        epoch_loss = 0.0
        steps = 0
        t0 = time.time()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{ts}] Epoch [{epoch+1}/{args.max_epochs}] starting.", flush=True)

        for batch in train_loader:
            steps += 1
            imgs   = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                logits, aux = model(imgs)
                loss = loss_func(logits, labels, aux)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            del imgs, labels, logits, aux, loss

        epoch_loss /= max(steps, 1)
        lr = optimizer.param_groups[0]["lr"]

        if scheduler is not None:
            scheduler.step()

        # Refresh BG slice sampling for the next epoch (after all batches consumed)
        if hasattr(train_loader.dataset, "on_epoch_end"):
            train_loader.dataset.on_epoch_end()
            writer.add_scalar("train/loss", epoch_loss, epoch + 1)
            writer.add_scalar("train/lr",   lr,         epoch + 1)

        epoch_dur = time.time() - t0
        total_train_time  += epoch_dur
        train_epochs_done += 1

        avg_train    = total_train_time / train_epochs_done
        avg_val      = total_val_time / max(val_passes_done, 1)
        rem_epochs   = args.max_epochs - (epoch + 1)
        rem_vals     = (args.max_epochs // args.val_every) - ((epoch + 1) // args.val_every)
        eta_sec      = rem_epochs * avg_train + rem_vals * avg_val
        elapsed      = time.time() - global_start

        ep_to_val    = args.val_every - ((epoch + 1) % args.val_every)
        if ep_to_val == args.val_every:
            ep_to_val = 0
        nv_str = _fmt_time(ep_to_val * avg_train) if ep_to_val > 0 else "Now"
        eta_str = _fmt_time(eta_sec) + ("" if val_passes_done > 0 else " (+Val pending)")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{ts}] [Epoch {epoch+1}/{args.max_epochs}] "
            f"Loss={epoch_loss:.4f}  LR={lr:.2e}  Time={epoch_dur:.1f}s | "
            f"Elapsed={_fmt_time(elapsed)}\n"
            f"      -> Next Val in: {nv_str} | ETA: {eta_str}",
            flush=True,
        )

        gc.collect()
        torch.cuda.empty_cache()

        # ── Validation ────────────────────────────────────────────────────
        if (epoch + 1) % args.val_every == 0:
            val_t0  = time.time()
            ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            est_str = _fmt_time(avg_val) if val_passes_done > 0 else "Pending"
            print(f"\n[{ts}] Validation at epoch {epoch+1}. Est.: {est_str}", flush=True)

            gc.collect()
            torch.cuda.empty_cache()

            mean_dice, val_loss, per_class = _validate(epoch + 1)
            one_minus_dice = 1.0 - mean_dice
            bg_d, tumor_d, nodule_d = per_class[0], per_class[1], per_class[2]

            if writer:
                writer.add_scalar("val/dice",          mean_dice,      epoch + 1)
                writer.add_scalar("val/loss",          val_loss,       epoch + 1)
                writer.add_scalar("val/one_minus_dice", one_minus_dice, epoch + 1)
                # Optional: Add them to TensorBoard to track them visually!
                writer.add_scalar("val/dice_bg",     bg_d,     epoch + 1)
                writer.add_scalar("val/dice_tumor",  tumor_d,  epoch + 1)
                writer.add_scalar("val/dice_nodule", nodule_d, epoch + 1)

            val_dur = time.time() - val_t0
            total_val_time  += val_dur
            val_passes_done += 1
            elapsed = time.time() - global_start
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"[{ts}] Val Dice={mean_dice:.4f} (Best={best_acc:.4f}) | "
                f"1-Dice={one_minus_dice:.4f} | Loss={val_loss:.4f}\n"
                f"      -> Per-Class: BG={bg_d:.4f} | Tumor={tumor_d:.4f} | Nodule={nodule_d:.4f}\n"
                f"      -> Val time={val_dur:.1f}s | Elapsed={_fmt_time(elapsed)}\n",
                flush=True,
            )

            def _ckpt(extra=None):
                d = {
                    "epoch":      epoch + 1,
                    "best_acc":   best_acc,
                    "state_dict": (model.module.state_dict()
                                   if hasattr(model, "module")
                                   else model.state_dict()),
                    "optimizer":  optimizer.state_dict(),
                    "scheduler":  scheduler.state_dict() if scheduler else None,
                    "scaler":     scaler.state_dict() if args.amp else None,
                }
                if extra:
                    d.update(extra)
                return d

            torch.save(_ckpt(), os.path.join(args.logdir, "model_last.pth"))

            if mean_dice > best_acc:
                best_acc   = mean_dice
                best_epoch = epoch + 1
                torch.save(
                    _ckpt({"best_acc": best_acc}),
                    os.path.join(args.logdir, "model_best.pth"),
                )
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(
                    f"[{ts}] [x] New best | Epoch={best_epoch} | Dice={best_acc:.4f}",
                    flush=True,
                )

    if writer:
        writer.close()

    print(f"\nTraining complete. Best Dice={best_acc:.4f} at epoch {best_epoch}")
    return best_acc
