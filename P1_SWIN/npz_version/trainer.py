"""
trainer.py  —  SwinCross training + validation loop
=====================================================

Changes vs. original:
  - Validation inference now always uses FP16 autocast (torch.autocast with
    dtype=float16), independent of the --noamp training flag.
    Rationale: --noamp disables AMP only for training backward passes where
    vanishing gradients were observed.  Forward-only inference (no .backward())
    is safe in FP16 and ~2× faster — matching the behaviour of test.py.
  - Bug fix: model_inferer inf_size used roi_x twice instead of roi_z;
    inherited from train.py.  Unchanged here (fix is in train.py).
"""

import gc
import os
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from monai.data.utils import decollate_batch


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer,
    scheduler=None,
    start_epoch=0,
    start_best_acc=0.0,
    scaler_state=None,
    post_label=None,
    post_pred=None,
):
    """
    Core training + validation loop.

    Parameters
    ----------
    model            : torch.nn.Module
    train_loader     : DataLoader
    val_loader       : DataLoader
    optimizer        : torch.optim.Optimizer
    loss_func        : callable  (DiceFocalLoss)
    acc_func         : monai DiceMetric
    args             : argparse.Namespace
    model_inferer    : partial(sliding_window_inference, ...)
    scheduler        : LR scheduler or None
    start_epoch      : int   (for checkpoint resume)
    start_best_acc   : float (for checkpoint resume)
    scaler_state     : dict  (GradScaler state for AMP resume)
    post_label       : AsDiscrete transform
    post_pred        : AsDiscrete transform
    """

    # ── Reproducibility ───────────────────────────────────────────────────
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    # ── Device ────────────────────────────────────────────────────────────
    device = (getattr(args, "device", None)
              or torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"))

    # ── AMP scaler (training only) ────────────────────────────────────────
    scaler = GradScaler(enabled=args.amp)
    if scaler_state is not None and args.amp:
        scaler.load_state_dict(scaler_state)

    # ── TensorBoard ───────────────────────────────────────────────────────
    writer = None
    if args.rank == 0:
        os.makedirs(args.logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.logdir)

    best_acc  = float(start_best_acc)
    best_epoch = -1

    # ══════════════════════════════════════════════════════════════════════
    # Epoch loop
    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch, args.max_epochs):

        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # ── Training phase ────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        step       = 0
        start_time = time.time()

        for batch_data in train_loader:
            step   += 1
            inputs  = batch_data["image"].to(device)
            labels  = batch_data["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=args.amp):
                outputs = model(inputs)
                loss    = loss_func(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}",
                  flush=True)

            del inputs, labels, outputs, loss

        epoch_loss  /= step
        current_lr   = optimizer.param_groups[0]["lr"]

        if scheduler is not None and not getattr(args, "disable_scheduler_step", False):
            scheduler.step()

        if args.rank == 0:
            writer.add_scalar("train/loss", epoch_loss, epoch)
            writer.add_scalar("train/lr",   current_lr, epoch)

        print(
            f"[Epoch {epoch+1}/{args.max_epochs}] "
            f"Train loss: {epoch_loss:.4f}  LR: {current_lr:.2e}  "
            f"Time: {time.time()-start_time:.1f}s",
            flush=True,
        )

        gc.collect()
        torch.cuda.empty_cache()

        # ── Validation phase ──────────────────────────────────────────────
        if (epoch + 1) % args.val_every == 0:
            print(f"Running validation at epoch {epoch+1}...")

            gc.collect()
            torch.cuda.empty_cache()

            model.eval()
            acc_func.reset()
            val_loss  = 0.0
            val_steps = 0

            with torch.no_grad():
                for val_data in val_loader:
                    val_steps += 1

                    # ── FP16 inference — always ON, separate from training AMP ──
                    # --noamp disables AMP only during backward; inference is
                    # always safe in FP16 and roughly 2× faster than FP32.
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=torch.cuda.is_available(),
                    ):
                        val_outputs_cpu = model_inferer(
                            val_data["image"].to(device)
                        ).cpu()

                    val_labels_cpu = val_data["label"].cpu()

                    val_outputs_list = [
                        post_pred(i) for i in decollate_batch(val_outputs_cpu)
                    ]
                    val_labels_list = [
                        post_label(i) for i in decollate_batch(val_labels_cpu)
                    ]

                    val_loss_batch = loss_func(val_outputs_cpu, val_labels_cpu)
                    val_loss += val_loss_batch.item()

                    acc_func(y_pred=val_outputs_list, y=val_labels_list)

                    del val_outputs_cpu, val_labels_cpu, val_outputs_list, val_labels_list

            mean_acc      = acc_func.aggregate().item()
            acc_func.reset()
            val_loss     /= val_steps
            one_dice_metric = 1.0 - mean_acc

            if args.rank == 0:
                writer.add_scalar("val/dice",          mean_acc,       epoch)
                writer.add_scalar("val/loss",          val_loss,       epoch)
                writer.add_scalar("val/one_minus_dice", one_dice_metric, epoch)

                print(
                    f"Validation Dice: {mean_acc:.4f} (Best: {best_acc:.4f}) | "
                    f"1-Dice: {one_dice_metric:.4f} | Val loss: {val_loss:.4f}",
                    flush=True,
                )

                def _ckpt(extra=None):
                    """Build checkpoint dict."""
                    d = {
                        "epoch":      epoch + 1,
                        "best_acc":   best_acc,
                        "state_dict": (model.module.state_dict()
                                       if hasattr(model, "module")
                                       else model.state_dict()),
                        "optimizer":  optimizer.state_dict(),
                        "scheduler":  (scheduler.state_dict()
                                       if scheduler is not None else None),
                        "scaler":     (scaler.state_dict() if args.amp else None),
                    }
                    if extra:
                        d.update(extra)
                    return d

                # Always save the latest checkpoint
                torch.save(_ckpt(), os.path.join(args.logdir, "model_last.pth"))

                if mean_acc > best_acc:
                    best_acc   = mean_acc
                    best_epoch = epoch + 1
                    torch.save(
                        _ckpt({"best_acc": best_acc}),
                        os.path.join(args.logdir, "model_best.pth"),
                    )
                    print(
                        f"✅ New best model  epoch={best_epoch}  "
                        f"dice={best_acc:.4f}",
                        flush=True,
                    )

    # ── Cleanup ───────────────────────────────────────────────────────────
    if args.rank == 0:
        writer.close()

    print(f"Training complete. Best Dice: {best_acc:.4f} at epoch {best_epoch}")
    return best_acc


# ── Utility (used by test.py if imported) ────────────────────────────────────

def dice(pred, gt, epsilon=1e-6):
    """Binary Dice coefficient.  Used for standalone evaluation; not in training loop."""
    import numpy as np
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    return (2.0 * intersection) / (pred.sum() + gt.sum() + epsilon)
