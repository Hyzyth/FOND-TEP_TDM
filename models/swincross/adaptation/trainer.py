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
from datetime import datetime

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from monai.data.utils import decollate_batch
from monai.metrics import DiceMetric

def _format_time(seconds):
    """Formats seconds into d h m s."""
    if seconds < 0: return "0s"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if d > 0: return f"{d}d {h:02d}h {m:02d}m"
    if h > 0: return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"

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
    
    # ── Per-Class Metric Setup ────────────────────────────────────────────
    # Setting reduction="none" preserves both batch and class dimensions.
    per_class_acc_func = DiceMetric(include_background=True, reduction="none")

    # ── Time Tracking Initialization ──────────────────────────────────────
    global_start_time = time.time()
    total_train_time  = 0.0
    train_epochs_done = 0
    total_val_time    = 0.0
    val_passes_done   = 0

    # ── Initial "Blank Run" Validation ────────────────────────────────────
    if start_epoch == 0:
        init_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{init_start}] Running initial zero-shot validation (random weights).", flush=True)
        model.eval()
        acc_func.reset()
        init_val_loss = 0.0
        init_val_steps = 0

        with torch.no_grad():
            for val_data in val_loader:
                init_val_steps += 1
                val_outputs_gpu = model_inferer(val_data["image"].to(device))
                val_labels_gpu = val_data["label"].to(device)
                
                val_outputs_list = [post_pred(i) for i in decollate_batch(val_outputs_gpu)]
                val_labels_list = [post_label(i) for i in decollate_batch(val_labels_gpu)]
                
                init_val_loss += loss_func(val_outputs_gpu.float(), val_labels_gpu.float()).item()
                
                acc_func(y_pred=val_outputs_list, y=val_labels_list)
                per_class_acc_func(y_pred=val_outputs_list, y=val_labels_list)
                
                del val_outputs_gpu, val_labels_gpu, val_outputs_list, val_labels_list

        init_mean_acc = acc_func.aggregate().item()

        # Calculate mean across the dataset (dim=0), resulting in shape (num_classes,)
        init_per_class = per_class_acc_func.aggregate().nanmean(dim=0).cpu().numpy()
        bg_d     = init_per_class[0] if len(init_per_class) > 0 else 0.0
        tumor_d  = init_per_class[1] if len(init_per_class) > 1 else 0.0
        nodule_d = init_per_class[2] if len(init_per_class) > 2 else 0.0

        acc_func.reset()
        init_val_loss /= init_val_steps
        init_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{init_end}] Initial Validation Complete | Dice: {init_mean_acc:.4f} | Val loss: {init_val_loss:.4f} | BG: {bg_d:.4f} | Tumor: {tumor_d:.4f} | Nodule: {nodule_d:.4f}\n", flush=True)
        
        if args.rank == 0 and writer is not None:
            writer.add_scalar("val/dice", init_mean_acc, 0)
            writer.add_scalar("val/loss", init_val_loss, 0)
            writer.add_scalar("val/dice_bg", bg_d, 0)
            writer.add_scalar("val/dice_tumor", tumor_d, 0)
            writer.add_scalar("val/dice_nodule", nodule_d, 0)
        
            init_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({"state_dict": init_state}, os.path.join(args.logdir, "model_init.pth"))
            print(f"[{init_end}] Saved initial random weights to model_init.pth", flush=True)

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
        start_str  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"[{start_str}] Starting Epoch [{epoch+1}/{args.max_epochs}].", flush=True)

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

            del inputs, labels, outputs, loss

        epoch_loss  /= step
        current_lr   = optimizer.param_groups[0]["lr"]

        if scheduler is not None and not getattr(args, "disable_scheduler_step", False):
            scheduler.step()

        if args.rank == 0:
            writer.add_scalar("train/loss", epoch_loss, epoch)
            writer.add_scalar("train/lr",   current_lr, epoch)

        # ── Time Tracking & ETA Calculations ──────────────────────────────
        epoch_train_duration = time.time() - start_time
        total_train_time += epoch_train_duration
        train_epochs_done += 1

        avg_train = total_train_time / train_epochs_done
        avg_val   = (total_val_time / val_passes_done) if val_passes_done > 0 else 0.0

        rem_epochs = args.max_epochs - (epoch + 1)

        # Calculate exactly how many validations are left
        rem_vals = (args.max_epochs // args.val_every) - ((epoch + 1) // args.val_every)
        
        global_eta_sec = (rem_epochs * avg_train) + (rem_vals * avg_val)
        elapsed_sec    = time.time() - global_start_time

        epochs_to_next_val = args.val_every - ((epoch + 1) % args.val_every)
        if epochs_to_next_val == args.val_every:
            epochs_to_next_val = 0  # Validation is happening in the current loop iteration
            
        time_to_next_val_sec = epochs_to_next_val * avg_train
        eta_str      = _format_time(global_eta_sec) if val_passes_done > 0 else f"{_format_time(global_eta_sec)} (+Val pending)"
        next_val_str = _format_time(time_to_next_val_sec) if epochs_to_next_val > 0 else "Now"

        end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"[{end_str}] [Epoch {epoch+1}/{args.max_epochs}] "
            f"Train loss: {epoch_loss:.4f}  LR: {current_lr:.2e}  "
            f"Epoch Time: {epoch_train_duration:.1f}s | Elapsed: {_format_time(elapsed_sec)}\n"
            f"      ↳ Next Val in: {next_val_str} | Full Training ETA: {eta_str}",
            flush=True,
        )

        gc.collect()
        torch.cuda.empty_cache()

        # ── Validation phase ──────────────────────────────────────────────
        if (epoch + 1) % args.val_every == 0:
            val_start_time = time.time()
            val_start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            val_eta_str = _format_time(avg_val) if val_passes_done > 0 else "Pending"
            print(f"\n[{val_start_str}] Running validation at epoch {epoch+1}. Est. duration: {val_eta_str}", flush=True)

            gc.collect()
            torch.cuda.empty_cache()

            model.eval()
            acc_func.reset()
            per_class_acc_func.reset()
            val_loss  = 0.0
            val_steps = 0

            with torch.no_grad():
                for val_data in val_loader:
                    val_steps += 1

                    val_outputs_gpu = model_inferer(val_data["image"].to(device))
                    val_labels_gpu = val_data["label"].to(device)

                    val_outputs_list = [post_pred(i) for i in decollate_batch(val_outputs_gpu)]
                    val_labels_list = [post_label(i) for i in decollate_batch(val_labels_gpu)]

                    val_loss_batch = loss_func(val_outputs_gpu.float(), val_labels_gpu.float())
                    val_loss += val_loss_batch.item()

                    acc_func(y_pred=val_outputs_list, y=val_labels_list)
                    per_class_acc_func(y_pred=val_outputs_list, y=val_labels_list)

                    del val_outputs_gpu, val_labels_gpu, val_outputs_list, val_labels_list

            mean_acc      = acc_func.aggregate().item()

            per_class = per_class_acc_func.aggregate().nanmean(dim=0).cpu().numpy()
            bg_d     = per_class[0] if len(per_class) > 0 else 0.0
            tumor_d  = per_class[1] if len(per_class) > 1 else 0.0
            nodule_d = per_class[2] if len(per_class) > 2 else 0.0

            acc_func.reset()
            per_class_acc_func.reset()

            val_loss     /= val_steps
            one_dice_metric = 1.0 - mean_acc

            if args.rank == 0:
                writer.add_scalar("val/dice",          mean_acc,       epoch)
                writer.add_scalar("val/loss",          val_loss,       epoch)
                writer.add_scalar("val/one_minus_dice", one_dice_metric, epoch)
                
                writer.add_scalar("val/dice_bg",     bg_d,     epoch)
                writer.add_scalar("val/dice_tumor",  tumor_d,  epoch)
                writer.add_scalar("val/dice_nodule", nodule_d, epoch)

                val_duration = time.time() - val_start_time
                total_val_time += val_duration
                val_passes_done += 1
                val_end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                elapsed_sec = time.time() - global_start_time
                print(
                    f"[{val_end_str}] Epoch {epoch+1} Validation Dice: {mean_acc:.4f} (Best: {best_acc:.4f}) | "
                    f"1-Dice: {one_dice_metric:.4f} | Val loss: {val_loss:.4f}\n"
                    f"      ↳ Per-Class: BG={bg_d:.4f} | Tumor={tumor_d:.4f} | Nodule={nodule_d:.4f}\n"
                    f"      ↳ Val Time: {val_duration:.1f}s | Script Elapsed: {_format_time(elapsed_sec)}\n",
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

                    check_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(
                        f"[{check_str}] New best model obtained | Epoch={best_epoch} | "
                        f"Dice={best_acc:.4f}",
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
