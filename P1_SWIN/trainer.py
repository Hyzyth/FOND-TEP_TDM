import os
import time
import gc
import torch
import numpy as np

from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from monai.data.utils import decollate_batch # endroit de importation modifiée par proposition de VSCode monai.data -> monai.data.utils


# ============================================================
# Main training function
# ============================================================
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
    start_best_acc=0.0, # MODIFICATION: Permet de reprendre le meilleur score connu lors du chargement d'un checkpoint
    scaler_state=None,  # MODIFICATION: Permet de reprendre l'état du GradScaler pour une reprise de formation plus fluide
    post_label=None,
    post_pred=None,
):
    """
    Core training + validation loop for 3D medical image segmentation
    using MONAI and PyTorch.

    This function supports:
    - PET/CT multi-modal input (multi-channel tensors)
    - Mixed Precision Training (AMP)
    - Sliding Window Inference for large 3D volumes
    - Distributed Data Parallel (DDP)
    - TensorBoard logging
    - Best-model checkpointing

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model (e.g. SwinUNETR Cross-Modality).
    train_loader : DataLoader
        MONAI DataLoader for training data.
    val_loader : DataLoader
        MONAI DataLoader for validation data.
    optimizer : torch.optim.Optimizer
        Optimizer (AdamW, SGD, etc.).
    loss_func : callable
        Loss function (e.g. DiceCELoss).
    acc_func : monai.metrics.Metric
        Metric function (DiceMetric).
    args : argparse.Namespace
        Training configuration.
    model_inferer : callable
        Sliding window inference wrapper.
    scheduler : torch.optim.lr_scheduler, optional
        Learning rate scheduler.
    start_epoch : int
        Starting epoch (useful for resuming training).
    start_best_acc : float
        Best validation accuracy to start with (useful for resuming training).
    scaler_state : dict
        State dict for GradScaler (useful for resuming AMP training).
    post_label : callable
        Post-processing for ground truth labels.
    post_pred : callable
        Post-processing for model predictions.
    """

    # ------------------------------------------------------------
    # Reproducibility (important for academic projects)
    # ------------------------------------------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    torch.cuda.manual_seed_all(0)

    # ------------------------------------------------------------
    # Device and AMP setup
    # ------------------------------------------------------------
    # Centralisation de la logique de sélection du device pour éviter les erreurs d'attribut lors de l'accès à args.device dans le reste du code
    if hasattr(args, 'device'):
        device = args.device
    else:
        # Fallback de sécurité si args.device n'existe pas
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    # ------------------------------------------------------------
    
    scaler = GradScaler(enabled=args.amp)
    if scaler_state is not None and args.amp:       # MODIFICATION: Restauration du scaler depuis checkpoint pour resume AMP correct.
        scaler.load_state_dict(scaler_state)

    # ------------------------------------------------------------
    # TensorBoard logging (only on rank 0 for DDP)
    # ------------------------------------------------------------
    writer = None
    if args.rank == 0:
        os.makedirs(args.logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.logdir)

    # ------------------------------------------------------------
    # Tracking best validation Dice
    # ------------------------------------------------------------
    best_acc = float(start_best_acc)    # MODIFICATION: Initialisé depuis checkpoint pour ne pas écraser le vrai meilleur score au resume.
    best_epoch = -1

    # ============================================================
    # Epoch loop
    # ============================================================
    for epoch in range(start_epoch, args.max_epochs):

        # Required for DistributedSampler shuffling
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # -------------------------
        # TRAINING PHASE
        # -------------------------
        model.train()
        epoch_loss = 0.0
        step = 0
        start_time = time.time()

        for batch_data in train_loader:
            step += 1

            # ----------------------------------------------------
            # Input tensors
            # image shape: [B, 2, D, H, W] (PET + CT)
            # label shape: [B, 1, D, H, W]
            # ----------------------------------------------------
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            # Debug disabled for performance - uncomment if needed:
            # print(f"DEBUG Labels shape: {labels.shape}, dtype: {labels.dtype}, unique: {torch.unique(labels)}")

            optimizer.zero_grad(set_to_none=True)

            # ----------------------------------------------------
            # Forward pass with Automatic Mixed Precision (AMP)
            # ----------------------------------------------------
            with autocast(enabled=args.amp):
                outputs = model(inputs)
                loss = loss_func(outputs, labels)

            # ----------------------------------------------------
            # Backward pass (scaled for AMP stability)
            # ----------------------------------------------------
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item():.4f}", flush=True) # MODIFICATION: Affichage du loss à chaque epoch et step pour un suivi plus fin
            
            # MEMORY FIX: Explicitly delete tensors to prevent GPU memory leak
            del inputs, labels, outputs, loss

        # Average training loss over the epoch
        epoch_loss /= step

        current_lr = optimizer.param_groups[0]["lr"] if len(optimizer.param_groups) > 0 else float("nan") # MODIFICATION: current_lr loggé, utile pour diagnostiquer les problèmes de convergence liés au scheduler ou à l'optimiseur.
        # Learning rate scheduling
        if scheduler is not None and not getattr(args, "disable_scheduler_step", False): # MODIFICATION: Guard disable_scheduler_step, permet de désactiver le step du scheduler lors du resume pour éviter les problèmes de LR trop bas après reprise.
            scheduler.step()

        # TensorBoard logging (training loss)
        if args.rank == 0:
            writer.add_scalar("train/loss", epoch_loss, epoch)  # Log du loss à chaque epoch
            writer.add_scalar("train/lr", current_lr, epoch)    # MODIFICATION: Log du learning rate à chaque epoch

        print(
            f"[Epoch {epoch + 1}/{args.max_epochs}] "   # Affichage de l'epoch en cours
            f"Train loss: {epoch_loss:.4f} "            # Affichage du loss moyen de l'epoch
            f"LR: {current_lr:.2e}",                    # MODIFICATION: Affichage du learning rate actuel
            f"Time: {time.time() - start_time:.2f}s",   # Affichage du temps écoulé pour l'epoch
            flush=True
        )

        # MEMORY FIX: Clear GPU cache after each epoch
        gc.collect()
        torch.cuda.empty_cache()

        # -------------------------
        # VALIDATION PHASE
        # -------------------------
        if (epoch + 1) % args.val_every == 0:

            print(f"Running validation at epoch {epoch + 1}...")

            # MODIFICATION: Clear memory before validation to prevent OOM
            gc.collect()
            torch.cuda.empty_cache()

            model.eval()
            acc_func.reset()

            val_loss = 0.0
            val_steps = 0

            with torch.no_grad():
                for val_data in val_loader:
                    val_steps += 1

                    # MODIFICATION: CPU offload immédiat, évite les OOM sur grands volumes 3D.
                    with autocast(enabled=args.amp):
                        val_outputs_cpu = model_inferer(val_data["image"].to(device)).cpu()

                    val_labels_cpu = val_data["label"].to(device="cpu")

                    val_outputs_list = [
                        post_pred(i) for i in decollate_batch(val_outputs_cpu)
                    ]
                    val_labels_list = [
                        post_label(i) for i in decollate_batch(val_labels_cpu)
                    ]

                    # MODIFICATION: val_loss restauré, signal de diagnostic train/val divergence.
                    val_loss_batch = loss_func(
                        val_outputs_cpu,
                        val_labels_cpu
                    )
                    val_loss += val_loss_batch.item()

                    acc_func(y_pred=val_outputs_list, y=val_labels_list)
                    
                    # MEMORY FIX: Explicitly delete validation tensors to prevent GPU memory leak during validation loop
                    del val_outputs_cpu, val_labels_cpu, val_outputs_list, val_labels_list

            # Aggregate Dice over validation dataset
            mean_acc = acc_func.aggregate().item()
            acc_func.reset()
            val_loss /= val_steps
            one_dice_metric = 1.0 - mean_acc
            
            # -------------------------
            # Logging & checkpointing
            # -------------------------
            if args.rank == 0:
                writer.add_scalar("val/dice", mean_acc, epoch)
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/one_minus_dice", one_dice_metric, epoch)

                print(
                    f"Validation Dice: {mean_acc:.4f} "
                    f"(Best: {best_acc:.4f}) | "
                    f"1-Dice: {one_dice_metric:.4f} | "
                    f"Val loss: {val_loss:.4f}",
                    flush=True
                )

                # Always save last model
                last_checkpoint = {
                    "epoch": epoch + 1,
                    "best_acc": best_acc,
                    "state_dict": model.module.state_dict()
                    if hasattr(model, "module")
                    else model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # MODIFICATION: Scheduler et scaler sauvegardés, permet un resume à l'identique.
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": scaler.state_dict() if args.amp else None,
                }

                torch.save(
                    last_checkpoint,
                    os.path.join(args.logdir, "model_last.pth"),
                )

                # Save best model based on Dice score
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_epoch = epoch + 1

                    best_checkpoint = {
                        "epoch": best_epoch,
                        "best_acc": best_acc,
                        "state_dict": model.module.state_dict()
                        if hasattr(model, "module")
                        else model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        # MODIFICATION: Scheduler et scaler sauvegardés, permet un resume à l'identique.
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "scaler": scaler.state_dict() if args.amp else None,
                    }

                    torch.save(
                        best_checkpoint,
                        os.path.join(args.logdir, "model_best.pth"),
                    )

                    print(
                        f"✅ Saved new best model "
                        f"(epoch {best_epoch}, dice {best_acc:.4f}, 1-dice {one_dice_metric:.4f})",
                        flush=True
                    )

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------
    if args.rank == 0:
        writer.close()

    print(
        f"Training completed. "
        f"Best Dice: {best_acc:.4f} at epoch {best_epoch}"
    )

    return best_acc

def dice(pred, gt, epsilon=1e-6):
    """
    Compute Dice coefficient between two binary volumes.

    This function is used during testing / inference (test.py),
    NOT during training.

    Parameters
    ----------
    pred : numpy.ndarray or torch.Tensor
        Binary prediction mask (D, H, W)
        True / 1 = predicted foreground (tumor)
    gt : numpy.ndarray or torch.Tensor
        Binary ground-truth mask (D, H, W)
        True / 1 = ground-truth foreground
    epsilon : float
        Small constant to avoid division by zero

    Returns
    -------
    dice_score : float
        Dice similarity coefficient in [0, 1]
    """

    # Ensure numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.cpu().numpy()

    # Convert to boolean masks
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # Compute intersection and volumes
    intersection = np.logical_and(pred, gt).sum()
    pred_volume = pred.sum()
    gt_volume = gt.sum()

    # Handle empty ground truth & prediction case
    if pred_volume == 0 and gt_volume == 0:
        return 1.0

    dice_score = (2.0 * intersection) / (pred_volume + gt_volume + epsilon)
    return dice_score
