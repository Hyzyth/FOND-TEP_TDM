#!/bin/bash
# =============================================================================
# MedSAM2 Training Script — SAM2.1-Hiera-Tiny Fine-tuning on HECKTOR
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-04
#
# Fine-tunes MedSAM2 on the HECKTOR Task-1 PET+CT NPZ dataset.
#
# Data source  : /data/ethan/MedSAM2/hecktor_npz/   (prepared by dataset_building.sh)
# Model init   : ./checkpoints/sam2.1_hiera_tiny.pt  (from download_checkpoints.sh)
# All outputs  : /data/ethan/MedSAM2/runs/<MODEL_DIR>/
#
# IMPORTANT — train.py prepends ./runs/ to --logdir, so absolute paths
# break.  Fix: symlink ./runs -> /data/ethan/MedSAM2/runs once, then use
# short relative names for --logdir.  The symlink is created in Step 1.
# =============================================================================

set -e

# =============================================================================
# STEP 0 — Environment setup (assumes medsam2_env already built by dataset_building.sh)
# =============================================================================

if [ ! -d "medsam2_env" ]; then
    echo "medsam2_env not found. Run MedSAM2_dataset_building.sh first."
    exit 1
fi
source medsam2_env/bin/activate

# =============================================================================
# STEP 1 — Path configuration
# =============================================================================

NPZ_DIR=/data/ethan/MedSAM2/hecktor_npz
RUNS_ROOT=/data/ethan/MedSAM2/runs
CONFIG="sam2/configs/sam2.1_hiera_tiny_hecktor.yaml"

# Short name used as the --logdir argument.
# Actual output path: /data/ethan/MedSAM2/runs/$MODEL_DIR/
MODEL_DIR=ethan_hecktor_hiera_tiny

mkdir -p "$RUNS_ROOT/$MODEL_DIR"
ln -sfn "$RUNS_ROOT" ./runs

echo "========================================"
echo "  MedSAM2 × HECKTOR — Fine-tuning"
echo "  Data    : $NPZ_DIR"
echo "  Config  : $CONFIG"
echo "  Outputs : $RUNS_ROOT/$MODEL_DIR/"
echo "  GPU     : $CUDA_VISIBLE_DEVICES"
echo "========================================"

# =============================================================================
# STEP 2A — Quick debug run (2 epochs, no caching)  [UNCOMMENT TO TEST PIPELINE]
#
# Use this before committing to a full run to verify the pipeline is intact.
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 python3.12 -u training/train.py \
#     --config  sam2.1_hiera_tiny_hecktor \
#     dataset.train_folder="$NPZ_DIR/train" \
#     dataset.val_folder="$NPZ_DIR/val" \
#     scratch.num_epochs=2 \
#     scratch.train_video_batch_size=1 \
#     launcher.experiment_log_dir="$RUNS_ROOT/ethan_debug" \
#     2>&1 | tee "$RUNS_ROOT/ethan_debug/debug_run.log"

# =============================================================================
# STEP 2B — Full training from scratch  [ACTIVE]
#
# Parameter notes:
#   sam2.1_hiera_tiny_hecktor : HECKTOR-specific config (HECKTORNPZRawDataset,
#                               max_num_objects=2, num_frames=8)
#   batch_size 2              : 512px x 3ch x bfloat16 — raise to 4 if VRAM allows
#   num_epochs 75             : adjust to your compute budget
#   base_lr 5e-5              : cosine decay to 5e-6
#   vision_lr 3e-5            : lower LR for the frozen-then-unfrozen image encoder
# =============================================================================

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3.12 -u training/train.py \
    --config  sam2.1_hiera_tiny_hecktor \
    dataset.train_folder="$NPZ_DIR/train" \
    dataset.val_folder="$NPZ_DIR/val" \
    launcher.experiment_log_dir="$RUNS_ROOT/$MODEL_DIR" \
    2>&1 | tee "$RUNS_ROOT/$MODEL_DIR/training_from_scratch.log"

# =============================================================================
# STEP 2C — Resume from checkpoint  [UNCOMMENT IF NEEDED]
#
# Adjust --resume to point to the desired checkpoint.
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python3.12 -u training/train.py \
#     --config  sam2.1_hiera_tiny_hecktor \
#     dataset.train_folder="$NPZ_DIR/train" \
#     dataset.val_folder="$NPZ_DIR/val" \
#     launcher.experiment_log_dir="$RUNS_ROOT/$MODEL_DIR" \
#     trainer.resume_from="$RUNS_ROOT/$MODEL_DIR/checkpoints/checkpoint_last.pth" \
#     2>&1 | tee "$RUNS_ROOT/$MODEL_DIR/training_resume.log"

# =============================================================================
# STEP 3 — Copy best checkpoint to the shared checkpoints directory
#          so the inference script can find it at ./checkpoints/MedSAM2_latest.pt
# =============================================================================

BEST_CKPT=$(ls -t "$RUNS_ROOT/$MODEL_DIR/checkpoints/"*.pth 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    echo "Copying best checkpoint: $BEST_CKPT"
    cp "$BEST_CKPT" ./checkpoints/MedSAM2_latest.pt
    echo "Saved → ./checkpoints/MedSAM2_latest.pt"
else
    echo "[WARN] No checkpoint found in $RUNS_ROOT/$MODEL_DIR/checkpoints/ — skipping copy."
fi
