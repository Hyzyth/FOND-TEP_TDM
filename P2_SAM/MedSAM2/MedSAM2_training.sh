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
# STEP 0 — Environment setup (uv + venv + requirements)
# =============================================================================

if ! command -v uv &> /dev/null; then
    echo "uv not found — installing..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

uv python install 3.10

if [ ! -d "medsam2_env" ]; then
    uv venv medsam2_env --python 3.10
fi

source medsam2_env/bin/activate

if [ -f requirements.txt ]; then
    uv pip install -r requirements.txt
else
    echo "requirements.txt not found! Aborting."
    exit 1
fi

# Install the package itself in editable mode if not already done
if ! python -c "import sam2" &>/dev/null; then
    pip install -e . --no-build-isolation
fi

# =============================================================================
# STEP 1 — Path configuration
# =============================================================================

NPZ_DIR=/data/ethan/MedSAM2/hecktor_npz
RUNS_ROOT=/data/ethan/MedSAM2/runs
ln -sfn "$RUNS_ROOT" ./runs

# =============================================================================
# STEP 2A — Quick debug run (2 epochs) [COMMENTED OUT]
# =============================================================================
# CONFIG_DEBUG="configs/sam2.1_hiera_tiny_hecktor_debug.yaml"
# MODEL_DIR="ethan_debug_run"
# mkdir -p "$RUNS_ROOT/$MODEL_DIR"
# echo "========================================"
# echo "  MedSAM2 × HECKTOR — debug"
# echo "  Data    : $NPZ_DIR"
# echo "  Config  : $CONFIG_DEBUG"
# echo "  Outputs : $RUNS_ROOT/$MODEL_DIR/"
# echo "  GPU     : $CUDA_VISIBLE_DEVICES"
# echo "========================================"
#
# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python3.10 -u -m training.train \
#     --config "$CONFIG_DEBUG" \
#     --dataset-path "$NPZ_DIR/train" \
#     --output-path "$RUNS_ROOT/$MODEL_DIR" \
#     2>&1 | tee "$RUNS_ROOT/$MODEL_DIR/training_debug.log"

# =============================================================================
# STEP 2B — Full training from scratch [COMMENTED OUT]
# Starts from: sam2.1_hiera_tiny.pt
# =============================================================================
# CONFIG_SCRATCH="configs/sam2.1_hiera_tiny_hecktor_scratch.yaml"
# MODEL_DIR="ethan_hecktor_scratch"
# mkdir -p "$RUNS_ROOT/$MODEL_DIR"
# echo "========================================"
# echo "  MedSAM2 × HECKTOR — training from scratch"
# echo "  Data    : $NPZ_DIR"
# echo "  Config  : $CONFIG_SCRATCH"
# echo "  Outputs : $RUNS_ROOT/$MODEL_DIR/"
# echo "  GPU     : $CUDA_VISIBLE_DEVICES"
# echo "========================================"
#
# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python3.10 -u -m training.train \
#     --config "$CONFIG_SCRATCH" \
#     --dataset-path "$NPZ_DIR/train" \
#     --output-path "$RUNS_ROOT/$MODEL_DIR" \
#     2>&1 | tee "$RUNS_ROOT/$MODEL_DIR/training_scratch.log"

# =============================================================================
# STEP 2C — Fine-Tuning [ACTIVE]
# Starts from: MedSAM2_latest.pt
# =============================================================================
CONFIG_FINETUNE="configs/sam2.1_hiera_tiny_hecktor_finetune.yaml"
MODEL_DIR="ethan_hecktor_finetuned"
mkdir -p "$RUNS_ROOT/$MODEL_DIR"

echo "========================================"
echo "  MedSAM2 × HECKTOR — Fine-tuning"
echo "  Data    : $NPZ_DIR"
echo "  Config  : $CONFIG_FINETUNE"
echo "  Outputs : $RUNS_ROOT/$MODEL_DIR/"
echo "========================================"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3.10 -u -m training.train \
    --config "$CONFIG_FINETUNE" \
    --dataset-path "$NPZ_DIR/train" \
    --output-path "$RUNS_ROOT/$MODEL_DIR" \
    2>&1 | tee "$RUNS_ROOT/$MODEL_DIR/training_finetune.log"


# =============================================================================
# STEP 3 — Copy best checkpoint
# =============================================================================

BEST_CKPT=$(ls -t "$RUNS_ROOT/$MODEL_DIR/checkpoints/"*.pth 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    echo "Copying best checkpoint: $BEST_CKPT"
    cp "$BEST_CKPT" ./checkpoints/MedSAM2_latest.pt
    echo "Saved → ./checkpoints/MedSAM2_latest.pt"
else
    echo "[WARN] No checkpoint found in $RUNS_ROOT/$MODEL_DIR/checkpoints/ — skipping copy."
fi
