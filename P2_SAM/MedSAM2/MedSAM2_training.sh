#!/bin/bash
# =============================================================================
# MedSAM2 Training Script - SAM2.1-Hiera-Tiny Fine-tuning on HECKTOR 2026
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-05
#
# Data source : /data/ethan/PP_hecktor2026_kfold_npz/  (SwinCross NPZ format)
#               Uses SAME split as SwinCross and DualwaveSAM for direct comparison.
#
# Split used
# ----------
#   JSON : dataset_swincross_2026kfold_classic.json
#     training   key -> model training
#     validation key -> epoch-level Dice monitoring (fold 0 of SwinCross k-fold)
#
# The locked test vault (dataset_swincross_2026kfold_test.json) is NEVER
# touched during training - only during inference (MedSAM2_inference_execution.sh).
#
# Model init   : ./checkpoints/sam2.1_hiera_tiny.pt  (from download_checkpoints.sh)
# All outputs  : /data/ethan/MedSAM2/runs/<MODEL_DIR>/
#
# IMPORTANT - train.py prepends ./runs/ to --logdir, so absolute paths
# break.  The symlink ./runs -> /data/ethan/MedSAM2/runs is created in Step 1.
# =============================================================================

set -e

# =============================================================================
# STEP 0 - Environment setup
# =============================================================================

if ! command -v uv &> /dev/null; then
    echo "uv not found - installing..."
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
# STEP 1 - Path configuration
# =============================================================================

SWINCROSS_NPZ_ROOT="/data/ethan/PP_hecktor2026_kfold_npz"
CLASSIC_JSON="dataset_swincross_2026kfold_classic.json"

RUNS_ROOT=/data/ethan/MedSAM2/runs
mkdir -p "$RUNS_ROOT"
ln -sfn "$RUNS_ROOT" ./runs

# Verify prerequisites
if [ ! -f "$SWINCROSS_NPZ_ROOT/$CLASSIC_JSON" ]; then
    echo "  JSON not found: $SWINCROSS_NPZ_ROOT/$CLASSIC_JSON"
    echo "  Run P1_SWIN/SwinCross_NPZ_Dataset_Building.sh first."
    exit 1
fi

N_TRAIN=$(python3.10 -c "
import json
with open('${SWINCROSS_NPZ_ROOT}/${CLASSIC_JSON}') as f: d=json.load(f)
print(len(d.get('training', [])))
")
N_VAL=$(python3.10 -c "
import json
with open('${SWINCROSS_NPZ_ROOT}/${CLASSIC_JSON}') as f: d=json.load(f)
print(len(d.get('validation', [])))
")

echo "========================================"
echo "  MedSAM2 - Fine-tuning on HECKTOR 2026"
echo "  Data root  : $SWINCROSS_NPZ_ROOT"
echo "  JSON       : $CLASSIC_JSON"
echo "  Train cases: $N_TRAIN"
echo "  Val cases  : $N_VAL"
echo "========================================"

# =============================================================================
# STEP 2A - Quick debug run (2 epochs)
# =============================================================================
# CONFIG_DEBUG="configs/sam2.1_hiera_tiny_hecktor_debug.yaml"
# MODEL_DIR="ethan_debug_run"
# mkdir -p "$RUNS_ROOT/$MODEL_DIR"
#
# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python3.10 -u -m training.train \
#     --config "$CONFIG_DEBUG" \
#     --output-path "$RUNS_ROOT/$MODEL_DIR" \
#     2>&1 | tee "$RUNS_ROOT/$MODEL_DIR/training_debug.log"

# =============================================================================
# STEP 2B - Full fine-tuning
#
# The YAML config (sam2.1_hiera_tiny_hecktor_finetune.yaml) must specify:
#   dataset:
#     _target_: training.dataset.hecktor_dataset.HECKTORNPZRawDataset
#     data_dir: /data/ethan/PP_hecktor2026_kfold_npz
#     json_list: dataset_swincross_2026kfold_classic.json
#     split: training
#
# The validation dataset block uses split: validation from the same JSON.
# =============================================================================

CONFIG_FINETUNE="configs/sam2.1_hiera_tiny_hecktor_finetune.yaml"
MODEL_DIR="ethan_hecktor_finetuned"
mkdir -p "$RUNS_ROOT/$MODEL_DIR"

echo ""
echo "Starting fine-tuning …"
echo "  Config  : $CONFIG_FINETUNE"
echo "  Outputs : $RUNS_ROOT/$MODEL_DIR/"
echo ""

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3.10 -u -m training.train \
    --config "$CONFIG_FINETUNE" \
    --output-path "$RUNS_ROOT/$MODEL_DIR" \
    2>&1 | tee "$RUNS_ROOT/$MODEL_DIR/training_finetune.log"

# =============================================================================
# STEP 3 - Copy best checkpoint to ./checkpoints/
# =============================================================================

BEST_CKPT=$(ls -t "$RUNS_ROOT/$MODEL_DIR/checkpoints/"*.pt 2>/dev/null | head -1)
if [ -n "$BEST_CKPT" ]; then
    echo "Copying best checkpoint: $BEST_CKPT"
    cp "$BEST_CKPT" ./checkpoints/MedSAM2_latest.pt
    echo "  Saved -> ./checkpoints/MedSAM2_latest.pt"
else
    echo "  [WARN] No checkpoint found in $RUNS_ROOT/$MODEL_DIR/checkpoints/ - skipping copy."
fi

echo ""
echo "Training complete."
echo "  Model     : $RUNS_ROOT/$MODEL_DIR"
echo "  Checkpoint: ./checkpoints/MedSAM2_latest.pt"
echo ""
echo "Run MedSAM2_inference_execution.sh to evaluate on the locked test vault."
