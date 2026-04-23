#!/bin/bash
# =============================================================================
# SwinCross Training Script — SwinUNETR Cross-Modality Fusion
# Project : ProjetMaster / StageM1_IA
# Author  : Santiago (original), updated for Ethan's run
# Updated : 2026-04
#
# Data source  : /data/santiago/Datast001_HECKTOR_SwinCross/  (read-only)
# All outputs  : /data/ethan/SwinCross/hecktor_runs/<MODEL_DIR>/
#
# IMPORTANT — train.py prepends ./runs/ to --logdir, so absolute paths
# break. Fix: symlink ./runs -> /data/ethan/SwinCross/hecktor_runs once, then use
# short relative names for --logdir. The symlink is created in Step 1.
# =============================================================================

set -e   # abort on first error

# =============================================================================
# STEP 0 — Environment setup (uv + venv + requirements)
# =============================================================================

if ! command -v uv &> /dev/null; then
    echo "uv not found — installing..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

uv python install 3.12

if [ ! -d "swincross_env" ]; then
    uv venv swincross_env --python 3.12
fi
source swincross_env/bin/activate

if [ -f requirements.txt ]; then
    uv pip install -r requirements.txt
else
    echo "requirements.txt not found! Aborting."
    exit 1
fi

# =============================================================================
# STEP 1 — Path configuration
# =============================================================================

PPDATA_FOLDER=/data/santiago/Datast001_HECKTOR_SwinCross/
JSON_LIST=dataset_swincross.json

# Short name used as the --logdir argument.
# Actual output path will be: /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/
MODEL_DIR=ethan_hecktor_2000ep_run

# Create output root and symlink ./runs -> /data/ethan/SwinCross/hecktor_runs
# so train.py's "./runs/$MODEL_DIR" resolves to /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR
mkdir -p /data/ethan/SwinCross/hecktor_runs
ln -sfn /data/ethan/SwinCross/hecktor_runs ./runs

# Pre-create the model directory so the log redirect below doesn't fail
mkdir -p /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR

echo "Data      : $PPDATA_FOLDER"
echo "Outputs   : /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/"
echo "GPU       : 0 (CUDA_VISIBLE_DEVICES=0)"

# =============================================================================
# STEP 2A — Training from scratch on GPU 0  [ACTIVE]
# =============================================================================
#
# Parameter notes:
#   --noamp          : AMP causes vanishing gradients with this loss — use FP32
#   --save_checkpoint: save model_best.pth and model_last.pth every val cycle
#   --batch_size 2   : 96³ x 2ch x FP32 is VRAM-heavy; raise to 4 if VRAM allows
#   --cache_rate 0.5 : caches 50% of training data in RAM; lower to 0.0 if OOM
#   --val_every 20   : validate every 20 epochs (3000 ep / 20 = 150 checkpoints)
#   --smooth_nr/dr   : 1e-5 each — more stable than 1e-6 on background-only patches
#   --RandFlipd_prob 0.5  : increased from default 0.2 to fight overfitting
#   --gamma 2.0      : DiceFocalLoss focal exponent — standard value
#   NOTE: --res_block and --dropout_rate are parsed by train.py but have NO effect
#         on the SwinUNETR model, which uses its own hardcoded config values.

# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python3.12 -u train.py \
#     --data_dir   $PPDATA_FOLDER \
#     --logdir     $MODEL_DIR \
#     --json_list  $JSON_LIST \
#     --batch_size 2 \
#     --val_every  20 \
#     --workers    4 \
#     --cache_rate 1 \
#     --RandFlipd_prob           0.5 \
#     --RandRotate90d_prob       0.5 \
#     --RandScaleIntensityd_prob 0.2 \
#     --RandShiftIntensityd_prob 0.2 \
#     --noamp \
#     --save_checkpoint \
#     2>&1 | tee /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/training_from_scratch.log

# =============================================================================
# STEP 2B — Resume from checkpoint (adjusted for new scheduler logic)
# =============================================================================

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3.12 -u train.py \
    --data_dir        $PPDATA_FOLDER \
    --json_list       $JSON_LIST \
    --logdir          $MODEL_DIR \
    --checkpoint      ./runs/$MODEL_DIR/model_last.pth \
    --max_epochs      400 \
    --warmup_epochs   50 \
    --batch_size      2 \
    --val_every       20 \
    --optim_lr        1e-4 \
    --reg_weight      1e-5 \
    --lrschedule      warmup_cosine \
    --RandFlipd_prob           0.5 \
    --RandRotate90d_prob       0.5 \
    --RandScaleIntensityd_prob 0.2 \
    --RandShiftIntensityd_prob 0.2 \
    --cache_rate  1 \
    --workers     4 \
    --noamp \
    --save_checkpoint \
    2>&1 | tee /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/training_resume.log

# =============================================================================
# STEP 2C — Quick debug run (2 epochs, no caching, no GPU wait)  [COMMENT OUT]
# =============================================================================
# Use this to verify the full pipeline runs end-to-end before committing to
# a 2000-epoch job. Outputs go to ./runs/ethan_debug/ (i.e. /data/ethan/SwinCross/hecktor_runs/ethan_debug/)
#
# CUDA_VISIBLE_DEVICES=0 python3.12 train.py \
#     --data_dir   $PPDATA_FOLDER \
#     --json_list  $JSON_LIST \
#     --logdir     ethan_debug \
#     --batch_size 2 \
#     --cache_rate 0.0 \
#     --max_epochs 2 \
#     --val_every  1 \
#     --workers    0 \
#     --noamp

# =============================================================================
# STEP 3 — Export training curves  [UNCOMMENT AFTER TRAINING]
# =============================================================================
# uv run export_graphs.py \
#     --logdir  ./runs/$MODEL_DIR \
#     --output  /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/training_graphics

# Continuous version (use if training was interrupted and resumed one or more times):
# uv run export_graphs.py \
#     --logdir     ./runs/$MODEL_DIR \
#     --output     /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/training_graphics_continuous \
#     --continuous
