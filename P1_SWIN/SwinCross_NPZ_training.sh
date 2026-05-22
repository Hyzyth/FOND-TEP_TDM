#!/bin/bash
# =============================================================================
# SwinCross Training Script — SwinUNETR Cross-Modality Fusion
# Project : ProjetMaster / StageM1_IA
# Author  : Santiago (original), updated for Ethan's NPZ pipeline
# Updated : 2026-05
#
# Data source  : /data/ethan/PP_hecktor_swincross_npz/   (NPZ, read by data_utils.py)
# All outputs  : /data/ethan/SwinCross/hecktor_runs/<MODEL_DIR>/
#
# Speed improvements in this version
# ------------------------------------
#  1. NPZ input: orient/resample/crop done offline — no heavy CPU work per batch.
#  2. persistent_workers=True: workers stay alive between epochs.
#  3. prefetch_factor=2: workers pre-load next batch while GPU trains.
#  4. cache_num cap removed: cache_rate now fully respected.
#  5. FP16 validation inference: ~2× faster validation, independent of --noamp.
#  6. roi_z bug fixed: inf_size now uses roi_z (was roi_x twice).
# =============================================================================

set -e

# ── STEP 0 — Environment ──────────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "uv not found — installing..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi
uv python install 3.12
[ ! -d "swincross_env" ] && uv venv swincross_env --python 3.12
source swincross_env/bin/activate
[ -f requirements.txt ] && uv pip install -r requirements.txt || { echo "requirements.txt missing"; exit 1; }

EPOCH_NUMBER=1000

# ── STEP 1 — Paths ────────────────────────────────────────────────────────────
PPDATA_FOLDER=/data/ethan/PP_hecktor_swincross_npz
JSON_LIST=dataset_swincross.json
MODEL_DIR=HECKTOR_run_${EPOCH_NUMBER}_epoch

mkdir -p /data/ethan/SwinCross
ln -sfn /data/ethan/SwinCross ./runs
mkdir -p /data/ethan/SwinCross/$MODEL_DIR

echo "Data    : $PPDATA_FOLDER"
echo "Outputs : /data/ethan/SwinCross/$MODEL_DIR/"
echo "GPU     : 0"

# ── STEP 2A — Training from scratch  ──────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3.12 -u npz_version/train.py \
    --data_dir   $PPDATA_FOLDER \
    --logdir     $MODEL_DIR \
    --json_list  $JSON_LIST \
    --batch_size 2 \
    --val_every  20 \
    --workers    4 \
    --cache_rate 1.0 \
    --max_epochs $EPOCH_NUMBER \
    --warmup_epochs 50 \
    --RandFlipd_prob           0.5 \
    --RandRotate90d_prob       0.5 \
    --RandScaleIntensityd_prob 0.2 \
    --RandShiftIntensityd_prob 0.2 \
    --noamp \
    --save_checkpoint \
    2>&1 | tee /data/ethan/SwinCross/$MODEL_DIR/training_from_scratch.log

# ── STEP 2B — Resume from checkpoint ─────────────────────────────────────────
# CUDA_VISIBLE_DEVICES=0 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python3.12 -u npz_version/train.py \
#     --data_dir        $PPDATA_FOLDER \
#     --json_list       $JSON_LIST \
#     --logdir          $MODEL_DIR \
#     --checkpoint      ./runs/$MODEL_DIR/model_last.pth \
#     --max_epochs      $EPOCH_NUMBER \
#     --warmup_epochs   50 \
#     --batch_size      2 \
#     --val_every       20 \
#     --optim_lr        1e-4 \
#     --reg_weight      1e-5 \
#     --lrschedule      warmup_cosine \
#     --RandFlipd_prob           0.5 \
#     --RandRotate90d_prob       0.5 \
#     --RandScaleIntensityd_prob 0.2 \
#     --RandShiftIntensityd_prob 0.2 \
#     --cache_rate  1.0 \
#     --workers     4 \
#     --noamp \
#     --save_checkpoint \
#     2>&1 | tee /data/ethan/SwinCross/$MODEL_DIR/training_resume.log

# ── STEP 2C — Quick debug run  [COMMENT OUT] ──────────────────────────────────
# CUDA_VISIBLE_DEVICES=0 python3.12 npz_version/train.py \
#     --data_dir   $PPDATA_FOLDER \
#     --json_list  $JSON_LIST \
#     --logdir     ethan_debug \
#     --batch_size 2 \
#     --cache_rate 0.0 \
#     --max_epochs 2 \
#     --val_every  1 \
#     --workers    4 \
#     --noamp

# ── STEP 3 — Export training curves  [UNCOMMENT AFTER TRAINING] ───────────────
# uv run export_graphs.py \
#     --logdir  ./runs/$MODEL_DIR \
#     --output  /data/ethan/SwinCross/$MODEL_DIR/training_graphics
