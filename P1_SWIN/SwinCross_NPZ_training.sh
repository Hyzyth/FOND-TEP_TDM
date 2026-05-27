#!/bin/bash
# =============================================================================
# SwinCross Training Script — SwinUNETR Cross-Modality Fusion
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-05
#
# Data source  : /data/ethan/PP_hecktor_swincross_npz/   (NPZ, read by data_utils.py)
# All outputs  : /data/ethan/SwinCross/<MODEL_DIR>/
#
# Two training modes:
#
#   MODE A — Single-fold (original)
#     Data : /data/ethan/PP_hecktor_swincross_npz/   (2025 only, 80/20 split)
#
#   MODE B — K-fold cross-validation (anti-overfitting)
#     Data : /data/ethan/PP_hecktor2026_kfold_npz/   (2025+2026 combined)
#     Trains k independent models, one per fold JSON.
#     Each model's validation set = fold holdout + fixed unseen-hospital cases.
#
#     Run prepare_hecktor2026_kfold_npz.py (Dataset Building script) first
#     to generate the NPZ files and per-fold JSONs.
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

# ── Shared hyper-parameters ───────────────────────────────────────────────────
EPOCH_NUMBER=400
MAX_GPU=0                  # GPU index to train on

mkdir -p /data/ethan/SwinCross
ln -sfn /data/ethan/SwinCross ./runs

# =============================================================================
# MODE A — Single-fold training (original 80/20 split, 2025 data only)
# =============================================================================
# PPDATA_FOLDER=/data/ethan/PP_hecktor_swincross_npz
# JSON_LIST=dataset_swincross.json
# JSON_DEBUG=dataset_swincross_debug.json
# MODEL_DIR=HECKTOR_run_${EPOCH_NUMBER}_epoch
#
# mkdir -p /data/ethan/SwinCross/$MODEL_DIR
# echo "Data    : $PPDATA_FOLDER"
# echo "Outputs : /data/ethan/SwinCross/$MODEL_DIR/"
#
# CUDA_VISIBLE_DEVICES=$MAX_GPU \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python3.12 -u npz_version/train.py \
#     --data_dir   $PPDATA_FOLDER \
#     --logdir     $MODEL_DIR \
#     --json_list  $JSON_LIST \
#     --batch_size 2 \
#     --val_every  20 \
#     --workers    4 \
#     --cache_rate 0.0 \
#     --max_epochs $EPOCH_NUMBER \
#     --warmup_epochs 50 \
#     --RandFlipd_prob           0.5 \
#     --RandRotate90d_prob       0.5 \
#     --RandScaleIntensityd_prob 0.2 \
#     --RandShiftIntensityd_prob 0.2 \
#     --noamp \
#     --save_checkpoint \
#     2>&1 | tee /data/ethan/SwinCross/$MODEL_DIR/training_from_scratch.log

# ── STEP A.2 — Resume from checkpoint ────────────────────────────────────────
# CUDA_VISIBLE_DEVICES=$MAX_GPU \
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
#     --cache_rate  0.0 \
#     --workers     4 \
#     --noamp \
#     --save_checkpoint \
#     2>&1 | tee /data/ethan/SwinCross/$MODEL_DIR/training_resume.log

# ── STEP A.3 — Quick debug run  [COMMENT OUT] ─────────────────────────────────
# CUDA_VISIBLE_DEVICES=$MAX_GPU python3.12 npz_version/train.py \
#     --data_dir   $PPDATA_FOLDER \
#     --json_list  $JSON_DEBUG \
#     --logdir     ethan_debug \
#     --batch_size 2 \
#     --cache_rate 0.0 \
#     --max_epochs 2 \
#     --val_every  1 \
#     --workers    4 \
#     --noamp   \



# =============================================================================
# MODE B — K-fold
# =============================================================================
# Prerequisites:
#   Run SwinCross_NPZ_Dataset_Building.sh first to generate:
#     /data/ethan/PP_hecktor2026_kfold_npz/dataset_swincross_2026kfold_fold{0..K-1}.json

PPDATA_FOLDER=/data/ethan/PP_hecktor2026_kfold_npz
K_FOLDS=5
JSON_PREFIX=dataset_swincross_2026kfold
BASE_MODEL_DIR=HECKTOR_kfold_${EPOCH_NUMBER}ep

echo "╔══════════════════════════════════════════════════╗"
echo "║  SwinCross K-Fold Training  (k=${K_FOLDS}, ${EPOCH_NUMBER} epochs)  ║"
echo "╚══════════════════════════════════════════════════╝"
echo "Data    : $PPDATA_FOLDER"
echo "GPU     : $MAX_GPU"
echo ""

# ── Optional: sanity-check that all fold JSONs exist before starting ──────────
for fold in $(seq 0 $((K_FOLDS - 1))); do
    JSON_FILE="$PPDATA_FOLDER/${JSON_PREFIX}_fold${fold}.json"
    if [ ! -f "$JSON_FILE" ]; then
        echo "    Missing JSON: $JSON_FILE"
        echo "    Run SwinCross_NPZ_Dataset_Building.sh first."
        exit 1
    fi
done
echo "    All ${K_FOLDS} fold JSONs found."
echo ""

# ── Main k-fold training loop ─────────────────────────────────────────────────
for fold in $(seq 0 $((K_FOLDS - 1))); do

    JSON_LIST="${JSON_PREFIX}_fold${fold}.json"
    MODEL_DIR="${BASE_MODEL_DIR}_fold${fold}"

    mkdir -p /data/ethan/SwinCross/$MODEL_DIR

    echo "┌─────────────────────────────────────────────────"
    echo "│  Fold ${fold} / $((K_FOLDS - 1))"
    echo "│  JSON   : $JSON_LIST"
    echo "│  LogDir : /data/ethan/SwinCross/$MODEL_DIR/"
    echo "└─────────────────────────────────────────────────"

    # ── Train from scratch ────────────────────────────────────────────────
    CUDA_VISIBLE_DEVICES=$MAX_GPU \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3.12 -u npz_version/train.py \
        --data_dir      $PPDATA_FOLDER \
        --logdir        $MODEL_DIR \
        --json_list     $JSON_LIST \
        --batch_size    2 \
        --val_every     20 \
        --workers       4 \
        --cache_rate    0.35 \
        --max_epochs    $EPOCH_NUMBER \
        --warmup_epochs 50 \
        --RandFlipd_prob           0.5 \
        --RandRotate90d_prob       0.5 \
        --RandScaleIntensityd_prob 0.2 \
        --RandShiftIntensityd_prob 0.2 \
        --noamp \
        --save_checkpoint \
        2>&1 | tee /data/ethan/SwinCross/$MODEL_DIR/training_fold${fold}.log

    echo ""
    echo "    Fold ${fold} complete → /data/ethan/SwinCross/$MODEL_DIR/model_best.pth"
    echo ""

done

echo "╔══════════════════════════════════════════╗"
echo "║  All ${K_FOLDS} folds trained successfully.       ║"
echo "╚══════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Run SwinCross_NPZ_inference_execution.sh to generate per-fold predictions."
echo "  2. Run ensemble_kfold_predictions.py to majority-vote across the ${K_FOLDS} models:"
echo ""
echo "     FOLD_DIRS=\"\""
for fold in $(seq 0 $((K_FOLDS - 1))); do
echo "     FOLD_DIRS=\"\$FOLD_DIRS /data/ethan/SwinCross/${BASE_MODEL_DIR}_fold${fold}/hecktor_inference\""
done
echo ""
echo "     python3.12 npz_version/ensemble_kfold_predictions.py \\"
echo "         --fold_dirs  \$FOLD_DIRS \\"
echo "         --output_dir /data/ethan/SwinCross/${BASE_MODEL_DIR}_ensemble \\"
echo "         --data_dir   $PPDATA_FOLDER \\"
echo "         --json_list  ${JSON_PREFIX}_full.json"

# =============================================================================
# MODE B — Resume a single interrupted fold
# =============================================================================
# FOLD_TO_RESUME=2
# MODEL_DIR=${BASE_MODEL_DIR}_fold${FOLD_TO_RESUME}
# JSON_LIST="${JSON_PREFIX}_fold${FOLD_TO_RESUME}.json"
#
# CUDA_VISIBLE_DEVICES=$MAX_GPU \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python3.12 -u npz_version/train.py \
#     --data_dir        $PPDATA_FOLDER \
#     --logdir          $MODEL_DIR \
#     --json_list       $JSON_LIST \
#     --checkpoint      ./runs/$MODEL_DIR/model_last.pth \
#     --max_epochs      $EPOCH_NUMBER \
#     --warmup_epochs   50 \
#     --batch_size      2 \
#     --val_every       20 \
#     --workers         4 \
#     --cache_rate      0.35 \
#     --RandFlipd_prob           0.5 \
#     --RandRotate90d_prob       0.5 \
#     --RandScaleIntensityd_prob 0.2 \
#     --RandShiftIntensityd_prob 0.2 \
#     --noamp \
#     --save_checkpoint \
#     2>&1 | tee /data/ethan/SwinCross/$MODEL_DIR/training_fold${FOLD_TO_RESUME}_resume.log

# =============================================================================
# MODE B — Full-training run after k-fold (all data, no held-out fold)
#           Use this for the final production model after k-fold experiments.
# =============================================================================
# FULL_MODEL_DIR=${BASE_MODEL_DIR}_full
# mkdir -p /data/ethan/SwinCross/$FULL_MODEL_DIR
#
# CUDA_VISIBLE_DEVICES=$MAX_GPU \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python3.12 -u npz_version/train.py \
#     --data_dir      $PPDATA_FOLDER \
#     --logdir        $FULL_MODEL_DIR \
#     --json_list     ${JSON_PREFIX}_full.json \
#     --batch_size    2 \
#     --val_every     20 \
#     --workers       4 \
#     --cache_rate    0.35 \
#     --max_epochs    $EPOCH_NUMBER \
#     --warmup_epochs 50 \
#     --RandFlipd_prob           0.5 \
#     --RandRotate90d_prob       0.5 \
#     --RandScaleIntensityd_prob 0.2 \
#     --RandShiftIntensityd_prob 0.2 \
#     --noamp \
#     --save_checkpoint \
#     2>&1 | tee /data/ethan/SwinCross/$FULL_MODEL_DIR/training_full.log

# =============================================================================
# Debug run  [UNCOMMENT TO USE]
# =============================================================================
# CUDA_VISIBLE_DEVICES=$MAX_GPU python3.12 npz_version/train.py \
#     --data_dir   $PPDATA_FOLDER \
#     --json_list  ${JSON_PREFIX}_fold0.json \
#     --logdir     kfold_debug \
#     --batch_size 2 \
#     --cache_rate 1 \
#     --max_epochs 2 \
#     --val_every  1 \
#     --workers    4 \
#     --noamp
