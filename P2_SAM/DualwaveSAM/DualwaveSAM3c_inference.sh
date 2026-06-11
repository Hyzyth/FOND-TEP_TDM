#!/bin/bash
# =============================================================================
# DualwaveSAM 3-Class Inference & Rich Evaluation Master Script
# Author  : Ethan
# Project : ProjetMaster / StageM1_IA
# Updated : 2026-05
#
# Data source : /data/ethan/PP_hecktor2026_kfold_npz/
# Outputs     : /data/ethan/DualwaveSAM3c/<run_dir>/<target>/
#
# Evaluation reuses SwinCross evaluate_predictions.py and plot_metrics.py
# directly (identical NIfTI output format → drop-in compatible).
#
# Modes:
#   Classic  single model  →  Test vault / Train pool / Val pool
#   K-Fold   ensemble      →  same targets
# =============================================================================

set -e

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                          GLOBAL CONFIGURATION                          ║
# ╚════════════════════════════════════════════════════════════════════════╝
# ── 1. Execution Toggles (false = skip) ────────────────────────────────────
SKIP_TEST_RUN=true
SKIP_CLASSIC_TEST=false
SKIP_CLASSIC_TRAIN=false
SKIP_CLASSIC_VAL=false

SKIP_KFOLD_TEST=true
SKIP_KFOLD_TRAIN=true
SKIP_KFOLD_VAL=true

# ── 2. Hardware & Inference Parameters ────────────────────────────────────
GPU=0
INFER_BATCH=32          # Slices per GPU forward pass (2D - can be large)

# ── 3. Data Paths & JSONs ──────────────────────────────────────────────────
HECKTOR_DATA="/data/ethan/PP_hecktor2026_kfold_npz"
JSON_PREFIX="dataset_swincross_2026kfold"

JSON_TEST="${JSON_PREFIX}_test.json"
JSON_TRAIN="${JSON_PREFIX}_classic_train.json"
JSON_VAL="${JSON_PREFIX}_classic_val.json"

# ── 4. Model Architecture (must match training config) ─────────────────────
IMG_SIZE=256
N_FILTERS=16
WAVELET="haar"
NUM_CLASSES=3

# ── 5. Classic Model Setup ─────────────────────────────────────────────────
TEST_MODEL_DIR="DualwaveSAM3c_test"
TEST_WEIGHTS="model_last.pth"
CLASSIC_MODEL_DIR="DualwaveSAM3c_classic_500ep"
CLASSIC_WEIGHTS="model_last.pth"

# ── 6. K-Fold Setup ────────────────────────────────────────────────────────
KFOLD_BASE_DIR="DualwaveSAM3c_kfold_100ep"
K_FOLDS=5

# ── 7. SwinCross evaluation scripts (reused as-is) ─────────────────────────
SWIN_DIR="../../P1_SWIN"   # relative to DualwaveSAM root
EVAL_SCRIPT="${SWIN_DIR}/npz_version/evaluate_predictions.py"
PLOT_METRICS_SCRIPT="${SWIN_DIR}/plot_metrics.py"


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                      ENVIRONMENT & HELPERS                             ║
# ╚════════════════════════════════════════════════════════════════════════╝
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d "dualwave_env" ]; then
    echo "dualwave_env not found - run the training script first."
    exit 1
fi
source dualwave_env/bin/activate

mkdir -p /data/ethan/DualwaveSAM3c
ln -sfn /data/ethan/DualwaveSAM3c ./runs

# ── Helper: single model test run inference + eval + plot ──────────────────
run_test_inference() {
    local TARGET_NAME=$1
    local DATA_DIR=$2
    local JSON_FILE=$3
    local SPLIT=$4          # "validation" | "training" (JSON key)

    echo "╔═══════════════════════════════════════════════╗"
    echo "║ TEST RUN INFERENCE : $TARGET_NAME"
    echo "║ (Quick sanity check with a single model)      ║"
    echo "╚═══════════════════════════════════════════════╝"

    local OUT_DIR="/data/ethan/DualwaveSAM3c/${TEST_MODEL_DIR}/${TARGET_NAME}"
    mkdir -p "$OUT_DIR"

    echo " [1/4] Inference with ${TEST_MODEL_DIR}/${TEST_WEIGHTS}"
    CUDA_VISIBLE_DEVICES=$GPU python3.12 adaptation/infer.py \
        --data_dir    "$DATA_DIR" \
        --json_list   "$JSON_FILE" \
        --checkpoint  "./runs/${TEST_MODEL_DIR}/${TEST_WEIGHTS}" \
        --output_dir  "$OUT_DIR" \
        --split       "$SPLIT" \
        --img_size    $IMG_SIZE \
        --n_filters   $N_FILTERS \
        --wavelet     "$WAVELET" \
        --num_classes $NUM_CLASSES \
        --batch_size  $INFER_BATCH \
        --gpu         0 \
        --skip_existing \
        2>&1 | tee "$OUT_DIR/inference.log"
    
    echo " [2/4] Evaluation"
    python3.12 "$EVAL_SCRIPT" \
        --data_dir   "$DATA_DIR" \
        --json_list  "$JSON_FILE" \
        --output_dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/evaluation.log"
    
    echo " [3/4] Plotting metrics"
    python3.12 "$PLOT_METRICS_SCRIPT" \
        --csv_path   "$OUT_DIR/per_case_evaluation_rich.csv" \
        --output_dir "$OUT_DIR/plots"
    
    echo " [4/4] Plotting Post-Processing Analytics"
    python3.12 adaptation/plot_postprocessing.py \
        --csv_path   "$OUT_DIR/postprocessing_logs.csv" \
        --output_dir "$OUT_DIR/plots"
    
    echo " Complete."
}

# ── Helper: single-model inference + eval + plot ───────────────────────────
run_single_inference() {
    local TARGET_NAME=$1
    local DATA_DIR=$2
    local JSON_FILE=$3
    local SPLIT=$4          # "validation" | "training" (JSON key)

    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ CLASSIC MODE : $TARGET_NAME"
    echo "╚════════════════════════════════════════════════════════════╝"

    local OUT_DIR="/data/ethan/DualwaveSAM3c/${CLASSIC_MODEL_DIR}/${TARGET_NAME}"
    mkdir -p "$OUT_DIR"

    echo " [1/4] Inference with ${CLASSIC_MODEL_DIR}/${CLASSIC_WEIGHTS}"
    CUDA_VISIBLE_DEVICES=$GPU python3.12 adaptation/infer.py \
        --data_dir    "$DATA_DIR" \
        --json_list   "$JSON_FILE" \
        --checkpoint  "./runs/${CLASSIC_MODEL_DIR}/${CLASSIC_WEIGHTS}" \
        --output_dir  "$OUT_DIR" \
        --split       "$SPLIT" \
        --img_size    $IMG_SIZE \
        --n_filters   $N_FILTERS \
        --wavelet     "$WAVELET" \
        --num_classes $NUM_CLASSES \
        --batch_size  $INFER_BATCH \
        --gpu         0 \
        --skip_existing \
        2>&1 | tee "$OUT_DIR/inference.log"

    echo " [2/4] Evaluation"
    python3.12 "$EVAL_SCRIPT" \
        --data_dir   "$DATA_DIR" \
        --json_list  "$JSON_FILE" \
        --output_dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/evaluation.log"

    echo " [3/4] Plotting metrics"
    python3.12 "$PLOT_METRICS_SCRIPT" \
        --csv_path   "$OUT_DIR/per_case_evaluation_rich.csv" \
        --output_dir "$OUT_DIR/plots"
    
    echo " [4/4] Plotting Post-Processing Analytics"
    python3.12 adaptation/plot_postprocessing.py \
        --csv_path   "$OUT_DIR/postprocessing_logs.csv" \
        --output_dir "$OUT_DIR/plots"

    echo " Complete."
}


# ── Helper: k-fold ensemble inference + eval + plot ────────────────────────
run_kfold_ensemble() {
    local TARGET_NAME=$1
    local DATA_DIR=$2
    local JSON_FILE=$3
    local SPLIT=$4

    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ K-FOLD ENSEMBLE : $TARGET_NAME"
    echo "╚════════════════════════════════════════════════════════════╝"

    local ENSEMBLE_OUT="/data/ethan/DualwaveSAM3c/${KFOLD_BASE_DIR}_ensemble/${TARGET_NAME}"
    mkdir -p "$ENSEMBLE_OUT"
    local FOLD_DIRS=""

    # 1. Per-fold inference
    echo " [1/4] Per-Fold Inference"
    for fold in $(seq 0 $((K_FOLDS - 1))); do
        local FOLD_MODEL_DIR="${KFOLD_BASE_DIR}_fold${fold}"
        local FOLD_OUT="/data/ethan/DualwaveSAM3c/${FOLD_MODEL_DIR}/${TARGET_NAME}"
        mkdir -p "$FOLD_OUT"
        FOLD_DIRS="$FOLD_DIRS $FOLD_OUT"

        echo "      ↳ Fold $fold"
        CUDA_VISIBLE_DEVICES=$GPU python3.12 adaptation/infer.py \
            --data_dir    "$DATA_DIR" \
            --json_list   "$JSON_FILE" \
            --checkpoint  "./runs/${FOLD_MODEL_DIR}/model_best.pth" \
            --output_dir  "$FOLD_OUT" \
            --split       "$SPLIT" \
            --img_size    $IMG_SIZE \
            --n_filters   $N_FILTERS \
            --wavelet     "$WAVELET" \
            --num_classes $NUM_CLASSES \
            --batch_size  $INFER_BATCH \
            --gpu         0 \
            --skip_existing \
            2>&1 | tee "$FOLD_OUT/inference.log"
    done

    # 2. Majority-vote ensemble
    echo " [2/4] Majority-Vote Ensemble"
    python3.12 adaptation/ensemble_kfold_predictions.py \
        --fold_dirs  $FOLD_DIRS \
        --output_dir "$ENSEMBLE_OUT" \
        --data_dir   "$DATA_DIR" \
        --json_list  "$JSON_FILE" \
        2>&1 | tee "$ENSEMBLE_OUT/ensemble.log"

    # 3. Evaluate ensemble
    echo " [3/4] Evaluating Ensemble"
    python3.12 "$EVAL_SCRIPT" \
        --data_dir   "$DATA_DIR" \
        --json_list  "$JSON_FILE" \
        --output_dir "$ENSEMBLE_OUT" \
        2>&1 | tee "$ENSEMBLE_OUT/evaluation.log"

    # 4. Plot ensemble metrics
    echo " [4/4] Plotting Ensemble Metrics"
    python3.12 "$PLOT_METRICS_SCRIPT" \
        --csv_path   "$ENSEMBLE_OUT/per_case_evaluation_rich.csv" \
        --output_dir "$ENSEMBLE_OUT/plots"

    echo " Complete."
}


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                              EXECUTION                                 ║
# ╚════════════════════════════════════════════════════════════════════════╝

# Test run (optional, quick sanity check)
if [ "$SKIP_TEST_RUN" = false ]; then
    run_test_inference "hecktor_TEST_vault" "$HECKTOR_DATA" "$JSON_TEST"  "validation"
fi

# ── 1. CLASSIC ────────────────────────────────────────────────────────────
if [ "$SKIP_CLASSIC_TEST" = false ]; then
    run_single_inference "hecktor_TEST_vault" "$HECKTOR_DATA" "$JSON_TEST"  "validation"
fi
if [ "$SKIP_CLASSIC_TRAIN" = false ]; then
    run_single_inference "hecktor_overfit_TRAIN" "$HECKTOR_DATA" "$JSON_TRAIN" "validation"
fi
if [ "$SKIP_CLASSIC_VAL" = false ]; then
    run_single_inference "hecktor_overfit_VAL" "$HECKTOR_DATA" "$JSON_VAL"  "validation"
fi

# ── 2. K-FOLD ENSEMBLE ────────────────────────────────────────────────────
if [ "$SKIP_KFOLD_TEST" = false ]; then
    run_kfold_ensemble "hecktor_TEST_vault" "$HECKTOR_DATA" "$JSON_TEST"  "validation"
fi
if [ "$SKIP_KFOLD_TRAIN" = false ]; then
    run_kfold_ensemble "hecktor_overfit_TRAIN" "$HECKTOR_DATA" "$JSON_TRAIN" "validation"
fi
if [ "$SKIP_KFOLD_VAL" = false ]; then
    run_kfold_ensemble "hecktor_overfit_VAL" "$HECKTOR_DATA" "$JSON_VAL"  "validation"
fi

echo "Pipeline complete."
