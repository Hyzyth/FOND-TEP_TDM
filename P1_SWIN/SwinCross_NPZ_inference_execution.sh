#!/bin/bash
# =============================================================================
# SwinCross Inference & Rich Evaluation Master Script
# Author : Ethan
# Project : ProjetMaster / StageM1_IA
# Updated : 2026-05 (NPZ pipeline)
#
# Data source  : /data/ethan/PP_hecktor2026_kfold_npz/  (HECKTOR NPZ)
#                /data/ethan/PP_temporal_swincross_npz/ (TemPoRAL NPZ)
#
# Modes: Classic Single Model & K-Fold Ensemble
# Targets: Test Vault, Train Pool, Val Pool, TemPoRAL Zero-Shot
# =============================================================================

set -e

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                          GLOBAL CONFIGURATION                          ║
# ╚════════════════════════════════════════════════════════════════════════╝
# ── 1. Execution Toggles (Set to true to skip, false to run) ───────────────
SKIP_CLASSIC_TEST=false
SKIP_CLASSIC_TRAIN=false
SKIP_CLASSIC_VAL=false

SKIP_KFOLD_TEST=true
SKIP_KFOLD_TRAIN=true
SKIP_KFOLD_VAL=true

SKIP_TEMPORAL_CLASSIC=false
SKIP_TEMPORAL_KFOLD=true

# ── 2. Hardware & Parameters ───────────────────────────────────────────────
GPU=0
INFER_OVERLAP=0.7   # 0.5 for speed, 0.7 for max quality
SW_BATCH=4

# ── 3. Data Paths & JSONs ──────────────────────────────────────────────────
HECKTOR_DATA="/data/ethan/PP_hecktor2026_kfold_npz"
TEMPORAL_DATA="/data/ethan/PP_temporal_npz"

JSON_TEST="dataset_swincross_2026kfold_test.json"
JSON_TRAIN="dataset_swincross_2026kfold_classic_train.json"
JSON_VAL="dataset_swincross_2026kfold_classic_val.json"
JSON_TEMPORAL="dataset_temporal.json"

# ── 4. Model Setup ─────────────────────────────────────────────────────────
# Classic Mode 
CLASSIC_MODEL_DIR="HECKTOR_run_1500_epoch"
CLASSIC_WEIGHTS="model_best.pth"

# K-Fold Mode
KFOLD_BASE_DIR="HECKTOR_kfold_400ep"
K_FOLDS=5

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                     ENVIRONMENT & HELPER FUNCTIONS                     ║
# ╚════════════════════════════════════════════════════════════════════════╝
if [ ! -d "swincross_env" ]; then
    echo "swincross_env not found — run the training script first."
    exit 1
fi
source swincross_env/bin/activate
[ -f requirements.txt ] && uv pip install -r requirements.txt matplotlib seaborn pandas

mkdir -p /data/ethan/SwinCross
ln -sfn /data/ethan/SwinCross ./runs


# ── Helper: Run Single Model Inference + Eval + Plot ───────────────────────
run_single_inference() {
    local TARGET_NAME=$1
    local DATA_DIR=$2
    local JSON_FILE=$3
    
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ CLASSIC MODE : $TARGET_NAME"
    echo "╚════════════════════════════════════════════════════════════╝"
    
    local OUT_DIR="/data/ethan/SwinCross/${CLASSIC_MODEL_DIR}/${TARGET_NAME}"
    mkdir -p "$OUT_DIR"

    echo " [1/3] Running Inference with model: $CLASSIC_MODEL_DIR/$CLASSIC_WEIGHTS."
    CUDA_VISIBLE_DEVICES=$GPU python3.12 adaptation/test.py \
        --pretrained_dir "./runs/$CLASSIC_MODEL_DIR" \
        --pretrained_model_name "$CLASSIC_WEIGHTS" \
        --output_dir "$OUT_DIR" \
        --data_dir "$DATA_DIR" \
        --json_list "$JSON_FILE" \
        --infer_overlap $INFER_OVERLAP --in_channels 2 --out_channels 3 \
        --roi_x 96 --roi_y 96 --roi_z 96 \
        --workers 2 --sw_batch_size $SW_BATCH --skip_existing \
        2>&1 | tee "$OUT_DIR/inference.log"

    echo " [2/3] Running Evaluation."
    CUDA_VISIBLE_DEVICES=$GPU python3.12 adaptation/evaluate_predictions.py \
        --data_dir "$DATA_DIR" \
        --json_list "$JSON_FILE" \
        --output_dir "$OUT_DIR" \
        2>&1 | tee "$OUT_DIR/evaluation.log"

    echo " [3/3] Generating Metrics & Post-Processing Plots"
    python3.12 adaptation/plot_metrics.py \
        --csv_path "$OUT_DIR/per_case_evaluation_rich.csv" \
        --output_dir "$OUT_DIR/plots"
    
    if [ -f "$OUT_DIR/postprocessing_logs.csv" ]; then
        python3.12 adaptation/plot_postprocessing.py \
            --csv_path "$OUT_DIR/postprocessing_logs.csv" \
            --output_dir "$OUT_DIR/plots"
    fi
    echo " Complete."
}

# ── Helper: Run K-Fold Ensemble Inference + Eval + Plot ────────────────────
run_kfold_ensemble() {
    local TARGET_NAME=$1
    local DATA_DIR=$2
    local JSON_FILE=$3

    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║ K-FOLD ENSEMBLE : $TARGET_NAME"
    echo "╚════════════════════════════════════════════════════════════╝"
    
    local ENSEMBLE_OUT="/data/ethan/SwinCross/${KFOLD_BASE_DIR}_ensemble/${TARGET_NAME}"
    mkdir -p "$ENSEMBLE_OUT"
    local FOLD_DIRS=""

    # 1. Generate predictions for each fold
    echo " [1/4] Running Per-Fold Inference."
    for fold in $(seq 0 $((K_FOLDS - 1))); do
        local FOLD_MODEL_DIR="${KFOLD_BASE_DIR}_fold${fold}"
        local FOLD_OUT="/data/ethan/SwinCross/${FOLD_MODEL_DIR}/${TARGET_NAME}"
        mkdir -p "$FOLD_OUT"
        FOLD_DIRS="$FOLD_DIRS $FOLD_OUT"

        echo "      ↳ Inferring Fold $fold."
        CUDA_VISIBLE_DEVICES=$GPU python3.12 adaptation/test.py \
            --pretrained_dir "./runs/$FOLD_MODEL_DIR" \
            --pretrained_model_name "model_best.pth" \
            --output_dir "$FOLD_OUT" \
            --data_dir "$DATA_DIR" \
            --json_list "$JSON_FILE" \
            --infer_overlap $INFER_OVERLAP --in_channels 2 --out_channels 3 \
            --roi_x 96 --roi_y 96 --roi_z 96 \
            --workers 2 --sw_batch_size $SW_BATCH --skip_existing \
            2>&1 | tee "$FOLD_OUT/inference.log"
        
        # Generate post-processing plots for this specific fold if the log exists
        if [ -f "$FOLD_OUT/postprocessing_logs.csv" ]; then
            python3.12 adaptation/plot_postprocessing.py \
                --csv_path "$FOLD_OUT/postprocessing_logs.csv" \
                --output_dir "$FOLD_OUT/plots" > /dev/null 2>&1
        fi
    done

    # 2. Ensemble majority vote
    echo " [2/4] Running Majority-Vote Ensemble."
    python3.12 adaptation/ensemble_kfold_predictions.py \
        --fold_dirs $FOLD_DIRS \
        --output_dir "$ENSEMBLE_OUT" \
        --data_dir "$DATA_DIR" \
        --json_list "$JSON_FILE" \
        2>&1 | tee "$ENSEMBLE_OUT/ensemble.log"

    # 3. Evaluate Ensemble
    echo " [3/4] Evaluating Ensemble."
    CUDA_VISIBLE_DEVICES=$GPU python3.12 adaptation/evaluate_predictions.py \
        --data_dir "$DATA_DIR" \
        --json_list "$JSON_FILE" \
        --output_dir "$ENSEMBLE_OUT" \
        2>&1 | tee "$ENSEMBLE_OUT/evaluation.log"
    
    # 4. Plot Ensemble Metrics
    echo " [4/4] Generating Ensemble Plots."
    python3.12 adaptation/plot_metrics.py \
        --csv_path "$ENSEMBLE_OUT/per_case_evaluation_rich.csv" \
        --output_dir "$ENSEMBLE_OUT/plots"
    
    # If the ensemble script generates its own post-processing logs, plot them
    if [ -f "$ENSEMBLE_OUT/postprocessing_logs.csv" ]; then
        python3.12 adaptation/plot_postprocessing.py \
            --csv_path "$ENSEMBLE_OUT/postprocessing_logs.csv" \
            --output_dir "$ENSEMBLE_OUT/plots"
    fi
    echo " Complete."
}

# ── Helper: TemPoRAL Stratification & Plots ────────────────────────────────
generate_temporal_sub_reports() {
    local TARGET_DIR=$1
    local RICH_CSV="$TARGET_DIR/per_case_evaluation_rich.csv"

    if [ -f "$RICH_CSV" ]; then
        echo " [Extra] Generating Timepoint Sub-reports for TemPoRAL..."
        local HEADER=$(head -1 "$RICH_CSV")
        local TP_COL=$(head -1 "$RICH_CSV" | tr ',' '\n' | grep -n "^timepoint$" | cut -d: -f1)

        if [ -n "$TP_COL" ]; then
            local UNIQUE_TPS=$(tail -n+2 "$RICH_CSV" | awk -F',' -v col="$TP_COL" '{print $col}' | sort -u | grep -v '^$')
            
            for tp in $UNIQUE_TPS; do
                local SUB_CSV="$TARGET_DIR/per_case_rich_${tp}.csv"
                {
                    echo "$HEADER"
                    grep -v "^case_id" "$RICH_CSV" | awk -F',' -v col="$TP_COL" -v tp="$tp" '$col == tp'
                } > "$SUB_CSV"
                
                local N=$(tail -n+2 "$SUB_CSV" | wc -l)
                echo "  ↳ $tp: $N cases → $SUB_CSV"
                
                # We pipe plotting output to /dev/null to avoid cluttering the terminal
                python3.12 adaptation/plot_metrics.py \
                    --csv_path   "$SUB_CSV" \
                    --output_dir "$TARGET_DIR/plots_${tp}" > /dev/null 2>&1
            done
            echo " Sub-reports Complete."
        fi
    fi
    echo ""
}

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                              EXECUTION                                 ║
# ╚════════════════════════════════════════════════════════════════════════╝
# ── 1. CLASSIC HECKTOR ────────────────────────────────────────────────────
if [ "$SKIP_CLASSIC_TEST" = false ]; then
    run_single_inference "hecktor_TEST_vault" $HECKTOR_DATA $JSON_TEST
fi

if [ "$SKIP_CLASSIC_TRAIN" = false ]; then
    run_single_inference "hecktor_overfit_TRAIN" $HECKTOR_DATA $JSON_TRAIN
fi

if [ "$SKIP_CLASSIC_VAL" = false ]; then
    run_single_inference "hecktor_overfit_VAL" $HECKTOR_DATA $JSON_VAL
fi

# ── 2. K-FOLD ENSEMBLE HECKTOR ────────────────────────────────────────────
if [ "$SKIP_KFOLD_TEST" = false ]; then
    run_kfold_ensemble "hecktor_TEST_vault" $HECKTOR_DATA $JSON_TEST
fi

if [ "$SKIP_KFOLD_TRAIN" = false ]; then
    # Checks if the ensemble overfits to the pool it trained on
    run_kfold_ensemble "hecktor_overfit_TRAIN" $HECKTOR_DATA $JSON_TRAIN
fi

if [ "$SKIP_KFOLD_VAL" = false ]; then
    run_kfold_ensemble "hecktor_overfit_VAL" $HECKTOR_DATA $JSON_VAL
fi

# ── 3. TEMPORAL ZERO-SHOT ─────────────────────────────────────────────────
if [ "$SKIP_TEMPORAL_CLASSIC" = false ]; then
    run_single_inference "temporal_zeroshot" $TEMPORAL_DATA $JSON_TEMPORAL
    generate_temporal_sub_reports "/data/ethan/SwinCross/${CLASSIC_MODEL_DIR}/temporal_zeroshot"
fi

if [ "$SKIP_TEMPORAL_KFOLD" = false ]; then
    run_kfold_ensemble "temporal_zeroshot" $TEMPORAL_DATA $JSON_TEMPORAL
    generate_temporal_sub_reports "/data/ethan/SwinCross/${KFOLD_BASE_DIR}_ensemble/temporal_zeroshot"
fi

echo "Pipeline complete."
