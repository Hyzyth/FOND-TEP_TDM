#!/bin/bash
# =============================================================================
# SwinCross Inference & Rich Evaluation Master Script
# Project : ProjetMaster / StageM1_IA
# Updated : 2026-05 (NPZ pipeline)
#
# Data source  : /data/ethan/PP_hecktor_swincross_npz/    (HECKTOR NPZ)
#                /data/ethan/PP_temporal_swincross_npz/   (TemPoRAL NPZ)
# Model weights: /data/ethan/SwinCross/<MODEL_DIR>/
# Predictions  : /data/ethan/SwinCross/<INFERENCE_OUTPUT>/
#
# Two inference modes:
#
#   MODE A — Single model (original)
#     From a single training run (Mode A training).
#
#   MODE B — K-fold: per-fold inference + ensemble
#     Runs test.py for each fold model, then calls ensemble_kfold_predictions.py
#     for majority-vote fusion across k models.
#     Evaluation is run on both individual folds and the ensemble.
#
# Speed improvements vs. original
# --------------------------------
#  - test.py loads NPZ directly (no MONAI Invertd overhead).
#  - FP16 autocast always active during inference.
#  - Optional --compile flag wraps model with torch.compile.
#  - Default infer_overlap lowered to 0.5 (was 0.7) for ~3× faster inference;
#    set back to 0.7 for best quality.
# =============================================================================

set -e

# ── STEP 0 — Environment ──────────────────────────────────────────────────────
if [ ! -d "swincross_env" ]; then
    echo "swincross_env not found — run the training script first."
    exit 1
fi
source swincross_env/bin/activate
[ -f requirements.txt ] && uv pip install -r requirements.txt matplotlib seaborn pandas

# ── STEP 1 — Paths ────────────────────────────────────────────────────────────
MODEL_DIR=HECKTOR_run_1000_epoch
MODEL_USED=model_last.pth
GPU=0

HECKTOR_DATA=/data/ethan/PP_hecktor_swincross_npz
TEMPORAL_DATA=/data/ethan/PP_temporal_swincross_npz

mkdir -p /data/ethan/SwinCross
ln -sfn /data/ethan/SwinCross ./runs

# ── Banner helpers ────────────────────────────────────────────────────────────
banner_infer_heck()    { 
    echo "╔═══════════════════════════════╗"; 
    echo "║  SwinCross HECKTOR Inference  ║"; 
    echo "╚═══════════════════════════════╝"; 
    }
banner_infer_temporal(){ 
    echo "╔══════════════════════════════════════════╗"; 
    echo "║  SwinCross TemPoRAL Zero-Shot Inference  ║"; 
    echo "╚══════════════════════════════════════════╝"; 
    }
banner_eval_heck()     { 
    echo "╔════════════════════════════════╗"; 
    echo "║  SwinCross HECKTOR Evaluation  ║"; 
    echo "╚════════════════════════════════╝"; 
    }
banner_eval_temporal() { 
    echo "╔═════════════════════════════════╗"; 
    echo "║  SwinCross TemPoRAL Evaluation  ║"; 
    echo "╚═════════════════════════════════╝"; 
    }
banner_plot()          { 
    echo "╔════════════════════╗"; 
    echo "║  Metrics Plotting  ║"; 
    echo "╚════════════════════╝"; 
    }

# =============================================================================
# 1. Single-model inference: HECKTOR Inference, Evaluation & Plotting
# =============================================================================
HECKTOR_OUT=/data/ethan/SwinCross/$MODEL_DIR/hecktor_last_model_overlap05
mkdir -p $HECKTOR_OUT

banner_infer_heck
CUDA_VISIBLE_DEVICES=$GPU python3.12 npz_version/test.py \
    --pretrained_dir        ./runs/$MODEL_DIR \
    --pretrained_model_name $MODEL_USED \
    --output_dir            $HECKTOR_OUT \
    --data_dir              $HECKTOR_DATA \
    --json_list             dataset_swincross_validation.json \
    --infer_overlap 0.5 --in_channels 2 --out_channels 3 \
    --roi_x 96 --roi_y 96 --roi_z 96 \
    --workers 2 --sw_batch_size 4 --skip_existing \
    2>&1 | tee $HECKTOR_OUT/inference.log

banner_eval_heck
CUDA_VISIBLE_DEVICES=$GPU python3.12 npz_version/evaluate_predictions.py \
    --data_dir $HECKTOR_DATA \
    --json_list dataset_swincross_validation.json \
    --output_dir $HECKTOR_OUT \
    2>&1 | tee $HECKTOR_OUT/evaluation.log

banner_plot
python3.12 plot_metrics.py \
    --csv_path "$HECKTOR_OUT/per_case_evaluation_rich.csv" \
    --output_dir "$HECKTOR_OUT/plots"

# ── Optional: high-overlap pass for best quality ──────────────────────────────
# HECKTOR_OUT_07=/data/ethan/SwinCross/$MODEL_DIR/hecktor_best_model_overlap07
# mkdir -p $HECKTOR_OUT_07
# banner_infer_heck
# CUDA_VISIBLE_DEVICES=$GPU python3.12 npz_version/test.py \
#     --pretrained_dir ./runs/$MODEL_DIR \
#     --pretrained_model_name $MODEL_USED \
#     --output_dir $HECKTOR_OUT_07 \
#     --data_dir $HECKTOR_DATA \
#     --json_list dataset_swincross_validation.json \
#     --infer_overlap 0.7 --in_channels 2 --out_channels 3 \
#     --roi_x 96 --roi_y 96 --roi_z 96 \
#     --workers 2 --sw_batch_size 4 --skip_existing \
#     2>&1 | tee $HECKTOR_OUT_07/inference.log

# banner_eval_heck
# CUDA_VISIBLE_DEVICES=$GPU python3.12 npz_version/evaluate_predictions.py \
#     --data_dir $HECKTOR_DATA \
#     --json_list dataset_swincross_validation.json \
#     --output_dir $HECKTOR_OUT_07 \
#     2>&1 | tee $HECKTOR_OUT_07/evaluation.log

# banner_plot
# python3.12 plot_metrics.py \
#     --csv_path "$HECKTOR_OUT_07/per_case_evaluation_rich.csv" \
#     --output_dir "$HECKTOR_OUT_07/plots"

# ── Optional: overfit check on training data ───────────────────────────────────
HECKTOR_OUT_TRAIN=/data/ethan/SwinCross/$MODEL_DIR/hecktor_last_model_overfit_check
mkdir -p $HECKTOR_OUT_TRAIN
banner_infer_heck
CUDA_VISIBLE_DEVICES=$GPU python3.12 npz_version/test.py \
    --pretrained_dir ./runs/$MODEL_DIR \
    --pretrained_model_name $MODEL_USED \
    --output_dir $HECKTOR_OUT_TRAIN \
    --data_dir $HECKTOR_DATA \
    --json_list dataset_swincross_train.json \
    --infer_overlap 0.5 --in_channels 2 --out_channels 3 \
    --roi_x 96 --roi_y 96 --roi_z 96 \
    --workers 2 --sw_batch_size 4 --skip_existing \
    2>&1 | tee $HECKTOR_OUT_TRAIN/inference.log

banner_eval_heck
CUDA_VISIBLE_DEVICES=$GPU python3.12 npz_version/evaluate_predictions.py \
    --data_dir $HECKTOR_DATA \
    --json_list dataset_swincross_train.json \
    --output_dir $HECKTOR_OUT_TRAIN \
    2>&1 | tee $HECKTOR_OUT_TRAIN/evaluation.log

banner_plot
python3.12 plot_metrics.py \
    --csv_path "$HECKTOR_OUT_TRAIN/per_case_evaluation_rich.csv" \
    --output_dir "$HECKTOR_OUT_TRAIN/plots"


# =============================================================================
# MODE B — K-fold inference + ensemble (2025+2026 data)
# =============================================================================
# K_FOLDS=5
# EPOCH_NUMBER=1000
# BASE_MODEL_DIR=HECKTOR_kfold_${EPOCH_NUMBER}ep
# PPDATA_FOLDER=/data/ethan/PP_hecktor2026_kfold_npz
# JSON_PREFIX=dataset_swincross_2026kfold
# INFER_OVERLAP=0.5
# GPU_INFER=0
# GPU_EVAL=0

# echo "╔═══════════════════════════════════════════════════════╗"
# echo "║  SwinCross K-Fold Inference + Evaluation + Ensemble   ║"
# echo "╚═══════════════════════════════════════════════════════╝"

# FOLD_INFER_DIRS=""   # populated below for ensemble step

# # ── Per-fold inference + evaluation ──────────────────────────────────────────
# for fold in $(seq 0 $((K_FOLDS - 1))); do

#     MODEL_DIR="${BASE_MODEL_DIR}_fold${fold}"
#     JSON_LIST="${JSON_PREFIX}_fold${fold}.json"
#     FOLD_OUT=/data/ethan/SwinCross/$MODEL_DIR/hecktor_inference_overlap${INFER_OVERLAP/./}
#     mkdir -p $FOLD_OUT

#     FOLD_INFER_DIRS="$FOLD_INFER_DIRS $FOLD_OUT"

#     echo ""
#     echo "┌─ Fold ${fold} inference ─────────────────────────────────"
#     echo "│  Model : /data/ethan/SwinCross/$MODEL_DIR/model_best.pth"
#     echo "│  JSON  : $JSON_LIST"
#     echo "└──────────────────────────────────────────────────────"

#     CUDA_VISIBLE_DEVICES=$GPU_INFER python3.12 npz_version/test.py \
#         --pretrained_dir        ./runs/$MODEL_DIR \
#         --pretrained_model_name model_best.pth \
#         --output_dir            $FOLD_OUT \
#         --data_dir              $PPDATA_FOLDER \
#         --json_list             $JSON_LIST \
#         --infer_overlap $INFER_OVERLAP \
#         --in_channels 2 --out_channels 3 \
#         --roi_x 96 --roi_y 96 --roi_z 96 \
#         --workers 2 --sw_batch_size 4 --skip_existing \
#         2>&1 | tee $FOLD_OUT/inference.log

#     CUDA_VISIBLE_DEVICES=$GPU_EVAL python3.12 npz_version/evaluate_predictions.py \
#         --data_dir  $PPDATA_FOLDER \
#         --json_list $JSON_LIST \
#         --output_dir $FOLD_OUT \
#         2>&1 | tee $FOLD_OUT/evaluation.log

#     python3.12 plot_metrics.py \
#         --csv_path   "$FOLD_OUT/per_case_evaluation_rich.csv" \
#         --output_dir "$FOLD_OUT/plots"

# done

# ── Ensemble: majority vote across all k fold models ─────────────────────────
# echo ""
# echo "╔═══════════════════════════════════╗"
# echo "║  Majority-vote ensemble  (k=${K_FOLDS})  ║"
# echo "╚═══════════════════════════════════╝"

# ENSEMBLE_DIR=/data/ethan/SwinCross/${BASE_MODEL_DIR}_ensemble
# mkdir -p $ENSEMBLE_DIR

# python3.12 npz_version/ensemble_kfold_predictions.py \
#     --fold_dirs  $FOLD_INFER_DIRS \
#     --output_dir $ENSEMBLE_DIR \
#     --data_dir   $PPDATA_FOLDER \
#     --json_list  ${JSON_PREFIX}_full.json \
#     2>&1 | tee $ENSEMBLE_DIR/ensemble.log

# # Evaluate ensemble on fixed validation (full.json = all pool train + fixed val)
# CUDA_VISIBLE_DEVICES=$GPU_EVAL python3.12 npz_version/evaluate_predictions.py \
#     --data_dir  $PPDATA_FOLDER \
#     --json_list ${JSON_PREFIX}_full.json \
#     --output_dir $ENSEMBLE_DIR \
#     2>&1 | tee $ENSEMBLE_DIR/evaluation.log

# python3.12 plot_metrics.py \
#     --csv_path   "$ENSEMBLE_DIR/per_case_evaluation_rich.csv" \
#     --output_dir "$ENSEMBLE_DIR/plots"

# echo ""
# echo "  K-Fold Pipeline complete."
# echo "  Per-fold predictions + metrics : /data/ethan/SwinCross/${BASE_MODEL_DIR}_fold*/hecktor_inference*/"
# echo "  Ensemble predictions + metrics : $ENSEMBLE_DIR/"


# =============================================================================
# 2. TemPoRAL Zero-Shot Inference, Evaluation & Plotting (Works with any mode, 
#     point to desired fold or ensemble directory)
# =============================================================================
# TEMPORAL_OUT=/data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/temporal_zeroshot
# mkdir -p $TEMPORAL_OUT

# banner_infer_temporal
# CUDA_VISIBLE_DEVICES=0 python3.12 npz_version/test.py \
#     --pretrained_dir        ./runs/$MODEL_DIR \
#     --pretrained_model_name $MODEL_USED \
#     --output_dir            $TEMPORAL_OUT \
#     --data_dir              $TEMPORAL_DATA \
#     --json_list             dataset_swincross_temporal.json \
#     --infer_overlap 0.7 --in_channels 2 --out_channels 3 \
#     --roi_x 96 --roi_y 96 --roi_z 96 \
#     --workers 2 --sw_batch_size 4 --skip_existing \
#     2>&1 | tee $TEMPORAL_OUT/inference.log

# banner_eval_temporal
# CUDA_VISIBLE_DEVICES=0 python3.12 npz_version/evaluate_predictions.py \
#     --data_dir $TEMPORAL_DATA \
#     --json_list dataset_swincross_temporal.json \
#     --output_dir $TEMPORAL_OUT \
#     2>&1 | tee $TEMPORAL_OUT/evaluation.log

# banner_plot
# python3.12 plot_metrics.py \
#     --csv_path "$TEMPORAL_OUT/per_case_evaluation_rich.csv" \
#     --output_dir "$TEMPORAL_OUT/plots"

# =============================================================================
# 3. Single-model inference: Temporal per-timepoint stratification & per-stratum plots
# =============================================================================
# RICH_CSV="$TEMPORAL_OUT/per_case_evaluation_rich.csv"
# if [ -f "$RICH_CSV" ]; then
#     echo "=== Timepoint sub-reports ==="
#     HEADER=$(head -1 "$RICH_CSV")
#     TP_COL=$(head -1 "$RICH_CSV" | tr ',' '\n' | grep -n "^timepoint$" | cut -d: -f1)

#     if [ -n "$TP_COL" ]; then
#         UNIQUE_TPS=$(tail -n+2 "$RICH_CSV" \
#             | awk -F',' -v col="$TP_COL" '{print $col}' | sort -u | grep -v '^$')
#         for tp in $UNIQUE_TPS; do
#             SUB_CSV="$TEMPORAL_OUT/per_case_rich_${tp}.csv"
#             {
#                 echo "$HEADER"
#                 grep -v "^case_id" "$RICH_CSV" \
#                     | awk -F',' -v col="$TP_COL" -v tp="$tp" '$col == tp'
#             } > "$SUB_CSV"
#             N=$(tail -n+2 "$SUB_CSV" | wc -l)
#             echo "  $tp: $N cases → $SUB_CSV"
#             python3.12 plot_metrics.py \
#                 --csv_path   "$SUB_CSV" \
#                 --output_dir "$TEMPORAL_OUT/plots_${tp}"
#         done
#     fi
# fi



echo "Pipeline complete."
