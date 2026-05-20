#!/bin/bash
# =============================================================================
# SwinCross Inference & Rich Evaluation Master Script
# Project : ProjetMaster / StageM1_IA
# Author  : Santiago (original), updated for Ethan's run
# Updated : 2026-04
#
# Data source  : /data/santiago/Datast001_HECKTOR_SwinCross/  (read-only)
# Model weights: /data/ethan/SwinCross/hecktor_runs/<MODEL_DIR>/
# Predictions  : /data/ethan/SwinCross/<INFERENCE_OUTPUT>/
# =============================================================================

set -e

# =============================================================================
# STEP 0 — Environment (assumes training script already ran env setup)
# =============================================================================
if [ ! -d "swincross_env" ]; then
    echo "swincross_env not found. Run the training script first to build it."
    exit 1
fi
source swincross_env/bin/activate

if [ -f requirements.txt ]; then
    uv pip install -r requirements.txt
    uv pip install matplotlib seaborn pandas # Ensure plotting libs are installed
else
    echo "requirements.txt not found! Aborting."
    exit 1
fi

# =============================================================================
# STEP 1 — Path configuration
# =============================================================================

MODEL_DIR=ethan_hecktor_2000ep_run
MODEL_USED=backup_checkpoints/model_280ep_dice0.6235_slim.pth
HECKTOR_DATA=/data/santiago/Datast001_HECKTOR_SwinCross/
TEMPORAL_DATA=/data/ethan/PP_temporal_dataset_SwinCross/

# Re-apply the symlink in case this script is run in a fresh shell
mkdir -p /data/ethan/SwinCross/hecktor_runs
ln -sfn /data/ethan/SwinCross/hecktor_runs ./runs

# Output folder for predicted segmentation masks
INFERENCE_OUTPUT=/data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/hecktor_2000ep_predictions
mkdir -p $INFERENCE_OUTPUT

#--- Shared info printout ---
run_echo_infer_hecktor(){
    echo "╔═══════════════════════════════╗"
    echo "║  SwinCross HECKTOR Inference  ║"
    echo "╚═══════════════════════════════╝"
}

run_echo_infer_temporal(){
    echo "╔══════════════════════════════════════════╗"
    echo "║  SwinCross TemPoRAL Zero-Shot Inference  ║"
    echo "╚══════════════════════════════════════════╝"
}

run_echo_val_hecktor(){
    echo "╔════════════════════════════════╗"
    echo "║  SwinCross HECKTOR Evaluation  ║"
    echo "╚════════════════════════════════╝"
}

run_echo_val_temporal(){
    echo "╔═════════════════════════════════╗"
    echo "║  SwinCross TemPoRAL Evaluation  ║"
    echo "╚═════════════════════════════════╝"
}

run_echo_plotting(){
    echo "╔════════════════════╗"
    echo "║  Metrics Plotting  ║"
    echo "╚════════════════════╝"
}

# =============================================================================
# 0. — Inference on dedicated test set
# Requires dataset_swincross_testing_group.json built with:
#   python3.12 test_or_inf_dataset_builer_spitk.py \
#       --input_folder HECKTOR_DATA \
#       --json_name dataset_swincross_testing_group.json \
# =============================================================================
# HECKTOR_OUT=/data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/inference_testset
# mkdir -p $HECKTOR_OUT
# run_echo_infer_hecktor
# CUDA_VISIBLE_DEVICES=0 python3.12 test.py \
#     --pretrained_dir        ./runs/$MODEL_DIR \
#     --pretrained_model_name $MODEL_USED \
#     --output_dir            $HECKTOR_OUT \
#     --data_dir              $HECKTOR_DATA \
#     --json_list             dataset_swincross_testing_group.json \
#     --infer_overlap 0.7 --in_channels 2 --out_channels 3 \
#     --roi_x 96 --roi_y 96 --roi_z 96 --workers 2 --skip_existing \
#     2>&1 | tee $HECKTOR_OUT/inference_testset.log

# run_echo_val_hecktor
# CUDA_VISIBLE_DEVICES=0 python3.12 evaluate_predictions.py \
#     --data_dir $HECKTOR_DATA \
#     --json_list dataset_swincross_testing_group.json \
#     --output_dir $HECKTOR_OUT \
#     2>&1 | tee $HECKTOR_OUT/evaluation.log

# run_echo_plotting
# python3.12 plot_metrics.py \
#     --csv_path "$HECKTOR_OUT/per_case_evaluation_rich.csv" \
#     --output_dir "$HECKTOR_OUT/plots"

# =============================================================================
# 1. HECKTOR Inference, Evaluation & Plotting (High Overlap)
# =============================================================================
HECKTOR_OUT=/data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/hecktor_best_model_overlap07
mkdir -p $HECKTOR_OUT

run_echo_infer_hecktor
CUDA_VISIBLE_DEVICES=0 python3.12 test.py \
    --pretrained_dir ./runs/$MODEL_DIR \
    --pretrained_model_name $MODEL_USED \
    --output_dir $HECKTOR_OUT \
    --data_dir $HECKTOR_DATA \
    --json_list dataset_swincross.json \
    --infer_overlap 0.7 --in_channels 2 --out_channels 3 \
    --roi_x 96 --roi_y 96 --roi_z 96 --workers 2 --skip_existing \
    2>&1 | tee $HECKTOR_OUT/inference.log

run_echo_val_hecktor
CUDA_VISIBLE_DEVICES=0 python3.12 evaluate_predictions.py \
    --data_dir $HECKTOR_DATA \
    --json_list dataset_swincross.json \
    --output_dir $HECKTOR_OUT \
    2>&1 | tee $HECKTOR_OUT/evaluation.log

run_echo_plotting
python3.12 plot_metrics.py \
    --csv_path "$HECKTOR_OUT/per_case_evaluation_rich.csv" \
    --output_dir "$HECKTOR_OUT/plots"

# =============================================================================
# 2. TEMPORAL Zero-Shot Inference, Evaluation & Plotting
# =============================================================================
TEMPORAL_OUT=/data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/temporal_zeroshot
mkdir -p $TEMPORAL_OUT

run_echo_infer_temporal
CUDA_VISIBLE_DEVICES=0 python3.12 test.py \
    --pretrained_dir ./runs/$MODEL_DIR \
    --pretrained_model_name $MODEL_USED \
    --output_dir $TEMPORAL_OUT \
    --data_dir $TEMPORAL_DATA \
    --json_list dataset_swincross_temporal.json \
    --infer_overlap 0.7 --in_channels 2 --out_channels 3 \
    --roi_x 96 --roi_y 96 --roi_z 96 --workers 2 --skip_existing \
    2>&1 | tee $TEMPORAL_OUT/inference.log

run_echo_val_temporal
CUDA_VISIBLE_DEVICES=0 python3.12 evaluate_predictions.py \
    --data_dir $TEMPORAL_DATA \
    --json_list dataset_swincross_temporal.json \
    --output_dir $TEMPORAL_OUT \
    2>&1 | tee $TEMPORAL_OUT/evaluation.log

run_echo_plotting
python3.12 plot_metrics.py \
    --csv_path "$TEMPORAL_OUT/per_case_evaluation_rich.csv" \
    --output_dir "$TEMPORAL_OUT/plots"

# =============================================================================
# Automated Metric Stratification via AWK + Plotting per stratum
# =============================================================================
RICH_CSV="$TEMPORAL_OUT/per_case_evaluation_rich.csv"
if [ -f "$RICH_CSV" ]; then
    echo "=== Generating Timepoint Sub-reports & Plots ==="
    HEADER=$(head -1 "$RICH_CSV")
    TP_COL=$(head -1 "$RICH_CSV" | tr ',' '\n' | grep -n "^timepoint$" | cut -d: -f1)

    if [ -n "$TP_COL" ]; then
        UNIQUE_TPS=$(tail -n+2 "$RICH_CSV" | awk -F',' -v col="$TP_COL" '{print $col}' | sort -u | grep -v '^$')
        
        for tp in $UNIQUE_TPS; do
            SUB_CSV="$TEMPORAL_OUT/per_case_rich_${tp}.csv"
            {
                echo "$HEADER"
                grep -v "^case_id" "$RICH_CSV" | awk -F',' -v col="$TP_COL" -v tp="$tp" '$col == tp'
            } > "$SUB_CSV"
            N=$(tail -n+2 "$SUB_CSV" | wc -l)
            echo "  Generated $tp report: $SUB_CSV ($N cases)"
            
            # Plot the subset dynamically!
            echo "  Plotting sub-report: $tp"
            python3.12 plot_metrics.py \
                --csv_path "$SUB_CSV" \
                --output_dir "$TEMPORAL_OUT/plots_${tp}"
        done
    fi
fi

echo "Pipeline Complete."
