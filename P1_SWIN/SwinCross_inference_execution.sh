#!/bin/bash
# =============================================================================
# SwinCross Inference Script — SwinUNETR Cross-Modality Fusion
# Project : ProjetMaster / StageM1_IA
# Author  : Santiago (original), updated for Ethan's run
# Updated : 2026-04
#
# Uses test.py — the MONAI Invertd + SimpleITK CopyInformation approach.
# This is the most spatially correct output method: transforms are inverted
# by MONAI, then the original LPS metadata is copied from the source SITK file.
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

# =============================================================================
# STEP 1 — Path configuration
# =============================================================================

PPDATA_FOLDER=/data/santiago/Datast001_HECKTOR_SwinCross/

# Which model run to use for inference
MODEL_DIR=ethan_hecktor_2000ep_run

# Re-apply the symlink in case this script is run in a fresh shell
mkdir -p /data/ethan/SwinCross/hecktor_runs
ln -sfn /data/ethan/SwinCross/hecktor_runs ./runs

# Output folder for predicted segmentation masks
INFERENCE_OUTPUT=/data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/hecktor_2000ep_predictions
mkdir -p $INFERENCE_OUTPUT

echo "Model dir : /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/"
echo "Output    : $INFERENCE_OUTPUT/"
echo "GPU       : $CUDA_VISIBLE_DEVICES"

# =============================================================================
# STEP 2A — Inference on validation set using BEST model  [ACTIVE]
#
# --infer_overlap 0.5 : good accuracy / speed tradeoff; raise to 0.7 for finals
# --json_list         : uses the same JSON as training (validation split)
# To run on a dedicated held-out test set, point --json_list to
#   dataset_swincross_testing_group.json (build it with test_or_inf_dataset_builer_spitk.py)
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 python3.12 test.py \
#     --pretrained_dir        ./runs/$MODEL_DIR \
#     --pretrained_model_name model_best.pth \
#     --output_dir            $INFERENCE_OUTPUT/best_model \
#     --data_dir              $PPDATA_FOLDER \
#     --json_list             dataset_swincross.json \
#     --infer_overlap         0.5 \
#     --in_channels           2 \
#     --out_channels          3 \
#     --roi_x 96 --roi_y 96 --roi_z 96 \
#     --workers               4 \
#     2>&1 | tee /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/inference_best.log

# =============================================================================
# STEP 2B — Inference using LAST model  [UNCOMMENT IF NEEDED]
# Useful to compare last vs best when training is still converging.
# =============================================================================
# CUDA_VISIBLE_DEVICES=0 python3.12 test.py \
#     --pretrained_dir        ./runs/$MODEL_DIR \
#     --pretrained_model_name model_last.pth \
#     --output_dir            $INFERENCE_OUTPUT/last_model \
#     --data_dir              $PPDATA_FOLDER \
#     --json_list             dataset_swincross.json \
#     --infer_overlap         0.5 \
#     --in_channels           2 \
#     --out_channels          3 \
#     --roi_x 96 --roi_y 96 --roi_z 96 \
#     --workers               4 \
#     2>&1 | tee /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/inference_last.log

# =============================================================================
# STEP 2C — Inference on dedicated test set
# Requires dataset_swincross_testing_group.json built with:
#   python3.12 test_or_inf_dataset_builer_spitk.py \
#       --input_folder /data/santiago/Datast001_HECKTOR_SwinCross/ \
#       --json_name dataset_swincross_testing_group.json \
# =============================================================================
CUDA_VISIBLE_DEVICES=0 python3.12 test.py \
    --pretrained_dir        ./runs/$MODEL_DIR \
    --pretrained_model_name model_best.pth \
    --output_dir            $INFERENCE_OUTPUT/test_set_inference \
    --data_dir              $PPDATA_FOLDER \
    --json_list             dataset_swincross_testing_group.json \
    --infer_overlap         0.7 \
    --in_channels           2 \
    --out_channels          3 \
    --roi_x 96 --roi_y 96 --roi_z 96 \
    --workers               4 \
    2>&1 | tee /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/inference_testset.log

# =============================================================================
# STEP 2Cbis — Inference on dedicated test set with inference only (no dice, no metrics)
# Requires dataset_swincross_testing_group.json built with:
#   python3.12 test_or_inf_dataset_builer_spitk.py \
#       --input_folder /data/santiago/Datast001_HECKTOR_SwinCross/ \
#       --json_name dataset_swincross_testing_group.json \
#       --inference_only
# =============================================================================
# CUDA_VISIBLE_DEVICES=0 python3.12 test.py \
#     --pretrained_dir        ./runs/$MODEL_DIR \
#     --pretrained_model_name model_best.pth \
#     --output_dir            $INFERENCE_OUTPUT/test_set_inference_only \
#     --data_dir              $PPDATA_FOLDER \
#     --json_list             dataset_swincross_testing_group.json \
#     --infer_overlap         0.7 \
#     --in_channels           2 \
#     --out_channels          3 \
#     --roi_x 96 --roi_y 96 --roi_z 96 \
#     --workers               4 \
#     --inference_only         \
#     2>&1 | tee /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/inference_only_testset.log

# =============================================================================
# STEP 2D — High-overlap inference (more accurate, slower)  [UNCOMMENT IF NEEDED]
# Use for final results / paper figures.
# =============================================================================
# CUDA_VISIBLE_DEVICES=0 python3.12 test.py \
#     --pretrained_dir        ./runs/$MODEL_DIR \
#     --pretrained_model_name model_best.pth \
#     --output_dir            $INFERENCE_OUTPUT/best_model_overlap07 \
#     --data_dir              $PPDATA_FOLDER \
#     --json_list             dataset_swincross.json \
#     --infer_overlap         0.7 \
#     --in_channels           2 \
#     --out_channels          3 \
#     --roi_x 96 --roi_y 96 --roi_z 96 \
#     --workers               4 \
#     2>&1 | tee /data/ethan/SwinCross/hecktor_runs/$MODEL_DIR/inference_best_overlap07.log
