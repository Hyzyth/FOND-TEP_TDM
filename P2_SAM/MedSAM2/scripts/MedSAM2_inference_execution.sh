#!/bin/bash
# =============================================================================
# MedSAM2 Inference & Evaluation Script — HECKTOR Task-1 GTVp/GTVn
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-04
#
# Runs MedSAM2 inference on HECKTOR NPZ files then evaluates DSC.
#
# Checkpoint   : ./checkpoints/MedSAM2_latest.pt   (from training or download)
# NPZ input    : /data/ethan/MedSAM2/hecktor_npz/val/
# Predictions  : /data/ethan/MedSAM2/predictions/<PRED_DIR>/
# Evaluation   : /data/ethan/MedSAM2/predictions/<PRED_DIR>/dsc_results.csv
#
# Uses GT bounding boxes as prompts (oracle mode — for development / sanity check).
# =============================================================================

set -e

# =============================================================================
# STEP 0 — Environment
# =============================================================================

if [ ! -d "medsam2_env" ]; then
    echo "medsam2_env not found. Run MedSAM2_dataset_building.sh first."
    exit 1
fi
source medsam2_env/bin/activate

# =============================================================================
# STEP 1 — Path configuration
# =============================================================================

NPZ_VAL=/data/ethan/MedSAM2/hecktor_npz/val
NPZ_TRAIN=/data/ethan/MedSAM2/hecktor_npz/train

CHECKPOINT=./checkpoints/MedSAM2_latest.pt
CFG=sam2/configs/sam2.1_hiera_tiny_hecktor.yaml

PRED_DIR=ethan_hecktor_hiera_tiny
PRED_ROOT=/data/ethan/MedSAM2/predictions/$PRED_DIR

mkdir -p "$PRED_ROOT"

# Re-apply symlink in case this is a fresh shell
mkdir -p /data/ethan/MedSAM2/runs
ln -sfn /data/ethan/MedSAM2/runs ./runs

echo "========================================"
echo "  MedSAM2 × HECKTOR — Inference"
echo "  Checkpoint : $CHECKPOINT"
echo "  GPU        : $CUDA_VISIBLE_DEVICES"
echo "========================================"

# =============================================================================
# STEP 2A — Inference on validation set using best checkpoint  [ACTIVE]
#
# --bbox_shift 5  : add a 5-voxel margin around GT bounding boxes (oracle mode)
# --save_nifti    : write predicted masks as .nii.gz for visual QC in ITK-SNAP
# --save_overlays : write axial PNG overlays (CT + GTVp/GTVn colour overlay)
# =============================================================================

CUDA_VISIBLE_DEVICES=0 python3.12 inference/infer_hecktor.py \
    --checkpoint    "$CHECKPOINT" \
    --cfg           "$CFG" \
    --imgs_path     "$NPZ_VAL" \
    --pred_save_dir "$PRED_ROOT/val_best" \
    --bbox_shift    5 \
    --save_nifti \
    --save_overlays \
    2>&1 | tee "$PRED_ROOT/inference_val_best.log"

# =============================================================================
# STEP 2B — Inference on training set (overfit check)  [UNCOMMENT IF NEEDED]
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 python3.12 inference/infer_hecktor.py \
#     --checkpoint    "$CHECKPOINT" \
#     --cfg           "$CFG" \
#     --imgs_path     "$NPZ_TRAIN" \
#     --pred_save_dir "$PRED_ROOT/train_best" \
#     --bbox_shift    5 \
#     --save_overlays \
#     2>&1 | tee "$PRED_ROOT/inference_train_best.log"

# =============================================================================
# STEP 2C — High-overlap inference on val (more accurate, slower)
#           Increase --bbox_shift to 0 to use tight GT boxes for a ceiling estimate.
#           [UNCOMMENT FOR FINAL RESULTS]
# =============================================================================

# CUDA_VISIBLE_DEVICES=0 python3.12 inference/infer_hecktor.py \
#     --checkpoint    "$CHECKPOINT" \
#     --cfg           "$CFG" \
#     --imgs_path     "$NPZ_VAL" \
#     --pred_save_dir "$PRED_ROOT/val_best_tightbox" \
#     --bbox_shift    0 \
#     --save_nifti \
#     --save_overlays \
#     2>&1 | tee "$PRED_ROOT/inference_val_tightbox.log"

# =============================================================================
# STEP 3 — Evaluate predicted masks against GT (DSC per patient + mean)
#
# Writes dsc_results.csv with columns:
#   patient | dsc_gtvp | dsc_gtvn | dsc_overall
# =============================================================================

echo ""
echo "========================================"
echo "  MedSAM2 × HECKTOR — Evaluation"
echo "========================================"

python3.12 inference/evaluate_hecktor.py \
    --pred_dir "$PRED_ROOT/val_best" \
    --gt_dir   "$NPZ_VAL" \
    --output   "$PRED_ROOT/val_best/dsc_results.csv" \
    2>&1 | tee "$PRED_ROOT/evaluation_val_best.log"

# Uncomment to also evaluate the train-set run (overfit check):
# python3.12 inference/evaluate_hecktor.py \
#     --pred_dir "$PRED_ROOT/train_best" \
#     --gt_dir   "$NPZ_TRAIN" \
#     --output   "$PRED_ROOT/train_best/dsc_results.csv" \
#     2>&1 | tee "$PRED_ROOT/evaluation_train_best.log"

echo ""
echo "Results: $PRED_ROOT/val_best/dsc_results.csv"
