#!/bin/bash
# =============================================================================
# MedSAM2 Dataset Building Script - SwinCross NPZ Bridge
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-05
#
# MedSAM2 now reads the SAME SwinCross-format NPZ files used by SwinCross
# and DualwaveSAM - no extra conversion step is needed.
#
# Prerequisites
# -------------
#   Run P1_SWIN/SwinCross_NPZ_Dataset_Building.sh with BUILD_HECKTOR_2026_KFOLD=true
#   BEFORE this script. 
#
# This script's responsibilities
# --------------------------------
#   1. Verify the SwinCross NPZ root exists.
#   2. Set up symlinks for the MedSAM2 checkpoint and ./runs directories.
#   3. Prepare the TEMPORAL dataset (zero-shot inference only).
#
# Training split used by MedSAM2
# --------------------------------
#   JSON : dataset_swincross_2026kfold_classic.json
#     training   -> used for model training
#     validation -> used for epoch-level Dice monitoring (fold 0)
#
# Test vault (inference only, never seen during training)
# --------------------------------------------------------
#   JSON : dataset_swincross_2026kfold_test.json
#     validation -> cases to run inference on
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
# STEP 1 - SwinCross NPZ root verification
# =============================================================================

SWINCROSS_NPZ_ROOT="/data/ethan/PP_hecktor2026_kfold_npz"
CLASSIC_JSON="${SWINCROSS_NPZ_ROOT}/dataset_swincross_2026kfold_classic.json"
TEST_JSON="${SWINCROSS_NPZ_ROOT}/dataset_swincross_2026kfold_test.json"

echo "========================================"
echo "  MedSAM2 - Dataset Setup"
echo "  SwinCross NPZ root : $SWINCROSS_NPZ_ROOT"
echo "========================================"

if [ ! -d "$SWINCROSS_NPZ_ROOT" ]; then
    echo ""
    echo "  SwinCross NPZ root not found: $SWINCROSS_NPZ_ROOT"
    echo ""
    echo "  Run P1_SWIN/SwinCross_NPZ_Dataset_Building.sh first with:"
    echo "    BUILD_HECKTOR_2026_KFOLD=true"
    echo ""
    exit 1
fi

for f in "$CLASSIC_JSON" "$TEST_JSON"; do
    if [ ! -f "$f" ]; then
        echo "  Missing JSON: $f"
        echo "  Re-run SwinCross_NPZ_Dataset_Building.sh to regenerate."
        exit 1
    fi
done

N_TRAIN=$(python3.10 -c "
import json
with open('${CLASSIC_JSON}') as f: d=json.load(f)
print(len(d.get('training', [])))
")
N_VAL=$(python3.10 -c "
import json
with open('${CLASSIC_JSON}') as f: d=json.load(f)
print(len(d.get('validation', [])))
")
N_TEST=$(python3.10 -c "
import json
with open('${TEST_JSON}') as f: d=json.load(f)
print(len(d.get('validation', [])))
")

echo ""
echo "  SwinCross NPZ root verified."
echo "     Classic train : $N_TRAIN cases"
echo "     Classic val   : $N_VAL cases"
echo "     Test vault    : $N_TEST cases (locked - never seen during training)"
echo ""

# =============================================================================
# STEP 2 - Checkpoint and runs directories
# =============================================================================

CKPT_DIR=/data/ethan/MedSAM2/checkpoints
mkdir -p "$CKPT_DIR"
ln -sfn "$CKPT_DIR" ./checkpoints

mkdir -p /data/ethan/MedSAM2/runs
ln -sfn /data/ethan/MedSAM2/runs ./runs

echo "  Symlinks ready:"
echo "     ./checkpoints -> $CKPT_DIR"
echo "     ./runs        -> /data/ethan/MedSAM2/runs"
echo ""

# =============================================================================
# STEP 3 - TemPoRAL dataset (zero-shot inference only)
# =============================================================================

TEMPORAL_DATA_DIR=/data/santiago/Database_nifti_TEMPORAL
TEMPORAL_OUTPUT_DIR=/data/ethan/MedSAM2/temporal_npz

if [ -d "$TEMPORAL_DATA_DIR" ]; then
    echo "========================================="
    echo "  MedSAM2 x TEMPORAL - NPZ preparation"
    echo "  Source  : $TEMPORAL_DATA_DIR"
    echo "  Output  : $TEMPORAL_OUTPUT_DIR"
    echo "========================================="

    mkdir -p "$TEMPORAL_OUTPUT_DIR"

    python3.10 data_preparation/prepare_temporal_npz.py \
        --input_folder  "$TEMPORAL_DATA_DIR" \
        --output_folder "$TEMPORAL_OUTPUT_DIR" \
        --timepoints    "all" \
        --ct_low        -200 \
        --ct_high        800 \
        --crop_margin     5 \
        2>&1 | tee "$TEMPORAL_OUTPUT_DIR/preparation.log"

    echo ""
    echo "  TemPoRAL Dataset preparation complete."
    echo "     Output NPZ  : $TEMPORAL_OUTPUT_DIR/"
    echo "     Manifest    : $TEMPORAL_OUTPUT_DIR/manifest.json"
else
    echo "  TemPoRAL source not found at $TEMPORAL_DATA_DIR - skipping."
fi

# =============================================================================
# STEP 4 - Summary
# =============================================================================

echo ""
echo "========================================"
echo "  MedSAM2 Dataset Setup Complete"
echo "========================================"
echo ""
echo "  Training data"
echo "    Root    : $SWINCROSS_NPZ_ROOT"
echo "    JSON    : dataset_swincross_2026kfold_classic.json"
echo "    Train   : $N_TRAIN cases  (from SwinCross pipeline)"
echo "    Val     : $N_VAL cases   (fold 0, used for epoch Dice)"
echo ""
echo "  Test vault (inference only)"
echo "    JSON    : dataset_swincross_2026kfold_test.json"
echo "    Cases   : $N_TEST  (never used during training)"
echo ""
echo "  To train, run: bash MedSAM2_training.sh"
echo "  To infer,  run: bash MedSAM2_inference_execution.sh"
