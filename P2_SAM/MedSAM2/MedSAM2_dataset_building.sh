#!/bin/bash
# =============================================================================
# MedSAM2 Dataset Building Script — HECKTOR NPZ Preparation
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-04
#
# Converts HECKTOR Task-1 NIfTI data into the NPZ format required by MedSAM2.
#
# Source data  : /data/santiago/HECKTOR_data/Task_1_segmentation/  (read-only)
# Output NPZ   : /data/ethan/MedSAM2/hecktor_npz/{train,val}/
#
# Output structure per patient:
#   ct_imgs  : (D, H, W) uint8  – CT windowed & normalised [0, 255]
#   pet_imgs : (D, H, W) uint8  – PET SUV normalised [0, 255]
#   gts      : (D, H, W) uint8  – 0=bg, 1=GTVp, 2=GTVn
#   spacing  : (3,) float64     – voxel spacing mm (z, y, x)
# =============================================================================

set -e

# =============================================================================
# STEP 0 — Environment setup (uv + venv + requirements)
# =============================================================================

if ! command -v uv &> /dev/null; then
    echo "uv not found — installing..."
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
# STEP 1 — Path configuration
# =============================================================================

HECKTOR_DATA_DIR=/data/santiago/HECKTOR_data/2025/Task_1_segmentation
TEMPORAL_DATA_DIR=/data/santiago/Database_nifti_TEMPORAL
HECKTOR_OUTPUT_DIR=/data/ethan/MedSAM2/hecktor_npz
TEMPORAL_OUTPUT_DIR=/data/ethan/MedSAM2/temporal_npz

mkdir -p "$HECKTOR_OUTPUT_DIR"
mkdir -p "$TEMPORAL_OUTPUT_DIR"

# =============================================================================
# STEP 2 — Run the NPZ converter
# =============================================================================

# echo "========================================"
# echo "  MedSAM2 × HECKTOR — NPZ preparation"
# echo "  Source  : $HECKTOR_DATA_DIR"
# echo "  Output  : $HECKTOR_OUTPUT_DIR"
# echo "========================================"

# python3.10 data_preparation/prepare_hecktor_npz.py \
#     --data_dir   "$HECKTOR_DATA_DIR" \
#     --output_dir "$HECKTOR_OUTPUT_DIR" \
#     --val_ratio  0.2 \
#     --seed       42 \
#     --ct_low    -200 \
#     --ct_high    800 \
#     --pet_suv_max 0.0 \
#     --crop_margin 5 \
#     2>&1 | tee "$HECKTOR_OUTPUT_DIR/preparation.log"

# echo ""
# echo "HECKTOR Dataset preparation complete."
# echo "Split saved to: $HECKTOR_OUTPUT_DIR/data_split.json"
# echo "Train NPZ:      $HECKTOR_OUTPUT_DIR/train/"
# echo "Val NPZ:        $HECKTOR_OUTPUT_DIR/val/"

echo "========================================="
echo "  MedSAM2 × TEMPORAL — NPZ preparation"
echo "  Source  : $TEMPORAL_DATA_DIR"
echo "  Output  : $TEMPORAL_OUTPUT_DIR"
echo "========================================="

python3.10 data_preparation/prepare_temporal_npz.py \
    --input_folder   "$TEMPORAL_DATA_DIR" \
    --output_folder  "$TEMPORAL_OUTPUT_DIR" \
    --timepoints     "all" \
    --ct_low         -200 \
    --ct_high        800 \
    --crop_margin     5 \
    2>&1 | tee "$TEMPORAL_OUTPUT_DIR/preparation.log"

echo ""
echo "TEMPORAL Dataset preparation complete."
echo "Output NPZ: $TEMPORAL_OUTPUT_DIR/"
