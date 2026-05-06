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

uv python install 3.12

if [ ! -d "medsam2_env" ]; then
    uv venv medsam2_env --python 3.12
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

DATA_DIR=/data/santiago/HECKTOR_data/Task_1_segmentation
OUTPUT_DIR=/data/ethan/MedSAM2/hecktor_npz

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  MedSAM2 × HECKTOR — NPZ preparation"
echo "  Source  : $DATA_DIR"
echo "  Output  : $OUTPUT_DIR"
echo "========================================"

# =============================================================================
# STEP 2 — Run the NPZ converter
#
# Key parameters:
#   --val_ratio 0.2      : 20 % of patients go to val/
#   --seed 42            : reproducible split
#   --ct_low / --ct_high : H&N soft-tissue window [-200, 800] HU
#   --pet_suv_max 0      : per-patient 99th-percentile SUV normalisation
#   --crop_margin 5      : keep 5 axial slices above/below the tumour
#
# To use a fixed SUV ceiling instead of per-patient percentile, set e.g.
#   --pet_suv_max 10.0
# =============================================================================

python3.12 data_preparation/prepare_hecktor_npz.py \
    --data_dir   "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --val_ratio  0.2 \
    --seed       42 \
    --ct_low    -200 \
    --ct_high    800 \
    --pet_suv_max 0.0 \
    --crop_margin 5 \
    2>&1 | tee "$OUTPUT_DIR/preparation.log"

echo ""
echo "Dataset preparation complete."
echo "Split saved to: $OUTPUT_DIR/data_split.json"
echo "Train NPZ:      $OUTPUT_DIR/train/"
echo "Val NPZ:        $OUTPUT_DIR/val/"
