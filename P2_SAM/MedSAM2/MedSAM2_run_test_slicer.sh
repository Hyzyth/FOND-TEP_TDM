#!/bin/bash
# =============================================================================
# Run Slicer Test
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
#
# Runs the visual smoke-test for slicer.py with dynamic modality selection.
# Usage: ./run_test_slicer.sh [both|ct|pet]
# =============================================================================

set -e

# =============================================================================
# STEP 1 — Configuration & Environment
# =============================================================================

# Default to 'both' if no argument is provided
MODALITY=${1:-both} 

# Validate input
if [[ "$MODALITY" != "ct" && "$MODALITY" != "pet" && "$MODALITY" != "both" ]]; then
    echo "Error: Invalid modality. Please use 'ct', 'pet', or 'both'."
    exit 1
fi

NPZ_DIR="/data/ethan/MedSAM2/hecktor_npz/val"
# Dynamically change the output dir based on the modality used
OUT_DIR="/data/ethan/MedSAM2/slicer_test_${MODALITY}" 
K=5
SEED=42

# Optional: If you use a virtual environment or conda, uncomment and adapt below:
# source /path/to/venv/bin/activate
# conda activate medsam2

mkdir -p "$OUT_DIR"

echo "========================================"
echo "  MedSAM2 — Running Slicer Test"
echo "  Modality:   $MODALITY"
echo "  Input dir:  $NPZ_DIR"
echo "  Output dir: $OUT_DIR"
echo "========================================"

# =============================================================================
# STEP 2 — Execution
# =============================================================================

python test_slicer.py \
    --npz_dir "$NPZ_DIR" \
    --output_dir "$OUT_DIR" \
    --k "$K" \
    --seed "$SEED" \
    --modality "$MODALITY"

echo ""
echo "Done! Check $OUT_DIR for the generated figures."