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
SLICE_PAD=1   # Padding along the "depth" axis of the view
PLANAR_PAD=5  # Padding along the 2D plane of the view

mkdir -p "$OUT_DIR"

echo "========================================"
echo "  MedSAM2 — Running Slicer Test"
echo "  Modality:   $MODALITY"
echo "  Padding:    Slice=$SLICE_PAD | Planar=$PLANAR_PAD"
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
    --modality "$MODALITY" \
    --slice_pad "$SLICE_PAD" \
    --planar_pad "$PLANAR_PAD"

echo ""
echo "Done! Check $OUT_DIR for the generated figures."