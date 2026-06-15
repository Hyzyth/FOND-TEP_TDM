#!/bin/bash
# =============================================================================
# MedSAM2 - Visual Smoke Tests
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-05
#
# Runs visual smoke-tests for:
#   1. slicer.py          -> test_slicer.py
#   2. auto_prompting/    -> test_auto_prompting.py
#      A) PET proposals   (always)
#      B) Proposal net    (when PROPOSAL_MODEL is set)
#      C) Hybrid          (when PROPOSAL_MODEL is set)
#
# Usage
# -----
#   bash MedSAM2_run_tests.sh [ct|pet|both] [test_slicer|test_auto|all]
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

if ! python -c "import sam2" &>/dev/null; then
    pip install -e . --no-build-isolation
fi

# =============================================================================
# STEP 1 - Arguments & shared config
# =============================================================================

MODALITY=${1:-both}
TESTS=${2:-all}

if [[ "$MODALITY" != "ct" && "$MODALITY" != "pet" && "$MODALITY" != "both" ]]; then
    echo "Error: First arg must be 'ct', 'pet', or 'both'."
    exit 1
fi

if [[ "$TESTS" != "test_slicer" && "$TESTS" != "test_auto" && "$TESTS" != "all" ]]; then
    echo "Error: Second arg must be 'test_slicer', 'test_auto', or 'all'."
    exit 1
fi

# ── SwinCross NPZ root (shared with SwinCross + DualwaveSAM) ──────────────────
# The test scripts read NPZ files with keys: ct, pet, label (SwinCross format).
# Point at the validation split of the classic JSON - these are real patient
# cases that were never used for training.
SWINCROSS_NPZ_ROOT="/data/ethan/PP_hecktor2026_kfold_npz"
NPZ_DIR="${SWINCROSS_NPZ_ROOT}/npz/train"   # val cases also reside here

# Verify
if [ ! -d "$NPZ_DIR" ]; then
    echo "  SwinCross NPZ directory not found: $NPZ_DIR"
    echo "  Run P1_SWIN/SwinCross_NPZ_Dataset_Building.sh first."
    exit 1
fi

K=5
SEED=42

PROPOSAL_MODEL="/data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt"
PROB_THRESHOLD=0.25

SLICE_PAD=1
PLANAR_PAD=5

# =============================================================================
# STEP 2 - Slicer smoke-test
# =============================================================================

if [[ "$TESTS" == "test_slicer" || "$TESTS" == "all" ]]; then
    OUT_DIR="/data/ethan/MedSAM2/test_slicer_${MODALITY}"
    mkdir -p "$OUT_DIR"

    echo "========================================"
    echo "  Test 1 - slicer.py"
    echo "  Modality   : $MODALITY"
    echo "  NPZ dir    : $NPZ_DIR  (SwinCross format)"
    echo "  Output dir : $OUT_DIR"
    echo "========================================"

    python3.10 test_slicer.py \
        --npz_dir    "$NPZ_DIR"    \
        --output_dir "$OUT_DIR"    \
        --k          "$K"          \
        --seed       "$SEED"       \
        --modality   "$MODALITY"   \
        --slice_pad  "$SLICE_PAD"  \
        --planar_pad "$PLANAR_PAD" \
        --npz_format swincross

    echo "  -> Figures saved to $OUT_DIR"
    echo ""
fi

# =============================================================================
# STEP 3 - Auto-prompting smoke-test
# =============================================================================

if [[ "$TESTS" == "test_auto" || "$TESTS" == "all" ]]; then
    OUT_DIR="/data/ethan/MedSAM2/test_auto_prompting"
    mkdir -p "$OUT_DIR"

    echo "========================================"
    echo "  Test 2 - auto_prompting/"
    echo "  NPZ dir    : $NPZ_DIR  (SwinCross format)"
    echo "  Output dir : $OUT_DIR"

    MODEL_ARG=""
    if [ -f "$PROPOSAL_MODEL" ]; then
        echo "  Model      : $PROPOSAL_MODEL"
        MODEL_ARG="--proposal_model $PROPOSAL_MODEL --prob_threshold $PROB_THRESHOLD"
        echo "  Tests      : A (PET)  B (UNet)  C (Hybrid)"
    else
        echo "  Model      : not found - running PET-only test A"
        echo "  Tests      : A (PET) only"
    fi
    echo "========================================"

    python3.10 test_auto_prompting.py \
        --npz_dir    "$NPZ_DIR"    \
        --output_dir "$OUT_DIR"    \
        --k          "$K"          \
        --seed       "$SEED"       \
        --slice_pad  "$SLICE_PAD"  \
        --planar_pad "$PLANAR_PAD" \
        --npz_format swincross     \
        $MODEL_ARG

    echo ""
    echo "  -> Figures saved to $OUT_DIR"
    echo ""
fi

echo "All tests complete."
