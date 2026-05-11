#!/bin/bash
# =============================================================================
# MedSAM2 — Visual Smoke Tests
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
#
# Runs visual smoke-tests for:
#   1. slicer.py          → test_slicer.py
#   2. auto_prompting/    → test_auto_prompting.py
#      A) PET proposals   (always)
#      B) Proposal net    (when PROPOSAL_MODEL is set)
#      C) Hybrid          (when PROPOSAL_MODEL is set)
#
# Usage
# -----
#   bash MedSAM2_run_tests.sh [ct|pet|both] [test_slicer|test_auto|all]
#
#   First arg  : modality for slicer test  (default: both)
#   Second arg : which test(s) to run      (default: all)
#
# Examples
#   bash MedSAM2_run_tests.sh both all
#   bash MedSAM2_run_tests.sh ct   test_slicer
#   bash MedSAM2_run_tests.sh both test_auto
# =============================================================================

set -e

# =============================================================================
# STEP 0 — Environment setup
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

if ! python -c "import sam2" &>/dev/null; then
    pip install -e . --no-build-isolation
fi

# =============================================================================
# STEP 1 — Arguments & shared config
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

NPZ_DIR="/data/ethan/MedSAM2/hecktor_npz/val"
K=5
SEED=42

# Path to trained proposal network — leave empty to skip net/hybrid tests
PROPOSAL_MODEL="/data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt"
PROB_THRESHOLD=0.25

# Slicer padding defaults
SLICE_PAD=1
PLANAR_PAD=5

# =============================================================================
# STEP 2 — Slicer smoke-test  (test_slicer.py)
# =============================================================================

if [[ "$TESTS" == "test_slicer" || "$TESTS" == "all" ]]; then
    OUT_DIR="/data/ethan/MedSAM2/test_slicer_${MODALITY}"
    mkdir -p "$OUT_DIR"

    echo "========================================"
    echo "  Test 1 — slicer.py"
    echo "  Modality   : $MODALITY"
    echo "  Padding    : slice=$SLICE_PAD  planar=$PLANAR_PAD"
    echo "  Input dir  : $NPZ_DIR"
    echo "  Output dir : $OUT_DIR"
    echo "========================================"

    python3.10 test_slicer.py \
        --npz_dir    "$NPZ_DIR"    \
        --output_dir "$OUT_DIR"    \
        --k          "$K"          \
        --seed       "$SEED"       \
        --modality   "$MODALITY"   \
        --slice_pad  "$SLICE_PAD"  \
        --planar_pad "$PLANAR_PAD"

    echo "  → Figures saved to $OUT_DIR"
    echo ""
fi

# =============================================================================
# STEP 3 — Auto-prompting smoke-test  (test_auto_prompting.py)
# =============================================================================

if [[ "$TESTS" == "test_auto" || "$TESTS" == "all" ]]; then
    OUT_DIR="/data/ethan/MedSAM2/test_auto_prompting"
    mkdir -p "$OUT_DIR"

    echo "========================================"
    echo "  Test 2 — auto_prompting/"
    echo "  Input dir  : $NPZ_DIR"
    echo "  Output dir : $OUT_DIR"

    # Determine whether to pass the model
    MODEL_ARG=""
    if [ -f "$PROPOSAL_MODEL" ]; then
        echo "  Model      : $PROPOSAL_MODEL"
        MODEL_ARG="--proposal_model $PROPOSAL_MODEL --prob_threshold $PROB_THRESHOLD"
        echo "  Tests      : A (PET)  B (UNet)  C (Hybrid)"
    else
        echo "  Model      : not found — running PET-only test A"
        echo "  (Train the proposal network first with MedSAM2_training_proposal_net.sh"
        echo "   to also enable tests B and C.)"
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
        $MODEL_ARG

    echo ""
    echo "  → Figures saved to $OUT_DIR"
    echo "    *_pet_proposals.png : Test A — PET thresholding (all 4 methods)"
    if [ -f "$PROPOSAL_MODEL" ]; then
        echo "    *_proposal_net.png  : Test B — UNet probability maps (3 thresholds)"
        echo "    *_hybrid.png        : Test C — Hybrid proposals vs GT"
    fi
    echo ""
fi

echo "All tests complete."
