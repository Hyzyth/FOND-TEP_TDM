#!/bin/bash
# =============================================================================
# DualwaveSAM 3-Class Training Log Plotter
# Author  : Ethan
# Project : ProjetMaster / StageM1_IA
# Updated : 2026-05
# =============================================================================

set -e

echo "╔═══════════════════════════════════════════════╗"
echo "║   DualwaveSAM 3-Class Training Log Plotter    ║"
echo "╚═══════════════════════════════════════════════╝"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 1. Environment ────────────────────────────────────────────────────────
if [ -d "dualwave_env" ]; then
    source dualwave_env/bin/activate
    echo "[INFO] Activated dualwave_env."
else
    echo "[WARN] dualwave_env not found - using system Python."
fi

if command -v uv &> /dev/null; then
    uv pip install --upgrade pandas matplotlib "numpy<2.0.0" seaborn
else
    pip install --upgrade pandas matplotlib "numpy<2.0.0" seaborn
fi

# ── 2. Configuration ──────────────────────────────────────────────────────
NUM_EPOCHS=500  # For plot titles - adjust if needed
K_FOLDS=5       # For plot titles - adjust if needed
KFOLD_EPOCHS=$(($NUM_EPOCHS / $K_FOLDS))   # For plot titles - adjust if needed
# Match these to your training script settings
#TEST_DIR="./runs/DualwaveSAM3c_test"
CLASSIC_DIR="./runs/DualwaveSAM3c_classic_${NUM_EPOCHS}ep"
KFOLD_BASE="./runs/DualwaveSAM3c_kfold_${KFOLD_EPOCHS}ep"

# ── 3.0 Test run plot (optional) ──────────────────────────────────────────
# if [ -d "$TEST_DIR" ]; then
#     echo "[INFO] Parsing Test Run logs..."
#     mkdir -p "$TEST_DIR/plots"
#     python3.12 adaptation/plot_training.py \
#         --log_dirs  "$TEST_DIR" \
#         --output_dir "$TEST_DIR/plots" \
#         --title "DualwaveSAM 3-class Test Run"
# else
#     echo "[SKIP] Test directory not found: $TEST_DIR"
# fi

# ── 3.1 Classic Plot ──────────────────────────────────────────────────────
if [ -d "$CLASSIC_DIR" ]; then
    echo "[INFO] Parsing Classic Run logs..."
    mkdir -p "$CLASSIC_DIR/plots"
    python3.12 adaptation/plot_training.py \
        --log_dirs  "$CLASSIC_DIR" \
        --output_dir "$CLASSIC_DIR/plots" \
        --title "DualwaveSAM 3-class Classic ($NUM_EPOCHS Epochs)"
else
    echo "[SKIP] Classic directory not found: $CLASSIC_DIR"
fi

# ── 3.2 K-Fold Plot ───────────────────────────────────────────────────────
FOLD_DIRS=$(ls -d ${KFOLD_BASE}_fold* 2>/dev/null || true)

if [ -n "$FOLD_DIRS" ]; then
    echo ""
    echo "[INFO] Parsing K-Fold logs..."
    ENSEMBLE_PLOT_DIR="./runs/${KFOLD_BASE##*/}_ensemble/plots"
    mkdir -p "$ENSEMBLE_PLOT_DIR"
    python3.12 adaptation/plot_training.py \
        --log_dirs  $FOLD_DIRS \
        --output_dir "$ENSEMBLE_PLOT_DIR" \
        --title "DualwaveSAM 3-class K-Fold Ensemble"
else
    echo "[SKIP] No K-Fold directories found matching: ${KFOLD_BASE}_fold*"
fi

echo ""
echo "[INFO] Plotting complete."
