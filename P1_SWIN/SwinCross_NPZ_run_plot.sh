#!/bin/bash
# =============================================================================
# Run script to parse logs and generate training plots (Classic + K-Fold)
# =============================================================================

set -e

echo "╔═══════════════════════════════════════╗"
echo "║   SwinCross Training Log Plotter      ║"
echo "╚═══════════════════════════════════════╝"

# ── 1. Setup Environment ──────────────────────────────────────────────────────
if [ -d "swincross_env" ]; then
    source swincross_env/bin/activate
    echo "[INFO] Activated swincross_env."
else
    echo "[WARNING] swincross_env not found. Running in system/current environment."
fi

# ── 2. Force Re-download / Install Modules ────────────────────────────────────
echo "[INFO] Ensuring required modules (pandas, matplotlib) are installed/updated..."
if command -v uv &> /dev/null; then
    # Use ultra-fast uv pip if it exists in your env
    uv pip install --upgrade pandas matplotlib
else
    # Fallback to standard pip
    pip install --upgrade pandas matplotlib
fi

# ── 3. Configuration ──────────────────────────────────────────────────────────
# Match these to your training directories
CLASSIC_DIR="./runs/HECKTOR_run_1000_epoch"
KFOLD_BASE="./runs/HECKTOR_kfold_400ep"

# ── 4. Execute Classic Plotting ───────────────────────────────────────────────
if [ -d "$CLASSIC_DIR" ]; then
    echo "[INFO] Parsing Classic Run logs..."
    mkdir -p "$CLASSIC_DIR/plots"
    python3.12 plot_training.py \
        --log_dirs "$CLASSIC_DIR" \
        --output_dir "$CLASSIC_DIR/plots" \
        --title "Classic Run (1000 Epochs)"
else
    echo "[SKIP] Classic directory not found: $CLASSIC_DIR"
fi

# ── 5. Execute K-Fold Plotting ────────────────────────────────────────────────
# Look for any directories matching the K-Fold pattern
FOLD_DIRS=$(ls -d ${KFOLD_BASE}_fold* 2>/dev/null || true)

if [ -n "$FOLD_DIRS" ]; then
    echo ""
    echo "[INFO] Parsing K-Fold Ensemble logs..."
    ENSEMBLE_PLOT_DIR="./runs/${KFOLD_BASE}_ensemble/plots"
    mkdir -p "$ENSEMBLE_PLOT_DIR"
    
    # We pass all fold directories directly into the --log_dirs argument
    python3.12 plot_training.py \
        --log_dirs $FOLD_DIRS \
        --output_dir "$ENSEMBLE_PLOT_DIR" \
        --title "K-Fold Ensemble Training"
else
    echo "[SKIP] No K-Fold directories found matching: ${KFOLD_BASE}_fold*"
fi

echo ""
echo "[INFO] Plotting pipeline complete."
