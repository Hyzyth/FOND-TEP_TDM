#!/bin/bash
# =============================================================================
# Run script to parse logs and generate training plots
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

# ── 3. Execute Plotting Script ────────────────────────────────────────────────
echo "[INFO] Executing plot_training.py..."
python3.12 plot_training.py

echo "[INFO] Pipeline complete."
