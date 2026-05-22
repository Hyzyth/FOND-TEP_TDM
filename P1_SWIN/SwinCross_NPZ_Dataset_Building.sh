#!/bin/bash
set -e

# =============================================================================
# SwinCross Dataset Building — NPZ offline preprocessing
# =============================================================================
# Replaces dataset_builder_simpleITK.py / dataset_builder_TEMPORAL.py.
# For each patient the script:
#   1. Orients to RAS
#   2. Resamples to 1 mm isotropic
#   3. Crops foreground
#   4. Saves a per-patient NPZ (image arrays + inverse-transform metadata)
#   5. Saves the original-space GT NIfTI for evaluate_predictions.py
#   6. Writes a MONAI-compatible JSON split
#
# Moving these heavy ops offline drops per-epoch data-loading time from
# ~70 min to a few minutes (GPU util goes from ~0 % to near 100 %).
# =============================================================================

# ── Environment ──────────────────────────────────────────────────────────────
if ! command -v uv &> /dev/null; then
    echo "uv not found — installing..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

uv python install 3.12

if [ ! -d "swincross_env" ]; then
    uv venv swincross_env --python 3.12
fi
source swincross_env/bin/activate

if [ -f requirements.txt ]; then
    uv pip install -r requirements.txt
else
    echo "requirements.txt not found — aborting." && exit 1
fi

# ── HECKTOR 2025 ──────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════╗"
echo "║  HECKTOR → SwinCross NPZ build   ║"
echo "╚══════════════════════════════════╝"
python3.12 npz_version/prepare_hecktor_npz_swincross.py \
    --data_dir   /data/santiago/HECKTOR_data/2025/Task_1_segmentation \
    --output_dir /data/ethan/PP_hecktor_swincross_npz \
    --json_name  dataset_swincross.json \
    --val_split  0.2 \
    --seed       42 \
    2>&1 | tee /data/ethan/PP_hecktor_swincross_npz/preprocessing.log

# ── TemPoRAL zero-shot ────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════╗"
echo "║  TemPoRAL → SwinCross NPZ build      ║"
echo "╚══════════════════════════════════════╝"
python3.12 npz_version/prepare_temporal_npz_swincross.py \
    --input_folder  /data/santiago/Database_nifti_TEMPORAL \
    --output_folder /data/ethan/PP_temporal_swincross_npz \
    --json_name     dataset_swincross_temporal.json \
    --timepoints    all \
    2>&1 | tee /data/ethan/PP_temporal_swincross_npz/preprocessing.log

echo ""
echo "Dataset building complete."
echo "  HECKTOR NPZ → /data/ethan/PP_hecktor_swincross_npz/"
echo "  TemPoRAL NPZ → /data/ethan/PP_temporal_swincross_npz/"
