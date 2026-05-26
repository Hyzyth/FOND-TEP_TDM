#!/bin/bash
set -e

# =============================================================================
# SwinCross Dataset Building — NPZ offline preprocessing
# =============================================================================
# A) HECKTOR 2026      — k-fold split (X% train, (1-X)% fixed val)
# B) TemPoRAL          — zero-shot validation dataset
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

# # ── HECKTOR 2025 ──────────────────────────────────────────────────────────────
# echo ""
# echo "╔══════════════════════════════════╗"
# echo "║  HECKTOR → SwinCross NPZ build   ║"
# echo "╚══════════════════════════════════╝"
# python3.12 npz_version/prepare_hecktor_npz_swincross.py \
#     --data_dir   /data/santiago/HECKTOR_data/2025/Task_1_segmentation \
#     --output_dir /data/ethan/PP_hecktor_swincross_npz \
#     --json_name  dataset_swincross.json \
#     --val_split  0.2 \
#     --seed       42 \
#     2>&1 | tee /data/ethan/PP_hecktor_swincross_npz/preprocessing.log

# =============================================================================
# HECKTOR 2026 — stratified k-fold split
#
#  --train_ratio X   X% of each hospital's patients → training pool.
#                    Default 0.8 (80/20). Lower to 0.7 for more validation coverage.
#  --k_folds         Number of CV folds. Default 5.
#
#  Outputs:
#    dataset_swincross_2026kfold_fold{0..k-1}.json
#    dataset_swincross_2026kfold_full.json
#    split_info.json
# =============================================================================
echo ""
echo "╔═════════════════════════════════════════════════╗"
echo "║  HECKTOR 2026 → SwinCross k-fold NPZ build      ║"
echo "╚═════════════════════════════════════════════════╝"

OUTPUT_DIR=/data/ethan/PP_hecktor2026_kfold_npz
mkdir -p $OUTPUT_DIR

python3.12 npz_version/prepare_hecktor2026_kfold_npz.py \
    --data_dir    "/data/santiago/HECKTOR_data/2026/HECKTOR 2026 Training Data" \
    --output_dir  $OUTPUT_DIR \
    --train_ratio 0.8 \
    --k_folds     5 \
    --json_prefix dataset_swincross_2026kfold \
    --seed        42 \
    2>&1 | tee $OUTPUT_DIR/preprocessing.log

# ── TemPoRAL zero-shot ────────────────────────────────────────────────────────
# echo ""
# echo "╔══════════════════════════════════════╗"
# echo "║  TemPoRAL → SwinCross NPZ build      ║"
# echo "╚══════════════════════════════════════╝"
# python3.12 npz_version/prepare_temporal_npz_swincross.py \
#     --input_folder  /data/santiago/Database_nifti_TEMPORAL \
#     --output_folder /data/ethan/PP_temporal_swincross_npz \
#     --json_name     dataset_swincross_temporal.json \
#     --timepoints    all \
#     2>&1 | tee /data/ethan/PP_temporal_swincross_npz/preprocessing.log

echo ""
echo "Dataset building complete."
echo "  HECKTOR NPZ → /data/ethan/PP_hecktor_swincross_npz/"
echo "  HECKTOR K-Fold NPZ → /data/ethan/PP_hecktor2026_kfold_npz/"
echo "  TemPoRAL NPZ → /data/ethan/PP_temporal_swincross_npz/"
