#!/bin/bash
# scripts/train_hecktor.sh
# ========================
# Launch MedSAM2 fine-tuning on HECKTOR Task-1.
#
# Prerequisites
# -------------
#   1. Activate the MedSAM2 conda environment.
#   2. Ensure checkpoints exist at /data/ethan/MedSAM2/checkpoints/.
#   3. NPZ data prepared via:
#        python data_preparation/prepare_hecktor_npz.py
#
# Usage
#   bash scripts/train_hecktor.sh
#   # Override GPU count:
#   NUM_GPUS=4 bash scripts/train_hecktor.sh

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
export PATH=/usr/local/cuda/bin:$PATH

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="sam2/configs/sam2.1_hiera_tiny_hecktor.yaml"
OUTPUT_PATH="/data/ethan/MedSAM2/exp_log/hecktor_finetune"
NUM_GPUS="${NUM_GPUS:-2}"

echo "========================================"
echo "  MedSAM2 × HECKTOR fine-tuning"
echo "  Repo root : ${REPO_ROOT}"
echo "  Config    : ${CONFIG}"
echo "  Output    : ${OUTPUT_PATH}"
echo "  GPUs      : ${NUM_GPUS}"
echo "========================================"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1))) \
python training/train.py \
    -c "${CONFIG}" \
    --output-path "${OUTPUT_PATH}" \
    --use-cluster 0 \
    --num-gpus "${NUM_GPUS}" \
    --num-nodes 1

echo "Training complete."
