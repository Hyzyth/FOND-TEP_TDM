#!/bin/bash
# scripts/infer_hecktor.sh
# ========================
# Run MedSAM2 inference and evaluation on the HECKTOR validation set.
#
# Environment overrides (all optional):
#   CHECKPOINT  – path to model checkpoint  (default: MedSAM2_latest.pt)
#   CFG         – model YAML config         (default: sam2.1_hiera_t512.yaml)
#   IMGS_PATH   – directory of input NPZ files
#   PRED_DIR    – directory for output NPZ files
#   GT_DIR      – directory of GT NPZ files for evaluation (default: IMGS_PATH)
#
# Usage
#   bash scripts/infer_hecktor.sh
#   CHECKPOINT=/path/to/custom.pt bash scripts/infer_hecktor.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="${CHECKPOINT:-/data/ethan/MedSAM2/checkpoints/MedSAM2_latest.pt}"
CFG="${CFG:-sam2/configs/sam2.1_hiera_t512.yaml}"
IMGS_PATH="${IMGS_PATH:-/data/ethan/MedSAM2/hecktor_npz/val}"
PRED_DIR="${PRED_DIR:-/data/ethan/MedSAM2/predictions/val}"
GT_DIR="${GT_DIR:-${IMGS_PATH}}"

echo "========================================"
echo "  MedSAM2 × HECKTOR inference"
echo "  Checkpoint : ${CHECKPOINT}"
echo "  Config     : ${CFG}"
echo "  Input      : ${IMGS_PATH}"
echo "  Output     : ${PRED_DIR}"
echo "  GT dir     : ${GT_DIR}"
echo "========================================"

# ── Step 1: inference ────────────────────────────────────────────────────────
python inference/infer_hecktor.py \
    --checkpoint   "${CHECKPOINT}" \
    --cfg          "${CFG}" \
    --imgs_path    "${IMGS_PATH}" \
    --pred_save_dir "${PRED_DIR}" \
    --save_nifti \
    --save_overlays \
    --bbox_shift 5 \
    --num_workers 1

# ── Step 2: evaluation ───────────────────────────────────────────────────────
python inference/evaluate_hecktor.py \
    --pred_dir "${PRED_DIR}" \
    --gt_dir   "${GT_DIR}" \
    --output   "${PRED_DIR}/dsc_results.csv"

echo "Done."
