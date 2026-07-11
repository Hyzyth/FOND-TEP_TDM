#!/bin/bash
# =============================================================================
# MedSAM2 Checkpoint Download Script
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-04
#
# Downloads the required base SAM2.1 checkpoints from Meta's official release
# and (optionally) a MedSAM2-finetuned checkpoint if available.
#
# Checkpoint destination: /data/ethan/MedSAM2/checkpoints/
# Symlink:                ./checkpoints -> /data/ethan/MedSAM2/checkpoints/
#
# The training and inference scripts expect weights at:
#   ./checkpoints/sam2.1_hiera_tiny.pt       (base, used for fine-tuning init)
#   ./checkpoints/MedSAM2_latest.pt          (MedSAM2-finetuned, used for inference)
# =============================================================================

set -e

# =============================================================================
# STEP 1 — Directories and symlink
# =============================================================================

CKPT_DIR=/data/ethan/MedSAM2/checkpoints
mkdir -p "$CKPT_DIR"

# Symlink ./checkpoints -> persistent storage so all scripts can use relative paths
ln -sfn "$CKPT_DIR" ./checkpoints

echo "========================================"
echo "  MedSAM2 — Checkpoint Download"
echo "  Destination: $CKPT_DIR"
echo "========================================"

# =============================================================================
# STEP 2 — SAM2.1 base weights (Meta official release)
#
# These are needed as the initialisation point for fine-tuning (see training
# script, model_weight_initializer in the YAML).
# All four model sizes are listed; only Hiera-Tiny is downloaded by default.
# Uncomment whichever size matches your training config.
# =============================================================================

BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

# --- Hiera Tiny (used by sam2.1_hiera_tiny_hecktor.yaml) ---
if [ ! -f "$CKPT_DIR/sam2.1_hiera_tiny.pt" ]; then
    echo "Downloading sam2.1_hiera_tiny.pt ..."
    wget -q --show-progress -O "$CKPT_DIR/sam2.1_hiera_tiny.pt" \
        "$BASE_URL/sam2.1_hiera_tiny.pt"
else
    echo "sam2.1_hiera_tiny.pt already present — skipping."
fi

# --- Hiera Small  [UNCOMMENT IF NEEDED] ---
# if [ ! -f "$CKPT_DIR/sam2.1_hiera_small.pt" ]; then
#     wget -q --show-progress -O "$CKPT_DIR/sam2.1_hiera_small.pt" \
#         "$BASE_URL/sam2.1_hiera_small.pt"
# fi

# --- Hiera Base+  [UNCOMMENT IF NEEDED] ---
# if [ ! -f "$CKPT_DIR/sam2.1_hiera_base_plus.pt" ]; then
#     wget -q --show-progress -O "$CKPT_DIR/sam2.1_hiera_base_plus.pt" \
#         "$BASE_URL/sam2.1_hiera_base_plus.pt"
# fi

# --- Hiera Large  [UNCOMMENT IF NEEDED] ---
# if [ ! -f "$CKPT_DIR/sam2.1_hiera_large.pt" ]; then
#     wget -q --show-progress -O "$CKPT_DIR/sam2.1_hiera_large.pt" \
#         "$BASE_URL/sam2.1_hiera_large.pt"
# fi

# =============================================================================
# STEP 3 — MedSAM2 fine-tuned weights
#
# If a pre-trained MedSAM2 checkpoint exists (e.g. from a prior training run
# or a public release), place it here.  The inference script expects it at
#   ./checkpoints/MedSAM2_latest.pt
# =============================================================================

MEDSAM2_URL="https://huggingface.co/wanglab/MedSAM2/blob/main/MedSAM2_latest.pt"
if [ ! -f "$CKPT_DIR/MedSAM2_latest.pt" ]; then
    echo "Downloading MedSAM2_latest.pt ..."
    wget -q --show-progress -O "$CKPT_DIR/MedSAM2_latest.pt" "$MEDSAM2_URL"
fi

# =============================================================================
# STEP 4 — Integrity check (file sizes, not hashes — fast sanity check)
# =============================================================================

echo ""
echo "Checkpoint directory contents:"
ls -lh "$CKPT_DIR"
echo ""
echo "Done. Symlink ./checkpoints -> $CKPT_DIR is ready."
