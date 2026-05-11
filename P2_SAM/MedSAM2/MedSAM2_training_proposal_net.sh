#!/bin/bash
# =============================================================================
# MedSAM2 — Proposal Network Training
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
#
# Trains the Small3DUNet on HECKTOR NPZ data for use as the 'unet' and
# 'hybrid' prompt modes in MedSAM2_inference_execution.sh.
#
# Outputs
# -------
#   /data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt
#   /data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_last.pt
#   /data/ethan/MedSAM2/proposal_net/checkpoints/training_log.csv
#
# Typical runtime : ~30–45 min on a single GPU (40 epochs, batch 1)
#
# Usage
# -----
#   bash MedSAM2_training_proposal_net.sh
#
# To resume from last checkpoint, set RESUME=1 below (not yet implemented
# in the training script — rerunning overwrites the last checkpoint only;
# the best checkpoint is never overwritten).
# =============================================================================

set -e

# =============================================================================
# STEP 0 — Environment
# =============================================================================

if [ ! -d "medsam2_env" ]; then
    echo "medsam2_env not found. Run MedSAM2_dataset_building.sh first."
    exit 1
fi
source medsam2_env/bin/activate

# =============================================================================
# STEP 1 — Paths & hyperparameters
# Edit these to match your setup / run experiments.
# =============================================================================

NPZ_TRAIN=/data/ethan/MedSAM2/hecktor_npz/train
NPZ_VAL=/data/ethan/MedSAM2/hecktor_npz/val
OUTPUT_DIR=/data/ethan/MedSAM2/proposal_net/checkpoints

# Architecture
BASE_FEATURES=16       # U-Net base channel count (16 → 32 → 64)
DROPOUT=0.10           # Dropout3d probability

# Training
NUM_EPOCHS=40
BATCH_SIZE=1           # Reduce to 1 if OOM; full-volume batches are large
LR=1e-3
WEIGHT_DECAY=1e-4
BCE_WEIGHT=0.30        # Recall weight = 1 - BCE_WEIGHT = 0.70
CROP_SIZE="64,128,128" # D,H,W crop (set to "" for full-volume, higher OOM risk)
THRESHOLD=0.25         # Probability threshold used during validation metrics
VAL_EVERY=5            # Run validation every N epochs

NUM_WORKERS=2
SEED=42
GPU=0

mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "  Proposal Network — Training"
echo "  Train : $NPZ_TRAIN"
echo "  Val   : $NPZ_VAL"
echo "  Output: $OUTPUT_DIR"
echo "  GPU   : $GPU"
echo "========================================"

# =============================================================================
# STEP 2 — Train
# =============================================================================

CUDA_VISIBLE_DEVICES=$GPU python3.10 -m auto_prompting.train_proposal_net \
    --train_dir     "$NPZ_TRAIN"    \
    --val_dir       "$NPZ_VAL"      \
    --output_dir    "$OUTPUT_DIR"   \
    --base_features "$BASE_FEATURES"\
    --dropout       "$DROPOUT"      \
    --num_epochs    "$NUM_EPOCHS"   \
    --batch_size    "$BATCH_SIZE"   \
    --lr            "$LR"           \
    --weight_decay  "$WEIGHT_DECAY" \
    --bce_weight    "$BCE_WEIGHT"   \
    --crop_size     "$CROP_SIZE"    \
    --threshold     "$THRESHOLD"    \
    --val_every     "$VAL_EVERY"    \
    --num_workers   "$NUM_WORKERS"  \
    --seed          "$SEED"         \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "========================================"
echo "  Training complete."
echo "  Best checkpoint : $OUTPUT_DIR/proposal_net_best.pt"
echo "  Last checkpoint : $OUTPUT_DIR/proposal_net_last.pt"
echo "  Training log    : $OUTPUT_DIR/training_log.csv"
echo "========================================"

# =============================================================================
# STEP 3 — Quick sanity check: does the best checkpoint load correctly?
# =============================================================================

echo ""
echo "Sanity-checking best checkpoint..."
python3.10 - <<'EOF'
import sys, os
sys.path.insert(0, os.getcwd())
from auto_prompting.proposal_net import Small3DUNet
import torch

path = "/data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt"
if not os.path.exists(path):
    print("  [WARN] Best checkpoint not found — training may have failed.")
    sys.exit(0)

net = Small3DUNet.load(path, device="cpu")
dummy = torch.zeros(1, 2, 32, 64, 64)
out = net(dummy)
print(f"  OK — output shape: {tuple(out.shape)}  (expected: (1, 1, 32, 64, 64))")
EOF

echo "Done."
