#!/bin/bash
# =============================================================================
# MedSAM2 - Proposal Network Training
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-05
#
# Trains the Small3DUNet on the HECKTOR 2026 SwinCross NPZ training split.
#
# UPDATED: now reads directly from the shared SwinCross NPZ root using the
# classic JSON split (same data as SwinCross + DualwaveSAM training).
# The old --train_dir / --val_dir arguments pointing to MedSAM2's own
# hecktor_npz/ directories are replaced.
#
# Outputs
# -------
#   /data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt
#   /data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_last.pt
#   /data/ethan/MedSAM2/proposal_net/checkpoints/training_log.csv
# =============================================================================

set -e

# =============================================================================
# STEP 0 - Environment
# =============================================================================

if [ ! -d "medsam2_env" ]; then
    echo "medsam2_env not found. Run MedSAM2_dataset_building.sh first."
    exit 1
fi
source medsam2_env/bin/activate

# =============================================================================
# STEP 1 - Paths & hyperparameters
# =============================================================================

SWINCROSS_NPZ_ROOT="/data/ethan/PP_hecktor2026_kfold_npz"
CLASSIC_JSON="dataset_swincross_2026kfold_classic.json"
OUTPUT_DIR=/data/ethan/MedSAM2/proposal_net/checkpoints

# Architecture
BASE_FEATURES=16
DROPOUT=0.10

# Prior probability - sets output_bias = log(p/(1-p)) at init
PRIOR_PROB=0.02

# Loss weights
FOCAL_WEIGHT=0.30
TVERSKY_ALPHA=0.30
TVERSKY_BETA=0.70

# Training
NUM_EPOCHS=100
BATCH_SIZE=1            # Full-volume batches - reduce if OOM
LR=1e-3
WEIGHT_DECAY=1e-4
CROP_SIZE="64,128,128"
THRESHOLD=0.25
VAL_EVERY=5
MIN_RECALL_FOR_SAVE=0.50

NUM_WORKERS=2
SEED=42
GPU=0

# Verify prerequisites
if [ ! -f "$SWINCROSS_NPZ_ROOT/$CLASSIC_JSON" ]; then
    echo "  JSON not found: $SWINCROSS_NPZ_ROOT/$CLASSIC_JSON"
    echo "  Run models/swincross/SwinCross_NPZ_Dataset_Building.sh first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Extract train/val NPZ directories from JSON
TRAIN_NPZ_DIR="${SWINCROSS_NPZ_ROOT}/npz/train"
VAL_NPZ_DIR="${SWINCROSS_NPZ_ROOT}/npz/train"  # fold-0 val cases also in npz/train

# The proposal net dataset reads NPZ files directly and combines ct/pet/label
# from the SwinCross format.  We point it at the full npz/train directory;
# the HECKTORProposalDataset in train_proposal_net.py will be updated to read
# the SwinCross keys (ct, pet, label) instead of the old (ct_imgs, pet_imgs, gts).
# A class-specific JSON override is also supported - see note below.
#
# NOTE: train_proposal_net.py's HECKTORProposalDataset is updated in this
# release to read SwinCross NPZ format.  See auto_prompting/train_proposal_net.py.

echo "========================================"
echo "  Proposal Network - Training"
echo "  Data root  : $SWINCROSS_NPZ_ROOT"
echo "  JSON       : $CLASSIC_JSON"
echo "  Train NPZ  : $TRAIN_NPZ_DIR"
echo "  Output     : $OUTPUT_DIR"
echo "  GPU        : $GPU"
echo "  Loss       : Focal($FOCAL_WEIGHT) + Tversky($(echo "1 - $FOCAL_WEIGHT" | bc))"
echo "              Tversky alpha=$TVERSKY_ALPHA  beta=$TVERSKY_BETA"
echo "  Prior      : $PRIOR_PROB  (output bias init)"
echo "========================================"

# =============================================================================
# STEP 2 - Train
# =============================================================================

CUDA_VISIBLE_DEVICES=$GPU python3.10 -m auto_prompting.train_proposal_net \
    --data_dir       "$SWINCROSS_NPZ_ROOT"   \
    --json_list      "$CLASSIC_JSON"         \
    --output_dir     "$OUTPUT_DIR"           \
    --base_features  "$BASE_FEATURES"        \
    --dropout        "$DROPOUT"              \
    --prior_prob     "$PRIOR_PROB"           \
    --focal_weight   "$FOCAL_WEIGHT"         \
    --tversky_alpha  "$TVERSKY_ALPHA"        \
    --tversky_beta   "$TVERSKY_BETA"         \
    --num_epochs     "$NUM_EPOCHS"           \
    --batch_size     "$BATCH_SIZE"           \
    --lr             "$LR"                   \
    --weight_decay   "$WEIGHT_DECAY"         \
    --crop_size      "$CROP_SIZE"            \
    --threshold      "$THRESHOLD"            \
    --val_every      "$VAL_EVERY"            \
    --min_recall_for_save "$MIN_RECALL_FOR_SAVE" \
    --num_workers    "$NUM_WORKERS"          \
    --seed           "$SEED"                 \
    2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "========================================"
echo "  Training complete."
echo "  Best checkpoint : $OUTPUT_DIR/proposal_net_best.pt"
echo "  Last checkpoint : $OUTPUT_DIR/proposal_net_last.pt"
echo "  Training log    : $OUTPUT_DIR/training_log.csv"
echo "========================================"

# =============================================================================
# STEP 3 - Sanity check
# =============================================================================

echo ""
echo "Sanity-checking best checkpoint..."
python3.10 - <<'EOF'
import sys, os
sys.path.insert(0, os.getcwd())
from auto_prompting.proposal_net import Small3DUNet
import torch, math

path = "/data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt"
if not os.path.exists(path):
    print("  [WARN] Best checkpoint not found - training may have failed.")
    sys.exit(0)

net = Small3DUNet.load(path, device="cpu")
dummy = torch.zeros(1, 2, 32, 64, 64)
out = net(dummy)
mean_pred = out.mean().item()
print(f"  OK - output shape : {tuple(out.shape)}  (expected: (1, 1, 32, 64, 64))")
print(f"       mean_pred     : {mean_pred:.4f}")
print(f"       output_bias   : {net.output_bias.item():.4f}  "
      f"(prior_prob ~ {torch.sigmoid(net.output_bias).item():.3f})")
if mean_pred > 0.3:
    print("  [WARN] mean_pred is still high - check training logs for collapse.")
else:
    print("  Looks healthy.")
EOF

echo "Done."
