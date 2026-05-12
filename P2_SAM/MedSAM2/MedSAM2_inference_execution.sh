#!/bin/bash
# =============================================================================
# MedSAM2 Inference & Evaluation Script — HECKTOR Task-1 GTVp/GTVn
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-05
#
# Runs all prompt-mode variants then evaluates every prediction directory.
#
# Sections
# --------
#   0. Environment & paths
#   1. GT oracle        (val / train overfit-check / val tight)
#   2. PET-only         (base41 / nestle / black / daisne)
#   3. UNet-only        (3 probability thresholds)
#   4. Hybrid           (4 PET methods × 3 thresholds = 12 runs)
#   5. Evaluate ALL prediction directories
#
# Usage
# -----
#   bash MedSAM2_inference_execution.sh
#
# To run only one section, comment out the others or set the skip flags below.
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
# STEP 1 — Path configuration
# =============================================================================

NPZ_VAL=/data/ethan/MedSAM2/hecktor_npz/val
NPZ_TRAIN=/data/ethan/MedSAM2/hecktor_npz/train
CHECKPOINT=./checkpoints/MedSAM2_latest.pt
CFG=sam2/configs/sam2.1_hiera_tiny_hecktor_infer.yaml
PRED_ROOT=/data/ethan/MedSAM2/predictions
PROPOSAL_MODEL=/data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt

mkdir -p "$PRED_ROOT"

# Re-apply symlink in case this is a fresh shell
mkdir -p /data/ethan/MedSAM2/runs
ln -sfn /data/ethan/MedSAM2/runs ./runs

# ── Per-section skip flags (1=skip, 0=run) ───────────────────────────────────
SKIP_GT=0
SKIP_PET=1
SKIP_UNET=1
SKIP_HYBRID=1
SKIP_EVAL=0

GPU=0

echo "========================================"
echo "  MedSAM2 × HECKTOR — Inference & Evaluation"
echo "  Checkpoint : $CHECKPOINT"
echo "  GPU        : $CUDA_VISIBLE_DEVICES"
echo "========================================"

# ── Shared inference wrapper ─────────────────────────────────────────────────
# Usage: run_infer <pred_dir_name> <extra_args...>
run_infer() {
    local PRED_DIR="$1"; shift
    local FULL_PATH="$PRED_ROOT/$PRED_DIR"
    mkdir -p "$FULL_PATH"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Running: $PRED_DIR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CUDA_VISIBLE_DEVICES=$GPU python3.10 inference/infer_hecktor.py \
        --checkpoint    "$CHECKPOINT" \
        --cfg           "$CFG"        \
        --pred_save_dir "$FULL_PATH"  \
        --save_overlays               \
        "$@"                          \
        2>&1 | tee "$FULL_PATH/inference.log"
}

# ── Shared eval wrapper ───────────────────────────────────────────────────────
run_eval() {
    local PRED_DIR="$1"
    local GT_DIR="${2:-$NPZ_VAL}"
    local FULL_PATH="$PRED_ROOT/$PRED_DIR"
    if [ ! -d "$FULL_PATH" ]; then
        echo "  [SKIP eval] $PRED_DIR — directory not found."
        return 0
    fi

    echo "  Evaluating: $PRED_DIR"
    python3.10 inference/evaluate_hecktor.py \
        --pred_dir "$FULL_PATH"                  \
        --gt_dir   "$GT_DIR"                     \
        --output   "$FULL_PATH/dsc_results.csv"  \
        2>&1 | tee "$FULL_PATH/evaluation.log"
}

# =============================================================================
# SECTION 1 — GT oracle  (ground-truth boxes; development / ceiling estimate)
# =============================================================================
if [ "$SKIP_GT" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║   SECTION 1 — GT Oracle      ║"
    echo "╚══════════════════════════════╝"

    # # 1a  Val — standard padding (planar=5 vox, slice=1)
    # run_infer "gt_val" \
    #     --imgs_path "$NPZ_VAL" \
    #     --bbox_mode gt --bbox_shift 5 --slice_pad 1 \
    #     --save_nifti

    # # 1b  Train — overfit check (same settings, different split)
    # run_infer "gt_train_overfit" \
    #     --imgs_path "$NPZ_TRAIN" \
    #     --bbox_mode gt --bbox_shift 5 --slice_pad 1

    # # 1c  Val tight — zero padding (DSC ceiling with exact GT boxes)
    # run_infer "gt_val_tight" \
    #     --imgs_path "$NPZ_VAL" \
    #     --bbox_mode gt --bbox_shift 0 --slice_pad 0 \
    #     --save_nifti
    
    # # 1d  Val reduced — negative padding (DSC ceiling with exact GT boxes)
    # run_infer "gt_val_reduced" \
    #     --imgs_path "$NPZ_VAL" \
    #     --bbox_mode gt --bbox_shift -5 --slice_pad -1 \
    #     --save_nifti
    
    # 1e Val extra reduced — more aggressive negative padding (DSC ceiling with exact GT boxes)
    run_infer "gt_val_extra_reduced" \
        --imgs_path "$NPZ_VAL" \
        --bbox_mode gt --bbox_shift -10 --slice_pad -2 \
        --save_nifti
fi


# =============================================================================
# SECTION 2 — PET-only auto-prompting
# All four methods; only base41 works without pet_suv_max in the NPZ.
# The others fall back to base41 gracefully if suv_max is missing.
# =============================================================================
if [ "$SKIP_PET" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║   SECTION 2 — PET-only       ║"
    echo "╚══════════════════════════════╝"

    for PET_METHOD in base41 nestle black daisne; do
        run_infer "pet_${PET_METHOD}" \
            --imgs_path  "$NPZ_VAL" \
            --bbox_mode  pet        \
            --pet_method "$PET_METHOD"
    done
fi


# =============================================================================
# SECTION 3 — UNet-only auto-prompting
# Three probability thresholds: 0.15 (high recall), 0.25 (balanced), 0.35 (precise)
# Requires a trained proposal network at $PROPOSAL_MODEL.
# Train with:
#   python -m auto_prompting.train_proposal_net \
#       --train_dir "$NPZ_TRAIN" --val_dir "$NPZ_VAL" --num_epochs 40
# =============================================================================
if [ "$SKIP_UNET" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║   SECTION 3 — UNet-only      ║"
    echo "╚══════════════════════════════╝"

    if [ ! -f "$PROPOSAL_MODEL" ]; then
        echo "  [WARN] Proposal model not found at $PROPOSAL_MODEL"
        echo "  Run the training script first. Skipping UNet section."
    else
        for THRESH in 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95; do
            THRESH_STR=$(echo "$THRESH" | tr '.' '_')
            run_infer "unet_t${THRESH_STR}" \
                --imgs_path       "$NPZ_VAL"        \
                --bbox_mode       unet              \
                --proposal_model  "$PROPOSAL_MODEL" \
                --prob_threshold  "$THRESH"
        done
    fi
fi


# =============================================================================
# SECTION 4 — Hybrid  (PET recall + UNet precision filter)
# 4 PET methods × 3 UNet thresholds = 12 runs.
# Recommended starting point: hybrid_base41_t0_25
# =============================================================================
if [ "$SKIP_HYBRID" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║   SECTION 4 — Hybrid         ║"
    echo "╚══════════════════════════════╝"

    if [ ! -f "$PROPOSAL_MODEL" ]; then
        echo "  [WARN] Proposal model not found at $PROPOSAL_MODEL — skipping Hybrid."
    else
        for PET_METHOD in base41 nestle black daisne; do
            for THRESH in 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95; do
                THRESH_STR=$(echo "$THRESH" | tr '.' '_')
                run_infer "hybrid_${PET_METHOD}_t${THRESH_STR}" \
                    --imgs_path       "$NPZ_VAL"        \
                    --bbox_mode       hybrid            \
                    --pet_method      "$PET_METHOD"     \
                    --proposal_model  "$PROPOSAL_MODEL" \
                    --prob_threshold  "$THRESH"
            done
        done
    fi
fi


# =============================================================================
# SECTION 5 — Evaluate ALL prediction directories
# Writes dsc_results.csv into every pred dir; collects a top-level summary.
# =============================================================================
if [ "$SKIP_EVAL" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║   SECTION 5 — Evaluate ALL   ║"
    echo "╚══════════════════════════════╝"

    # Train-split evaluation uses the train NPZ as GT
    GT_FOR_DIR() {
        if [[ "$1" == *"train"* ]]; then echo "$NPZ_TRAIN"; else echo "$NPZ_VAL"; fi
    }

    SUMMARY_CSV="$PRED_ROOT/all_results_summary.csv"
    echo "pred_dir,dsc_gtvp,dsc_gtvn,dsc_overall" > "$SUMMARY_CSV"

    for PRED_FULL in "$PRED_ROOT"/*/; do
        PRED_DIR=$(basename "$PRED_FULL")
        # Skip per-patient log sub-dirs or other non-prediction dirs
        [[ "$PRED_DIR" == "logs" ]] && continue
        [[ "$PRED_DIR" == "nifti" ]] && continue
        [[ "$PRED_DIR" == "overlays" ]] && continue

        GT_DIR=$(GT_FOR_DIR "$PRED_DIR")
        run_eval "$PRED_DIR" "$GT_DIR"

        # Append the MEAN row from this run's CSV to the top-level summary
        RESULTS_CSV="$PRED_ROOT/$PRED_DIR/dsc_results.csv"
        if [ -f "$RESULTS_CSV" ]; then
            MEAN_ROW=$(grep "^MEAN" "$RESULTS_CSV" || true)
            if [ -n "$MEAN_ROW" ]; then
                DSC_GTVP=$(echo "$MEAN_ROW"  | cut -d',' -f2)
                DSC_GTVN=$(echo "$MEAN_ROW"  | cut -d',' -f3)
                DSC_OVER=$(echo "$MEAN_ROW"  | cut -d',' -f4)
                echo "${PRED_DIR},${DSC_GTVP},${DSC_GTVN},${DSC_OVER}" >> "$SUMMARY_CSV"
            fi
        fi
    done

    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  All-results summary → $SUMMARY_CSV"
    echo "══════════════════════════════════════════════════════════"
    # Print the summary table to stdout for quick comparison
    column -t -s',' "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
fi

echo ""
echo "Done."
