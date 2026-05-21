#!/bin/bash
# =============================================================================
# MedSAM2 Inference, Evaluation & Plotting — HECKTOR + TemPoRAL
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-05
#
# Sections:
#   0. Environment & paths
#   1. HECKTOR Inference  (GT / PET / UNet / Hybrid)
#   2. Temporal Zero-Shot Inference
#   3. Evaluate ALL prediction directories
#   4. Plot ALL evaluation CSVs
#   5. Temporal per-timepoint stratification & plots
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

NPZ_HECKTOR_VAL=/data/ethan/MedSAM2/hecktor_npz/val
NPZ_HECKTOR_TRAIN=/data/ethan/MedSAM2/hecktor_npz/train
NPZ_TEMPORAL=/data/ethan/MedSAM2/temporal_npz
TEMPORAL_MANIFEST=$NPZ_TEMPORAL/manifest.json

CHECKPOINT_BASE=./checkpoints/MedSAM2_latest.pt
CHECKPOINT_FINETUNED=./runs/ethan_hecktor_finetuned/checkpoints/checkpoint_300_slim.pt
CFG=sam2/configs/sam2.1_hiera_tiny_hecktor_infer.yaml
PRED_ROOT=/data/ethan/MedSAM2/predictions
PROPOSAL_MODEL=/data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt

mkdir -p "$PRED_ROOT"
mkdir -p /data/ethan/MedSAM2/runs
ln -sfn /data/ethan/MedSAM2/runs ./runs

GPU=0

# ── Per-section skip flags (0=run, 1=skip) ───────────────────────────────────
SKIP_GT=0
SKIP_PET=1
SKIP_UNET=1
SKIP_HYBRID=1
SKIP_TEMPORAL_INFER=0
SKIP_EVAL=0
SKIP_PLOT=0
SKIP_TP_STRAT=0

echo "========================================"
echo "  MedSAM2 × HECKTOR + TemPoRAL"
echo "  Checkpoint : $CHECKPOINT_FINETUNED"
echo "========================================"

# ── Shared inference wrapper ─────────────────────────────────────────────────
run_infer() {
    local PRED_DIR="$1"; shift
    local FULL_PATH="$PRED_ROOT/$PRED_DIR"
    mkdir -p "$FULL_PATH"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Inference: $PRED_DIR"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    CUDA_VISIBLE_DEVICES=$GPU python3.10 inference/infer_npz.py \
        --checkpoint    "$CHECKPOINT_FINETUNED" \
        --cfg           "$CFG"       \
        --pred_save_dir "$FULL_PATH" \
        --save_overlays              \
        "$@"                         \
        2>&1 | tee "$FULL_PATH/inference.log"
}

# ── Shared eval wrapper ───────────────────────────────────────────────────────
# Usage: run_eval <pred_dir_name> <gt_dir> [manifest_path]
run_eval() {
    local PRED_DIR="$1"
    local GT_DIR="$2"
    local MANIFEST="${3:-}"
    local FULL_PATH="$PRED_ROOT/$PRED_DIR"
    if [ ! -d "$FULL_PATH" ]; then
        echo "  [SKIP eval] $PRED_DIR — directory not found."
        return 0
    fi
    echo "  Evaluating: $PRED_DIR"
    MANIFEST_ARG=""
    if [ -n "$MANIFEST" ] && [ -f "$MANIFEST" ]; then
        MANIFEST_ARG="--manifest $MANIFEST"
    fi
    python3.10 inference/evaluate_npz.py \
        --pred_dir   "$FULL_PATH"                  \
        --gt_dir     "$GT_DIR"                     \
        --output_dir "$FULL_PATH"                  \
        $MANIFEST_ARG                              \
        2>&1 | tee "$FULL_PATH/evaluation.log"
}

# ── Shared plot wrapper ───────────────────────────────────────────────────────
run_plot() {
    local PRED_DIR="$1"
    local FULL_PATH="$PRED_ROOT/$PRED_DIR"
    local CSV_PATH="$FULL_PATH/per_case_evaluation_rich.csv"
    if [ ! -f "$CSV_PATH" ]; then
        echo "  [SKIP plot] $PRED_DIR — CSV not found."
        return 0
    fi
    echo "  Plotting: $PRED_DIR"
    python3.10 inference/plot_metrics.py \
        --csv_path   "$CSV_PATH"         \
        --output_dir "$FULL_PATH/plots"  \
        2>&1 | tee "$FULL_PATH/plot.log"
}

# =============================================================================
# SECTION 1 — HECKTOR Inference (GT oracle + auto-prompting modes)
# =============================================================================

if [ "$SKIP_GT" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║  SECTION 1A — GT Oracle      ║"
    echo "╚══════════════════════════════╝"

    # 1a  Val — standard padding (planar=5 vox, slice=1)
    run_infer "gt_val" \
        --imgs_path "$NPZ_HECKTOR_VAL" \
        --bbox_mode gt --bbox_shift 5 --slice_pad 1 --save_nifti

    # 1b  Train — overfit check (same settings, different split)
    run_infer "gt_train_overfit" \
        --imgs_path "$NPZ_HECKTOR_TRAIN" \
        --bbox_mode gt --bbox_shift 5 --slice_pad 1

    # 1c  Val tight — zero padding (DSC ceiling with exact GT boxes)
    run_infer "gt_val_tight" \
        --imgs_path "$NPZ_HECKTOR_VAL" \
        --bbox_mode gt --bbox_shift 0 --slice_pad 0 --save_nifti

    # 1d  Val reduced — negative padding (DSC ceiling with exact GT boxes)
    run_infer "gt_val_reduced" \
        --imgs_path "$NPZ_HECKTOR_VAL" \
        --bbox_mode gt --bbox_shift -5 --slice_pad -1 --save_nifti

    # 1e Val extra reduced — more aggressive negative padding (DSC ceiling with exact GT boxes)
    run_infer "gt_val_extra_reduced" \
        --imgs_path "$NPZ_HECKTOR_VAL" \
        --bbox_mode gt --bbox_shift -10 --slice_pad -2 --save_nifti
fi

# PET-only auto-prompting
if [ "$SKIP_PET" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║  SECTION 1B — PET-only       ║"
    echo "╚══════════════════════════════╝"
    for PET_METHOD in base41 nestle black daisne; do
        run_infer "pet_${PET_METHOD}" \
            --imgs_path "$NPZ_HECKTOR_VAL" \
            --bbox_mode pet --pet_method "$PET_METHOD"
    done
fi

# UNet-only auto-prompting
if [ "$SKIP_UNET" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║  SECTION 1C — UNet-only      ║"
    echo "╚══════════════════════════════╝"
    if [ ! -f "$PROPOSAL_MODEL" ]; then
        echo "  [WARN] Proposal model not found — skipping."
    else
        for THRESH in 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95; do
            THRESH_STR=$(echo "$THRESH" | tr '.' '_')
            run_infer "unet_t${THRESH_STR}" \
                --imgs_path "$NPZ_HECKTOR_VAL" \
                --bbox_mode unet \
                --proposal_model "$PROPOSAL_MODEL" \
                --prob_threshold "$THRESH"
        done
    fi
fi

# Hybrid auto-prompting (UNet + PET)
if [ "$SKIP_HYBRID" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║  SECTION 1D — Hybrid         ║"
    echo "╚══════════════════════════════╝"
    if [ ! -f "$PROPOSAL_MODEL" ]; then
        echo "  [WARN] Proposal model not found — skipping."
    else
        for PET_METHOD in base41 nestle black daisne; do
            for THRESH in 0.05 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95; do
                THRESH_STR=$(echo "$THRESH" | tr '.' '_')
                run_infer "hybrid_${PET_METHOD}_t${THRESH_STR}" \
                    --imgs_path "$NPZ_HECKTOR_VAL" \
                    --bbox_mode hybrid \
                    --pet_method "$PET_METHOD" \
                    --proposal_model "$PROPOSAL_MODEL" \
                    --prob_threshold "$THRESH"
            done
        done
    fi
fi

# =============================================================================
# SECTION 2 — Temporal Zero-Shot Inference
# =============================================================================
if [ "$SKIP_TEMPORAL_INFER" -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════╗"
    echo "║  SECTION 2 — Temporal Zero-Shot Inference  ║"
    echo "╚════════════════════════════════════════════╝"

    # GT oracle on temporal (uses RTStruct masks as prompts where available)
    run_infer "temporal_gt_zeroshot" \
        --imgs_path "$NPZ_TEMPORAL" \
        --bbox_mode gt --bbox_shift 5 --slice_pad 1 --save_nifti

    # Best auto-prompting config (adapt as needed after HECKTOR evaluation)
    run_infer "temporal_pet_base41" \
        --imgs_path "$NPZ_TEMPORAL" \
        --bbox_mode pet --pet_method base41

    if [ -f "$PROPOSAL_MODEL" ]; then
        run_infer "temporal_hybrid_base41" \
            --imgs_path "$NPZ_TEMPORAL" \
            --bbox_mode hybrid --pet_method base41 \
            --proposal_model "$PROPOSAL_MODEL" --prob_threshold 0.25
    fi
fi

# =============================================================================
# SECTION 3 — Evaluate ALL prediction directories
# =============================================================================
if [ "$SKIP_EVAL" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║  SECTION 3 — Evaluate ALL    ║"
    echo "╚══════════════════════════════╝"

    SUMMARY_CSV="$PRED_ROOT/all_results_summary.csv"
    echo "pred_dir,mean_dice,mean_dice_gtvp,mean_dice_gtvn" > "$SUMMARY_CSV"

    for PRED_FULL in "$PRED_ROOT"/*/; do
        PRED_DIR=$(basename "$PRED_FULL")
        # Skip non-prediction subdirs
        [[ "$PRED_DIR" == "logs" || "$PRED_DIR" == "nifti" || "$PRED_DIR" == "overlays" ]] && continue

        # Choose GT dir and manifest based on dir name
        if [[ "$PRED_DIR" == temporal_* ]]; then
            GT_DIR="$NPZ_TEMPORAL"
            MANIFEST="$TEMPORAL_MANIFEST"
        elif [[ "$PRED_DIR" == *train* ]]; then
            GT_DIR="$NPZ_HECKTOR_TRAIN"
            MANIFEST=""
        else
            GT_DIR="$NPZ_HECKTOR_VAL"
            MANIFEST=""
        fi

        run_eval "$PRED_DIR" "$GT_DIR" "$MANIFEST"

        # Append mean row to summary
        RICH_CSV="$PRED_FULL/per_case_evaluation_rich.csv"
        if [ -f "$RICH_CSV" ]; then
            MEAN_DICE=$(awk -F',' 'NR>1 && $13!="" {s+=$13; n++} END {if(n>0) printf "%.4f", s/n}' "$RICH_CSV")
            MEAN_P=$(awk -F',' 'NR>1 && $11!="" {s+=$11; n++} END {if(n>0) printf "%.4f", s/n}' "$RICH_CSV")
            MEAN_N=$(awk -F',' 'NR>1 && $12!="" {s+=$12; n++} END {if(n>0) printf "%.4f", s/n}' "$RICH_CSV")
            echo "${PRED_DIR},${MEAN_DICE},${MEAN_P},${MEAN_N}" >> "$SUMMARY_CSV"
        fi
    done

    echo ""
    echo "══════════════════════════════════════════════"
    echo "  All-results summary → $SUMMARY_CSV"
    echo "══════════════════════════════════════════════"
    column -t -s',' "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
fi

# =============================================================================
# SECTION 4 — Plot ALL evaluation CSVs
# =============================================================================
if [ "$SKIP_PLOT" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════╗"
    echo "║  SECTION 4 — Plot ALL        ║"
    echo "╚══════════════════════════════╝"
    for PRED_FULL in "$PRED_ROOT"/*/; do
        PRED_DIR=$(basename "$PRED_FULL")
        [[ "$PRED_DIR" == "logs" || "$PRED_DIR" == "nifti" || "$PRED_DIR" == "overlays" ]] && continue
        run_plot "$PRED_DIR"
    done
fi

# =============================================================================
# SECTION 5 — Temporal per-timepoint stratification & plots
# =============================================================================
if [ "$SKIP_TP_STRAT" -eq 0 ]; then
    echo ""
    echo "╔════════════════════════════════════════════╗"
    echo "║  SECTION 5 — Temporal Timepoint Sub-Reports║"
    echo "╚════════════════════════════════════════════╝"

    for TEMPORAL_PRED_DIR in temporal_gt_zeroshot temporal_pet_base41 temporal_hybrid_base41; do
        RICH_CSV="$PRED_ROOT/$TEMPORAL_PRED_DIR/per_case_evaluation_rich.csv"
        if [ ! -f "$RICH_CSV" ]; then
            echo "  [SKIP] $TEMPORAL_PRED_DIR — CSV not found."
            continue
        fi

        echo "=== Timepoint sub-reports for: $TEMPORAL_PRED_DIR ==="
        HEADER=$(head -1 "$RICH_CSV")
        TP_COL=$(head -1 "$RICH_CSV" | tr ',' '\n' | grep -n "^timepoint$" | cut -d: -f1)

        if [ -z "$TP_COL" ]; then
            echo "  [WARN] 'timepoint' column not found in CSV."
            continue
        fi

        UNIQUE_TPS=$(tail -n+2 "$RICH_CSV" | awk -F',' -v col="$TP_COL" '{print $col}' | sort -u | grep -v '^$')

        for tp in $UNIQUE_TPS; do
            SUB_DIR="$PRED_ROOT/$TEMPORAL_PRED_DIR"
            SUB_CSV="$SUB_DIR/per_case_rich_${tp}.csv"
            {
                echo "$HEADER"
                tail -n+2 "$RICH_CSV" | awk -F',' -v col="$TP_COL" -v tp="$tp" '$col == tp'
            } > "$SUB_CSV"
            N=$(tail -n+2 "$SUB_CSV" | wc -l)
            echo "  Generated $tp report: $SUB_CSV ($N cases)"

            python3.10 inference/plot_metrics.py \
                --csv_path   "$SUB_CSV"                           \
                --output_dir "$SUB_DIR/plots_${tp}"               \
                2>&1 | tee "$SUB_DIR/plot_${tp}.log"
        done
    done
fi

echo ""
echo "Pipeline complete."
