#!/bin/bash
# =============================================================================
# MedSAM2 Inference, Evaluation & Plotting - HECKTOR 2026 + TemPoRAL
# Project : ProjetMaster / StageM1_IA
# Author  : Ethan
# Updated : 2026-05
#
# Data sources
# ------------
#   HECKTOR 2026  : /data/ethan/PP_hecktor2026_kfold_npz/  (SwinCross NPZ)
#   TemPoRAL      : /data/ethan/MedSAM2/temporal_npz/      (TemPoRAL NPZ)
#
# Split alignment with SwinCross + DualwaveSAM
# ---------------------------------------------
#   Training      : dataset_swincross_2026kfold_classic.json  "training" key
#   Val (overfit) : dataset_swincross_2026kfold_classic.json  "validation" key
#   Locked test   : dataset_swincross_2026kfold_test.json     "validation" key
#   TemPoRAL      : manifest.json  "cases" list  (zero-shot, never used in training)
#
# Per-target pipeline
# --------------------
#   [1] infer_npz.py          -> <case_id>_Pred.nii.gz  +  postprocessing_logs.csv
#   [2] evaluate_npz.py       -> per_case_evaluation_rich.csv  (+ MEAN row)
#   [3] plot_metrics.py       -> 14-plot suite  (shared with SwinCross)
#   [4] plot_postprocessing.py-> PP01-PP04 analytics
#
# NPZ format auto-detection
# --------------------------
#   infer_npz.py detects SwinCross vs TemPoRAL per file from NPZ keys.
#   Raw SUV (float32) is threaded to the auto-prompter for HECKTOR 2026
#   (nestle/black/daisne use it directly - no uint8 round-trip loss).
#   For TemPoRAL, pet_suv_max is used to reconstruct SUV from uint8 as before.
# =============================================================================

set -e

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                          GLOBAL CONFIGURATION                          ║
# ╚════════════════════════════════════════════════════════════════════════╝

# ── 1. Execution Toggles (0=run, 1=skip) ──────────────────────────────────
SKIP_GT_TEST=1          # GT oracle - locked test vault  (recommended first)
SKIP_GT_TRAIN=1         # GT oracle - train pool overfit check
SKIP_GT_VAL=1           # GT oracle - val pool overfit check

SKIP_PET=1              # PET auto-prompting - test vault
SKIP_UNET=1             # UNet auto-prompting - test vault
SKIP_HYBRID=1           # Hybrid auto-prompting - test vault

SKIP_TEMPORAL=0         # TemPoRAL zero-shot inference
SKIP_TP_STRAT=0         # Per-timepoint sub-reports for TemPoRAL

# ── 2. Hardware & Paths ────────────────────────────────────────────────────
GPU=0

HECKTOR_DATA="/data/ethan/PP_hecktor2026_kfold_npz"
JSON_TEST="dataset_swincross_2026kfold_test.json"
JSON_CLASSIC="dataset_swincross_2026kfold_classic.json"

# Separate train-only and val-only JSONs (produced by prepare_hecktor2026_kfold_npz.py)
JSON_TRAIN="dataset_swincross_2026kfold_classic_train.json"
JSON_VAL="dataset_swincross_2026kfold_classic_val.json"

TEMPORAL_DATA="/data/ethan/MedSAM2/temporal_npz"
TEMPORAL_MANIFEST="${TEMPORAL_DATA}/manifest.json"

CHECKPOINT="./runs/ethan_hecktor_finetuned/checkpoints/checkpoint.pt"
CFG="sam2/configs/sam2.1_hiera_tiny_hecktor_infer.yaml"
PROPOSAL_MODEL="/data/ethan/MedSAM2/proposal_net/checkpoints/proposal_net_best.pt"

PRED_ROOT="/data/ethan/MedSAM2/predictions"
mkdir -p "$PRED_ROOT"
mkdir -p /data/ethan/MedSAM2/runs
ln -sfn /data/ethan/MedSAM2/runs ./runs

# ── 3. Shared SwinCross plot_metrics.py ───────────────────────────────────
SWIN_DIR="../../P1_SWIN"
PLOT_METRICS="${SWIN_DIR}/adaptation/plot_metrics.py"

# ── 4. Auto-prompting sweep config ────────────────────────────────────────
PET_METHODS="base41 nestle black daisne"
UNET_THRESHOLDS="0.05 0.15 0.25 0.35 0.45"


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                          ENVIRONMENT                                   ║
# ╚════════════════════════════════════════════════════════════════════════╝
if [ ! -d "medsam2_env" ]; then
    echo "medsam2_env not found. Run MedSAM2_dataset_building.sh first."
    exit 1
fi
source medsam2_env/bin/activate
uv pip install -q matplotlib seaborn pandas SimpleITK tqdm

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "  Checkpoint not found: $CHECKPOINT"
    echo "  Run MedSAM2_training.sh first."
    exit 1
fi

# Verify TemPoRAL manifest when needed
if [ "$SKIP_TEMPORAL" -eq 0 ] && [ ! -f "$TEMPORAL_MANIFEST" ]; then
    echo "  TemPoRAL manifest not found: $TEMPORAL_MANIFEST"
    echo "  Run MedSAM2_dataset_building.sh first (TemPoRAL preparation step)."
    exit 1
fi


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                        HELPER FUNCTIONS                                ║
# ╚════════════════════════════════════════════════════════════════════════╝

# ── run_infer <out_dir_name> <data_dir> <json_list> <split> [extra_args]
run_infer() {
    local OUT_NAME="$1"; shift
    local DATA="$1";     shift
    local JSON="$1";     shift
    local SPLIT="$1";    shift
    local FULL="$PRED_ROOT/$OUT_NAME"
    mkdir -p "$FULL"

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Inference -> $OUT_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    CUDA_VISIBLE_DEVICES=$GPU python3.10 inference/infer_npz.py \
        --checkpoint    "$CHECKPOINT" \
        --cfg           "$CFG" \
        --data_dir      "$DATA" \
        --json_list     "$JSON" \
        --split         "$SPLIT" \
        --pred_save_dir "$FULL" \
        --save_overlays \
        --skip_existing \
        "$@" \
        2>&1 | tee "$FULL/inference.log"
}

# ── run_eval <out_dir_name> <data_dir> <json_list>
run_eval() {
    local OUT_NAME="$1"
    local DATA="$2"
    local JSON="$3"
    local FULL="$PRED_ROOT/$OUT_NAME"

    [ -d "$FULL" ] || { echo "  [SKIP eval] $OUT_NAME - dir not found."; return 0; }
    echo "  [Eval] $OUT_NAME"

    python3.10 inference/evaluate_npz.py \
        --data_dir   "$DATA" \
        --json_list  "$JSON" \
        --output_dir "$FULL" \
        2>&1 | tee "$FULL/evaluation.log"
}

# ── run_plot_metrics <out_dir_name>
run_plot_metrics() {
    local OUT_NAME="$1"
    local CSV="$PRED_ROOT/$OUT_NAME/per_case_evaluation_rich.csv"
    [ -f "$CSV" ] || { echo "  [SKIP plot_metrics] $OUT_NAME"; return 0; }
    echo "  [Plot metrics] $OUT_NAME"
    python3.10 "$PLOT_METRICS" \
        --csv_path   "$CSV" \
        --output_dir "$PRED_ROOT/$OUT_NAME/plots" \
        2>&1 | tee "$PRED_ROOT/$OUT_NAME/plot_metrics.log"
}

# ── run_plot_pp <out_dir_name>
run_plot_pp() {
    local OUT_NAME="$1"
    local CSV="$PRED_ROOT/$OUT_NAME/postprocessing_logs.csv"
    [ -f "$CSV" ] || { echo "  [SKIP plot_pp] $OUT_NAME"; return 0; }
    echo "  [Plot post-proc] $OUT_NAME"
    python3.10 inference/plot_postprocessing.py \
        --csv_path   "$CSV" \
        --output_dir "$PRED_ROOT/$OUT_NAME/plots" \
        2>&1 | tee "$PRED_ROOT/$OUT_NAME/plot_postprocessing.log"
}

# ── full_pipeline <out_dir_name> <data_dir> <json_list> <split> [extra_args]
full_pipeline() {
    local OUT_NAME="$1"; shift
    local DATA="$1";     shift
    local JSON="$1";     shift
    local SPLIT="$1";    shift
    run_infer        "$OUT_NAME" "$DATA" "$JSON" "$SPLIT" "$@"
    run_eval         "$OUT_NAME" "$DATA" "$JSON"
    run_plot_metrics "$OUT_NAME"
    run_plot_pp      "$OUT_NAME"
}

# ── temporal_sub_reports <out_dir_name>
temporal_sub_reports() {
    local OUT_NAME="$1"
    local TARGET="$PRED_ROOT/$OUT_NAME"
    local CSV="$TARGET/per_case_evaluation_rich.csv"

    [ -f "$CSV" ] || return 0
    echo " [Sub-reports] $OUT_NAME"

    local SAFE="${CSV}.tmp"
    tr -d '\r' < "$CSV" > "$SAFE"

    local HEADER
    HEADER=$(head -1 "$SAFE")
    local TP_COL
    TP_COL=$(head -1 "$SAFE" | tr ',' '\n' | grep -n "^timepoint$" | cut -d: -f1)

    if [ -n "$TP_COL" ]; then
        local TPS
        TPS=$(tail -n+2 "$SAFE" | grep -v "^MEAN" \
              | awk -F',' -v col="$TP_COL" '{print $col}' \
              | sort -u | grep -v '^$')
        for tp in $TPS; do
            local SUB="$TARGET/per_case_rich_${tp}.csv"
            {
                echo "$HEADER"
                grep -v "^case_id" "$SAFE" | grep -v "^MEAN" \
                | awk -F',' -v col="$TP_COL" -v tp="$tp" '$col == tp'
            } > "$SUB"
            local N
            N=$(tail -n+2 "$SUB" | wc -l)
            echo "  ↳ $tp: $N cases -> $SUB"
            
            # ── NEW: Calculate and append the local MEAN row using Pandas ──
            python3.10 -c '
import sys
try:
    import pandas as pd
    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)
    if len(df) > 0:
        num_cols = df.select_dtypes(include=["float64", "int64"]).columns
        mean_vals = df[num_cols].mean().round(4).to_dict()
        mean_row = {c: "" for c in df.columns}
        mean_row.update(mean_vals)
        mean_row["case_id"] = "MEAN"
        mean_row["patient"] = "ALL_CASES"
        mean_row["comments"] = f"Average across {len(df)} cases"
        pd.DataFrame([mean_row]).to_csv(csv_path, mode="a", header=False, index=False)
except ImportError:
    pass
' "$SUB"
            # ───────────────────────────────────────────────────────────────

            python3.10 "$PLOT_METRICS" \
                --csv_path   "$SUB" \
                --output_dir "$TARGET/plots_${tp}" > /dev/null 2>&1
        done
    fi
    rm -f "$SAFE"
    echo " Sub-reports complete."
    echo ""
}


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                              EXECUTION                                 ║
# ╚════════════════════════════════════════════════════════════════════════╝
echo "========================================"
echo "  MedSAM2 Inference & Evaluation"
echo "  Checkpoint : $CHECKPOINT"
echo "========================================"

# ── 1. GT Oracle ──────────────────────────────────────────────────────────
if [ "$SKIP_GT_TEST" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  GT Oracle - Locked Test Vault       ║"
    echo "╚══════════════════════════════════════╝"
    # a. Same conditions as during training (tight bounding boxes)
    full_pipeline "gt_test_vault_base" \
        "$HECKTOR_DATA" "$JSON_TEST" "validation" \
        --bbox_mode gt --bbox_shift 0 --slice_pad 0
    
    # b. Slightly looser bounding boxes (5 voxels shift) to test robustness
    full_pipeline "gt_test_vault_shift5" \
        "$HECKTOR_DATA" "$JSON_TEST" "validation" \
        --bbox_mode gt --bbox_shift 5 --slice_pad 1
    
    # c. Slightly tighter bounding boxes (5 voxels shift) to test robustness
    full_pipeline "gt_test_vault_shift-5" \
        "$HECKTOR_DATA" "$JSON_TEST" "validation" \
        --bbox_mode gt --bbox_shift -5 --slice_pad -1
fi

if [ "$SKIP_GT_TRAIN" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  GT Oracle - Train Pool (overfit)    ║"
    echo "╚══════════════════════════════════════╝"
    full_pipeline "gt_overfit_train" \
        "$HECKTOR_DATA" "$JSON_TRAIN" "validation" \
        --bbox_mode gt --bbox_shift 5 --slice_pad 1
fi

if [ "$SKIP_GT_VAL" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  GT Oracle - Val Pool (overfit)      ║"
    echo "╚══════════════════════════════════════╝"
    full_pipeline "gt_overfit_val" \
        "$HECKTOR_DATA" "$JSON_VAL" "validation" \
        --bbox_mode gt --bbox_shift 5 --slice_pad 1
fi

# ── 2. PET Auto-prompting ─────────────────────────────────────────────────
if [ "$SKIP_PET" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  PET Auto-prompting - Test Vault     ║"
    echo "║  (raw SUV from SwinCross NPZ float16)║"
    echo "╚══════════════════════════════════════╝"
    for METHOD in $PET_METHODS; do
        full_pipeline "pet_${METHOD}_test" \
            "$HECKTOR_DATA" "$JSON_TEST" "validation" \
            --bbox_mode pet --pet_method "$METHOD" \
            --bbox_shift 5 --slice_pad 1
    done
fi

# ── 3. UNet Auto-prompting ────────────────────────────────────────────────
if [ "$SKIP_UNET" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  UNet Auto-prompting - Test Vault    ║"
    echo "╚══════════════════════════════════════╝"
    if [ ! -f "$PROPOSAL_MODEL" ]; then
        echo "  [WARN] Proposal model not found: $PROPOSAL_MODEL - skipping."
    else
        for THRESH in $UNET_THRESHOLDS; do
            THRESH_STR=$(echo "$THRESH" | tr '.' '_')
            full_pipeline "unet_t${THRESH_STR}_test" \
                "$HECKTOR_DATA" "$JSON_TEST" "validation" \
                --bbox_mode unet \
                --proposal_model "$PROPOSAL_MODEL" \
                --prob_threshold "$THRESH" \
                --bbox_shift 5 --slice_pad 1
        done
    fi
fi

# ── 4. Hybrid Auto-prompting ──────────────────────────────────────────────
if [ "$SKIP_HYBRID" -eq 0 ]; then
    echo ""
    echo "╔══════════════════════════════════════╗"
    echo "║  Hybrid Auto-prompting - Test Vault  ║"
    echo "╚══════════════════════════════════════╝"
    if [ ! -f "$PROPOSAL_MODEL" ]; then
        echo "  [WARN] Proposal model not found: $PROPOSAL_MODEL - skipping."
    else
        for METHOD in $PET_METHODS; do
            for THRESH in $UNET_THRESHOLDS; do
                THRESH_STR=$(echo "$THRESH" | tr '.' '_')
                full_pipeline "hybrid_${METHOD}_t${THRESH_STR}_test" \
                    "$HECKTOR_DATA" "$JSON_TEST" "validation" \
                    --bbox_mode hybrid \
                    --pet_method "$METHOD" \
                    --proposal_model "$PROPOSAL_MODEL" \
                    --prob_threshold "$THRESH" \
                    --bbox_shift 5 --slice_pad 1
            done
        done
    fi
fi

# ── 5. TemPoRAL Zero-Shot ─────────────────────────────────────────────────
if [ "$SKIP_TEMPORAL" -eq 0 ]; then
    echo ""
    echo "╔═════════════════════════════════════════════╗"
    echo "║  TemPoRAL Zero-Shot Inference               ║"
    echo "║  (TemPoRAL NPZ format, manifest.json)       ║"
    echo "╚═════════════════════════════════════════════╝"

    # 5a. GT oracle (uses RTStruct masks where available)
    full_pipeline "temporal_gt_zeroshot" \
        "$TEMPORAL_DATA" "manifest.json" "validation" \
        --bbox_mode gt --bbox_shift 0 --slice_pad 0

    # # 5b. PET base41 (scale-invariant, always available)
    # full_pipeline "temporal_pet_base41" \
    #     "$TEMPORAL_DATA" "manifest.json" "validation" \
    #     --bbox_mode pet --pet_method base41 \
    #     --bbox_shift 5 --slice_pad 1

    # # 5c. PET nestle (uses pet_suv_max reconstructed from uint8)
    # full_pipeline "temporal_pet_nestle" \
    #     "$TEMPORAL_DATA" "manifest.json" "validation" \
    #     --bbox_mode pet --pet_method nestle \
    #     --bbox_shift 5 --slice_pad 1

    # # 5d. Hybrid base41 (if proposal model available)
    # if [ -f "$PROPOSAL_MODEL" ]; then
    #     full_pipeline "temporal_hybrid_base41" \
    #         "$TEMPORAL_DATA" "manifest.json" "validation" \
    #         --bbox_mode hybrid --pet_method base41 \
    #         --proposal_model "$PROPOSAL_MODEL" --prob_threshold 0.25 \
    #         --bbox_shift 5 --slice_pad 1
    # fi
fi

# ── 6. TemPoRAL Timepoint Sub-Reports ─────────────────────────────────────
if [ "$SKIP_TP_STRAT" -eq 0 ]; then
    echo ""
    echo "╔═════════════════════════════════════════════╗"
    echo "║  TemPoRAL Timepoint Stratification          ║"
    echo "╚═════════════════════════════════════════════╝"
    for RUN in temporal_gt_zeroshot;
                # temporal_pet_base41 temporal_pet_nestle temporal_hybrid_base41; 
               do
        [ -d "$PRED_ROOT/$RUN" ] && temporal_sub_reports "$RUN"
    done
fi

# ── 7. All-results summary ────────────────────────────────────────────────
SUMMARY="$PRED_ROOT/all_results_summary.csv"
echo "pred_dir,mean_dice,mean_dice_gtvp,mean_dice_gtvn" > "$SUMMARY"

for PRED_FULL in "$PRED_ROOT"/*/; do
    PRED_DIR=$(basename "$PRED_FULL")
    [[ "$PRED_DIR" == "logs" ]] && continue
    CSV="$PRED_FULL/per_case_evaluation_rich.csv"
    
    if [ -f "$CSV" ]; then
        # Use Pandas to safely extract the exact MEAN row, ignoring CSV quoting issues
        METRICS=$(python3.10 -c '
import sys
try:
    import pandas as pd
    df = pd.read_csv(sys.argv[1])
    
    # Grab the pre-calculated MEAN row, or calculate it if it is missing
    if "MEAN" in df["case_id"].values:
        row = df[df["case_id"] == "MEAN"].iloc[0]
    else:
        row = df.mean(numeric_only=True)
    
    def safe_fmt(val):
        return "0.0000" if pd.isna(val) or val == "" else f"{float(val):.4f}"
        
    # FIX: Assign to variables first to avoid backslashes inside the f-string
    md = safe_fmt(row.get("mean_dice"))
    tp = safe_fmt(row.get("GTVp_dice"))
    tn = safe_fmt(row.get("GTVn_dice"))
    print(f"{md},{tp},{tn}")
    
except Exception:
    print("0.0000,0.0000,0.0000")
' "$CSV")
        
        echo "${PRED_DIR},${METRICS}" >> "$SUMMARY"
    fi
done

echo ""
echo "══════════════════════════════════════════════"
echo "  Results summary -> $SUMMARY"
echo "══════════════════════════════════════════════"
column -t -s',' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo ""
echo "Pipeline complete."
