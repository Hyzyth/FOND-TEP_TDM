#!/bin/bash
# =============================================================================
# DualwaveSAM 3-Class Zero-Shot Inference on TempoRAL
# =============================================================================

set -e

# ── 1. Configuration ───────────────────────────────────────────────────────
GPU=1
INFER_BATCH=16
IMG_SIZE=256
N_FILTERS=16
WAVELET="haar"
NUM_CLASSES=3

TEMPORAL_DATA="/data/ethan/PP_temporal_npz"
JSON_FILE="dataset_temporal.json"

# Set this to the best model directory and weights from your DualWave training
MODEL_DIR="DualwaveSAM3c_classic_500ep"
WEIGHTS="model_last.pth"
TARGET_NAME="temporal_zeroshot_vault"

# Paths to the reused SwinCross evaluation scripts
SWIN_DIR="../../P1_SWIN"
EVAL_SCRIPT="${SWIN_DIR}/npz_version/evaluate_predictions.py"
PLOT_METRICS_SCRIPT="${SWIN_DIR}/npz_version/plot_metrics.py"

# ── 2. Environment ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
source dualwave_env/bin/activate

OUT_DIR="/data/ethan/DualwaveSAM3c/${MODEL_DIR}/${TARGET_NAME}"
mkdir -p "$OUT_DIR"

# ── 3. Helper: Sub-Report Generator ────────────────────────────────────────
generate_temporal_sub_reports() {
    local TARGET_DIR=$1
    local RICH_CSV="$TARGET_DIR/per_case_evaluation_rich.csv"

    if [ -f "$RICH_CSV" ]; then
        echo " [Extra] Generating Timepoint Sub-reports for TemPoRAL..."
        
        # Safely strip Windows line endings (\r) to prevent bash/awk silent failures
        local SAFE_CSV="${RICH_CSV}.tmp"
        tr -d '\r' < "$RICH_CSV" > "$SAFE_CSV"

        local HEADER=$(head -1 "$SAFE_CSV")
        local TP_COL=$(head -1 "$SAFE_CSV" | tr ',' '\n' | grep -n "^timepoint$" | cut -d: -f1)

        if [ -n "$TP_COL" ]; then
            # Get unique timepoints, ignoring the header and the MEAN row
            local UNIQUE_TPS=$(tail -n+2 "$SAFE_CSV" | grep -v "^MEAN" | awk -F',' -v col="$TP_COL" '{print $col}' | sort -u | grep -v '^$')
            
            for tp in $UNIQUE_TPS; do
                local SUB_CSV="$TARGET_DIR/per_case_rich_${tp}.csv"
                {
                    echo "$HEADER"
                    # Filter out the header and the MEAN row, then match the timepoint
                    grep -v "^case_id" "$SAFE_CSV" | grep -v "^MEAN" | awk -F',' -v col="$TP_COL" -v tp="$tp" '$col == tp'
                } > "$SUB_CSV"
                
                local N=$(tail -n+2 "$SUB_CSV" | wc -l)
                echo "  ↳ $tp: $N cases → $SUB_CSV"
                
                # Plotting for the specific timepoint
                python3.12 "$PLOT_METRICS_SCRIPT" \
                    --csv_path   "$SUB_CSV" \
                    --output_dir "$TARGET_DIR/plots_${tp}" > /dev/null 2>&1
            done
        fi
        rm -f "$SAFE_CSV" # Clean up temp file
        echo " Sub-reports Complete."
    fi
    echo ""
}

# ── 4. Execution ───────────────────────────────────────────────────────────
echo "╔═════════════════════════════════════════╗"
echo "║ ZERO-SHOT INFERENCE : TempoRAL Dataset" ║
echo "╚═════════════════════════════════════════╝"

# 1. Inference
echo " [1/5] Running Inference..."
CUDA_VISIBLE_DEVICES=$GPU python3.12 adaptation/infer.py \
    --data_dir    "$TEMPORAL_DATA" \
    --json_list   "$JSON_FILE" \
    --checkpoint  "./runs/${MODEL_DIR}/${WEIGHTS}" \
    --output_dir  "$OUT_DIR" \
    --split       "validation" \
    --img_size    $IMG_SIZE \
    --n_filters   $N_FILTERS \
    --wavelet     "$WAVELET" \
    --num_classes $NUM_CLASSES \
    --batch_size  $INFER_BATCH \
    --gpu         0 \
    --skip_existing \
    2>&1 | tee "$OUT_DIR/inference.log"

# 2. Evaluation
echo " [2/5] Evaluating Predictions..."
python3.12 "$EVAL_SCRIPT" \
    --data_dir   "$TEMPORAL_DATA" \
    --json_list  "$JSON_FILE" \
    --output_dir "$OUT_DIR" \
    2>&1 | tee "$OUT_DIR/evaluation.log"

# 3. Global Plotting
echo " [3/5] Plotting Global Metrics..."
python3.12 "$PLOT_METRICS_SCRIPT" \
    --csv_path   "$OUT_DIR/per_case_evaluation_rich.csv" \
    --output_dir "$OUT_DIR/plots"

# 4. Post-Processing Analytics
echo " [4/5] Plotting Post-Processing Analytics..."
python3.12 adaptation/plot_postprocessing.py \
    --csv_path   "$OUT_DIR/postprocessing_logs.csv" \
    --output_dir "$OUT_DIR/plots"

# 5. Stratified Sub-reports
echo " [5/5] Generating Sub-reports by Timepoint..."
generate_temporal_sub_reports "$OUT_DIR"

echo " Zero-Shot Pipeline Complete!"
