#!/bin/bash
# =============================================================================
# DualwaveSAM 3-Class Training Script
# Author  : Ethan
# Project : ProjetMaster / StageM1_IA
# Updated : 2026-05
#
# Data source : /data/ethan/PP_hecktor2026_kfold_npz/  (SwinCross NPZ format)
# All outputs : /data/ethan/DualwaveSAM3c/<MODEL_DIR>/
#
# Uses the SAME folds as SwinCross for direct model comparison.
# Run the SwinCross Dataset Building script first if NPZ data does not exist.
# =============================================================================

set -e
set -o pipefail

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                          GLOBAL CONFIGURATION                          ║
# ╚════════════════════════════════════════════════════════════════════════╝
# ── 1. Execution Toggles ───────────────────────────────────────────────────
RUN_TEST=false
RUN_CLASSIC_TRAIN=true
RUN_CLASSIC_RESUME=false

RUN_KFOLD_TRAIN=true
RUN_KFOLD_PRODUCTION_FULL=false   # Train on 100% of pool (no held-out val)

# ── 2. Hardware & Hyperparameters ──────────────────────────────────────────
GPU=0
BATCH_SIZE=48           # 2D slices - can be much larger than 3D patch batches
VAL_EVERY=1             # Validate every N epochs
WARMUP_CLASSIC=10
WARMUP_KFOLD=3
LRATE=1e-4

EPOCH_NUMBER_CLASSIC=500
K_FOLDS=5
KFOLD_START=0
# Scale kfold epochs to keep total GPU time comparable to classic run
EPOCH_NUMBER_KFOLD=$(($EPOCH_NUMBER_CLASSIC / $K_FOLDS))

# ── 3. Model Architecture ──────────────────────────────────────────────────
IMG_SIZE=256
N_FILTERS=16            # WaveEncoder base filters (16 → 256-ch bottleneck)
WAVELET="haar"          # haar | db2 | db3 | sym4 (PyWavelets name)
NUM_CLASSES=3           # 0=background, 1=GTVp, 2=GTVn

# ── 4. Loss Hyperparameters ────────────────────────────────────────────────
LAMBDA1=0.01            # MAE (L1) loss weight for regularization
LAMBDA2=0.1             # MSE (L2) loss weight for regularization
GAMMA=2.0               # Focal exponent
BG_RATIO=0.3            # Fraction of background slices

# ── 5. Data Paths ──────────────────────────────────────────────────────────
PPDATA_FOLDER="/data/ethan/PP_hecktor2026_kfold_npz"
JSON_PREFIX="dataset_swincross_2026kfold"
LMDB_OUT_DIR="/data/ethan/DualwaveSAM3c/lmdb_cache"  # Optional LMDB cache for faster loading

# ── 6. Output Naming ───────────────────────────────────────────────────────
TEST_DIR="DualwaveSAM3c_test"
CLASSIC_MODEL_DIR="DualwaveSAM3c_classic_${EPOCH_NUMBER_CLASSIC}ep"
KFOLD_BASE_DIR="DualwaveSAM3c_kfold_${EPOCH_NUMBER_KFOLD}ep"


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                              ENVIRONMENT                               ║
# ╚════════════════════════════════════════════════════════════════════════╝
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"   # go to DualwaveSAM root

if ! command -v uv &> /dev/null; then
    echo "uv not found - installing..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

uv python install 3.12
[ ! -d "dualwave_env" ] && uv venv dualwave_env --python 3.12
source dualwave_env/bin/activate
[ -f requirements.txt ] && uv pip install -r requirements.txt || \
    { echo "[WARN] requirements.txt not found - ensure deps are installed"; }

mkdir -p /data/ethan/DualwaveSAM3c
ln -sfn /data/ethan/DualwaveSAM3c ./runs
mkdir -p "$LMDB_OUT_DIR"


# ╔════════════════════════════════════════════════════════════════════════╗
# ║                          JIT CACHE MANAGER                             ║
# ╚════════════════════════════════════════════════════════════════════════╝

BUILD_PID=""

wait_for_build() {
    if [ -n "$BUILD_PID" ]; then
        echo "  [Cache] Waiting for background build (PID: $BUILD_PID) to finish..."
        wait $BUILD_PID
        BUILD_PID=""
    fi
}

prep_cache() {
    local current_json=$1
    local next_json=$2
    
    echo "──────────────────────────────────────────────────────────────"
    echo "  [Cache] Preparing cache for: $current_json"
    
    # 1. Clear unneeded caches to save disk space
    for cache_item in "$LMDB_OUT_DIR"/*; do
        [ -e "$cache_item" ] || continue
        local item_name=$(basename "$cache_item")
        local current_base="${current_json%.json}"
        local next_base=""
        [ -n "$next_json" ] && next_base="${next_json%.json}"

        if [[ "$item_name" != "$current_base"* && "$item_name" != "$next_base"* ]]; then
            echo "    -> Deleting old cache data: $item_name"
            rm -rf "$cache_item"
        fi
    done

    # 2. Ensure CURRENT cache is fully built
    local current_done="$LMDB_OUT_DIR/${current_json%.json}.done"
    if [ ! -f "$current_done" ]; then
        wait_for_build # In case it's currently building from a previous lookahead
        if [ ! -f "$current_done" ]; then
            echo "    -> Building CURRENT cache (foreground): $current_json"
            python3.12 adaptation/build_lmdb_cache.py \
                --data_dir "$PPDATA_FOLDER" \
                --json_list "$current_json" \
                --out_dir "$LMDB_OUT_DIR"
            touch "$current_done"
        fi
    else
        echo "    -> CURRENT cache already exists."
    fi

    # 3. Lookahead: Start NEXT build in background (if provided and needed)
    if [ -n "$next_json" ]; then
        local next_done="$LMDB_OUT_DIR/${next_json%.json}.done"
        if [ ! -f "$next_done" ]; then
            echo "    -> Building NEXT cache (background, low I/O priority): $next_json"
            (
                # Use nice (CPU) and ionice (Disk) to prevent starving the GPU
                nice -n 10 ionice -c 2 -n 7 python3.12 adaptation/build_lmdb_cache.py \
                    --data_dir "$PPDATA_FOLDER" \
                    --json_list "$next_json" \
                    --out_dir "$LMDB_OUT_DIR"
                touch "$next_done"
            ) &
            BUILD_PID=$!
        else
            echo "    -> NEXT cache already exists."
        fi
    fi
    echo "──────────────────────────────────────────────────────────────"
}

# ╔════════════════════════════════════════════════════════════════════════╗
# ║                               EXECUTION                                ║
# ╚════════════════════════════════════════════════════════════════════════╝

# ── 0. Sanity Check ───────────────────────────────────────────────────────
if [ "$RUN_TEST" = true ]; then
    echo "╔════════════╗"
    echo "║  Test Run  ║"
    echo "╚════════════╝"
    mkdir -p /data/ethan/DualwaveSAM3c/$TEST_DIR

    prep_cache "${JSON_PREFIX}_classic.json" ""

    CUDA_VISIBLE_DEVICES=$GPU \
    python3.12 adaptation/train.py \
        --data_dir      "$PPDATA_FOLDER" \
        --json_list     "${JSON_PREFIX}_classic.json" \
        --logdir        "$TEST_DIR" \
        --max_epochs    20 \
        --batch_size    $BATCH_SIZE \
        --val_every     1 \
        --optim_lr      $LRATE \
        --warmup_epochs 5 \
        --img_size      $IMG_SIZE \
        --n_filters     $N_FILTERS \
        --wavelet       "$WAVELET" \
        --num_classes   $NUM_CLASSES \
        --bg_ratio      $BG_RATIO \
        --lambda1       $LAMBDA1 \
        --lambda2       $LAMBDA2 \
        --gamma         $GAMMA \
        --lrschedule    warmup_cosine \
        --gpu           0 \
        2>&1 | tee /data/ethan/DualwaveSAM3c/$TEST_DIR/test_run.log
    echo "  Test run complete."
fi

# ── A. CLASSIC TRAINING ───────────────────────────────────────────────────
if [ "$RUN_CLASSIC_TRAIN" = true ]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  CLASSIC TRAINING (${EPOCH_NUMBER_CLASSIC} Epochs)                              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    mkdir -p /data/ethan/DualwaveSAM3c/$CLASSIC_MODEL_DIR

    # Determine what to cache next
    NEXT_JSON=""
    if [ "$RUN_CLASSIC_RESUME" = true ]; then
        NEXT_JSON="${JSON_PREFIX}_classic.json" # No-op, essentially
    elif [ "$RUN_KFOLD_TRAIN" = true ]; then
        NEXT_JSON="${JSON_PREFIX}_fold${KFOLD_START}.json"
    elif [ "$RUN_KFOLD_PRODUCTION_FULL" = true ]; then
        NEXT_JSON="${JSON_PREFIX}_full.json"
    fi

    prep_cache "${JSON_PREFIX}_classic.json" "$NEXT_JSON"

    CUDA_VISIBLE_DEVICES=$GPU \
    python3.12 adaptation/train.py \
        --data_dir      "$PPDATA_FOLDER" \
        --json_list     "${JSON_PREFIX}_classic.json" \
        --logdir        "$CLASSIC_MODEL_DIR" \
        --max_epochs    $EPOCH_NUMBER_CLASSIC \
        --batch_size    $BATCH_SIZE \
        --val_every     $VAL_EVERY \
        --optim_lr      $LRATE \
        --warmup_epochs $WARMUP_CLASSIC \
        --img_size      $IMG_SIZE \
        --n_filters     $N_FILTERS \
        --wavelet       "$WAVELET" \
        --num_classes   $NUM_CLASSES \
        --bg_ratio      $BG_RATIO \
        --lambda1       $LAMBDA1 \
        --lambda2       $LAMBDA2 \
        --gamma         $GAMMA \
        --lrschedule    warmup_cosine \
        --gpu           0 \
        2>&1 | tee /data/ethan/DualwaveSAM3c/$CLASSIC_MODEL_DIR/training_scratch.log
fi

# ── A'. CLASSIC RESUME ────────────────────────────────────────────────────
if [ "$RUN_CLASSIC_RESUME" = true ]; then
    echo "  > Resuming Classic Training from model_last.pth."

    NEXT_JSON=""
    if [ "$RUN_KFOLD_TRAIN" = true ]; then
        NEXT_JSON="${JSON_PREFIX}_fold${KFOLD_START}.json"
    elif [ "$RUN_KFOLD_PRODUCTION_FULL" = true ]; then
        NEXT_JSON="${JSON_PREFIX}_full.json"
    fi

    prep_cache "${JSON_PREFIX}_classic.json" "$NEXT_JSON"

    CUDA_VISIBLE_DEVICES=$GPU \
    python3.12 adaptation/train.py \
        --data_dir      "$PPDATA_FOLDER" \
        --json_list     "${JSON_PREFIX}_classic.json" \
        --logdir        "$CLASSIC_MODEL_DIR" \
        --checkpoint    "./runs/$CLASSIC_MODEL_DIR/model_last.pth" \
        --max_epochs    $EPOCH_NUMBER_CLASSIC \
        --batch_size    $BATCH_SIZE \
        --val_every     $VAL_EVERY \
        --optim_lr      $LRATE \
        --warmup_epochs $WARMUP_CLASSIC \
        --img_size      $IMG_SIZE \
        --n_filters     $N_FILTERS \
        --wavelet       "$WAVELET" \
        --num_classes   $NUM_CLASSES \
        --bg_ratio      $BG_RATIO \
        --lambda1       $LAMBDA1 \
        --lambda2       $LAMBDA2 \
        --gamma         $GAMMA \
        --lrschedule    warmup_cosine \
        --gpu           0 \
        2>&1 | tee /data/ethan/DualwaveSAM3c/$CLASSIC_MODEL_DIR/training_resume.log
fi

# ── B. K-FOLD TRAINING ────────────────────────────────────────────────────
if [ "$RUN_KFOLD_TRAIN" = true ]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  K-FOLD TRAINING (k=${K_FOLDS}, ${EPOCH_NUMBER_KFOLD} Epochs/fold)                      ║"
    echo "╚══════════════════════════════════════════════════════════════╝"

    # Ensure all json exist
    for fold in $(seq $KFOLD_START $((K_FOLDS - 1))); do
        JSON_FILE="$PPDATA_FOLDER/${JSON_PREFIX}_fold${fold}.json"
        if [ ! -f "$JSON_FILE" ]; then
            echo "  /!\ Missing JSON for Fold $fold: $JSON_FILE"
            echo "      Run SwinCross Dataset Builder first."
            exit 1
        fi
    done

    for fold in $(seq $KFOLD_START $((K_FOLDS - 1))); do
        CURRENT_JSON="${JSON_PREFIX}_fold${fold}.json"
        NEXT_JSON=""
        
        # Determine the next phase for the background cache builder
        if [ $fold -lt $((K_FOLDS - 1)) ]; then
            NEXT_JSON="${JSON_PREFIX}_fold$((fold + 1)).json"
        elif [ "$RUN_KFOLD_PRODUCTION_FULL" = true ]; then
            NEXT_JSON="${JSON_PREFIX}_full.json"
        fi

        prep_cache "$CURRENT_JSON" "$NEXT_JSON"

        MODEL_DIR="${KFOLD_BASE_DIR}_fold${fold}"
        mkdir -p /data/ethan/DualwaveSAM3c/$MODEL_DIR

        echo "  > Fold ${fold} / $((K_FOLDS - 1))"
        CUDA_VISIBLE_DEVICES=$GPU \
        python3.12 adaptation/train.py \
            --data_dir      "$PPDATA_FOLDER" \
            --json_list     "$CURRENT_JSON" \
            --logdir        "$MODEL_DIR" \
            --max_epochs    $EPOCH_NUMBER_KFOLD \
            --batch_size    $BATCH_SIZE \
            --val_every     $VAL_EVERY \
            --optim_lr      $LRATE \
            --warmup_epochs $WARMUP_KFOLD \
            --img_size      $IMG_SIZE \
            --n_filters     $N_FILTERS \
            --wavelet       "$WAVELET" \
            --num_classes   $NUM_CLASSES \
            --bg_ratio      $BG_RATIO \
            --lambda1       $LAMBDA1 \
            --lambda2       $LAMBDA2 \
            --gamma         $GAMMA \
            --lrschedule    warmup_cosine \
            --gpu           0 \
            2>&1 | tee /data/ethan/DualwaveSAM3c/$MODEL_DIR/training_fold${fold}.log
        
        echo "  [V] Fold ${fold} complete."
    done
fi

# ── C. PRODUCTION (100% Train Pool) ──────────────────────────────────────
if [ "$RUN_KFOLD_PRODUCTION_FULL" = true ]; then
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  PRODUCTION MODEL (100% of Train Pool)                       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    
    prep_cache "${JSON_PREFIX}_full.json" ""

    FULL_MODEL_DIR="${KFOLD_BASE_DIR}_full_production"
    mkdir -p /data/ethan/DualwaveSAM3c/$FULL_MODEL_DIR

    CUDA_VISIBLE_DEVICES=$GPU \
    python3.12 adaptation/train.py \
        --data_dir      "$PPDATA_FOLDER" \
        --json_list     "${JSON_PREFIX}_full.json" \
        --logdir        "$FULL_MODEL_DIR" \
        --max_epochs    $EPOCH_NUMBER_KFOLD \
        --batch_size    $BATCH_SIZE \
        --val_every     $VAL_EVERY \
        --optim_lr      $LRATE \
        --warmup_epochs $WARMUP_KFOLD \
        --img_size      $IMG_SIZE \
        --n_filters     $N_FILTERS \
        --wavelet       "$WAVELET" \
        --num_classes   $NUM_CLASSES \
        --bg_ratio      $BG_RATIO \
        --lambda1       $LAMBDA1 \
        --lambda2       $LAMBDA2 \
        --gamma         $GAMMA \
        --lrschedule    warmup_cosine \
        --gpu           0 \
        2>&1 | tee /data/ethan/DualwaveSAM3c/$FULL_MODEL_DIR/training_production_full.log
fi

wait_for_build # Ensure final cleanup/build completes before exiting
echo "  Training script complete."
