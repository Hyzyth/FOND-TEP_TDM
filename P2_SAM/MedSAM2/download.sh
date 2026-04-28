#!/usr/bin/env bash
# Download MedSAM2 model checkpoints
mkdir -p checkpoints

if command -v wget > /dev/null 2>&1; then
    CMD="wget -P checkpoints"
elif command -v curl > /dev/null 2>&1; then
    CMD="curl -L -o"
    CURL=1
else
    echo "Please install wget or curl."
    exit 1
fi

HF_BASE_URL="https://huggingface.co/wanglab/MedSAM2/resolve/main"

# Latest general model (recommended starting point for HECKTOR fine-tuning)
MODEL1="MedSAM2_latest.pt"
# CT lesion model (useful if fine-tuning from a CT-specialized checkpoint)
MODEL2="MedSAM2_CTLesion.pt"

for model in $MODEL1 $MODEL2; do
    echo "Downloading ${model}..."
    model_url="${HF_BASE_URL}/${model}"
    if [ -n "$CURL" ]; then
        $CMD "checkpoints/${model}" "$model_url" || { echo "Failed to download $model_url"; exit 1; }
    else
        $CMD "$model_url" || { echo "Failed to download $model_url"; exit 1; }
    fi
done

# SAM2 base checkpoint (needed for training initialization)
SAM2_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SAM2_MODEL="sam2.1_hiera_tiny.pt"
echo "Downloading ${SAM2_MODEL}..."
sam2_model_url="${SAM2_BASE_URL}/${SAM2_MODEL}"
if [ -n "$CURL" ]; then
    $CMD "checkpoints/${SAM2_MODEL}" "$sam2_model_url" || { echo "Failed to download $sam2_model_url"; exit 1; }
else
    $CMD "$sam2_model_url" || { echo "Failed to download $sam2_model_url"; exit 1; }
fi

echo "Done. Checkpoints saved to ./checkpoints/"