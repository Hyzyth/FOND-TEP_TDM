#!/usr/bin/env bash
# scripts/download_checkpoints.sh
# =================================
# Downloads MedSAM2 model checkpoints to /data/ethan/MedSAM2/checkpoints/.
#
# Checkpoints
# -----------
#   MedSAM2_latest.pt   – General-purpose MedSAM2 (recommended starting point)
#   MedSAM2_CTLesion.pt – CT-lesion specialised variant
#   sam2.1_hiera_tiny.pt – Base SAM2.1 weights (needed for training from scratch)

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/data/ethan/MedSAM2/checkpoints}"
mkdir -p "${CKPT_DIR}"

HF_BASE="https://huggingface.co/wanglab/MedSAM2/resolve/main"
SAM2_BASE="https://dl.fbaipublicfiles.com/segment_anything_2/092824"

download() {
    local url="$1"
    local dest="$2"
    if [ -f "${dest}" ]; then
        echo "  Already exists: ${dest}"
        return 0
    fi
    echo "  Downloading $(basename "${dest}") ..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "${dest}" "${url}"
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "${dest}" "${url}"
    else
        echo "ERROR: wget or curl is required." >&2; exit 1
    fi
}

echo "Checkpoint directory: ${CKPT_DIR}"
download "${HF_BASE}/MedSAM2_latest.pt"       "${CKPT_DIR}/MedSAM2_latest.pt"
download "${HF_BASE}/MedSAM2_CTLesion.pt"     "${CKPT_DIR}/MedSAM2_CTLesion.pt"
download "${SAM2_BASE}/sam2.1_hiera_tiny.pt"  "${CKPT_DIR}/sam2.1_hiera_tiny.pt"

echo "All checkpoints downloaded to ${CKPT_DIR}"
