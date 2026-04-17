#!/bin/bash
set -e

# =============================================================================
# Environment setup (uv + venv + requirements)
# =============================================================================

if ! command -v uv &> /dev/null; then
    echo "uv not found — installing..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

uv python install 3.12

if [ ! -d "swincross_env" ]; then
    uv venv swincross_env --python 3.12
fi

source swincross_env/bin/activate

if [ -f requirements.txt ]; then
    uv pip install -r requirements.txt
else
    echo "requirements.txt not found! Aborting."
    exit 1
fi

# =============================================================================
# Run dataset builder
# =============================================================================

python3.12 dataset_builder_TEMPORAL.py