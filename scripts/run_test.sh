#!/usr/bin/env bash
# ==============================================================================
# CLARA Smoke Test Launcher — Native Ubuntu / ROCm
# ==============================================================================
# Activates the local virtual environment, configures AMD ROCm variables,
# and runs python/tools/test_model.py to verify the training stack before a long run.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${SCRIPT_DIR}/venv_rocm"

# 1. Check for virtual environment
if [ ! -f "${VENV}/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at ${VENV}/bin/activate"
    echo "Create it first: python -m venv ${VENV} && source ${VENV}/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 2. Activate environment
source "${VENV}/bin/activate"

# 3. Configure AMD ROCm Environment Variables for Radeon 860M (gfx1152)
echo "--> Configuring AMD ROCm env variables..."
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export HSA_ENABLE_SDMA=0
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:256"
export MIOPEN_FIND_MODE=5

echo "    HSA_OVERRIDE_GFX_VERSION = ${HSA_OVERRIDE_GFX_VERSION}"
echo "    PYTORCH_ROCM_ARCH        = ${PYTORCH_ROCM_ARCH}"
echo "    PYTORCH_HIP_ALLOC_CONF   = ${PYTORCH_HIP_ALLOC_CONF}"

# 4. Execute smoke test
echo "--> Launching CLARA model smoke test..."
python "${SCRIPT_DIR}/python/tools/test_model.py" "$@"

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================================================"
    echo " Smoke test PASSED — proceed to training"
    echo "   python python/training/train_clara.py --data /path/to/musdb18hq --device cuda"
    echo "======================================================================"
else
    echo "======================================================================"
    echo " Smoke test FAILED — fix issues before training"
    echo "======================================================================"
fi
exit $EXIT_CODE
