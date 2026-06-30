#!/usr/bin/env bash
# ==============================================================================
# CLARA Local Environment Setup — Native Ubuntu / ROCm
# ==============================================================================
# Creates a local virtual environment and installs ROCm-compatible PyTorch,
# torchaudio, and training dependencies. Run once per machine or after venv loss.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv_rocm"

echo "======================================================================"
echo " CLARA Local Environment Setup"
echo " Project: ${SCRIPT_DIR}"
echo " Venv:    ${VENV_DIR}"
echo "======================================================================"

# 1. Create virtual environment
if [ -d "${VENV_DIR}" ]; then
    echo "--> Existing virtual environment found. Re-using..."
else
    echo "--> Creating Python virtual environment..."
    # Prefer python3.12 or python3.13 for PyTorch compatibility
    PYTHON_EXEC="python3"
    if command -v python3.12 >/dev/null 2>&1; then
        PYTHON_EXEC="python3.12"
    elif command -v python3.13 >/dev/null 2>&1; then
        PYTHON_EXEC="python3.13"
    fi
    echo "    Using Python executable: ${PYTHON_EXEC}"
    ${PYTHON_EXEC} -m venv "${VENV_DIR}"

    # Initialize pip
    echo "--> Initializing pip..."
    "${VENV_DIR}/bin/python3" -m ensurepip || {
        echo "    ensurepip failed. Fetching get-pip.py..."
        curl -sS https://bootstrap.pypa.io/get-pip.py -o "${SCRIPT_DIR}/get-pip.py"
        "${VENV_DIR}/bin/python3" "${SCRIPT_DIR}/get-pip.py"
        rm "${SCRIPT_DIR}/get-pip.py"
    }
fi

# 2. Upgrade packaging utilities
echo "--> Upgrading core packaging tools..."
"${VENV_DIR}/bin/python3" -m pip install --upgrade pip setuptools wheel

# 3. Install ROCm-enabled PyTorch and Torchaudio
# Use ROCm 6.2.x for compatibility with gfx1103/gfx1100
echo "--> Installing ROCm-enabled PyTorch and Torchaudio (this may take a few minutes)..."
"${VENV_DIR}/bin/python3" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 4. Install other dependencies
echo "--> Installing training dependencies..."
"${VENV_DIR}/bin/python3" -m pip install numpy scipy soundfile tqdm auraloss

# Optional: torchcrepe for future f0 conditioning (not used in Stage 0)
echo "--> Installing torchcrepe (optional, for future conditioned training)..."
"${VENV_DIR}/bin/python3" -m pip install torchcrepe || echo "    WARNING: torchcrepe install failed — will retry later if needed"

# 5. Sanity check
echo "--> Verifying PyTorch and ROCm visibility..."
"${VENV_DIR}/bin/python3" -c "
import torch
import torchaudio
print('✓ PyTorch Version:', torch.__version__)
print('✓ Torchaudio Version:', torchaudio.__version__)
print('✓ CUDA/ROCm Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('✓ GPU Device:', torch.cuda.get_device_name(0))
    if hasattr(torch.version, 'hip'):
        print('✓ ROCm/HIP Build Version:', torch.version.hip)
else:
    print('! WARNING: GPU is not yet visible to PyTorch.')
    print('  Ensure HSA_OVERRIDE_GFX_VERSION=11.0.0 is set before running.')
"

echo "======================================================================"
echo " Setup Completed Successfully!"
echo " Activate environment with: source ${VENV_DIR}/bin/activate"
echo "======================================================================"
