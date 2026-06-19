#!/usr/bin/env bash
# ==============================================================================
# CLARA Training Environment Setup (run inside clara-training distrobox)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HOME}/clara-training/venv"

echo "======================================================================"
echo " CLARA Training Environment Setup"
echo " Container: $(hostname)"
echo " Date:      $(date)"
echo "======================================================================"

# 1. Create venv if needed
if [ -d "${VENV_DIR}" ]; then
    echo "--> Existing venv found at ${VENV_DIR}"
else
    echo "--> Creating virtual environment..."
    python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

# 2. Upgrade packaging
pip install -q --upgrade pip setuptools wheel

# 3. Install ROCm PyTorch
echo "--> Installing ROCm PyTorch (rocm6.2.4)..."
pip install torch torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.2.4 2>&1 | tail -3

# 4. Install all dependencies
echo "--> Installing training dependencies..."
pip install -q \
  numpy scipy soundfile \
  onnx onnxruntime \
  tqdm auraloss torchcrepe \
  matplotlib 2>&1 | tail -3

# 5. Verify ROCm GPU access
echo ""
echo "======================================================================"
echo " Verifying ROCm GPU Access"
echo "======================================================================"
export HSA_OVERRIDE_GFX_VERSION=11.0.0

python -c "
import torch, os
print(f'PyTorch: {torch.__version__}')
print(f'CUDA avail: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'ROCm/HIP: {torch.version.hip}')
    x = torch.randn(100, 100, device=\"cuda\")
    y = x @ x.T
    print(f'GPU compute OK: {y.mean().item():.4f}')
else:
    print('WARNING: GPU NOT visible')
    print(f'/dev/kfd: {os.path.exists(\"/dev/kfd\")}')
    print(f'/dev/dri/renderD128: {os.path.exists(\"/dev/dri/renderD128\")}')
"

echo ""
echo "======================================================================"
echo " Setup Complete!"
echo " Activate: source ${VENV_DIR}/bin/activate"
echo " Run:      export HSA_OVERRIDE_GFX_VERSION=11.0.0"
echo " Then:     python train.py ..."
echo "======================================================================"
