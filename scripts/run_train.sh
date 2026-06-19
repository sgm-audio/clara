#!/usr/bin/env bash
# ==============================================================================
# CLARA Training Runner — Direct Launch, No Tmux
# ==============================================================================
# Usage (local AMD APU / 860M, gfx1152 → gfx1100 override):
#   export HSA_OVERRIDE_GFX_VERSION=11.0.0
#   ./scripts/run_train.sh [DATA_DIR] [extra args]
#
# Usage (MI300X, gfx942 — do NOT use gfx1100 override):
#   HSA_OVERRIDE_GFX_VERSION="" PYTORCH_ROCM_ARCH=gfx942 ./scripts/run_train.sh
#
# Launches training directly with automatic checkpoint resume.
# For GPU monitoring, open a second terminal and run:
#   watch -n 2 rocm-smi --showtemp --showpower --showuse --showmemuse
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HOME}/clara-training/venv"
DEFAULT_DATA_DIR="/home/sgm/datasets/musdb18hq/train"
CKPT_DIR="${SCRIPT_DIR}/checkpoints_clara"

# ROCm / GPU env (set before any python/torch import)
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export HSA_ENABLE_SDMA=0
export PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:256"
export MIOPEN_FIND_MODE=5

# ── Parse args ───────────────────────────────────────────────────────────
DATA_DIR="${1:-${DEFAULT_DATA_DIR}}"
shift 1>/dev/null 2>&1 || true
EXTRA_ARGS=("$@")

# ── Verify prerequisites ─────────────────────────────────────────────────
if [ ! -f "${VENV}/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at ${VENV}/bin/activate"
    echo "Create it first: python -m venv ${VENV} && source ${VENV}/bin/activate && pip install -r requirements.txt"
    exit 1
fi

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: Dataset not found at ${DATA_DIR}"
    echo "Set DATA_DIR env var or pass as first arg: ./scripts/run_train.sh /path/to/data"
    exit 1
fi

source "${VENV}/bin/activate"

python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null || {
    echo "ERROR: GPU not available. Is HSA_OVERRIDE_GFX_VERSION=11.0.0 set?"
    echo "       Run: export HSA_OVERRIDE_GFX_VERSION=11.0.0"
    exit 1
}

# ── Info ─────────────────────────────────────────────────────────────────
NUM_TRACKS=$(find "${DATA_DIR}" -name "*.wav" | wc -l)
echo "======================================================================"
echo " CLARA Mamba RT Parallel Training"
echo " Container: $(hostname)"
echo " HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}"
echo " Dataset:   ${DATA_DIR} (${NUM_TRACKS} tracks)"
echo "======================================================================"
echo ""

# ── Auto-detect checkpoint to resume ─────────────────────────────────────
LATEST=""
if [ -d "${CKPT_DIR}" ]; then
    LATEST=$(ls -t "${CKPT_DIR}"/checkpoint_*.pt \
                  "${CKPT_DIR}"/interrupt_*.pt 2>/dev/null | head -n 1 || true)
fi

# ── Build training args ──────────────────────────────────────────────────
ARGS=(
    --data "${DATA_DIR}"
    --device cuda
    --batch 4
    --grad-accum 4
    --steps 100000
    --clip-seconds 0.5
    --ckpt-every 5000
    --val-every 1000
    --log-every 100
    --num-workers 0
    --cache
    --lr 1e-3
    --weight-decay 1e-5
    --clip-grad-norm 1.0
    --warmup 2000
)

if [ -n "${LATEST}" ]; then
    echo "--> [RESUME] Found checkpoint: ${LATEST}"
    ARGS+=(--resume "${LATEST}")
else
    echo "--> [FRESH] Starting new training run"
fi

ARGS+=("${EXTRA_ARGS[@]}")
mkdir -p "${CKPT_DIR}"
LOGFILE="${CKPT_DIR}/train.log"

echo "======================================================================"
echo " Command: python train_clara_mamba3.py ${ARGS[*]}"
echo " Log:     ${LOGFILE}"
echo " Monitor: watch -n 2 rocm-smi --showtemp --showpower --showuse --showmemuse"
echo "======================================================================"
echo ""

# ── Launch ────────────────────────────────────────────────────────────────
echo "=== CLARA Training Start: $(date) ===" | tee -a "${LOGFILE}"
python "${SCRIPT_DIR}/train_clara_mamba3.py" "${ARGS[@]}" 2>&1 | tee -a "${LOGFILE}"
EXIT_CODE=$?
echo "=== CLARA Training End: $(date) (exit=${EXIT_CODE}) ===" | tee -a "${LOGFILE}"

exit ${EXIT_CODE}