#!/usr/bin/env bash
# ============================================================================
# CLARA Training Launch Script — launch_train.sh
#
# Handles distrobox entry, ROCm env vars, and training invocation.
# Supports both tmux (detachable) and foreground modes.
#
# Usage:
#   # Tmux session (detachable — survives terminal close)
#   bash scripts/launch_train.sh tmux --steps 100000 --batch 2 --amp
#
#   # Foreground (runs in current terminal)
#   bash scripts/launch_train.sh fg --steps 100 --batch 2
#
#   # Phase 1 only (contrastive pretraining)
#   bash scripts/launch_train.sh tmux --steps 20000 --batch 4 --amp --warmup 2000
#
#   # Phase 2 only (single-track) from Phase 1 checkpoint
#   bash scripts/launch_train.sh tmux \
#     --resume checkpoints/best.pt \
#     --steps 50000 --batch 2 --amp
#
#   # View training log
#   tail -f checkpoints/train.log
#
#   # GPU monitoring (separate terminal)
#   watch -n 2 rocm-smi --showtemp --showpower --showuse --showmemuse
# ============================================================================

set -euo pipefail

DISTROBOX_NAME="${DISTROBOX_NAME:-clara-training}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/checkpoints}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${CHECKPOINT_DIR}/train_${TIMESTAMP}.log"

# ROCm env vars (must match what works on this hardware)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_ROCM_ARCH=gfx1100
export HSA_ENABLE_SDMA=0
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:256
export MIOPEN_FIND_MODE=5

TRAIN_CMD="python train_clara.py \
  --data ~/datasets/musdb18hq/train \
  --save ${CHECKPOINT_DIR} \
  --device cuda"

usage() {
    grep "^# " "$0" | sed 's/^# \?//' | sed '1,3d'
    exit 1
}

cmd_foreground() {
    mkdir -p "${CHECKPOINT_DIR}"
    echo "🚀 CLARA Training (foreground)"
    echo "   Log: ${LOG_FILE}"
    echo "   PID: $$"
    echo ""
    # shellcheck disable=SC2086
    distrobox enter "${DISTROBOX_NAME}" -- bash -l -c "
        source ~/clara_env/bin/activate
        cd ${PROJECT_DIR}
        ${TRAIN_CMD} $* 2>&1 | tee ${LOG_FILE}
    "
}

cmd_tmux() {
    local session_name="clara-train-${TIMESTAMP}"
    mkdir -p "${CHECKPOINT_DIR}"

    echo "🚀 CLARA Training (tmux session: ${session_name})"
    echo "   Detach: Ctrl+B, D"
    echo "   Reattach: tmux attach -t ${session_name}"
    echo "   Kill: tmux kill-session -t ${session_name}"
    echo "   Log: ${LOG_FILE}"
    echo ""

    # shellcheck disable=SC2086
    distrobox enter "${DISTROBOX_NAME}" -- bash -l -c "
        source ~/clara_env/bin/activate
        cd ${PROJECT_DIR}
        tmux new-session -d -s ${session_name} \
            '${TRAIN_CMD} $* 2>&1 | tee ${LOG_FILE}; echo \"Training exited.\"'
        echo \"Session: ${session_name}\"
        tmux attach -t ${session_name}
    "
}

# Parse subcommand
MODE="${1:-}"
if [ -z "${MODE}" ] || [[ "${MODE}" == -* ]]; then
    MODE="fg"  # default to foreground
else
    shift
fi

case "${MODE}" in
    fg|foreground) cmd_foreground "$@" ;;
    tmux)          cmd_tmux "$@" ;;
    *)             usage ;;
esac
