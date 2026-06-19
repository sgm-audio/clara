#!/usr/bin/env bash
# ============================================================================
# CLARA Local Training — One Script Does Everything
#
# Downloads MUSDB18-HQ, extracts it, runs training. That's it.
#
# Usage:
#   bash scripts/train_local.sh
#   bash scripts/train_local.sh --steps 50000 --batch 2
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_DIR="${DATASETS_DIR:-$HOME/datasets}"
MUSDB_DIR="$DATASETS_DIR/musdb18hq"
CKPT_DIR="${SCRIPT_DIR}/checkpoints_clara"

# Colors
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[CLARA]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

# ============================================================================
# Step 1: Download MUSDB18-HQ (if not present)
# ============================================================================
download_dataset() {
    local n_wavs
    n_wavs=$(find "$MUSDB_DIR" -name "*.wav" 2>/dev/null | wc -l)
    if [ "$n_wavs" -gt 0 ]; then
        log "Dataset already exists: $n_wavs WAV files at $MUSDB_DIR"
        return
    fi

    mkdir -p "$DATASETS_DIR"
    local archive="$DATASETS_DIR/musdb18hq.tar.gz"

    if [ ! -f "$archive" ]; then
        log "Downloading MUSDB18-HQ (~21 GB, 4 connections)..."
        aria2c -x 4 -s 4 --continue=true \
            "https://zenodo.org/record/5061438/files/musdb18hq.tar.gz" \
            -d "$DATASETS_DIR" -o musdb18hq.tar.gz
    else
        log "Archive already downloaded: $archive"
    fi

    log "Extracting (~21 GB → ~41 GB)..."
    tar -xzf "$archive" -C "$DATASETS_DIR" --checkpoint=10000 2>&1

    local extracted
    extracted=$(find "$MUSDB_DIR" -name "*.wav" 2>/dev/null | wc -l)
    log "Dataset ready: $extracted WAV files"
}

# ============================================================================
# Step 2: Verify environment
# ============================================================================
check_env() {
    python3 -c "import torch" 2>/dev/null || {
        warn "PyTorch not found. Install: pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4"
        exit 1
    }

    if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log "GPU: $(python3 -c "import torch; print(torch.cuda.get_device_name(0))")"
    else
        warn "No GPU detected, will use CPU (much slower)"
        warn "If you have AMD GPU: export HSA_OVERRIDE_GFX_VERSION=11.0.0"
    fi
}

# ============================================================================
# Step 3: Run training
# ============================================================================
run_training() {
    mkdir -p "$CKPT_DIR"

    log "Starting training..."
    log "Dataset: $MUSDB_DIR/train"
    log "Checkpoints: $CKPT_DIR"
    log "Log: $CKPT_DIR/train.log"

    python3 "$SCRIPT_DIR/train_clara_mamba3.py" \
        --data "$MUSDB_DIR/train" \
        --device cuda \
        --batch 4 \
        --grad-accum 4 \
        --steps 100000 \
        --clip-seconds 0.5 \
        --ckpt-every 5000 \
        --val-every 1000 \
        --log-every 100 \
        --lr 1e-3 \
        --weight-decay 1e-5 \
        --clip-grad-norm 1.0 \
        --warmup 2000 \
        --cache \
        --num-workers 0 \
        "$@" \
        2>&1 | tee "$CKPT_DIR/train.log"
}

# ============================================================================
# Main
# ============================================================================
main() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          CLARA Local Training Pipeline            ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
    echo ""

    download_dataset
    check_env
    run_training "$@"
}

main "$@"
