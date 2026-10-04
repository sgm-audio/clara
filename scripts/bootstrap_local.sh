#!/usr/bin/env bash
# ==============================================================================
# CLARA Local Bootstrap — AMD AI 7 350 / Radeon 860M (gfx1152 RDNA3.5)
# ==============================================================================
# One-shot setup for local training on this machine.
# Sets up venv, verifies GPU, runs CREPE pitch extraction, precomputes mel/STFT,
# then starts training.
#
# Usage:
#   bash scripts/bootstrap_local.sh           # interactive, full
#   bash scripts/bootstrap_local.sh --train  # setup + train
#   bash scripts/bootstrap_local.sh --smoke  # smoke test only
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="/home/sgm/micromamba/envs/rocm_torch"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints_clara"
DATA_DIR="${HOME}/datasets/musdb18hq/train"
PRECOMP_DIR="${HOME}/datasets/musdb18hq/precomputed"

# ── ROCm / GPU env — gfx1152 (Radeon 860M) needs gfx1100 spoof ─────────────
# AMD AI 7 350 (Ryzen AI) has integrated gfx1152 (RDNA3.5).
# ROCm 6.x doesn't officially support gfx1152 — spoof as gfx1100 (gfx11 FL).
# gfx1100 is fully supported in ROCm 6.x.
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1100}"
export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROC_VISIBLE_DEVICES="${ROC_VISIBLE_DEVICES:-0}"

# Disable SDMA engines — on APU with shared system RAM (up to 30GB UMA),
# SDMA pinning conflicts cause memory corruption and OOM. Disable for stability.
export HSA_ENABLE_SDMA="${HSA_ENABLE_SDMA:-0}"

# Optimize PyTorch memory allocator for a ~16GB UMA buffer shared between CPU/GPU.
# Larger splits fragment the address space; GC threshold keeps free lists lean.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:256}"

# MIOpen fine-grained search mode — exhaustive kernel search (mode 5) finds
# the fastest kernel for this specific shape. Costs ~1-2s at init, saves 10-30%
# per forward pass. Worth it for training where init is amortized.
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-5}"

# ── Helpers ────────────────────────────────────────────────────────────────

info()  { echo -e "  \033[1;34m[info]\033[0m  $*"; }
ok()    { echo -e "  \033[92m[ ok ]\033[0m  $*"; }
warn()  { echo -e "  \033[93m[warn]\033[0m  $*"; }
fail()  { echo -e "  \033[91m[fail]\033[0m  $*" >&2; }

section() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo " $1"
    echo "═══════════════════════════════════════════════════════"
}

# ── Step 1: ROCm / GPU Check ────────────────────────────────────────────────

step_rocm() {
    section "1. ROCm / GPU Check"
    echo "  HSA_OVERRIDE_GFX_VERSION = ${HSA_OVERRIDE_GFX_VERSION}"
    echo "  PYTORCH_ROCM_ARCH        = ${PYTORCH_ROCM_ARCH}"
    echo "  HSA_ENABLE_SDMA           = ${HSA_ENABLE_SDMA}"
    echo "  PYTORCH_ALLOC_CONF       = ${PYTORCH_ALLOC_CONF}"
    echo "  MIOPEN_FIND_MODE          = ${MIOPEN_FIND_MODE}"

    if ! command -v python3 &>/dev/null; then
        fail "python3 not found"
        return 1
    fi
    ok "python3 found"

    # Venv — use micromamba env directly (no activate needed)
    if [ -x "${VENV_DIR}/bin/python3" ]; then
        ok "micromamba env found at ${VENV_DIR}"
    else
        fail "micromamba env not found at ${VENV_DIR}"
        return 1
    fi

    # PyTorch + ROCm check (with env vars set)
    if \
        HSA_OVERRIDE_GFX_VERSION=11.0.0 \
        HIP_VISIBLE_DEVICES=0 \
        ROC_VISIBLE_DEVICES=0 \
        HSA_ENABLE_SDMA=0 \
        "${VENV_DIR}/bin/python3" -c "import torch; print(f'PyTorch {torch.__version__}')" 2>/dev/null; then
        ok "PyTorch installed"
    else
        info "Installing PyTorch..."
        "${VENV_DIR}/bin/pip" install --upgrade pip
        "${VENV_DIR}/bin/pip" install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    fi

    # GPU check
    if \
        HSA_OVERRIDE_GFX_VERSION=11.0.0 \
        HIP_VISIBLE_DEVICES=0 \
        ROC_VISIBLE_DEVICES=0 \
        HSA_ENABLE_SDMA=0 \
        "${VENV_DIR}/bin/python3" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        GPU=$("${VENV_DIR}/bin/python3" -c "import torch; print(torch.cuda.get_device_name(0))")
        VRAM=$("${VENV_DIR}/bin/python3" -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')")
        ok "GPU: ${GPU}"
        ok "VRAM: ${VRAM} (UMA — shared with system RAM)"

        # SGEMM smoke test
        RESULT=$( \
            HSA_OVERRIDE_GFX_VERSION=11.0.0 \
            HIP_VISIBLE_DEVICES=0 \
            ROC_VISIBLE_DEVICES=0 \
            HSA_ENABLE_SDMA=0 \
            "${VENV_DIR}/bin/python3" -c "
import torch
a = torch.randn(128, 128, device='cuda')
b = torch.randn(128, 128, device='cuda')
c = (a @ b).mean()
print('ok' if torch.isfinite(c).all().item() else 'nan')
" 2>/dev/null)
        if [ "${RESULT}" = "ok" ]; then
            ok "GPU SGEMM: no NaN"
        else
            warn "GPU produced NaN — check HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION}"
        fi
    else
        warn "No ROCm GPU detected — will run CPU only"
    fi

    # Training deps
    "${VENV_DIR}/bin/pip" install numpy scipy soundfile tqdm auraloss 2>/dev/null || true
}

# ── Step 2: Dataset ──────────────────────────────────────────────────────

step_dataset() {
    section "2. Dataset"

    if [ -d "${DATA_DIR}" ] && [ "$(find "${DATA_DIR}" -name '*.wav' 2>/dev/null | wc -l)" -gt 0 ]; then
        N=$(find "${DATA_DIR}" -name '*.wav' | wc -l)
        ok "Dataset found: ${DATA_DIR} (${N} files)"
    else
        info "Dataset not found at ${DATA_DIR}"
        echo "  1. Download MUSDB18-HQ from https://zenodo.org/record/5061434 (~2GB)"
        echo "  2. Place musdb18hq.tar.gz at ~/Downloads/"
        echo "  3. Re-run this script"
        return 1
    fi
}

# ── Step 3: CREPE + Pitch Extraction ──────────────────────────────────

step_crepe() {
    section "3. CREPE Pitch Extraction"
    info "Extracting CREPE f0 + pitch for all stems..."
    info "This runs once, saves .crepe.npy + .pitched.npy next to each .wav"
    info "(~10-30 min for full MUSDB18-HQ, depending on CPU)"

    if [ ! -f "${SCRIPT_DIR}/python/tools/extract_crepe.py" ]; then
        warn "python/tools/extract_crepe.py not found — skipping CREPE extraction"
        warn "  (StemDataset will use zeros for f0 conditioning — graceful degradation)"
        return 0
    fi

    # Check if already extracted (look for a sample .crepe.npy)
    SAMPLE=$(find "${DATA_DIR}" -name "*.crepe.npy" | head -1 2>/dev/null || true)
    if [ -n "${SAMPLE}" ]; then
        ok "CREPE features already extracted: ${SAMPLE}"
        return 0
    fi

    HSA_OVERRIDE_GFX_VERSION=11.0.0 HIP_VISIBLE_DEVICES=0 ROC_VISIBLE_DEVICES=0 \
        HSA_ENABLE_SDMA=0 \
        "${VENV_DIR}/bin/python3" "${SCRIPT_DIR}/python/tools/extract_crepe.py" \
        --data "${DATA_DIR}" \
        --device cuda \
        --model full \
        2>&1 | tee /tmp/crepe_extract.log

    EXTRACTED=$(find "${DATA_DIR}" -name "*.crepe.npy" 2>/dev/null | wc -l || true)
    if [ "${EXTRACTED}" -gt 0 ]; then
        ok "CREPE extracted: ${EXTRACTED} files"
    else
        warn "No CREPE files found — StemDataset will use zeros (OK for training)"
    fi
}

# ── Step 4: Precompute Mel/STFT Chunks ─────────────────────────────────

step_precompute() {
    section "4. Precompute Mel/STFT Features"
    info "Precomputing chunked mel + STFT features for fast DataLoader..."
    info "Saves .npz chunks to: ${PRECOMP_DIR}"

    if [ ! -f "${PRECOMP_DIR}/metadata.json" ]; then
        HSA_OVERRIDE_GFX_VERSION=11.0.0 HIP_VISIBLE_DEVICES=0 ROC_VISIBLE_DEVICES=0 \
            HSA_ENABLE_SDMA=0 \
            "${VENV_DIR}/bin/python3" "${SCRIPT_DIR}/python/tools/precompute_features.py" \
            --input "${DATA_DIR}" \
            --output "${PRECOMP_DIR}" \
            --chunk-seconds 0.5 \
            --sample-rate 44100 \
            --n-fft 2048 \
            --hop-length 512 \
            --n-mels 128 \
            --workers 4 \
            2>&1 | tee /tmp/precompute.log
        ok "Precomputed features saved to ${PRECOMP_DIR}"
    else
        info "Precomputed features already exist at ${PRECOMP_DIR}"
    fi
}

# ── Step 5: Smoke Test ────────────────────────────────────────────────

step_smoke() {
    section "5. Smoke Test"
    info "Running PerTrackProcessor smoke test (GPU)..."
    # PYTHONPATH=. needed so test_model.py can import train_mamba3
    # ROCm env vars MUST be set before Python starts for torch.cuda to detect GPU
    if \
        HSA_OVERRIDE_GFX_VERSION=11.0.0 \
        HIP_VISIBLE_DEVICES=0 \
        ROC_VISIBLE_DEVICES=0 \
        HSA_ENABLE_SDMA=0 \
        PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb=256 \
        MIOPEN_FIND_MODE=5 \
        PYTHONPATH="${SCRIPT_DIR}/python/training" \
        "${VENV_DIR}/bin/python3" "${SCRIPT_DIR}/python/tools/test_model.py" --device cuda 2>&1 | tee /tmp/clara_smoke.log; then
        ok "Smoke test PASSED"
    else
        fail "Smoke test FAILED — see /tmp/clara_smoke.log"
        return 1
    fi
}

# ── Step 6: Train ────────────────────────────────────────────────────────

step_train() {
    section "6. Training"
    mkdir -p "${CHECKPOINT_DIR}"

    if [ ! -d "${DATA_DIR}" ]; then
        fail "Dataset not available"
        return 1
    fi

    # Auto-resume
    LATEST=$(ls -t "${CHECKPOINT_DIR}"/checkpoint_*.pt "${CHECKPOINT_DIR}"/interrupt_*.pt 2>/dev/null | head -1 || true)

    # ROCm env vars MUST be set before Python starts (checked at library load time)
    export HSA_OVERRIDE_GFX_VERSION=11.0.0
    export HIP_VISIBLE_DEVICES=0
    export ROC_VISIBLE_DEVICES=0
    export HSA_ENABLE_SDMA=0
    export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb=256
    export MIOPEN_FIND_MODE=5

    # Build command — conservative batch=2 for 16GB UMA, effective batch=8 via grad accum 4
    # SCRIPT_DIR is the repository root, and the training script lives under
    # python/training/ (this used to point at ${SCRIPT_DIR}/train_clara.py,
    # which does not exist).
    CMD=("${VENV_DIR}/bin/python3" "${SCRIPT_DIR}/python/training/train_clara.py"
        --data "${DATA_DIR}"
        --save "${CHECKPOINT_DIR}"
        --device cuda
        --batch 2
        --grad-accum 4
        --steps 100000
        --val-every 2000
        --ckpt-every 1000
        --log-every 50
        --warmup 2000
        --early-stop-patience 10
        --num-workers 0
        --cache-limit 16
    )

    # Use precomputed features if available
    if [ -f "${PRECOMP_DIR}/metadata.json" ]; then
        info "Using precomputed features from: ${PRECOMP_DIR}"
        CMD+=(--precomputed "${PRECOMP_DIR}")
    fi

    if [ -n "${LATEST}" ]; then
        info "Resuming: ${LATEST}"
        CMD+=(--resume "${LATEST}")
    fi

    echo ""
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
    echo "  Monitor:  watch -n 5 rocm-smi"
    echo "  Command:  ${CMD[*]}"
    echo ""

    read -p "Start training? [y] " CONFIRM
    CONFIRM="${CONFIRM:-y}"
    if [[ "${CONFIRM}" =~ ^[Yy]$ ]]; then
        "${CMD[@]}"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────

usage() {
    echo "Usage: $0 [--train|--smoke|--crepe-only]"
    echo "  --train      setup + smoke test + training"
    echo "  --smoke     smoke test only"
    echo "  --crepe-only  extract CREPE features only"
}

MODE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --train)      MODE=train; shift ;;
        --smoke)     MODE=smoke; shift ;;
        --crepe-only) MODE=crepe; shift ;;
        --help|-h)   usage; exit 0 ;;
        *) fail "Unknown: $1"; usage; exit 1 ;;
    esac
done

section "CLARA Bootstrap — Radeon 860M (gfx1152 → gfx1100)"

step_rocm

if [ "${MODE}" = "smoke" ]; then
    step_smoke; exit 0
fi

if [ "${MODE}" = "crepe" ]; then
    step_dataset
    step_crepe; exit 0
fi

step_dataset
step_smoke
step_train

ok "Bootstrap complete!"
