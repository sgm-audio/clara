#!/usr/bin/env bash
# ============================================================================
# CLARA MI300X Cloud — single launch, all stages, auto-kill at $90
#
# AMD ADC bills $1.99/hr even when VM is STOPPED. To stop billing you must
# DESTROY the VM, which DELETES ALL DATA. This script:
#   1. Sets up everything needed
#   2. Downloads ONLY MUSDB18-HQ (~41 GB, 15 min) — the one dataset needed
#   3. Precomputes mel+STFT features
#   4. Trains Stage 0 (identity, ~4 hrs)
#   5. Exports weights to .bin
#   6. Syncs checkpoints to S3 every 5 min
#   7. Monitors cost and warns at thresholds
#   8. At $90, does a final sync and tells you to destroy the VM
#
# DATASETS:
#   MUSDB18-HQ (~41 GB) — the ONLY dataset needed right now
#   FMA, MAESTRO, Slakh, FSD50K — all SKIPPED, not needed for MVP
#   F0/CREPE — was in old Mamba-1 pipeline, not used in Mamba-3
#
# S3 STORAGE (NOT included with AMD ADC):
#   You need your own AWS account. Cost: ~$0.023/GB/month.
#   All checkpoints for a full run: ~10 GB = $0.23/month. Negligible.
#   IF YOU DON'T SET UP S3, KILLING THE VM = LOSING ALL CHECKPOINTS.
#
# Usage:
#   # One-time setup on cloud VM:
#   curl -fsSL https://raw.githubusercontent.com/SGM-Studios/clara/main/scripts/setup_cloud.sh | bash
#
#   # Set up S3 (one time, after script installs pip packages):
#   aws configure
#
#   # Then just let it run. Script monitors cost and warns at thresholds.
#   # At $90 (~45 hrs): sync everything → destroy VM → billing stops.
# ============================================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[CLARA]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERROR]${NC} $1"; }
info() { echo -e "${CYAN}[INFO]${NC} $1"; }

# ── Configuration ──────────────────────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/SGM-Studios/clara.git}"
REPO_DIR="${REPO_DIR:-$HOME/clara}"
DATASETS_DIR="${DATASETS_DIR:-$HOME/datasets}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-$HOME/checkpoints}"
S3_BUCKET="${S3_BUCKET:-s3://clara-checkpoints}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
CHUNK_SECONDS="${CHUNK_SECONDS:-0.5}"

# Cost constants
COST_PER_HOUR=1.99
KILL_THRESHOLD=90    # $90 — hard stop before $100 credit runs out
WARN_THRESHOLD=75    # $75 — warn, start wrapping up
HOURS_TO_KILL=$(echo "$KILL_THRESHOLD / $COST_PER_HOUR" | bc -l | xargs printf "%.0f")
HOURS_TO_WARN=$(echo "$WARN_THRESHOLD / $COST_PER_HOUR" | bc -l | xargs printf "%.0f")
START_FILE="$CHECKPOINTS_DIR/start_time.txt"

# ── Record start time for cost tracking ──
record_start() {
    mkdir -p "$CHECKPOINTS_DIR"
    date +%s > "$START_FILE"
    log "  Start time recorded. Cost clock: \$1.99/hr"
    log "  Warn at  \$${WARN_THRESHOLD}  (~${HOURS_TO_WARN}h) = $(date -d "@$(($(cat $START_FILE) + $HOURS_TO_WARN * 3600))" '+%Y-%m-%d %H:%M')"
    log "  Kill at  \$${KILL_THRESHOLD}  (~${HOURS_TO_KILL}h) = $(date -d "@$(($(cat $START_FILE) + $HOURS_TO_KILL * 3600))" '+%Y-%m-%d %H:%M')"
}

# ── Cost monitor — runs in background, warns and auto-kills ──
start_cost_monitor() {
    log "=== Cost Monitor ==="
    log "  Running in background, checks every 5 min"
    log "  Warns at \$${WARN_THRESHOLD}  (${HOURS_TO_WARN}h elapsed)"
    log "  Kills training at \$${KILL_THRESHOLD} (${HOURS_TO_KILL}h elapsed)"

    nohup bash -c '
        START_TS=$(cat '"$START_FILE"')
        WARN_SEC=$(( '"$HOURS_TO_WARN"' * 3600 ))
        KILL_SEC=$(( '"$HOURS_TO_KILL"' * 3600 ))
        WARNED=false

        while true; do
            NOW=$(date +%s)
            ELAPSED=$(( NOW - START_TS ))
            ELAPSED_H=$(echo "scale=2; $ELAPSED / 3600" | bc)
            COST=$(echo "scale=2; $ELAPSED_H * '"$COST_PER_HOUR"'" | bc)

            if [ $ELAPSED -ge $KILL_SEC ]; then
                echo "[COST] ⛔ \$${COST} reached at ${ELAPSED_H}h. KILLING training." \
                    | tee -a '"$CHECKPOINTS_DIR/cost_monitor.log"'
                echo "[COST] Syncing final checkpoints to S3..." \
                    | tee -a '"$CHECKPOINTS_DIR/cost_monitor.log"'
                bash '"$REPO_DIR"'/scripts/sync_checkpoints.sh \
                    --once '"$CHECKPOINTS_DIR"' --s3 '"$S3_BUCKET"'/checkpoints/ \
                    >> '"$CHECKPOINTS_DIR/cost_monitor.log"' 2>&1
                echo "[COST] ⛔ DESTROY VM NOW to stop billing. Data is safe in S3." \
                    | tee -a '"$CHECKPOINTS_DIR/cost_monitor.log"'
                echo "[COST] Kill command: aws s3 ls $S3_BUCKET/checkpoints/ --recursive" \
                    | tee -a '"$CHECKPOINTS_DIR/cost_monitor.log"'
                # Kill training (if still running)
                pkill -f train_clara_mamba3.py 2>/dev/null || true
                exit 0
            fi

            if [ $ELAPSED -ge $WARN_SEC ] && [ "$WARNED" = false ]; then
                echo "[COST] ⚠️  \$${COST} spent at ${ELAPSED_H}h. Wrap up soon!" \
                    | tee -a '"$CHECKPOINTS_DIR/cost_monitor.log"'
                WARNED=true
            fi

            # Log every 30 min
            if [ $(( ELAPSED % 1800 )) -lt 300 ]; then
                echo "[COST] \$${COST} @ ${ELAPSED_H}h (next warn at \$${WARN_THRESHOLD})" \
                    >> '"$CHECKPOINTS_DIR/cost_monitor.log"' 2>/dev/null || true
            fi

            sleep 300
        done
    ' > /dev/null 2>&1 &
    log "  Cost monitor PID: $!"
    log "  Logs: tail -f $CHECKPOINTS_DIR/cost_monitor.log"
}

# ── Phase 0: System Dependencies ──────────────────────────────────────────────
phase_system() {
    log "=== Phase 0: System Dependencies ==="
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3 python3-pip python3-venv \
        git wget curl \
        cmake g++ \
        libsndfile1-dev \
        bc

    python3 -m venv "$HOME/clara_env"
    source "$HOME/clara_env/bin/activate"
    pip install --quiet --upgrade pip

    # ROCm PyTorch (MI300X AMI should have ROCm pre-installed)
    pip install --quiet --upgrade \
        torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4

    python3 -c "
import torch
gpu = torch.cuda.get_device_name(0)
print(f'  GPU: {gpu}')
" 2>/dev/null || warn "GPU not detected. Set HSA_OVERRIDE_GFX_VERSION if needed."

    pip install --quiet numpy scipy soundfile tqdm
    log "  System ready. ROCm: $(rocm-smi --showversion 2>/dev/null || echo 'N/A')"
    log "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
}

# ── Phase 1: Repo ─────────────────────────────────────────────────────────────
phase_repo() {
    log "=== Phase 1: Repository ==="
    if [ -d "$REPO_DIR" ]; then
        cd "$REPO_DIR" && git pull --ff-only
    else
        git clone "$REPO_URL" "$REPO_DIR" && cd "$REPO_DIR"
    fi
    log "  Branch: $(git branch --show-current)  Commit: $(git rev-parse --short HEAD)"

    # Install awscli for S3 sync (user must `aws configure` themselves)
    pip install --quiet awscli 2>/dev/null || warn "awscli install failed — install manually"
}

# ── Phase 2: Dataset — MUSDB18-HQ ONLY ────────────────────────────────────────
# F0/CREPE/CREPE pitch was used in the OLD Mamba-1 pipeline (__train_laptop_ref.py).
# The current Mamba-3 architecture does NOT use F0 conditioning for Stage 0/1.
# F0/pitch extraction would go in cond slots 0-7 (DAW context) if added later.
# Skipped for MVP — not blocking Stage 0 identity training.
phase_datasets() {
    log "=== Phase 2: Dataset ==="
    log "  ONLY MUSDB18-HQ is needed for Stage 0 identity training."
    log "  FMA Full:     SKIPPED (100 GB download, VM gets destroyed)"
    log "  MAESTRO:      SKIPPED (not needed until conditioned training)"
    log "  Slakh/FSD50K: SKIPPED (not needed for MVP)"
    log "  F0/CREPE:     SKIPPED (old Mamba-1 pipeline, not in Mamba-3)"

    local musdb_dir="$DATASETS_DIR/musdb18hq"

    # Check if already extracted
    local n_wavs
    n_wavs=$(find "$musdb_dir" -name "*.wav" 2>/dev/null | wc -l)
    if [ "$n_wavs" -gt 0 ]; then
        log "  MUSDB18-HQ: ALREADY EXTRACTED ($n_wavs WAV files, $(du -sh "$musdb_dir" | cut -f1))"
        return
    fi

    # Check if compressed file already exists locally
    local archive="$DATASETS_DIR/musdb18hq.tar.gz"
    if [ -f "$archive" ]; then
        log "  MUSDB18-HQ: found existing archive ($(du -h "$archive" | cut -f1))"
    else
        log "  MUSDB18-HQ: downloading (~21 GB compressed)..."
        log "  Using aria2c (4 connections, auto-resume)"
        mkdir -p "$DATASETS_DIR"
        cd "$DATASETS_DIR"
        aria2c -x 4 -s 4 --continue=true \
            "https://zenodo.org/record/5061438/files/musdb18hq.tar.gz" \
            -o musdb18hq.tar.gz
        log "  Download complete"
    fi

    log "  Extracting (~21 GB → ~41 GB)..."
    cd "$DATASETS_DIR"
    tar -xzf musdb18hq.tar.gz -C "$DATASETS_DIR" --checkpoint=10000 2>&1
    local extracted_wavs
    extracted_wavs=$(find "$DATASETS_DIR/musdb18hq" -name "*.wav" 2>/dev/null | wc -l)
    log "  MUSDB18-HQ: READY ($extracted_wavs WAV files, $(du -sh "$musdb_dir" | cut -f1))"

    # Optionally remove archive to save space
    # rm -f "$archive"
}

# ── Phase 3: Precompute Features ──────────────────────────────────────────────
phase_precompute() {
    log "=== Phase 3: Precompute Features ==="
    cd "$REPO_DIR"
    source "$HOME/clara_env/bin/activate"

    local input_dir="$DATASETS_DIR/musdb18hq/train"
    local output_dir="$DATASETS_DIR/musdb18hq/precomputed"

    if [ -f "$output_dir/metadata.json" ]; then
        log "  Features already exist: $(grep train_chunks "$output_dir/metadata.json")"
        return
    fi

    local n_workers=$(( $(nproc) / 2 ))
    n_workers=$(( n_workers > 1 ? n_workers : 2 ))
    log "  Precomputing mel + STFT (workers=$n_workers, chunk=${CHUNK_SECONDS}s)..."
    python3 scripts/precompute_features.py \
        --input "$input_dir" \
        --output "$output_dir" \
        --chunk-seconds "$CHUNK_SECONDS" \
        --workers "$n_workers"
    log "  Precompute complete: $(grep train_chunks "$output_dir/metadata.json")"
}

# ── Phase 4: Stage 0 Identity Training ────────────────────────────────────────
phase_train() {
    log "=== Phase 4: Stage 0 Identity Training ==="
    log "  WARNING: AMD ADC bills \$1.99/hr even when STOPPED."
    log "  WARNING: DESTROYING the VM DELETES ALL DATA."
    log "  S3 sync is your only safety net."

    cd "$REPO_DIR"
    source "$HOME/clara_env/bin/activate"

    local precomputed_dir="$DATASETS_DIR/musdb18hq/precomputed"
    local save_dir="$CHECKPOINTS_DIR/mamba3_stage0"
    mkdir -p "$save_dir"

    # Start cost monitor (background)
    start_cost_monitor

    # Start GPU monitor (background)
    nohup watch -n 60 "rocm-smi --showtemp --showpower --showuse --showmemuse" \
        > "$CHECKPOINTS_DIR/gpu_monitor.log" 2>&1 &

    # Start S3 sync (background, every 5 min)
    if command -v aws &>/dev/null && [ -f "$HOME/.aws/credentials" ]; then
        log "  S3 configured — syncing checkpoints every 5 min"
        nohup bash "$REPO_DIR/scripts/sync_checkpoints.sh" \
            --watch "$save_dir" \
            --s3 "$S3_BUCKET/mamba3_stage0/" \
            --interval 300 \
            > "$CHECKPOINTS_DIR/sync_checkpoints.log" 2>&1 &
    else
        warn "  S3 NOT configured. Run: aws configure"
        warn "  Without S3: DESTROYING the VM = LOSING all checkpoints."
        warn "  S3 cost: ~\$0.02/GB/month. A full training run is ~\$0.23."
    fi

    # ── Launch training ──
    log "  Steps: $TRAIN_STEPS  Batch: $BATCH_SIZE  Accum: $GRAD_ACCUM"
    log "  Logs: tail -f $save_dir/train.log"
    log "  Cost: tail -f $CHECKPOINTS_DIR/cost_monitor.log"

    python3 train_clara_mamba3.py \
        --precomputed "$precomputed_dir" \
        --cache \
        --device cuda \
        --batch "$BATCH_SIZE" \
        --grad-accum "$GRAD_ACCUM" \
        --steps "$TRAIN_STEPS" \
        --lr 1e-3 --weight-decay 1e-5 --clip-grad-norm 1.0 \
        --num-workers 4 --log-every 100 --ckpt-every 5000 \
        --warmup 2000 \
        --save "$save_dir" \
        2>&1 | tee "$save_dir/train.log"

    # Training complete — final sync
    log "  Training complete. Final sync to S3..."
    if command -v aws &>/dev/null; then
        bash "$REPO_DIR/scripts/sync_checkpoints.sh" \
            --once "$save_dir" --s3 "$S3_BUCKET/mamba3_stage0/"
    fi
    log "  Final sync done. All checkpoints in S3."
}

# ── Phase 5: Export + Validate ────────────────────────────────────────────────
phase_export() {
    log "=== Phase 5: Export + Validate ==="
    cd "$REPO_DIR"
    source "$HOME/clara_env/bin/activate"

    local ckpt="${1:-$CHECKPOINTS_DIR/mamba3_stage0/best.pt}"
    if [ ! -f "$ckpt" ]; then
        ckpt="$CHECKPOINTS_DIR/mamba3_stage0/final.pt"
    fi
    if [ ! -f "$ckpt" ]; then
        err "No checkpoint found at $CHECKPOINTS_DIR/mamba3_stage0/"
        return 1
    fi

    local export_path="$REPO_DIR/models/MambaTransparent/model.bin"
    mkdir -p "$(dirname "$export_path")"

    log "  Exporting: $ckpt → $export_path"
    python3 train_clara_mamba3.py --export "$ckpt" --output "$export_path"
    python3 scripts/validate_export.py "$export_path"
    log "  Export valid."

    # Sync exported model to S3
    if command -v aws &>/dev/null; then
        bash "$REPO_DIR/scripts/sync_checkpoints.sh" \
            --once "$REPO_DIR/models/" --s3 "$S3_BUCKET/exports/"
        log "  Exported model synced to S3: $S3_BUCKET/exports/MambaTransparent/"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║     CLARA MI300X — Single Launch Training         ║${NC}"
    echo -e "${CYAN}║     Auto-kill at \$${KILL_THRESHOLD} (~${HOURS_TO_KILL}h)               ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  AMD ADC bills \$1.99/hr even when STOPPED.${NC}"
    echo -e "${YELLOW}⚠️  DESTROY VM to stop billing → DELETES ALL DATA.${NC}"
    echo -e "${YELLOW}⚠️  S3 sync protects your checkpoints.${NC}"
    echo -e "${YELLOW}⚠️  No S3? Run 'aws configure' after this script finishes.${NC}"
    echo ""

    local cmd="${1:-all}"
    case "$cmd" in
        system)    phase_system ;;
        repo)      phase_repo ;;
        datasets)  phase_datasets ;;
        precompute) phase_precompute ;;
        train)     record_start; phase_train ;;
        export)    phase_export ;;
        all)
            phase_system
            phase_repo
            phase_datasets
            phase_precompute
            record_start
            phase_train
            phase_export
            ;;
        *)
            echo "Usage: $0 [system|repo|datasets|precompute|train|export|all]"
            echo ""
            echo "  system     — Install deps + ROCm PyTorch"
            echo "  repo       — Clone CLARA repo"
            echo "  datasets   — Download MUSDB18-HQ only"
            echo "  precompute — Precompute mel+STFT"
            echo "  train      — Stage 0 identity training (records cost start)"
            echo "  export     — Export + validate model.bin"
            echo "  all        — Run everything (default)"
            exit 1
            ;;
    esac

    echo ""
    log "Phase '$cmd' complete."
    echo ""
    echo -e "${CYAN}── Next steps ──${NC}"
    echo "  Check training: tail -f $CHECKPOINTS_DIR/mamba3_stage0/train.log"
    echo "  Check cost:     tail -f $CHECKPOINTS_DIR/cost_monitor.log"
    echo "  Check GPU:      tail -f $CHECKPOINTS_DIR/gpu_monitor.log"
    echo ""
    echo -e "${YELLOW}  At \$90: cost monitor kills training and syncs to S3.${NC}"
    echo -e "${YELLOW}  Then DESTROY the VM to stop billing.${NC}"
    echo ""
}

main "$@"
