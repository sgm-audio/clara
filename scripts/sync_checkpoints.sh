#!/usr/bin/env bash
# ============================================================================
# CLARA Checkpoint Sync — saves checkpoints to cloud storage
#
# WHY THIS EXISTS:
#   AMD AI Developer Cloud bills $1.99/hr even when the VM is STOPPED.
#   To stop billing you must DESTROY the VM, which DELETES ALL DATA.
#   If credits run out ($100), the VM is killed and everything is lost.
#
#   This script auto-syncs checkpoints to S3 every N minutes so you never
#   lose a trained model, even if the VM gets nuked.
#
# USAGE:
#   # 1. Start sync in background (runs during training):
#   nohup scripts/sync_checkpoints.sh --watch checkpoints/ --s3 s3://clara-checkpoints/ &
#
#   # 2. One-shot sync:
#   scripts/sync_checkpoints.sh --once checkpoints/ --s3 s3://clara-checkpoints/
#
#   # 3. Without S3 (just local backup):
#   scripts/sync_checkpoints.sh --watch checkpoints/ --backup /mnt/backup/
#
# SETUP (one-time, on cloud VM):
#   pip install awscli boto3
#   aws configure  # set your AWS credentials
#
# COST NOTE:
#   S3 storage is ~$0.023/GB/month. Checkpoints are ~3-20 MB each.
#   Even 1000 checkpoints = ~10 GB = $0.23/month. Negligible.
# ============================================================================

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[SYNC]${NC} $1"; }
warn() { echo -e "${YELLOW}[SYNC]${NC} $1"; }
err()  { echo -e "${RED}[SYNC]${NC} $1"; }

# ── Defaults ──────────────────────────────────────────────────────────────────
SYNC_INTERVAL=300          # seconds between syncs (default: 5 min)
S3_BUCKET="${S3_BUCKET:-}" # e.g., s3://clara-checkpoints/
BACKUP_DIR="${BACKUP_DIR:-}"
WATCH_DIR=""
ONESHOT=false
KEEP_LOCAL_DAYS=0          # 0 = keep forever, N = delete local copies after N days

# ── Parse args ────────────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "  --watch DIR      Watch directory and sync every --interval seconds"
    echo "  --once DIR       One-shot sync of directory"
    echo "  --s3 URI         S3 bucket (s3://bucket-name/path/)"
    echo "  --backup DIR     Local backup directory (secondary)"
    echo "  --interval SEC   Sync interval (default: 300 = 5 min)"
    echo "  --keep-days N    Delete local checkpoints older than N days (default: keep all)"
    echo "  --help           Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --watch checkpoints/ --s3 s3://clara-checkpoints/"
    echo "  $0 --once checkpoints/ --s3 s3://clara-checkpoints/"
    echo "  $0 --watch checkpoints/ --backup /mnt/usb/backups/"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --watch)    WATCH_DIR="$2"; shift 2 ;;
        --once)     WATCH_DIR="$2"; ONESHOT=true; shift 2 ;;
        --s3)       S3_BUCKET="$2"; shift 2 ;;
        --backup)   BACKUP_DIR="$2"; shift 2 ;;
        --interval) SYNC_INTERVAL="$2"; shift 2 ;;
        --keep-days) KEEP_LOCAL_DAYS="$2"; shift 2 ;;
        --help|-h)  usage ;;
        *) err "Unknown option: $1"; usage ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────
if [ -z "$WATCH_DIR" ]; then
    err "No directory specified. Use --watch DIR or --once DIR"
    usage
fi

if [ ! -d "$WATCH_DIR" ]; then
    err "Directory does not exist: $WATCH_DIR"
    exit 1
fi

if [ -z "$S3_BUCKET" ] && [ -z "$BACKUP_DIR" ]; then
    warn "No sync target specified. Use --s3 or --backup."
    warn "Syncing to nowhere — checkpoints will only exist locally."
fi

# ── S3 check ──────────────────────────────────────────────────────────────────
check_s3() {
    if [ -n "$S3_BUCKET" ]; then
        if command -v aws &>/dev/null; then
            log "S3: configured ($S3_BUCKET)"
            return 0
        else
            warn "aws CLI not installed. S3 sync disabled."
            warn "  Install: pip install awscli && aws configure"
            S3_BUCKET=""
            return 1
        fi
    fi
}

# ── Sync function ─────────────────────────────────────────────────────────────
sync_checkpoints() {
    local src="$1"
    local ts
    ts=$(date '+%Y-%m-%d %H:%M:%S')
    local synced=false

    # Count files
    local n_ckpt
    n_ckpt=$(find "$src" -name "*.pt" -o -name "*.bin" -o -name "*.log" 2>/dev/null | wc -l)
    local size
    size=$(du -sh "$src" 2>/dev/null | cut -f1)
    log "[$ts] Syncing $src ($n_ckpt files, $size)..."

    # ── Sync to S3 ──
    if [ -n "$S3_BUCKET" ]; then
        if aws s3 sync "$src" "$S3_BUCKET" \
            --exclude "*.tmp" \
            --exclude "__pycache__/*" \
            --no-progress \
            2>&1; then
            log "  S3: synced to $S3_BUCKET"
            synced=true
        else
            warn "  S3: sync failed (check credentials/network)"
        fi
    fi

    # ── Sync to local backup ──
    if [ -n "$BACKUP_DIR" ]; then
        mkdir -p "$BACKUP_DIR"
        if rsync -a --progress "$src/" "$BACKUP_DIR/" 2>&1; then
            log "  Local: synced to $BACKUP_DIR"
            synced=true
        else
            warn "  Local: sync failed"
        fi
    fi

    # ── Cleanup old local checkpoints ──
    if [ "$KEEP_LOCAL_DAYS" -gt 0 ]; then
        local deleted
        deleted=$(find "$src" -name "*.pt" -type f -mtime "+$KEEP_LOCAL_DAYS" -delete -print 2>/dev/null | wc -l)
        if [ "$deleted" -gt 0 ]; then
            log "  Cleanup: deleted $deleted checkpoint(s) older than $KEEP_LOCAL_DAYS days"
        fi
    fi

    if [ "$synced" = true ]; then
        log "  Done"
    else
        warn "  No sync targets configured — run with --s3 or --backup"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    check_s3

    if [ "$ONESHOT" = true ]; then
        log "One-shot sync of $WATCH_DIR"
        sync_checkpoints "$WATCH_DIR"
        log "One-shot sync complete."
        exit 0
    fi

    # Continuous watch mode
    log "Starting checkpoint watcher..."
    log "  Watching:  $WATCH_DIR"
    log "  Interval:  ${SYNC_INTERVAL}s"
    log "  S3:        ${S3_BUCKET:-disabled}"
    log "  Backup:    ${BACKUP_DIR:-disabled}"
    log ""

    while true; do
        sync_checkpoints "$WATCH_DIR"
        log "  Next sync in ${SYNC_INTERVAL}s..."
        sleep "$SYNC_INTERVAL"
    done
}

main
