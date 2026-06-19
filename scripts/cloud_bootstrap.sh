#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# CLARA Cloud Bootstrap — Provision a fresh MI300X instance
# ═══════════════════════════════════════════════════════════════════
# Usage on a fresh AMD AI Dev Cloud instance:
#   curl -fsSL https://raw.githubusercontent.com/<user>/clara/main/scripts/cloud_bootstrap.sh | bash
#
# Or download and run:
#   wget https://raw.githubusercontent.com/<user>/clara/main/scripts/cloud_bootstrap.sh
#   chmod +x cloud_bootstrap.sh && ./cloud_bootstrap.sh
# ═══════════════════════════════════════════════════════════════════

REPO_URL="${REPO_URL:-https://github.com/SGM-Studios/clara.git}"
DATASET_URL="${DATASET_URL:-https://zenodo.org/records/1118372/files/musdb18hq.zip}"
DATASET_DIR="${DATASET_DIR:-$HOME/datasets/musdb18hq}"
VENV_DIR="${VENV_DIR:-$HOME/clara-venv}"
CLARA_DIR="${CLARA_DIR:-$HOME/clara}"
TRAINING_STEPS="${TRAINING_STEPS:-100000}"
TMUX_SESSION="${TMUX_SESSION:-clara-train}"
GIT_BRANCH="${GIT_BRANCH:-main}"

echo "=== CLARA Cloud Bootstrap ==="
echo "  Repo:       $REPO_URL"
echo "  Dataset:    $DATASET_DIR"
echo "  Steps:      $TRAINING_STEPS"
echo "  Session:    $TMUX_SESSION"
echo ""

# ── 1. Clone repo ──
if [ ! -d "$CLARA_DIR" ]; then
    echo "[1/5] Cloning repo..."
    git clone --branch "$GIT_BRANCH" "$REPO_URL" "$CLARA_DIR"
else
    echo "[1/5] Repo exists, pulling..."
    cd "$CLARA_DIR" && git pull
fi
cd "$CLARA_DIR"

# ── 2. Download dataset ──
if [ ! -d "$DATASET_DIR/train" ]; then
    echo "[2/5] Downloading MUSDB18-HQ..."
    mkdir -p "$DATASET_DIR"
    aria2c -x 4 -s 4 "$DATASET_URL" -d "$DATASET_DIR" -o musdb18hq.zip
    echo "  Unzipping..."
    unzip -q "$DATASET_DIR/musdb18hq.zip" -d "$DATASET_DIR"
    rm "$DATASET_DIR/musdb18hq.zip"
    echo "  Dataset ready: $(ls "$DATASET_DIR/train" | wc -l) tracks"
else
    echo "[2/5] Dataset already exists at $DATASET_DIR"
fi

# ── 3. Python venv with ROCm ──
if [ ! -d "$VENV_DIR" ]; then
    echo "[3/5] Creating Python venv..."
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip setuptools wheel
    
    # ROCm PyTorch (MI300X compatible)
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    
    # Training dependencies
    pip install numpy soundfile
else
    echo "[3/5] Venv exists, updating..."
    source "$VENV_DIR/bin/activate"
    pip install --upgrade torch torchaudio numpy soundfile
fi

# Also install deps in system path for git hooks etc.
pip install --user numpy soundfile 2>/dev/null || true

# ── 4. Verify the setup ──
echo "[4/5] Verifying setup..."
python3 -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  ROCm available: {torch.cuda.is_available()}')
print(f'  Devices: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'    {i}: {torch.cuda.get_device_name(i)}')
"

# Quick smoke test
python3 -c "
import train_clara_mamba3 as t
m = t.Mamba3Model(t.Mamba3Config())
print(f'  Model: {m.num_params():,} params')
x = torch.randn(2, 64)
y = m(x)
assert torch.isfinite(y).all(), 'NaN in forward'
print('  Forward: OK')
" 2>&1 | tail -5

echo ""
echo "[5/5] Starting training in tmux session: $TMUX_SESSION"

# ── 5. Launch training in tmux ──
train_cmd="cd $CLARA_DIR && source $VENV_DIR/bin/activate && \
python3 train_clara_mamba3.py \
  --data $DATASET_DIR/train \
  --save checkpoints \
  --steps $TRAINING_STEPS \
  --batch 8 \
  --grad-accum 4 \
  --log-every 100 \
  --ckpt-every 5000 \
  --device cuda"

# Also push checkpoints every N steps (separate background loop)
push_cmd="cd $CLARA_DIR && source $VENV_DIR/bin/activate && \
while true; do sleep 3600; \
  git add checkpoints/ && \
  git commit -m \"auto-checkpoint \$(date +%Y-%m-%d_%H:%M)\" && \
  git push origin $GIT_BRANCH; \
done"

tmux new-session -d -s "$TMUX_SESSION" -n training "$train_cmd"
tmux new-window -t "$TMUX_SESSION" -n push "$push_cmd"

echo ""
echo "=== Bootstrap complete ==="
echo "  Attach: tmux attach -t $TMUX_SESSION"
echo "  Detach: Ctrl+B, D"
echo ""
echo "  Training:   tmux select-window -t $TMUX_SESSION:0"
echo "  Push loop:  tmux select-window -t $TMUX_SESSION:1"
echo ""
echo "  Monitor:    watch -n 10 rocm-smi --showtemp --showpower --showuse --showmemuse"
echo "  Logs:       tail -f $CLARA_DIR/train.log (if redirected)"
