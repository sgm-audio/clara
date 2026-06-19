#!/usr/bin/env bash
# ==============================================================================
# CLARA Pipeline Quick Smoke Test
# ==============================================================================
# Tests the complete pipeline end-to-end with synthetic data:
#   1. Generate synthetic training data
#   2. Train MambaRTModelParallel for 50 steps
#   3. Export to ONNX
#   4. Verify ONNX loads in onnxruntime
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${HOME}/clara-training/venv"
source "${VENV}/bin/activate"
export HSA_OVERRIDE_GFX_VERSION=11.0.0

echo "======================================================================"
echo " CLARA Pipeline Smoke Test"
python -c 'import torch; print("PyTorch: " + torch.__version__); print("ROCm: " + (torch.version.hip or "N/A")); print("GPU: " + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"))'
echo "======================================================================"

cd "${SCRIPT_DIR}"

# Step 1: Generate synthetic dataset
echo ""
echo "--- Step 1: Generate synthetic training data ---"
SYNTH_DIR="/tmp/clara_smoke_test/train"
rm -rf /tmp/clara_smoke_test
mkdir -p "${SYNTH_DIR}"
python << 'PYEOF1'
import numpy as np, soundfile as sf, os
sr = 44100
synth_dir = "/tmp/clara_smoke_test/train"
os.makedirs(synth_dir, exist_ok=True)
for i in range(3):
    wav = np.random.randn(sr * 5).astype(np.float32) * 0.1
    sf.write(f"{synth_dir}/track_{i}.wav", wav, sr)
print(f"Created 3 synthetic WAV files in {synth_dir}")
PYEOF1

# Step 2: Short training run (50 steps)
echo ""
echo "--- Step 2: Train MambaRTModelParallel for 10 steps ---"
python "${SCRIPT_DIR}/train_clara_parallel.py" \
    --data "${SYNTH_DIR}" \
    --steps 10 \
    --batch 2 \
    --grad-accum 2 \
    --clip-seconds 0.025 \
    --log-every 5 \
    --ckpt-every 10 \
    --val-every 50 \
    --num-workers 0 \
    --save /tmp/clara_smoke_test/checkpoints \
    2>&1

echo ""

# Step 3: Export to ONNX
echo "--- Step 3: Export to ONNX ---"
python << 'PYEOF2'
import sys, os
sys.path.insert(0, "/var/home/sgm/Github Repo's/clara")
from train_clara_export import export_to_onnx

ckpt_path = None
ckpt_dir = "/tmp/clara_smoke_test/checkpoints"
for f in sorted(os.listdir(ckpt_dir)):
    if f.startswith("checkpoint_") and f.endswith(".pt"):
        ckpt_path = os.path.join(ckpt_dir, f)

if not ckpt_path:
    ckpt_path = os.path.join(ckpt_dir, "final.pt")

print(f"Using checkpoint: {ckpt_path}")

export_to_onnx(
    checkpoint_path=ckpt_path,
    output_path="/tmp/clara_smoke_test/mamba_rt.onnx",
    seq_len=64,
)
PYEOF2

echo ""

# Step 4: Verify ONNX
echo "--- Step 4: Verify ONNX in onnxruntime ---"
python << 'PYEOF3'
import onnxruntime as ort, numpy as np

sess = ort.InferenceSession("/tmp/clara_smoke_test/mamba_rt.onnx")
print(f"Inputs: {[i.name for i in sess.get_inputs()]}")
print(f"Outputs: {[o.name for o in sess.get_outputs()]}")

# Run with zero input
feeds = {"audio_chunk": np.zeros((1, 64, 1), dtype=np.float32)}
for i in range(4):
    feeds[f"conv{i}"] = np.zeros((1, 64, 4), dtype=np.float32)
    feeds[f"ssm{i}"] = np.zeros((1, 64, 16), dtype=np.float32)
feeds["cond"] = np.zeros((1, 8), dtype=np.float32)

out = sess.run(None, feeds)
print(f"Output shape: {out[0].shape}")
print(f"Output range: [{out[0].min():.6f}, {out[0].max():.6f}]")
print(f"Output variance: {out[0].var():.8f}")
print("ONNX verification: PASSED" if out[0].var() > 1e-12 else "ONNX: untrained (random weights)")
PYEOF3

echo ""
echo "======================================================================"
echo " Smoke Test Complete!"
echo " Checkpoints: /tmp/clara_smoke_test/checkpoints/"
echo " ONNX model:  /tmp/clara_smoke_test/mamba_rt.onnx"
echo "======================================================================"
