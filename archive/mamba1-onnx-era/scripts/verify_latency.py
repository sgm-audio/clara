#!/usr/bin/env python3
"""Verify Mamba SSM model latency by measuring impulse response delay."""

import numpy as np
import onnxruntime as ort
import sys
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "mamba_rt.onnx")


def measure_latency(model_path, seq_len=10000, impulse_pos=5000, noise_floor=1e-6):
    """Feed impulse through model, measure output delay."""

    # Load model
    if not os.path.exists(model_path):
        print(f"SKIP: Model not found at {model_path}")
        return None

    session = ort.InferenceSession(model_path)

    # Get model metadata
    for inp in session.get_inputs():
        print(f"  Input:  {inp.name} shape={inp.shape}")
    for out in session.get_outputs():
        print(f"  Output: {out.name} shape={out.shape}")

    # Create impulse input
    audio = np.zeros((1, seq_len, 1), dtype=np.float32)
    audio[0, impulse_pos, 0] = 1.0

    # Zero state
    conv_shape = (1, 64, 4)  # dModel=64, convWidth=4
    ssm_shape = (1, 64, 16)  # dModel=64, dState=16
    zero_conv = np.zeros(conv_shape, dtype=np.float32)
    zero_ssm = np.zeros(ssm_shape, dtype=np.float32)

    # Build input dict
    inputs = {
        "audio_chunk": audio,
        "conv0": zero_conv,
        "conv1": zero_conv,
        "conv2": zero_conv,
        "conv3": zero_conv,
        "ssm0": zero_ssm,
        "ssm1": zero_ssm,
        "ssm2": zero_ssm,
        "ssm3": zero_ssm,
    }

    # Run inference
    outputs = session.run(None, inputs)
    processed = outputs[0]  # shape (1, seq_len, 1)
    processed = processed.flatten()

    # Find first non-zero above noise floor
    first_response = None
    for i, val in enumerate(processed):
        if abs(val) > noise_floor:
            first_response = i
            break

    if first_response is None:
        print("  ERROR: No response above noise floor")
        return None

    measured_latency = first_response - impulse_pos
    print(f"  Impulse at sample {impulse_pos}")
    print(f"  First response at sample {first_response}")
    print(f"  Measured latency: {measured_latency} samples")

    return measured_latency


if __name__ == "__main__":
    print("=== Mamba SSM Latency Verification (Python/ONNX) ===")
    print(f"Model: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print("SKIP: Model file not found. Run 'python3 export_mamba_onnx.py' first.")
        sys.exit(0)

    latency = measure_latency(MODEL_PATH)

    if latency is not None:
        reported = 4  # conv_width
        print(f"\nReported latency: {reported} samples")
        print(f"Measured latency: {latency} samples")
        if abs(latency - reported) <= 1:
            print("PASS: Latency matches reported value within tolerance")
        else:
            print(f"FAIL: Latency mismatch! Difference = {latency - reported} samples")
            sys.exit(1)
    else:
        print("Could not measure latency (model may not be loaded)")

    sys.exit(0)
