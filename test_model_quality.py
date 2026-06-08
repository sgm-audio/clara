#!/usr/bin/env python3
"""
Model Quality Test Suite
========================
Tests that the model actually processes audio correctly.
Run after training to verify the model works before export.

Usage:
    python test_model_quality.py --checkpoint checkpoints_clara/best.pt
    python test_model_quality.py --checkpoint checkpoints_clara/best.pt --onnx models/mamba_rt.onnx
"""

import sys, os, argparse, json
import numpy as np
import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, ".")

from train_clara_model import (
    create_training_model, create_export_model,
    make_zero_states, make_zero_cond,
)


def si_sdr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute SI-SDR in dB."""
    eps = 1e-8
    target_energy = np.sum(target ** 2)
    dot = np.sum(pred * target)
    proj = dot / (target_energy + eps) * target
    noise = pred - proj
    sdr = 10 * np.log10(np.sum(proj ** 2) / (np.sum(noise ** 2) + eps) + eps)
    return float(sdr)


def test_model_quality(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    d_model, d_state, n_layers = 64, 16, 4
    conv_width, dt_rank, cond_dim = 4, 16, 8
    sr = 44100
    
    # ── Load model ──
    print(f"\n{'='*60}")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"{'='*60}")
    
    model = create_export_model(d_model, d_state, n_layers, conv_width, dt_rank, cond_dim)
    model.eval()
    
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    if 'state_dict' in sd:
        sd = sd['state_dict']
    
    compatible = {k: v for k, v in sd.items() if k in model.state_dict()}
    missing = set(model.state_dict().keys()) - set(compatible.keys())
    if missing:
        print(f"  WARNING: {len(missing)} missing keys (will use random init)")
        for m in sorted(missing):
            print(f"    - {m}")
    
    model.load_state_dict(compatible, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  Loaded {len(compatible)}/{len(model.state_dict())} parameters")
    print(f"  Step: {ckpt.get('step', '?')}, Loss: {ckpt.get('loss', '?'):.4f}")
    
    # ── Test 1: Identity passthrough ──
    print(f"\n[Test 1] Identity passthrough (sine wave)")
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    # Process in 64-sample chunks (matching inference engine)
    chunk_size = 64
    conv_states = [np.zeros((1, d_model, conv_width), dtype=np.float32) for _ in range(n_layers)]
    ssm_states = [np.zeros((1, d_model, d_state), dtype=np.float32) for _ in range(n_layers)]
    cond = np.zeros((1, cond_dim), dtype=np.float32)
    
    output_chunks = []
    for start in range(0, len(sine), chunk_size):
        chunk = sine[start:start + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        
        inp = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(-1).to(device)
        cond_t = torch.from_numpy(cond).to(device)
        convs_t = tuple(torch.from_numpy(s).to(device) for s in conv_states)
        ssms_t = tuple(torch.from_numpy(s).to(device) for s in ssm_states)
        
        with torch.no_grad():
            out, conv_out, ssm_out = model(inp, *convs_t, *ssms_t, cond_t)
        
        out_np = out.cpu().numpy().flatten()
        output_chunks.append(out_np[:len(sine) - start] if start + chunk_size > len(sine) else out_np)
        
        # Update states
        for i in range(n_layers):
            conv_states[i] = conv_out[0, i].cpu().numpy()
            ssm_states[i] = ssm_out[0, i].cpu().numpy()
    
    output = np.concatenate(output_chunks)
    
    # Metrics
    sdr_identity = si_sdr(output, sine)
    l1_err = np.mean(np.abs(output - sine))
    max_err = np.max(np.abs(output - sine))
    
    print(f"  SI-SDR: {sdr_identity:.2f} dB")
    print(f"  L1 error: {l1_err:.6f}")
    print(f"  Max error: {max_err:.6f}")
    print(f"  Output RMS: {np.sqrt(np.mean(output**2)):.4f}")
    print(f"  {'PASS' if sdr_identity > 10 else 'FAIL'}: SI-SDR > 10 dB required")
    
    # ── Test 2: Multi-chunk state consistency ──
    print(f"\n[Test 2] Multi-chunk state carryover")
    chunk1 = np.random.randn(1, 64, 1).astype(np.float32)
    chunk2 = np.random.randn(1, 64, 1).astype(np.float32)
    
    # Process chunk 1 + chunk 2 as separate calls with state carryover
    conv_s = [np.zeros((1, d_model, conv_width), dtype=np.float32) for _ in range(n_layers)]
    ssm_s = [np.zeros((1, d_model, d_state), dtype=np.float32) for _ in range(n_layers)]
    
    def run_chunk(chunk_np, convs_in, ssms_in):
        inp = torch.from_numpy(chunk_np).to(device)
        cond_t = torch.zeros(1, cond_dim, device=device)
        convs_t = tuple(torch.from_numpy(s).to(device) for s in convs_in)
        ssms_t = tuple(torch.from_numpy(s).to(device) for s in ssms_in)
        with torch.no_grad():
            out, co, so = model(inp, *convs_t, *ssms_t, cond_t)
        convs_out = [co[0, i].cpu().numpy() for i in range(n_layers)]
        ssms_out = [so[0, i].cpu().numpy() for i in range(n_layers)]
        return out.cpu().numpy(), convs_out, ssms_out
    
    out1, conv1, ssm1 = run_chunk(chunk1, conv_s, ssm_s)
    out2, conv2, ssm2 = run_chunk(chunk2, conv1, ssm1)
    
    # Compare with concatenated processing
    concat = np.concatenate([chunk1, chunk2], axis=1)
    conv_ref = [np.zeros((1, d_model, conv_width), dtype=np.float32) for _ in range(n_layers)]
    ssm_ref = [np.zeros((1, d_model, d_state), dtype=np.float32) for _ in range(n_layers)]
    out_ref, _, _ = run_chunk(concat, conv_ref, ssm_ref)
    
    diff_chunk2 = np.max(np.abs(out2 - out_ref[:, 64:, :]))
    print(f"  Chunk 2 vs reference max diff: {diff_chunk2:.8f}")
    print(f"  {'PASS' if diff_chunk2 < 1e-4 else 'FAIL'}: State carryover consistency")
    
    # ── Test 3: Export cross-check ──
    print(f"\n[Test 3] Training model vs export model (weight transfer)")
    if args.checkpoint is not None:
        train_model = create_training_model(d_model, d_state, n_layers, conv_width, dt_rank, cond_dim)
        train_sd = {}
        # Map export model keys back to training model
        for k, v in compatible.items():
            if k in train_model.state_dict():
                train_sd[k] = v
        train_model.load_state_dict(train_sd, strict=False)
        train_model = train_model.to(device)
        train_model.eval()
        
        test_audio = torch.randn(1, 64, 1, device=device)
        test_cond = torch.zeros(1, cond_dim, device=device)
        t_convs, t_ssms = make_zero_states(1, d_model, d_state, n_layers, conv_width, device)
        
        with torch.no_grad():
            p_out, _, _ = train_model(test_audio, *t_convs, *t_ssms, test_cond)
            s_out, _, _ = model(test_audio, *t_convs, *t_ssms, test_cond)
        
        diff = (p_out - s_out).abs().max().item()
        print(f"  Training vs export max diff: {diff:.10f}")
        print(f"  {'PASS' if diff < 1e-5 else 'FAIL'}: Export model matches training model")
    
    # ── Test 4: ONNX model validation (if path provided) ──
    if args.onnx:
        print(f"\n[Test 4] ONNX model validation: {args.onnx}")
        import onnxruntime as ort
        
        sess = ort.InferenceSession(args.onnx)
        
        # Test shapes
        audio = np.random.randn(1, 64, 1).astype(np.float32)
        feeds = {'audio_chunk': audio}
        for i in range(n_layers):
            feeds[f'conv{i}'] = np.zeros((1, d_model, conv_width), dtype=np.float32)
            feeds[f'ssm{i}'] = np.zeros((1, d_model, d_state), dtype=np.float32)
        feeds['cond'] = np.zeros((1, cond_dim), dtype=np.float32)
        
        out_audio, out_conv, out_ssm = sess.run(None, feeds)
        
        assert out_audio.shape == (1, 64, 1), f"Audio shape: {out_audio.shape}"
        assert out_conv.shape == (1, n_layers, d_model, conv_width), f"Conv shape: {out_conv.shape}"
        assert out_ssm.shape == (1, n_layers, d_model, d_state), f"SSM shape: {out_ssm.shape}"
        assert not np.allclose(out_audio, 0.0), "Output is all zeros!"
        assert np.all(np.isfinite(out_audio)), "Output has NaN/Inf!"
        
        # Compare ONNX vs PyTorch
        audio_pt = torch.from_numpy(audio)
        cond_pt = torch.zeros(1, cond_dim, device=device)
        convs_pt = tuple(torch.zeros(1, d_model, conv_width, device=device) for _ in range(n_layers))
        ssms_pt = tuple(torch.zeros(1, d_model, d_state, device=device) for _ in range(n_layers))
        
        with torch.no_grad():
            pt_out, pt_conv, pt_ssm = model(audio_pt, *convs_pt, *ssms_pt, cond_pt)
        
        flat_pt = pt_out.cpu().numpy().flatten()
        flat_ort = out_audio.flatten()
        cos_sim = np.dot(flat_pt, flat_ort) / (
            np.sqrt(np.dot(flat_pt, flat_pt)) * np.sqrt(np.dot(flat_ort, flat_ort)) + 1e-10
        )
        max_diff = np.max(np.abs(flat_pt - flat_ort))
        
        print(f"  Cosine similarity: {cos_sim:.8f}")
        print(f"  Max diff: {max_diff:.8f}")
        print(f"  {'PASS' if cos_sim > 0.999 and max_diff < 1e-4 else 'FAIL'}: ONNX vs PyTorch")
    
    # ── Overall ──
    quality_pass = sdr_identity > 10
    state_pass = diff_chunk2 < 1e-4
    
    print(f"\n{'='*60}")
    print(f"MODEL QUALITY: {'PASS' if quality_pass else 'FAIL'}")
    print(f"STATE CARRYOVER: {'PASS' if state_pass else 'FAIL'}")
    print(f"SI-SDR: {sdr_identity:.2f} dB")
    print(f"{'='*60}")
    
    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'si_sdr_db': sdr_identity,
        'l1_error': float(l1_err),
        'max_error': float(max_err),
        'state_carryover_diff': float(diff_chunk2),
        'quality_pass': quality_pass,
        'state_pass': state_pass,
    }
    with open('model_quality_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to model_quality_report.json")
    
    if args.save_wav:
        if quality_pass:
            sf.write(args.save_wav, output, sr)
            print(f"Saved output WAV: {args.save_wav}")
        else:
            # Save failure case too for debugging
            fail_path = args.save_wav.replace('.wav', '_FAIL.wav')
            sf.write(fail_path, output, sr)
            print(f"Saved FAIL output: {fail_path}")
    
    return quality_pass and state_pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CLARA Model Quality Test Suite")
    parser.add_argument('--checkpoint', required=True, help="Path to trained checkpoint .pt")
    parser.add_argument('--onnx', default=None, help="Path to ONNX model for cross-validation")
    parser.add_argument('--save-wav', default=None, help="Save output WAV for listening")
    args = parser.parse_args()
    
    success = test_model_quality(args)
    sys.exit(0 if success else 1)
