"""Compare node count for seq_len=64 vs 256 export."""
import os, sys, math, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, onnx, tempfile
sys.path.insert(0, r'C:\dev\Audio Development\0 CLARA')

from train_clara_model import MambaRTModelSequential, make_zero_states, make_zero_cond

D = 64; S = 16; N = 4; W = 4; R = 16; C = 8

model = MambaRTModelSequential(D, S, N, W, R, C)
model.eval()

for seq_len in [4, 16, 64, 256]:
    audio = torch.randn(1, seq_len, 1)
    convs, ssms = make_zero_states(1, D, S, N, W)
    cond = make_zero_cond(1, C)
    
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        p = f.name
    
    torch.onnx.export(model, (audio, *convs, *ssms, cond), p,
        export_params=True, opset_version=18,
        input_names=['a','c0','c1','c2','c3','s0','s1','s2','s3','co'],
        output_names=['o','nc','ns'])
    
    m = onnx.load(p)
    print(f'seq_len={seq_len:3d}: {len(m.graph.node):5d} nodes')
    os.unlink(p)
