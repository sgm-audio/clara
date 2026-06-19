"""
CLARA ONNX Export — wraps the sequential MambaRT model for ONNX export.
Reconstructed from verify_training_pipeline.py interface contract.
"""
import os
import torch
import numpy as np
from train_clara_model import MambaRTModelSequential

D_MODEL = 64
D_STATE = 16
N_LAYERS = 4
CONV_WIDTH = 4
DT_RANK = 16
COND_DIM = 8


def export_to_onnx(checkpoint_path, output_path, seq_len=256, opset_version=18):
    """
    Export trained sequential model to ONNX.

    Args:
        checkpoint_path: Path to .pt checkpoint with 'model_state_dict' key.
        output_path: Where to save the .onnx file.
        seq_len: Sequence length for export (dynamic on NPU, fixed is better).
        opset_version: ONNX opset version.
    """
    model = MambaRTModelSequential(
        d_model=D_MODEL, d_state=D_STATE, n_layers=N_LAYERS,
        conv_width=CONV_WIDTH, dt_rank=DT_RANK, cond_dim=COND_DIM,
    )
    model.eval()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded weights from {checkpoint_path}")

    # Export with dynamic batch, fixed seq_len
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    dummy_audio = torch.randn(1, seq_len, 1)
    dummy_convs = tuple(torch.zeros(1, D_MODEL, CONV_WIDTH) for _ in range(N_LAYERS))
    dummy_ssms = tuple(torch.zeros(1, D_MODEL, D_STATE) for _ in range(N_LAYERS))
    dummy_cond = torch.zeros(1, COND_DIM)
    all_args = (dummy_audio,) + dummy_convs + dummy_ssms + (dummy_cond,)

    input_names = ['audio_chunk'] + [f'conv{i}' for i in range(N_LAYERS)] + [f'ssm{i}' for i in range(N_LAYERS)] + ['cond']
    output_names = ['processed_audio', 'new_conv_state', 'new_ssm_state']

    torch.onnx.export(
        model,
        all_args,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'audio_chunk': {0: 'batch'},
            'processed_audio': {0: 'batch'},
            'new_conv_state': {0: 'batch'},
            'new_ssm_state': {0: 'batch'},
        },
    )
    print(f"  ONNX exported: {output_path}")
    print(f"  Inputs: {input_names}")
    print(f"  Outputs: {output_names}")
    return output_path
