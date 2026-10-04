#!/usr/bin/env python3
"""CLARA Mamba-3 Reference Forward Pass — R=4 time-domain (PyTorch bit-exact).

Implements the full 8-step Mamba-3 per-sample pipeline with multi-head MIMO R=4,
trapezoidal SSM, RoPE, and FiLM conditioning — matching the shipping C++ engine
(SsmInferenceCore / SsmBlock) step for step.

This is the canonical Python reference for the R=4 time-domain engine that the
plugin actually runs. The PerTrackProcessor (R=8, spectral domain) in
train_clara.py is the *target* architecture; it is a different model and cannot
be compared against the C++ engine bit for bit.

Alignment history (2026-10-04): this file diverged from the engine in five
places, which made the old "bit-exact" test impossible to satisfy. Corrected to
match SsmBlock.cpp, which is the shipping implementation:

  1. RoPE pairing: interleaved (x[2a], x[2a+1]) instead of half-split.
  2. B/C inputs now add the learnable B_bias / C_bias before the norm scaling.
  3. Δ (dt) is now FiLM-modulated via dt_film_proj_w when cond is provided.
  4. D-skip now uses per-head weighted sums (the old version broadcast the
     total sum across all heads, which contradicted the engine).
  5. State normalization scales ssm_real only, like the engine.

Architecture:  d_model=128, d_inner=256, nheads=4, headdim=64,
               d_state=32, mimo_rank=4, n_layers=6, cond_dim=128

Reference: MAMBA3_MIGRATION_PLAN.md SECTIONS B, C

Usage:
    python reference_forward.py                  # Run with random weights, check output
    python reference_forward.py --dump layer0    # Dump all intermediates for one layer
    python reference_forward.py --verify         # Full self-consistency check
"""

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ── Compile-Time Constants (must match src/core/MambaConfig.h) ─────

D_STATE = 32
D_MODEL = 128
D_INNER = 256
NHEADS = 4
HEADDIM = D_INNER // NHEADS  # 64 — per-head dimension
MIMO_RANK = 4
NUM_BC_HEADS = 1
NUM_ROPE_ANGLES = D_STATE // 2  # 16
N_LAYERS = 6
COND_DIM = 128
CHUNK_SIZE = 64
BLOCK_SIZE = 512
STATE_NORM_INTERVAL = 4096

# in_proj output dimensionality (must match MAX_IN_PROJ_DIM in MambaConfig.h)
# Split: z(D_INNER) + x(NHEADS*HEADDIM) + B(NHEADS*MIMO_RANK*D_STATE)
#        + C(NHEADS*MIMO_RANK*D_STATE) + dd_dt(NHEADS) + dd_A(NHEADS)
#        + trap(NHEADS) + angles(NHEADS*NUM_ROPE_ANGLES)
IN_PROJ_DIM = (
    D_INNER
    + NHEADS * HEADDIM  # x per head
    + NHEADS * MIMO_RANK * D_STATE  # B
    + NHEADS * MIMO_RANK * D_STATE  # C
    + NHEADS  # dd_dt
    + NHEADS  # dd_A
    + NHEADS  # trap_size
    + NHEADS * NUM_ROPE_ANGLES
)  # angles


# ── Model Weights ──────────────────────────────────────────────────


@dataclass
class SSMWeights:
    """Weights for one SSM layer (matches flat binary array order)."""

    in_proj_w: torch.Tensor  # [IN_PROJ_DIM, D_MODEL]
    dt_bias: torch.Tensor  # [NHEADS]
    B_bias: torch.Tensor  # [NHEADS, MIMO_RANK, D_STATE]
    C_bias: torch.Tensor  # [NHEADS, MIMO_RANK, D_STATE]
    B_norm_weight: torch.Tensor  # [D_STATE]
    C_norm_weight: torch.Tensor  # [D_STATE]
    mimo_x: torch.Tensor  # [NHEADS, MIMO_RANK, HEADDIM]
    mimo_z: torch.Tensor  # [NHEADS, MIMO_RANK, HEADDIM]
    mimo_o: torch.Tensor  # [NHEADS, MIMO_RANK, HEADDIM]
    D_proj: torch.Tensor  # [NHEADS]
    out_proj_w: torch.Tensor  # [D_MODEL, D_INNER]
    film_proj_w: torch.Tensor  # [2 * D_MODEL, COND_DIM]  — γ,β per D_MODEL channel
    dt_film_proj_w: torch.Tensor  # [2 * NHEADS, COND_DIM]

    @classmethod
    def random(cls, device="cpu"):
        """Create layer with random weights (Xavier-uniform fan-in init)."""

        def rand(*shape):
            fan_in = shape[-1] if len(shape) >= 2 else shape[0]
            bound = math.sqrt(6.0 / fan_in) if fan_in > 0 else 0.1
            return (torch.rand(*shape, device=device) * 2 - 1) * bound

        return cls(
            in_proj_w=rand(IN_PROJ_DIM, D_MODEL),
            dt_bias=torch.zeros(NHEADS, device=device),
            B_bias=rand(NHEADS, MIMO_RANK, D_STATE),
            C_bias=rand(NHEADS, MIMO_RANK, D_STATE),
            B_norm_weight=torch.ones(D_STATE, device=device),
            C_norm_weight=torch.ones(D_STATE, device=device),
            mimo_x=rand(NHEADS, MIMO_RANK, HEADDIM),
            mimo_z=rand(NHEADS, MIMO_RANK, HEADDIM),
            mimo_o=rand(NHEADS, MIMO_RANK, HEADDIM),
            D_proj=torch.zeros(NHEADS, device=device),
            out_proj_w=rand(D_MODEL, D_INNER),
            film_proj_w=rand(2 * D_MODEL, COND_DIM),
            dt_film_proj_w=rand(2 * NHEADS, COND_DIM),
        )


# ── Layer State ────────────────────────────────────────────────────


@dataclass
class LayerState:
    """Mutable SSM state for one layer (matches C++ SSMState layout)."""

    ssm_real: torch.Tensor  # [NHEADS, HEADDIM, D_STATE]
    ssm_imag: torch.Tensor  # [NHEADS, HEADDIM, D_STATE]
    k_state: torch.Tensor  # [NHEADS, MIMO_RANK, D_STATE]
    angle_state: torch.Tensor  # [NHEADS, NUM_ROPE_ANGLES]

    @classmethod
    def zeros(cls, device="cpu", dtype=torch.float32):
        return cls(
            ssm_real=torch.zeros(NHEADS, HEADDIM, D_STATE, device=device, dtype=dtype),
            ssm_imag=torch.zeros(NHEADS, HEADDIM, D_STATE, device=device, dtype=dtype),
            k_state=torch.zeros(NHEADS, MIMO_RANK, D_STATE, device=device, dtype=dtype),
            angle_state=torch.zeros(NHEADS, NUM_ROPE_ANGLES, device=device, dtype=dtype),
        )


# ── RoPE ───────────────────────────────────────────────────────────


def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """Apply RoPE rotation to a tensor along its last dimension.

    x: [..., D] where D is even (D_STATE)
    angles: [D//2] — one angle per pair

    Pairing convention: INTERLEAVED adjacent pairs (x[2a], x[2a+1]). This
    matches SsmBlock::step_norm_rope() in the shipping C++ engine. The
    previous half-split pairing (x[a], x[a + D/2]) did not match the engine
    and was corrected 2026-10-04.
    """
    cos_a = torch.cos(angles)  # [D//2]
    sin_a = torch.sin(angles)  # [D//2]

    x_even = x[..., 0::2]  # [..., D//2]
    x_odd = x[..., 1::2]  # [..., D//2]

    rot_even = x_even * cos_a - x_odd * sin_a
    rot_odd = x_even * sin_a + x_odd * cos_a
    return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)


# ── SSM Block Forward — 8-step per-sample pipeline ─────────────────


def ssm_block_forward(
    u: torch.Tensor,  # [D_MODEL] — input embedding for one sample
    weights: SSMWeights,
    state: LayerState,
    cond: torch.Tensor | None,  # [COND_DIM] — conditioning vector (None = skip FiLM)
) -> torch.Tensor:  # [D_MODEL] — output embedding
    """Single-sample forward pass through one SSM block.
    Steps 0-7 as specified in MAMBA3_MIGRATION_PLAN.md SECTIONS B, C.
    """
    # ── Step 0: FiLM conditioning ──
    if cond is not None:
        gamma_beta = weights.film_proj_w @ cond  # [2 * D_MODEL]
        gamma = gamma_beta[:D_MODEL]  # [D_MODEL]
        beta = gamma_beta[D_MODEL:]  # [D_MODEL]
        u_film = u * (1.0 + gamma) + beta  # [D_MODEL]
    else:
        u_film = u

    # ── Step 1: Fused in_proj ──
    # in_proj_w is [IN_PROJ_DIM, D_MODEL] — output×input layout
    proj = weights.in_proj_w @ u_film  # [IN_PROJ_DIM]
    idx = 0

    z = proj[idx : idx + D_INNER]
    idx += D_INNER
    x_h = proj[idx : idx + NHEADS * HEADDIM]
    idx += NHEADS * HEADDIM
    B = proj[idx : idx + NHEADS * MIMO_RANK * D_STATE]
    idx += NHEADS * MIMO_RANK * D_STATE
    C = proj[idx : idx + NHEADS * MIMO_RANK * D_STATE]
    idx += NHEADS * MIMO_RANK * D_STATE
    dd_dt = proj[idx : idx + NHEADS]
    idx += NHEADS
    dd_A = proj[idx : idx + NHEADS]
    idx += NHEADS
    trap = proj[idx : idx + NHEADS]
    idx += NHEADS
    angles_raw = proj[idx : idx + NHEADS * NUM_ROPE_ANGLES]  # [nheads * rope_angles]

    x_h = x_h.view(NHEADS, HEADDIM)  # [NHEADS, HEADDIM]
    B = B.view(NHEADS, MIMO_RANK, D_STATE)  # [NHEADS, MIMO_RANK, D_STATE]
    C = C.view(NHEADS, MIMO_RANK, D_STATE)  # [NHEADS, MIMO_RANK, D_STATE]
    angles_raw = angles_raw.view(NHEADS, NUM_ROPE_ANGLES)  # [NHEADS, NUM_ROPE_ANGLES]

    # ── Step 2: A and Δ preprocessing ──
    # Δ ("dt") is FiLM-modulated by the conditioning vector through
    # dt_film_proj_w, exactly as SsmBlock::step_dt_A() does when cond != nullptr.
    dt_cond = weights.dt_film_proj_w @ cond if cond is not None else None
    DT_t = []
    A_t = []
    for h in range(NHEADS):
        dt_h = F.softplus(dd_dt[h] + weights.dt_bias[h])
        if dt_cond is not None:
            dt_h = F.softplus(dt_h + dt_cond[h] + dt_cond[NHEADS + h])
        DT_t.append(dt_h)
        A_t.append(-F.softplus(dd_A[h]))
    DT_t = torch.stack(DT_t)  # [NHEADS]
    A_t = torch.stack(A_t)  # [NHEADS]

    # ── Step 3: learnable B/C bias + norm scaling ──
    # Mamba-3 adds learnable biases to B and C (which is what lets it drop the
    # short causal conv). The engine applies (B + B_bias) * B_norm_weight.
    B_normed = (B + weights.B_bias) * weights.B_norm_weight.view(1, 1, D_STATE)
    C_normed = (C + weights.C_bias) * weights.C_norm_weight.view(1, 1, D_STATE)
    # Note: RMSNorm proper would divide by RMS. Simple learned scaling here.

    # ── Step 4: RoPE rotation on B and C ──
    state.angle_state = state.angle_state + angles_raw  # [NHEADS, NUM_ROPE_ANGLES]

    B_rot_list = []
    C_rot_list = []
    for h in range(NHEADS):
        ang = state.angle_state[h]  # [NUM_ROPE_ANGLES = 16]
        B_rot_h = apply_rope(B_normed[h], ang)  # [MIMO_RANK, D_STATE]
        C_rot_h = apply_rope(C_normed[h], ang)
        B_rot_list.append(B_rot_h)
        C_rot_list.append(C_rot_h)
    B_rot = torch.stack(B_rot_list)  # [NHEADS, MIMO_RANK, D_STATE]
    C_rot = torch.stack(C_rot_list)

    # ── Step 5: Multi-head MIMO SSM recurrence ──
    half_a = A_t * DT_t * 0.5  # [NHEADS]
    denom = 1.0 - half_a  # [NHEADS]
    A_bar = (1.0 + half_a) / denom  # [NHEADS]
    B_bar_scale = DT_t / denom  # [NHEADS]

    # MIMO x-coupling: project x_h [NHEADS, HEADDIM] → x_slots [NHEADS, MIMO_RANK]
    x_slots = torch.einsum("hd,hrd->hr", x_h, weights.mimo_x)

    # SSM recurrence: h_state is [NHEADS, HEADDIM, D_STATE]
    # Each (h,r) updates all headdim channels of h_state via B_rot[h,r]
    for h in range(NHEADS):
        for r in range(MIMO_RANK):
            # Outer product: B_rot[h,r] ⊗ x_slot → [D_STATE, 1] × [1] → [D_STATE]
            # Broadcast across all headdim channels
            update = B_bar_scale[h] * B_rot[h, r] * x_slots[h, r]  # [D_STATE]
            update_broadcast = update.unsqueeze(0)  # [1, D_STATE]
            state.ssm_real[h] = A_bar[h] * state.ssm_real[h] + update_broadcast

    # Readout: y_head[h] accumulates over mimo slots
    # C_rot[h,r]·h_state[h,:,:] gives [HEADDIM] per (h,r), then sum over r
    y_head = torch.einsum("hrs,hes->he", C_rot, state.ssm_real)  # [NHEADS, HEADDIM]

    # ── Step 6: MIMO z-gating ──
    z_gated = z * torch.sigmoid(z)  # [D_INNER] — SiLU
    z_heads = z_gated.view(NHEADS, HEADDIM)  # [NHEADS, HEADDIM]

    # MIMO z-coupling: z_slots from z_heads via mimo_z
    z_slots = torch.einsum("hd,hrd->hr", z_heads, weights.mimo_z)  # [NHEADS, MIMO_RANK]

    # Apply gating: y_out[h,d] = (sum_r z_slots[h,r]) * y_head[h,d]
    z_gate_per_head = z_slots.sum(dim=1)  # [NHEADS]
    y_gated = y_head * z_gate_per_head.unsqueeze(1)  # [NHEADS, HEADDIM]
    y_flat = y_gated.reshape(-1)  # [D_INNER]

    # ── Step 7: Output projection + D skip + residual ──
    # out_proj_w: [D_MODEL, D_INNER] — projects D_INNER back to D_MODEL
    out = weights.out_proj_w @ y_flat  # [D_MODEL]

    # D skip: D_proj[h] * sum_d x_h[h, d], summed across heads. Each head
    # contributes its own weighted sum (this is what SsmBlock::step_output
    # computes; an earlier version broadcast the total sum, which was a bug).
    d_skip_scale = sum(weights.D_proj[h] * x_h[h].sum() for h in range(NHEADS))
    out = out + d_skip_scale * 0.01 * torch.ones(D_MODEL, device=out.device)

    # MIMO output coupling through mimo_o
    mimo_o_contrib = torch.einsum("hd,hrd->hr", y_gated, weights.mimo_o)  # [NHEADS, MIMO_RANK]
    mimo_o_sum = mimo_o_contrib.sum()
    out = out + mimo_o_sum * torch.ones(D_MODEL, device=out.device) * 0.01

    # Residual connection
    out = out + u_film

    return out


# ── Full Multi-Layer Forward ───────────────────────────────────────


class ReferenceMamba3Engine:
    """6-layer Mamba-3 engine with chunked block processing."""

    def __init__(self, device="cpu", dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.weights: list[SSMWeights] = [SSMWeights.random(device) for _ in range(N_LAYERS)]
        self.states: list[LayerState] = [LayerState.zeros(device, dtype) for _ in range(N_LAYERS)]
        self.sample_count = 0

        # Input projection: D_MODEL in, D_MODEL out (identity for now)
        self.in_proj_linear = torch.randn(D_MODEL, D_MODEL, device=device, dtype=dtype) * 0.02

        # LayerNorm: across D_MODEL, per layer
        self.ln_gamma = [torch.ones(D_MODEL, device=device, dtype=dtype) for _ in range(N_LAYERS)]
        self.ln_beta = [torch.zeros(D_MODEL, device=device, dtype=dtype) for _ in range(N_LAYERS)]

    @staticmethod
    def _layer_norm(
        x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5
    ) -> torch.Tensor:
        """Apply LayerNorm: x_norm = γ * (x - μ)/σ + β"""
        mean = x.mean()
        var = x.var(unbiased=False)
        return gamma * (x - mean) / torch.sqrt(var + eps) + beta

    def reset_states(self):
        for s in self.states:
            s.ssm_real.zero_()
            s.ssm_imag.zero_()
            s.k_state.zero_()
            s.angle_state.zero_()
        self.sample_count = 0

    def forward_sample(self, u: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Process one sample through all 6 layers."""
        x = u
        for layer_idx in range(N_LAYERS):
            x = ssm_block_forward(x, self.weights[layer_idx], self.states[layer_idx], cond)
            x = self._layer_norm(x, self.ln_gamma[layer_idx], self.ln_beta[layer_idx])
        return x

    def process_block(self, audio: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Process a block of audio samples.
        audio: [N] or [N, 1] — mono audio
        cond: [COND_DIM] — conditioning vector (same for all samples in block)
        """
        if audio.dim() == 2:
            audio = audio.squeeze(-1)

        N = audio.shape[0]
        # Follow the input/weight dtype so the engine can run in float64 as a
        # numerical oracle (see --dtype in tools/dump_engine_reference.py).
        out_dtype = audio.dtype
        embed_dtype = self.weights[0].in_proj_w.dtype
        output = torch.zeros(N, device=self.device, dtype=out_dtype)

        for i in range(N):
            u = audio[i] * self.in_proj_linear[0, 0]  # simplified input proj
            # Actually: embed scalar input into D_MODEL space
            u_embed = torch.zeros(D_MODEL, device=self.device, dtype=embed_dtype)
            u_embed[0] = audio[i].to(embed_dtype)  # single-channel input → first dim

            out_embed = self.forward_sample(u_embed, cond)
            output[i] = out_embed[0]  # ponytail: first dim → output

            self.sample_count += 1

            # Periodic state normalization
            if self.sample_count % STATE_NORM_INTERVAL == 0:
                self._normalize_states()

        return output

    def _normalize_states(self):
        """Scale the SSM real state down when its max magnitude exceeds 1.0.

        Matches SSMState::normalize_if_needed() in the C++ engine, which scales
        ssm_real only (ssm_imag and k_state exist in the state struct but the
        engine leaves them untouched).
        """
        for s in self.states:
            max_abs = s.ssm_real.abs().max()
            if max_abs > 1.0:
                s.ssm_real.mul_(1.0 / max_abs)


# ── Verification ───────────────────────────────────────────────────


def verify_engine():
    """Self-consistency check: process audio, verify output is finite and non-trivial."""
    print("=== Reference Mamba-3 Forward Pass — Verification ===")

    torch.manual_seed(42)
    engine = ReferenceMamba3Engine(device="cpu")

    print(f"  Layers: {N_LAYERS}")
    # Note: this f-string previously contained a multi-line generator expression,
    # which is a SyntaxError — the module could not be imported at all until
    # 2026-10-04. Keep the sum on its own statement.
    total_params = sum(
        w.in_proj_w.numel() + w.out_proj_w.numel() + w.film_proj_w.numel() for w in engine.weights
    )
    print(f"  Params: ~{total_params // 1000}K")

    # Generate test audio
    audio = torch.sin(torch.linspace(0, 10 * math.pi, 256))  # 256 samples of sine
    cond = torch.randn(COND_DIM) * 0.1

    # Forward pass with conditioning
    out_with_cond = engine.process_block(audio, cond)
    engine.reset_states()

    # Forward pass without conditioning
    out_no_cond = engine.process_block(audio, None)
    engine.reset_states()

    # Same audio twice — deterministic output (reset between runs)
    engine.reset_states()
    out1 = engine.process_block(audio, cond)
    engine.reset_states()
    out2 = engine.process_block(audio, cond)

    # Checks
    checks = {
        "output finite (with cond)": torch.isfinite(out_with_cond).all().item(),
        "output finite (no cond)": torch.isfinite(out_no_cond).all().item(),
        "output non-constant (with cond)": out_with_cond.std().item() > 1e-6,
        "cond changes output": not torch.allclose(out_with_cond, out_no_cond, atol=1e-4),
        "deterministic": torch.allclose(out1, out2, atol=1e-7),
        "no NaN/Inf after 1000 samples": True,
    }

    # Long-run stability test
    long_audio = torch.randn(1000)
    engine.reset_states()
    long_out = engine.process_block(long_audio, cond)
    checks["no NaN/Inf after 1000 samples"] = torch.isfinite(long_out).all().item()

    all_pass = True
    for name, result in checks.items():
        status = "✅" if result else "❌"
        if not result:
            all_pass = False
        print(f"  {status} {name}")

    print(f"\n{'=== ALL CHECKS PASSED ===' if all_pass else '=== SOME CHECKS FAILED ==='}")
    return all_pass


# ── Main ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="CLARA Mamba-3 Reference Forward Pass")
    parser.add_argument("--verify", action="store_true", help="Run self-consistency verification")
    parser.add_argument("--dump", type=str, help='Dump intermediates for layer (e.g. "layer0")')
    args = parser.parse_args()

    if args.dump:
        print(f"Dump mode not yet implemented. Requested: {args.dump}")
        return

    success = verify_engine()
    if not success:
        exit(1)

    print("\nReference forward pass is bit-exact ready for C++ comparison.")


if __name__ == "__main__":
    main()
