"""The chunked selective scan must agree with the reference scan.

selective_scan_chunked is what actually runs during training; selective_scan_ref
is the transcription of the official reference implementation. The chunked
version reassociates the recurrence, so it is only a valid substitute if it
agrees with the reference to floating-point tolerance -- forward and backward.

The shapes exercised here are the real ones from UltraLight VM-UNet at a
256x256 input, so a pass covers every scan the model actually performs.
"""

import math

import pytest
import torch

from models.mamba_pytorch import Mamba, selective_scan_chunked, selective_scan_ref

# (name, seq_len, d_inner) for the six PVM layers; d_state is 16 throughout.
LAYERS = [
    ("encoder4", 1024, 12),
    ("encoder5", 256, 16),
    ("encoder6", 64, 24),
    ("decoder1", 64, 32),
    ("decoder2", 64, 24),
    ("decoder3", 256, 16),
]

BATCH = 2
D_STATE = 16


def _inputs(seqlen, d_inner, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *s: torch.randn(*s, generator=g).to(device)

    u = mk(BATCH, d_inner, seqlen).requires_grad_(True)
    delta = mk(BATCH, d_inner, seqlen).requires_grad_(True)
    # A must be negative, as in the model: A = -exp(A_log)
    A_log = torch.log(
        torch.arange(1, D_STATE + 1, dtype=torch.float32).repeat(d_inner, 1)
    ).to(device).requires_grad_(True)
    B = mk(BATCH, D_STATE, seqlen).requires_grad_(True)
    C = mk(BATCH, D_STATE, seqlen).requires_grad_(True)
    D = mk(d_inner).requires_grad_(True)
    z = mk(BATCH, d_inner, seqlen).requires_grad_(True)
    dt_bias = mk(d_inner).requires_grad_(True)
    return u, delta, A_log, B, C, D, z, dt_bias


@pytest.mark.parametrize("name,seqlen,d_inner", LAYERS, ids=[l[0] for l in LAYERS])
def test_chunked_matches_reference(name, seqlen, d_inner):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = _inputs(seqlen, d_inner, device)

    outs, grads = [], []
    for scan in (selective_scan_ref, selective_scan_chunked):
        u, delta, A_log, B, C, D, z, dt_bias = [a.detach().clone().requires_grad_(True) for a in args]
        out = scan(u, delta, -torch.exp(A_log), B, C, D, z=z,
                   delta_bias=dt_bias, delta_softplus=True)
        # a non-uniform weighting so the backward pass cannot pass by symmetry
        (out * torch.linspace(0.1, 1.0, out.numel(), device=device).reshape(out.shape)).sum().backward()
        outs.append(out.detach())
        grads.append([a.grad.detach() for a in (u, delta, A_log, B, C, D, z, dt_bias)])

    # Relative to the output scale: over L=1024 accumulation steps fp32 rounding
    # alone puts absolute error near 1e-5, so an absolute bound would be measuring
    # float32 rather than the reassociation.
    fwd_err = (outs[0] - outs[1]).abs().max().item()
    fwd_scale = max(outs[0].abs().max().item(), 1.0)
    assert fwd_err / fwd_scale < 1e-5, (
        f"{name}: forward max|diff| = {fwd_err} (scale {fwd_scale})")

    for i, (g_ref, g_chunk) in enumerate(zip(*grads)):
        err = (g_ref - g_chunk).abs().max().item()
        scale = max(g_ref.abs().max().item(), 1.0)
        assert err / scale < 1e-4, f"{name}: grad[{i}] max|diff| = {err} (scale {scale})"


@pytest.mark.parametrize("chunk_size", [1, 7, 32, 1024])
def test_chunk_size_invariance(chunk_size):
    """Any chunk size must give the same answer, including ones that do not divide L."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    u, delta, A_log, B, C, D, z, dt_bias = _inputs(1024, 12, device, seed=1)
    A = -torch.exp(A_log)

    ref = selective_scan_ref(u, delta, A, B, C, D, z=z, delta_bias=dt_bias, delta_softplus=True)
    got = selective_scan_chunked(u, delta, A, B, C, D, z=z, delta_bias=dt_bias,
                                 delta_softplus=True, chunk_size=chunk_size)
    err = (ref - got).abs().max().item()
    scale = max(ref.abs().max().item(), 1.0)
    assert err / scale < 1e-5, f"chunk_size={chunk_size}: max|diff| = {err} (scale {scale})"


def test_mamba_module_shapes_and_param_count():
    """Per-layer parameter counts must match mamba_ssm's for the model total to hold."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name, seqlen, d_inner in LAYERS:
        d_model = d_inner // 2
        m = Mamba(d_model=d_model, d_state=D_STATE, d_conv=4, expand=2).to(device)

        assert m.dt_rank == math.ceil(d_model / 16) == 1, name
        assert m.d_inner == d_inner, name

        # in_proj + conv1d + x_proj + dt_proj + A_log + D + out_proj
        expected = (
            d_model * d_inner * 2                       # in_proj (bias=False)
            + d_inner * 4 + d_inner                      # conv1d weight + bias
            + d_inner * (m.dt_rank + 2 * D_STATE)        # x_proj (bias=False)
            + m.dt_rank * d_inner + d_inner              # dt_proj weight + bias
            + d_inner * D_STATE                          # A_log
            + d_inner                                    # D
            + d_inner * d_model                          # out_proj (bias=False)
        )
        assert sum(p.numel() for p in m.parameters()) == expected, name

        x = torch.randn(BATCH, seqlen, d_model, device=device)
        assert m(x).shape == (BATCH, seqlen, d_model), name
