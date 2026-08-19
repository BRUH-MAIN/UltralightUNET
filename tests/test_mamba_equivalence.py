"""mamba_ssm's fused CUDA kernel must compute what the reference scan computes.

The model runs ``mamba_ssm``. Nothing about a fused kernel is inspectable from the
outside, so this file pins it against ``models/mamba_pytorch.py`` -- an
independent transcription of the reference scan from the official Mamba
repository -- at the six shapes UltraLight VM-UNet actually uses:

  1. the oracle checks itself: ``selective_scan_chunked``, which reassociates the
     recurrence over chunks, against the plain per-timestep ``selective_scan_ref``
  2. the kernel against the oracle, at the level of ``selective_scan_fn``
  3. the kernel against the oracle, at the level of the whole ``Mamba`` module and
     then the whole network -- initialisation, forward, and every gradient

(2) and (3) need a GPU; (1) and the initialisation checks do not.

Tolerances are relative to the scale of the quantity being compared: over L=1024
accumulation steps fp32 rounding alone puts absolute error near 1e-5, so an
absolute bound would be measuring float32 rather than the thing under test.
"""

import math

import pytest
import torch

import models.UltraLight_VM_UNet as model_module
from models.mamba_pytorch import Mamba as MambaRef
from models.mamba_pytorch import selective_scan_chunked, selective_scan_ref

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

# mamba_ssm dispatches straight into a CUDA extension: there is no CPU path to
# fall back to, so everything that touches it is skipped off-GPU.
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="mamba_ssm is CUDA-only")


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


def _assert_close(name, ref, got, tol):
    err = (ref - got).abs().max().item()
    scale = max(ref.abs().max().item(), 1.0)
    assert err / scale < tol, f"{name}: max|diff| = {err} (scale {scale}, tol {tol})"


# --------------------------------------------------------------------------
# 1. the oracle against itself
# --------------------------------------------------------------------------

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

    _assert_close(f"{name} forward", outs[0], outs[1], 1e-5)
    for i, (g_ref, g_chunk) in enumerate(zip(*grads)):
        _assert_close(f"{name} grad[{i}]", g_ref, g_chunk, 1e-4)


@pytest.mark.parametrize("chunk_size", [1, 7, 32, 1024])
def test_chunk_size_invariance(chunk_size):
    """Any chunk size must give the same answer, including ones that do not divide L."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    u, delta, A_log, B, C, D, z, dt_bias = _inputs(1024, 12, device, seed=1)
    A = -torch.exp(A_log)

    ref = selective_scan_ref(u, delta, A, B, C, D, z=z, delta_bias=dt_bias, delta_softplus=True)
    got = selective_scan_chunked(u, delta, A, B, C, D, z=z, delta_bias=dt_bias,
                                 delta_softplus=True, chunk_size=chunk_size)
    _assert_close(f"chunk_size={chunk_size}", ref, got, 1e-5)


# --------------------------------------------------------------------------
# 2. the CUDA kernel against the oracle, scan by scan
# --------------------------------------------------------------------------

@cuda_only
@pytest.mark.parametrize("name,seqlen,d_inner", LAYERS, ids=[l[0] for l in LAYERS])
def test_selective_scan_cuda_matches_reference(name, seqlen, d_inner):
    """selective_scan_fn (the fused kernel) vs selective_scan_ref, fwd and bwd."""
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    args = _inputs(seqlen, d_inner, "cuda")

    outs, grads = [], []
    for scan in (selective_scan_ref, selective_scan_fn):
        u, delta, A_log, B, C, D, z, dt_bias = [a.detach().clone().requires_grad_(True) for a in args]
        out = scan(u, delta, -torch.exp(A_log), B, C, D, z=z,
                   delta_bias=dt_bias, delta_softplus=True)
        (out * torch.linspace(0.1, 1.0, out.numel(), device="cuda").reshape(out.shape)).sum().backward()
        outs.append(out.detach())
        grads.append([a.grad.detach() for a in (u, delta, A_log, B, C, D, z, dt_bias)])

    _assert_close(f"{name} forward", outs[0], outs[1], 1e-5)
    for i, (g_ref, g_cuda) in enumerate(zip(*grads)):
        _assert_close(f"{name} grad[{i}]", g_ref, g_cuda, 1e-4)


# --------------------------------------------------------------------------
# 3. the module and the whole network
# --------------------------------------------------------------------------

def _both_mambas(d_model, seed=0):
    """One mamba_ssm.Mamba and one MambaRef, built from the same RNG state."""
    from mamba_ssm import Mamba

    built = []
    for cls in (Mamba, MambaRef):
        torch.manual_seed(seed)
        built.append(cls(d_model=d_model, d_state=D_STATE, d_conv=4, expand=2))
    return built


@pytest.mark.parametrize("name,seqlen,d_inner", LAYERS, ids=[l[0] for l in LAYERS])
def test_module_init_matches_reference(name, seqlen, d_inner):
    """Same seed, same weights -- bit for bit.

    This is what makes the two interchangeable at all. Both classes draw from the
    global RNG in the same order (four nn.Linear/nn.Conv1d inits, then dt_proj's
    uniform_, then the dt/inv_dt exponential draw), so a single differing draw
    anywhere would show up here as a mismatched tensor.
    """
    pytest.importorskip("mamba_ssm")
    ssm, ref = _both_mambas(d_model=d_inner // 2)

    assert ssm.state_dict().keys() == ref.state_dict().keys(), name
    for k, v in ssm.state_dict().items():
        assert torch.equal(v, ref.state_dict()[k]), f"{name}: {k} differs"


@pytest.mark.parametrize("name,seqlen,d_inner", LAYERS, ids=[l[0] for l in LAYERS])
def test_module_param_count(name, seqlen, d_inner):
    """Per-layer parameter counts, from first principles rather than by comparison."""
    pytest.importorskip("mamba_ssm")
    from mamba_ssm import Mamba

    d_model = d_inner // 2
    m = Mamba(d_model=d_model, d_state=D_STATE, d_conv=4, expand=2)

    assert m.dt_rank == math.ceil(d_model / 16) == 1, name
    assert m.d_inner == d_inner, name

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


@cuda_only
@pytest.mark.parametrize("name,seqlen,d_inner", LAYERS, ids=[l[0] for l in LAYERS])
def test_module_forward_and_grads_match(name, seqlen, d_inner):
    """The full Mamba block: fused kernel vs oracle, on identical weights."""
    d_model = d_inner // 2
    ssm, ref = _both_mambas(d_model)
    ssm, ref = ssm.cuda(), ref.cuda()

    torch.manual_seed(1)
    x = torch.randn(BATCH, seqlen, d_model, device="cuda")

    outs, grads = [], []
    for m in (ref, ssm):  # oracle first, so it is the reference in _assert_close
        m.zero_grad()
        out = m(x.clone())
        (out * torch.linspace(0.1, 1.0, out.numel(), device="cuda").reshape(out.shape)).sum().backward()
        outs.append(out.detach())
        grads.append({n: p.grad.detach().clone() for n, p in m.named_parameters()})

    _assert_close(f"{name} forward", outs[0], outs[1], 1e-5)
    for n in grads[0]:
        _assert_close(f"{name} grad {n}", grads[0][n], grads[1][n], 1e-4)


def _build_model(mamba_cls, seed=42):
    """UltraLight_VM_UNet with PVMLayer's Mamba swapped for mamba_cls.

    PVMLayer looks the class up as a module global at construction time, so this
    patch is enough to build the identical network on either backend.
    """
    original = model_module.Mamba
    model_module.Mamba = mamba_cls
    try:
        torch.manual_seed(seed)
        return model_module.UltraLight_VM_UNet()
    finally:
        model_module.Mamba = original


def test_model_init_identical_across_backends():
    """Switching backend must not perturb a single initial weight.

    UltraLight_VM_UNet.__init__ ends with self.apply(self._init_weights), which
    reinitialises every nn.Linear and nn.Conv1d in the model -- including the ones
    inside Mamba, and including dt_proj.bias, whose _no_reinit marker it does not
    check. That only lands on the same tensors if the vendored module keeps the
    same submodule names and types, so this asserts the whole 49,457-parameter
    state_dict, not just the Mamba blocks.
    """
    pytest.importorskip("mamba_ssm")
    from mamba_ssm import Mamba

    ssm = _build_model(Mamba).state_dict()
    ref = _build_model(MambaRef).state_dict()

    assert ssm.keys() == ref.keys()
    assert sum(v.numel() for v in ssm.values()) == 49457
    for k, v in ssm.items():
        assert torch.equal(v, ref[k]), f"{k} differs"


@cuda_only
def test_model_forward_matches_reference_backend():
    """End to end: the network's output must not depend on which scan ran.

    Looser than the per-layer bounds above, for the same reason as in
    tests/test_pvm_batching.py: fp32 rounding accumulates through six PVM layers,
    the bridge and five interpolations. Measured, the two backends land within
    2e-7 of each other on average and 2e-4 at the worst of 131k output pixels.

    The assertion that carries the weight is the second one. The output is a
    probability thresholded at 0.5 to make a prediction, so what matters is
    whether any pixel changes class -- and none does, which makes the two
    backends interchangeable in the only sense the metrics can see.
    """
    from mamba_ssm import Mamba

    ssm = _build_model(Mamba).cuda().eval()
    ref = _build_model(MambaRef).cuda().eval()

    torch.manual_seed(2)
    x = torch.randn(2, 3, 256, 256, device="cuda")
    with torch.no_grad():
        got, want = ssm(x), ref(x)

    _assert_close("model forward", want, got, 1e-3)
    flipped = ((got >= 0.5) != (want >= 0.5)).sum().item()
    assert flipped == 0, f"{flipped} pixels changed segmentation class"
