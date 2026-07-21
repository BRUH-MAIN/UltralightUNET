"""
Pure-PyTorch replacement for ``mamba_ssm.modules.mamba_simple.Mamba`` (v1.0.1).

Why this exists
---------------
The reference implementation of UltraLight VM-UNet depends on ``mamba_ssm==1.0.1``
and ``causal_conv1d==1.0.0``. Both are CUDA extensions with Linux-only official
support, and ``mamba_ssm/ops/selective_scan_interface.py`` performs a top-level
``import selective_scan_cuda`` -- so on Windows the package cannot even be
imported, let alone run.

This module reimplements the same computation in plain PyTorch. It is not an
approximation: ``selective_scan_ref`` below is a transcription of the reference
scan that ships with the official Mamba repository (the same function its CUDA
kernel is tested against), and ``selective_scan_chunked`` is that identical
recurrence with the sequential product reassociated over chunks.

The fused kernel exists to make ``d_inner`` in the thousands tractable. In
UltraLight VM-UNet the six PVM layers run at d_inner = 12..32 over sequences of
at most 1024 tokens, so there is very little for it to accelerate.

Fidelity requirement
--------------------
``UltraLight_VM_UNet.__init__`` finishes with ``self.apply(self._init_weights)``,
which reinitialises *every* ``nn.Linear`` (trunc_normal_ std=0.02, zero bias) and
every ``nn.Conv1d`` inside the model -- including the ones in here, and including
``dt_proj.bias``, whose ``_no_reinit`` marker that function does not honour. Only
``A_log`` and ``D`` survive, because they are bare ``nn.Parameter``s.

That means the submodule *names and types* below are load-bearing: they must stay
``in_proj``/``x_proj``/``dt_proj``/``out_proj`` as ``nn.Linear`` and ``conv1d`` as
``nn.Conv1d`` for ``apply()`` to touch exactly what it touches upstream. Changing
one to, say, a fused parameter would silently alter the initialisation and hence
the results.
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

__all__ = ["Mamba", "selective_scan_ref", "selective_scan_chunked"]


def _scan_inputs(u, delta, A, B, C, delta_bias, delta_softplus):
    """Shared preamble: apply the dt bias/softplus and form deltaA / deltaB_u.

    u, delta : (b, d, l)
    A        : (d, n)
    B, C     : (b, n, l)
    returns deltaA, deltaB_u : (b, d, l, n)
    """
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
    deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B.float(), u)
    return deltaA, deltaB_u


def _finalize(y, u, D, z, dtype_in):
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    return out.to(dtype=dtype_in)


def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                       delta_softplus=False):
    """Reference selective scan -- one Python step per timestep.

    Transcribed from the official ``selective_scan_ref``, narrowed to the case
    this model actually uses: real ``A``, variable ``B``/``C`` of shape
    ``(b, n, l)``, and no returned last state.

    Kept as the correctness oracle for ``selective_scan_chunked``. Correct but
    slow -- roughly 6.9k Python iterations per forward pass of the full network.

    u, delta : (b, d, l)   A : (d, n)   B, C : (b, n, l)   D : (d,)   z : (b, d, l)
    out      : (b, d, l)
    """
    dtype_in = u.dtype
    u = u.float()
    deltaA, deltaB_u = _scan_inputs(u, delta, A, B, C, delta_bias, delta_softplus)
    C = C.float()

    x = A.new_zeros((u.shape[0], A.shape[0], A.shape[1]))
    ys = []
    # unbind rather than indexing per step: a[:, :, i] would register a separate
    # slice_backward for every timestep, each allocating a full-size zero tensor to
    # scatter into. unbind's backward is one stack for the whole loop.
    dA, dBu, Cs = deltaA.unbind(dim=2), deltaB_u.unbind(dim=2), C.unbind(dim=2)
    for i in range(u.shape[2]):
        x = dA[i] * x + dBu[i]
        ys.append(torch.einsum("bdn,bn->bd", x, Cs[i]))
    y = torch.stack(ys, dim=2)

    return _finalize(y, u, D, z, dtype_in)


def selective_scan_chunked(u, delta, A, B, C, D=None, z=None, delta_bias=None,
                           delta_softplus=False, chunk_size=None):
    """Same recurrence as :func:`selective_scan_ref`, reassociated over chunks.

    The state update ``x_i = a_i * x_{i-1} + b_i`` is elementwise, so a chunk of
    K steps can be summarised by two quantities computed with the carry-in set to
    zero: the running product ``p_k = prod_{j<=k} a_j`` and the local state
    ``s_k``. The true state is then ``x_k = p_k * H + s_k`` where ``H`` is the
    carry into that chunk. So we sweep K steps in parallel across all M chunks,
    then sweep M steps to thread the carries -- K + M Python steps instead of L.

    With K = round(sqrt(L)) that turns the worst layer here (L=1024) from 1024
    steps into 64. Each step is only a handful of tiny CUDA launches, so launch
    count is essentially the entire cost at this size.

    Underflow of ``p`` is benign rather than a stability problem: every ``a_i``
    is ``exp(delta * A)`` with ``A < 0``, hence in (0, 1), so a vanishing product
    means the state genuinely decayed away and 0 is the right answer. (This is
    the reason for preferring a chunked sweep over the log-space
    divide-by-cumprod formulation, which blows up in exactly that regime.)
    """
    dtype_in = u.dtype
    u = u.float()
    deltaA, deltaB_u = _scan_inputs(u, delta, A, B, C, delta_bias, delta_softplus)
    C = C.float()

    b, d, L, n = deltaA.shape
    if chunk_size is None:
        chunk_size = max(1, int(round(math.sqrt(L))))
    K = min(chunk_size, L)
    M = math.ceil(L / K)
    pad = M * K - L

    if pad:
        # a=1, b=0 -> padded steps carry the state through untouched; discarded below.
        deltaA = F.pad(deltaA.transpose(2, 3), (0, pad), value=1.0).transpose(2, 3)
        deltaB_u = F.pad(deltaB_u.transpose(2, 3), (0, pad), value=0.0).transpose(2, 3)

    a = deltaA.reshape(b, d, M, K, n)
    bu = deltaB_u.reshape(b, d, M, K, n)

    # Pass 1: K sequential steps, vectorised across the M chunks.
    #
    # unbind up front instead of indexing a[:, :, :, k] inside the loop. Each such
    # index registers its own slice_backward, which allocates a full (b,d,M,K,n)
    # zero tensor and scatters one slice into it -- per step, per layer, per
    # iteration. It dominated the training step before this change. unbind returns
    # the same views but backs off to a single stack for the whole loop.
    a_steps, bu_steps = a.unbind(dim=3), bu.unbind(dim=3)

    p_k = torch.ones(b, d, M, n, dtype=a.dtype, device=a.device)
    s_k = torch.zeros(b, d, M, n, dtype=a.dtype, device=a.device)
    prods, states = [], []
    for k in range(K):
        a_k = a_steps[k]
        p_k = p_k * a_k
        s_k = a_k * s_k + bu_steps[k]
        prods.append(p_k)
        states.append(s_k)
    p = torch.stack(prods, dim=3)
    s = torch.stack(states, dim=3)

    # Pass 2: M sequential steps threading the carry between chunks.
    # prods[-1] / states[-1] are already p[:, :, :, K-1] and s[:, :, :, K-1], so the
    # chunk summaries need no slicing at all.
    p_last, s_last = prods[-1].unbind(dim=2), states[-1].unbind(dim=2)
    H = torch.zeros(b, d, n, dtype=a.dtype, device=a.device)
    carries = []
    for m in range(M):
        carries.append(H)                       # carry *into* chunk m
        H = p_last[m] * H + s_last[m]           # carry out of chunk m
    carry = torch.stack(carries, dim=2)

    x = p * carry.unsqueeze(3) + s
    x = x.reshape(b, d, M * K, n)[:, :, :L]

    y = torch.einsum("bdln,bnl->bdl", x, C)
    return _finalize(y, u, D, z, dtype_in)


class Mamba(nn.Module):
    """Drop-in for ``mamba_ssm.modules.mamba_simple.Mamba`` (v1.0.1).

    Follows the upstream *slow path* (the ``use_fast_path=False`` branch), which
    is the branch that does not require the fused kernel. Inference caching
    (``inference_params``) is not implemented -- UltraLight VM-UNet never uses it.
    """

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # accepted for signature parity; ignored
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse of softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Upstream marks this so a caller's own init skips it. UltraLight VM-UNet's
        # _init_weights does not check the flag and zeroes the bias anyway; the
        # marker is kept only so this class stays interchangeable with upstream.
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        self.A_log = nn.Parameter(torch.log(A))  # Keep A_log in fp32
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None):
        """hidden_states: (B, L, d_model) -> (B, L, d_model)"""
        if inference_params is not None:
            raise NotImplementedError("inference_params caching is not supported")

        _, seqlen, _ = hidden_states.shape

        # matmul and BLH -> HBL transpose at once, as upstream does
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        x, z = xz.chunk(2, dim=1)

        # Short causal depthwise conv; padding is d_conv-1 on both sides, so trim
        # the right overhang to recover causality.
        x = self.act(self.conv1d(x)[..., :seqlen])

        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        y = selective_scan_chunked(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )

        y = rearrange(y, "b d l -> b l d")
        return self.out_proj(y)
