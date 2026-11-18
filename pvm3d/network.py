"""PVMUNet3D: the PVM-3D model family, generalized from models/UltraLight_VM_UNet_3D.py.

What is the same as the legacy 3D model (and therefore as the 2D paper model)
-------------------------------------------------------------------------------
* The encoder/decoder topology: one conv (or one PVM layer) per stage,
  GroupNorm(4) -> max-pool -> GELU on the way down, block -> GroupNorm ->
  trilinear upsample -> GELU -> additive skip on the way up, logits produced at
  half resolution and upsampled trilinearly, SC_Att_Bridge on the skips.
* PVMLayer3D with ``scan='raster', n_branches=4, share_weights=True,
  mixer='mamba'`` performs exactly the arithmetic of the legacy ``PVMLayer``
  (verified in tests/test_pvm3d_network.py against
  models/UltraLight_VM_UNet_3D.PVMLayer with copied weights).
* The weight init (``_init_weights``), including its documented side effect of
  re-initialising Mamba's internal Linear/Conv1d layers.
* ``legacy_config()`` reproduces the 64,804-parameter Hippocampus model, and
  ``convert_legacy_state_dict`` loads its checkpoints.

What is new (all needed by the roadmap's experiments)
------------------------------------------------------
* Depth, per-stage strides (anisotropic, e.g. ACDC's [1,2,2]) and per-stage
  conv kernel sizes come from the nnU-Net plans instead of being hard-coded to
  4 isotropic x2 pools on a 64^3 patch.
* Configurable PVM placement (``n_pvm_enc`` deepest encoder stages,
  ``n_pvm_dec`` deepest decoder blocks).
* Deep-supervision outputs in nnU-Net's expected order and scales.
* PVM ablation axes: scan order (raster / direction-diverse / bidirectional /
  tri-orientation / random), number of branches, shared vs. independent
  weights, and the token mixer itself (Mamba / identity / depthwise conv).
* SC_Att_Bridge over any number of skips, with spatial-only / channel-only /
  none variants.

Index conventions
-----------------
Stage i (0..n-1) runs its block at cumulative stride prod(strides[0..i]).
Encoder stages 0..n-2 are then pooled by strides[i+1], so skip t_i lives at
cumulative stride R_i = prod(strides[0..i+1]) and t_{n-2} is at bottleneck
resolution. Decoder block j (n-2 down to 0) maps widths[j+1] -> widths[j].
"""

import math
from typing import List, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

try:  # timm >= 0.9
    from timm.layers import trunc_normal_
except ImportError:  # pragma: no cover
    from timm.models.layers import trunc_normal_


SCAN_MODES = ('raster', 'dd', 'cross', 'bidir', 'tri', 'random')
MIXERS = ('mamba', 'identity', 'conv')
BRIDGE_MODES = ('both', 'spatial', 'channel', 'none')

# UltraLight's channel progression (2D: 8,16,24,32,48,64; legacy 3D: the first 5)
# expressed as multipliers of a base width, extended by repetition for deeper plans.
WIDTH_MULTIPLIERS = (1, 2, 3, 4, 6, 8)


def pvm_widths(base: int, n_stages: int) -> List[int]:
    mult = list(WIDTH_MULTIPLIERS) + [WIDTH_MULTIPLIERS[-1]] * max(0, n_stages - len(WIDTH_MULTIPLIERS))
    return [base * m for m in mult[:n_stages]]


def _build_mamba(d_model, d_state, d_conv, expand, impl):
    if impl == 'mamba_ssm':
        from mamba_ssm import Mamba
    elif impl == 'pytorch':
        from models.mamba_pytorch import Mamba
    else:
        raise ValueError(f'unknown mamba impl {impl!r}')
    return Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


class _ConvMixer(nn.Module):
    """Local-mixing control for ablation A1: inverted-bottleneck depthwise Conv3d
    over the token grid, standing in for Mamba's global sequence model."""
    def __init__(self, dim, expand=2, kernel_size=3):
        super().__init__()
        hid = dim * expand
        self.pw1 = nn.Conv3d(dim, hid, 1)
        self.dw = nn.Conv3d(hid, hid, kernel_size, padding=kernel_size // 2, groups=hid)
        self.pw2 = nn.Conv3d(hid, dim, 1)

    def forward(self, tokens, spatial):
        B, L, c = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, c, *spatial)
        x = self.pw2(F.gelu(self.dw(F.gelu(self.pw1(x)))))
        return x.reshape(B, c, L).transpose(1, 2)


class PVMLayer3D(nn.Module):
    """Parallel Vision Mamba layer on a 3D token grid.

    Defaults reproduce the legacy PVMLayer exactly. Ablation knobs:
      scan        raster | dd (direction-diverse: branch k scans orientation k%4,
                  zero extra cost) | cross (the earlier pvm3D study's scan: raster,
                  reversed raster, axes-reversed, axes-reversed+reversed -- 2 axis
                  orders x 2 directions; VMamba's 2D cross-scan carried to 3D)
                  | bidir (fwd+reverse per branch, 2x scan cost)
                  | tri (3 axis orientations per branch, 3x) | random (fixed
                  random token permutation per branch -- control for whether a
                  spatially coherent scan matters)
      n_branches  channel split count (legacy 4)
      share_weights  one Mamba for all branches (legacy) vs. one per branch
      mixer       mamba | identity (Mamba removed, LN+proj kept) | conv
      inner_skip  keep the ``skip_scale * x`` bypass around the mixer (legacy).
                  False forces every branch's information through the mixer --
                  the test for whether Mamba can carry the signal at all.
    """
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2,
                 scan='raster', n_branches=4, share_weights=True, mixer='mamba',
                 mamba_impl='mamba_ssm', random_seed=0, inner_skip=True):
        super().__init__()
        assert scan in SCAN_MODES, scan
        assert mixer in MIXERS, mixer
        assert input_dim % n_branches == 0, f'input_dim {input_dim} not divisible by n_branches {n_branches}'
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scan = scan
        self.n_branches = n_branches
        self.share_weights = share_weights
        self.mixer = mixer
        self.random_seed = random_seed
        d = input_dim // n_branches

        self.norm = nn.LayerNorm(input_dim)
        n_mods = 1 if share_weights else n_branches
        if mixer == 'mamba':
            self.mamba = nn.ModuleList([_build_mamba(d, d_state, d_conv, expand, mamba_impl) for _ in range(n_mods)])
        elif mixer == 'conv':
            self.convmix = nn.ModuleList([_ConvMixer(d) for _ in range(n_mods)])
        self.proj = nn.Linear(input_dim, output_dim)
        self.inner_skip = inner_skip
        self.skip_scale = nn.Parameter(torch.ones(1)) if inner_skip else None
        self._perm_cache = {}

    # --- scan orders -------------------------------------------------------
    @staticmethod
    def _orient_perm(spatial, axes, device):
        idx = torch.arange(spatial.numel(), device=device).reshape(*spatial)
        return idx.permute(*axes).reshape(-1)

    def _branch_perms(self, spatial, device):
        """For each branch, a list of (perm, inv) pairs -- one per sequence that
        branch scans. ``None`` means identity (raster) and skips the gather."""
        key = (tuple(spatial), str(device))
        if key in self._perm_cache:
            return self._perm_cache[key]

        def pair(p):
            return (p, torch.argsort(p))

        L = spatial.numel()
        rev = torch.arange(L - 1, -1, -1, device=device)
        o1 = self._orient_perm(spatial, (2, 0, 1), device)  # H fastest
        o2 = self._orient_perm(spatial, (1, 2, 0), device)  # D fastest
        out = []
        for k in range(self.n_branches):
            if self.scan == 'raster':
                out.append([None])
            elif self.scan == 'dd':
                choice = k % 4
                out.append([None] if choice == 0 else [pair(o1)] if choice == 1
                           else [pair(o2)] if choice == 2 else [pair(rev)])
            elif self.scan == 'cross':
                ax = self._orient_perm(spatial, (2, 1, 0), device)  # all axes reversed: D fastest, W slowest
                choice = k % 4
                out.append([None] if choice == 0 else [pair(rev)] if choice == 1
                           else [pair(ax)] if choice == 2 else [pair(ax.flip(0))])
            elif self.scan == 'bidir':
                out.append([None, pair(rev)])
            elif self.scan == 'tri':
                out.append([None, pair(o1), pair(o2)])
            elif self.scan == 'random':
                g = torch.Generator().manual_seed(self.random_seed * 1000 + k)
                out.append([pair(torch.randperm(L, generator=g).to(device))])
        self._perm_cache[key] = out
        return out

    # --- token mixing ------------------------------------------------------
    def _mix(self, xs, spatial):
        if self.mixer == 'identity':
            return list(xs)
        if self.mixer == 'conv':
            return [self.convmix[0 if self.share_weights else k](xk, spatial) for k, xk in enumerate(xs)]

        perms = self._branch_perms(spatial, xs[0].device)
        seqs, owner = [], []
        for k, xk in enumerate(xs):
            for p in perms[k]:
                seqs.append(xk if p is None else xk[:, p[0], :])
                owner.append(k)

        if self.share_weights:
            # batched over all sequences of all branches: one Mamba call, as the
            # legacy layer does for its 4 branches (models/UltraLight_VM_UNet.py:66)
            ys = list(torch.chunk(self.mamba[0](torch.cat(seqs, dim=0)), len(seqs), dim=0))
        else:
            ys = [None] * len(seqs)
            for k in range(self.n_branches):
                ids = [i for i, o in enumerate(owner) if o == k]
                yk = torch.chunk(self.mamba[k](torch.cat([seqs[i] for i in ids], dim=0)), len(ids), dim=0)
                for i, y in zip(ids, yk):
                    ys[i] = y

        out, i = [], 0
        for k in range(self.n_branches):
            acc = None
            for p in perms[k]:
                y = ys[i] if p is None else ys[i][:, p[1], :]
                acc = y if acc is None else acc + y
                i += 1
            out.append(acc if len(perms[k]) == 1 else acc / len(perms[k]))
        return out

    def forward(self, x):
        if x.dtype == torch.float16:  # legacy behaviour, kept verbatim
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        spatial = x.shape[2:]
        x_flat = x.reshape(B, C, spatial.numel()).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        xs = torch.chunk(x_norm, self.n_branches, dim=2)
        ys = self._mix(xs, spatial)
        if self.inner_skip:
            x_m = torch.cat([y + self.skip_scale * xk for y, xk in zip(ys, xs)], dim=2)
        else:
            x_m = torch.cat(ys, dim=2)
        x_m = self.proj(self.norm(x_m))
        return x_m.transpose(-1, -2).reshape(B, self.output_dim, *spatial)


class ChannelAttBridge3D(nn.Module):
    def __init__(self, chs: Sequence[int]):
        super().__init__()
        s = sum(chs)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.atts = nn.ModuleList([nn.Linear(s, c) for c in chs])

    def forward(self, ts):
        att = torch.cat([self.avgpool(t) for t in ts], dim=1)
        att = self.get_all_att(att.flatten(1).unsqueeze(1))
        return [torch.sigmoid(lin(att)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t)
                for t, lin in zip(ts, self.atts)]


class SpatialAttBridge3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv3d = nn.Sequential(nn.Conv3d(2, 1, 7, stride=1, padding=9, dilation=3), nn.Sigmoid())

    def forward(self, ts):
        return [self.shared_conv3d(torch.cat([t.mean(1, keepdim=True), t.max(1, keepdim=True)[0]], dim=1))
                for t in ts]


class SCAttBridge3D(nn.Module):
    def __init__(self, chs, mode='both'):
        super().__init__()
        assert mode in BRIDGE_MODES and mode != 'none'
        self.mode = mode
        if mode in ('both', 'channel'):
            self.catt = ChannelAttBridge3D(chs)
        if mode in ('both', 'spatial'):
            self.satt = SpatialAttBridge3D()

    def forward(self, ts):
        if self.mode == 'spatial':
            return [s * t + t for s, t in zip(self.satt(ts), ts)]
        if self.mode == 'channel':
            return [c * t + t for c, t in zip(self.catt(ts), ts)]
        # legacy SC_Att_Bridge3D arithmetic
        ss = [s * t for s, t in zip(self.satt(ts), ts)]
        ts2 = [a + r for a, r in zip(ss, ts)]
        return [c * t + s for c, t, s in zip(self.catt(ts2), ts2, ss)]


def _as3(v):
    return list(v) if isinstance(v, (list, tuple)) else [v] * 3


class PVMUNet3D(nn.Module):
    def __init__(self, input_channels: int, num_classes: int, widths: Sequence[int],
                 strides: Sequence, kernel_sizes: Optional[Sequence] = None,
                 n_pvm_enc: int = 3, n_pvm_dec: int = 3, bridge: str = 'both',
                 deep_supervision: bool = False, scan: str = 'raster', n_branches: int = 4,
                 share_weights: bool = True, mixer: str = 'mamba', d_state: int = 16,
                 expand: int = 2, mamba_impl: str = 'mamba_ssm', mamba_init: str = 'ultralight',
                 inner_skip: bool = True):
        super().__init__()
        n = len(widths)
        assert len(strides) == n, f'{len(strides)} strides for {n} stages'
        strides = [_as3(s) for s in strides]
        assert strides[0] == [1, 1, 1], f'first stage must be unstrided (nnU-Net convention), got {strides[0]}'
        kernel_sizes = [_as3(3)] * n if kernel_sizes is None else [_as3(k) for k in kernel_sizes]
        assert 0 <= n_pvm_enc <= n - 1, 'stage 0 (stem) is always a conv'
        assert 0 <= n_pvm_dec <= n - 1
        assert all(w % 4 == 0 for w in widths), 'GroupNorm(4) needs widths divisible by 4'
        assert bridge in BRIDGE_MODES

        self.n_stages = n
        self.widths = list(widths)
        self.strides = strides
        self.deep_supervision = deep_supervision
        self.config = dict(widths=list(widths), strides=strides, kernel_sizes=kernel_sizes,
                           n_pvm_enc=n_pvm_enc, n_pvm_dec=n_pvm_dec, bridge=bridge, scan=scan,
                           n_branches=n_branches, share_weights=share_weights, mixer=mixer,
                           d_state=d_state, expand=expand, mamba_init=mamba_init, inner_skip=inner_skip)
        assert mamba_init in ('ultralight', 'native'), mamba_init
        pvm_kw = dict(d_state=d_state, expand=expand, scan=scan, n_branches=n_branches,
                      share_weights=share_weights, mixer=mixer, mamba_impl=mamba_impl, inner_skip=inner_skip)

        def conv(cin, cout, k):
            return nn.Conv3d(cin, cout, k, padding=[kk // 2 for kk in k])

        enc = []
        for i in range(n):
            cin = input_channels if i == 0 else widths[i - 1]
            if i >= n - n_pvm_enc:
                enc.append(PVMLayer3D(cin, widths[i], **pvm_kw))
            else:
                enc.append(conv(cin, widths[i], kernel_sizes[i]))
        self.encoder = nn.ModuleList(enc)
        self.enc_norms = nn.ModuleList([nn.GroupNorm(4, widths[i]) for i in range(n - 1)])

        self.bridge = SCAttBridge3D(widths[:n - 1], bridge) if bridge != 'none' else None

        dec, dec_norms = [], []
        for order, j in enumerate(range(n - 2, -1, -1)):  # deepest decoder block first
            if order < n_pvm_dec:
                dec.append(PVMLayer3D(widths[j + 1], widths[j], **pvm_kw))
            else:
                dec.append(conv(widths[j + 1], widths[j], kernel_sizes[min(j + 2, n - 1)]))
            dec_norms.append(nn.GroupNorm(4, widths[j]))
        self.decoder = nn.ModuleList(dec)
        self.dec_norms = nn.ModuleList(dec_norms)

        self.final = nn.Conv3d(widths[0], num_classes, kernel_size=1)
        # extra deep-supervision heads at R_1..R_{n-3}; R_0 reuses self.final
        self.ds_heads = nn.ModuleList([nn.Conv3d(widths[j], num_classes, 1) for j in range(1, n - 2)])

        if mamba_init == 'ultralight':
            # legacy: re-initialises Mamba's own Linear/Conv1d too, which leaves its
            # output ~1e-3 of the skip path at init (vs ~3e-2 with Mamba's own init)
            self.apply(self._init_weights)
        else:  # 'native': everything except the Mamba internals, which keep mamba_ssm's init
            inside = [n for n, m in self.named_modules() if type(m).__name__ == 'Mamba']
            for n, m in self.named_modules():
                if not any(n == q or n.startswith(q + '.') for q in inside):
                    self._init_weights(m)

    def _init_weights(self, m):  # identical to models/UltraLight_VM_UNet_3D.py
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv3d):
            fan_out = 1
            for k in m.kernel_size:
                fan_out *= k
            fan_out *= m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        n = self.n_stages
        in_size = x.shape[2:]
        skips = []
        out = x
        for i in range(n - 1):
            out = F.gelu(F.max_pool3d(self.enc_norms[i](self.encoder[i](out)),
                                      kernel_size=self.strides[i + 1], stride=self.strides[i + 1]))
            skips.append(out)
        if self.bridge is not None:
            skips = self.bridge(skips)
        out = F.gelu(self.encoder[n - 1](out))  # bottleneck, unpooled, no norm (legacy)

        ds_feats = {}
        for order, j in enumerate(range(n - 2, -1, -1)):
            y = self.dec_norms[order](self.decoder[order](out))
            if order > 0:
                y = F.interpolate(y, size=skips[j].shape[2:], mode='trilinear', align_corners=True)
            out = F.gelu(y) + skips[j]
            ds_feats[j] = out

        low = self.final(out)  # at R_0
        logits = F.interpolate(low, size=in_size, mode='trilinear', align_corners=True)
        if not self.deep_supervision:
            return logits
        return [logits, low] + [head(ds_feats[j]) for j, head in zip(range(1, n - 2), self.ds_heads)]


# --------------------------------------------------------------------------
# legacy (models/UltraLight_VM_UNet_3D.py) compatibility
# --------------------------------------------------------------------------
def legacy_config():
    """Kwargs that rebuild the 64,804-param Hippocampus PVM-3D (64^3 input)."""
    return dict(widths=[8, 16, 24, 32, 48], strides=[[1, 1, 1]] + [[2, 2, 2]] * 4,
                kernel_sizes=None, n_pvm_enc=2, n_pvm_dec=1, bridge='both')


def convert_legacy_state_dict(sd: dict, n_stages: int = 5) -> dict:
    """Map a models/UltraLight_VM_UNet_3D state_dict onto PVMUNet3D keys."""
    out = {}
    for k, v in sd.items():
        parts = k.split('.')
        head = parts[0]
        if head.startswith('encoder') and head[7:].isdigit():
            i = int(head[7:]) - 1
            rest = parts[2:]  # drop the nn.Sequential index
            if rest and rest[0] == 'mamba':
                rest = ['mamba', '0'] + rest[1:]
            out['.'.join(['encoder', str(i)] + rest)] = v
        elif head.startswith('decoder') and head[7:].isdigit():
            o = int(head[7:]) - 1
            rest = parts[2:]
            if rest and rest[0] == 'mamba':
                rest = ['mamba', '0'] + rest[1:]
            out['.'.join(['decoder', str(o)] + rest)] = v
        elif head.startswith('ebn'):
            out['.'.join(['enc_norms', str(int(head[3:]) - 1)] + parts[1:])] = v
        elif head.startswith('dbn'):
            out['.'.join(['dec_norms', str(int(head[3:]) - 1)] + parts[1:])] = v
        elif head == 'scab':
            if parts[1] == 'catt' and parts[2].startswith('att') and parts[2][3:].isdigit():
                out['.'.join(['bridge', 'catt', 'atts', str(int(parts[2][3:]) - 1)] + parts[3:])] = v
            else:
                out['.'.join(['bridge'] + parts[1:])] = v
        else:
            out[k] = v
    return out
