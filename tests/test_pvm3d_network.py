"""PVMUNet3D (pvm3d/network.py) and its nnU-Net trainers.

The load-bearing checks are the legacy ones: the generalized network must
reproduce models/UltraLight_VM_UNet_3D.py exactly (64,804 params, identical
outputs from identical weights), otherwise "PVM-3D in nnU-Net" would be a
different model from the one the preliminary Hippocampus result describes.

GPU tests use the mamba_ssm kernel; CPU tests use the models/mamba_pytorch.py
oracle (equivalent per tests/test_mamba_equivalence.py).
"""

import importlib
import math
import os
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from pvm3d.network import (PVMLayer3D, PVMUNet3D, SCAN_MODES, convert_legacy_state_dict,  # noqa: E402
                           legacy_config, pvm_widths)
from pvm3d.nnunet_glue import (_ClosedFormCosineLR, build_pvm_from_plans,  # noqa: E402
                               build_tiny_unet_matched, count_inference_params)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device for mamba_ssm')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
IMPL = 'mamba_ssm' if torch.cuda.is_available() else 'pytorch'


def fake_cm(strides, kernel_sizes=None, features=None):
    """Minimal stand-in for nnU-Net's ConfigurationManager (3d, PlainConvUNet)."""
    n = len(strides)
    features = features or [min(32 * 2 ** i, 320) for i in range(n)]
    ak = dict(n_stages=n, features_per_stage=features, conv_op='torch.nn.modules.conv.Conv3d',
              kernel_sizes=kernel_sizes or [[3, 3, 3]] * n, strides=strides,
              n_conv_per_stage=[2] * n, n_conv_per_stage_decoder=[2] * (n - 1), conv_bias=True,
              norm_op='torch.nn.modules.instancenorm.InstanceNorm3d',
              norm_op_kwargs={'eps': 1e-5, 'affine': True}, dropout_op=None, dropout_op_kwargs=None,
              nonlin='torch.nn.LeakyReLU', nonlin_kwargs={'inplace': True})
    return SimpleNamespace(network_arch_init_kwargs=ak,
                           network_arch_class_name='dynamic_network_architectures.architectures.unet.PlainConvUNet',
                           network_arch_init_kwargs_req_import=['conv_op', 'norm_op', 'dropout_op', 'nonlin'],
                           configuration={})


# ---------------------------------------------------------------- legacy
def test_legacy_param_count_exact():
    net = PVMUNet3D(input_channels=1, num_classes=3, mamba_impl=IMPL, **legacy_config())
    assert count_inference_params(net) == 64_804


def test_width_schedule_matches_ultralight():
    assert pvm_widths(8, 6) == [8, 16, 24, 32, 48, 64]   # 2D paper c_list
    assert pvm_widths(8, 5) == [8, 16, 24, 32, 48]       # legacy 3D c_list
    assert pvm_widths(16, 7) == [16, 32, 48, 64, 96, 128, 128]


@cuda_only
def test_legacy_pvmlayer_equivalence():
    from models.UltraLight_VM_UNet_3D import PVMLayer
    torch.manual_seed(0)
    old = PVMLayer(24, 32).cuda()
    new = PVMLayer3D(24, 32).cuda()
    sd = {('mamba.0.' + k[6:] if k.startswith('mamba.') else k): v for k, v in old.state_dict().items()}
    new.load_state_dict(sd)
    x = torch.randn(2, 24, 8, 8, 8, device='cuda')
    torch.testing.assert_close(new(x), old(x), rtol=1e-5, atol=1e-6)


@cuda_only
def test_legacy_full_model_equivalence():
    from models.UltraLight_VM_UNet_3D import UltraLight_VM_UNet_3D
    torch.manual_seed(0)
    old = UltraLight_VM_UNet_3D(num_classes=3, input_channels=1, c_list=(8, 16, 24, 32, 48)).cuda().eval()
    new = PVMUNet3D(input_channels=1, num_classes=3, **legacy_config()).cuda().eval()
    missing, unexpected = new.load_state_dict(convert_legacy_state_dict(old.state_dict()), strict=False)
    assert not unexpected, unexpected
    assert all(k.startswith('ds_heads.') for k in missing), missing
    x = torch.randn(1, 1, 64, 64, 64, device='cuda')
    with torch.no_grad():
        torch.testing.assert_close(new(x), old(x), rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------- scans
@pytest.mark.parametrize('scan', SCAN_MODES)
def test_scan_permutations_invert(scan):
    layer = PVMLayer3D(16, 16, scan=scan, mamba_impl='pytorch')
    spatial = torch.Size([3, 4, 5])
    for seqs in layer._branch_perms(spatial, torch.device('cpu')):
        for p in seqs:
            if p is None:
                continue
            perm, inv = p
            assert sorted(perm.tolist()) == list(range(60))
            assert torch.equal(perm[inv], torch.arange(60))


def test_dd_branches_scan_distinct_orders():
    layer = PVMLayer3D(16, 16, scan='dd', mamba_impl='pytorch')
    perms = layer._branch_perms(torch.Size([3, 4, 5]), torch.device('cpu'))
    orders = [tuple(range(60)) if s[0] is None else tuple(s[0][0].tolist()) for s in perms]
    assert len(set(orders)) == 4


def test_dd_costs_no_extra_params():
    a = PVMLayer3D(32, 32, scan='raster', mamba_impl='pytorch')
    b = PVMLayer3D(32, 32, scan='dd', mamba_impl='pytorch')
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


# ---------------------------------------------------------------- geometry
ANISO = [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]


def test_deep_supervision_outputs_match_nnunet_scales():
    import numpy as np
    net = PVMUNet3D(1, 4, pvm_widths(8, 5), ANISO, kernel_sizes=[[1, 3, 3]] * 2 + [[3, 3, 3]] * 3,
                    deep_supervision=True, mamba_impl=IMPL).to(DEV)
    x = torch.randn(1, 1, 8, 64, 64, device=DEV)
    outs = net(x)
    # nnUNetTrainer._get_deep_supervision_scales
    scales = list(list(i) for i in 1 / np.cumprod(np.vstack(ANISO), axis=0))[:-1]
    assert len(outs) == len(scales)
    for o, s in zip(outs, scales):
        assert list(o.shape[2:]) == [int(round(d * f)) for d, f in zip(x.shape[2:], s)]
        assert o.shape[1] == 4
    net.deep_supervision = False
    assert net(x).shape == (1, 4, 8, 64, 64)


def _trainer_module():
    return importlib.import_module('pvm3d.nnunet_trainers.trainers_pvm3d')


def _pvm_trainers():
    mod = _trainer_module()
    return sorted(n for n in dir(mod) if n.startswith('nnUNetTrainerPVM3D') and hasattr(getattr(mod, n), 'PVM'))


@pytest.mark.parametrize('name', _pvm_trainers())
def test_every_variant_builds_and_backprops(name):
    cls = getattr(_trainer_module(), name)
    cfg = {**cls.PVM, 'mamba_impl': IMPL}
    net = build_pvm_from_plans(cfg, fake_cm(ANISO), 2, 3, enable_deep_supervision=True).to(DEV)
    outs = net(torch.randn(1, 2, 8, 32, 32, device=DEV))
    sum(o.float().mean() for o in outs).backward()
    no_grad = [n_ for n_, p in net.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f'parameters disconnected from the loss: {no_grad}'


def test_ablation_param_effects():
    mod = _trainer_module()
    p = {}
    for tag in ('S', 'S_dd', 'S_noMamba', 'S_unshared', 'S_br1'):
        cfg = {**getattr(mod, f'nnUNetTrainerPVM3D_{tag}').PVM, 'mamba_impl': 'pytorch'}
        p[tag] = count_inference_params(build_pvm_from_plans(cfg, fake_cm(ANISO), 1, 3, False))
    assert p['S_dd'] == p['S']              # DD-PVM is free
    assert p['S_noMamba'] < p['S']          # Mamba removed
    assert p['S_unshared'] > p['S']         # 4 Mambas instead of 1
    assert p['S_br1'] > p['S']              # one full-width Mamba > one quarter-width shared


# ---------------------------------------------------------------- baselines / glue
def test_tiny_unet_matches_target_params():
    cm = fake_cm([[1, 1, 1]] + [[2, 2, 2]] * 5)
    ref = build_pvm_from_plans({**_trainer_module().nnUNetTrainerPVM3DFinal_S.PVM, 'mamba_impl': 'pytorch'},
                               cm, 1, 3, False)
    target = count_inference_params(ref)
    tiny = build_tiny_unet_matched(target, cm, 1, 3, enable_deep_supervision=True)
    assert abs(count_inference_params(tiny) - target) / target < 0.05, (count_inference_params(tiny), target)


def test_closed_form_cosine_matches_config_setting_3d():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sch = _ClosedFormCosineLR(opt, T_max=50, eta_min=1e-5, base_lr=1e-3)
    lrs = {}
    for e in (0, 25, 50, 100, 150):
        sch.step(e)
        lrs[e] = opt.param_groups[0]['lr']
    assert math.isclose(lrs[0], 1e-3) and math.isclose(lrs[50], 1e-5) and math.isclose(lrs[100], 1e-3)
    assert math.isclose(lrs[150], 1e-5)


def test_trainers_discoverable_via_nnUNet_extTrainer():
    """Runs in a fresh interpreter, like nnUNetv2_train does: nnU-Net's lookup
    iterates sys.modules, which pytest's own lazily-loaded modules can mutate
    mid-iteration (a test-harness artifact, not a CLI failure)."""
    import subprocess
    names = ['nnUNetTrainerPVM3D_S', 'nnUNetTrainerPVM3D_S_dd', 'nnUNetTrainerPVM3DFinal_S_frac25',
             'nnUNetTrainerTinyUNet_matchS', 'nnUNetTrainer_frac10', 'nnUNetTrainerPVM3D_S_AdamWCos50']
    code = ('from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name as f\n'
            'from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer\n'
            f'for n in {names!r}:\n'
            '    c = f(n); assert issubclass(c, nnUNetTrainer) and c.__name__ == n, n\n'
            'print("ALL_FOUND")\n')
    env = {**os.environ, 'nnUNet_extTrainer': os.path.join(REPO, 'pvm3d', 'nnunet_trainers')}
    r = subprocess.run([sys.executable, '-c', code], env=env, cwd='/', capture_output=True, text=True, timeout=300)
    assert 'ALL_FOUND' in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


@cuda_only
def test_pytorch_backend_matches_kernel_dd_scan():
    """The CPU benchmarking path (mamba_impl='pytorch') must compute what the
    training path computes, including the direction-diverse gathers."""
    torch.manual_seed(0)
    a = PVMLayer3D(32, 32, scan='dd', mamba_impl='mamba_ssm').cuda()
    b = PVMLayer3D(32, 32, scan='dd', mamba_impl='pytorch').cuda()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 32, 4, 6, 5, device='cuda')
    torch.testing.assert_close(b(x), a(x), rtol=1e-4, atol=1e-5)


@cuda_only
def test_native_mamba_init_keeps_mamba_output_alive():
    """The inherited UltraLight init re-initialises Mamba's projections (std 0.02),
    leaving its output ~1e-3 of the skip path at init; 'native' keeps mamba_ssm's
    own init (~3e-2). This is what ablation A9 (_S_nativeInit) tests."""
    def ratio(net):
        r = []
        hs = [mb.register_forward_hook(lambda m, i, o, l=l: r.append((o.norm() / (l.skip_scale * i[0]).norm()).item()))
              for l in net.modules() if isinstance(l, PVMLayer3D) for mb in l.mamba]
        with torch.no_grad():
            net(torch.randn(1, 1, 64, 64, 64, device='cuda'))
        for h in hs:
            h.remove()
        return sum(r) / len(r)
    torch.manual_seed(0)
    ul = ratio(PVMUNet3D(1, 3, **legacy_config()).cuda().eval())
    torch.manual_seed(0)
    nat = ratio(PVMUNet3D(1, 3, mamba_init='native', **legacy_config()).cuda().eval())
    assert ul < 5e-3 and nat > 10 * ul, (ul, nat)


def test_no_inner_skip_routes_everything_through_mixer():
    net = PVMLayer3D(16, 16, inner_skip=False, mamba_impl='pytorch')
    assert net.skip_scale is None
    with torch.no_grad():
        for p in net.mamba.parameters():
            p.zero_()
    x = torch.randn(1, 16, 3, 4, 5)
    # with the mixer silenced and no bypass, the output is proj(norm(0)) = proj bias, input-independent
    torch.testing.assert_close(net(x), net(torch.randn_like(x)))
