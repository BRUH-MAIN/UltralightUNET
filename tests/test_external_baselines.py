"""External baseline wrappers (pvm3d/nnunet_trainers/trainers_external.py).

These import third-party code fetched by scripts/cluster/fetch_baselines.sh, so
every test skips until that code is present ($PVM3D_THIRD_PARTY or
$REPO/third_party). Run once after fetching, on the machine that will train:

    bash scripts/cluster/fetch_baselines.sh && pip install "monai>=1.3"
    python -m pytest tests/test_external_baselines.py -v

Checks: each network builds from nnU-Net-style plans, returns full-resolution
logits with the right channel count, back-propagates, and the patch-divisibility
guard fires for patches nnU-Net plans but the architecture cannot take.
SegMamba runs only inside its own venv (its mamba fork); elsewhere it skips.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
TP = os.environ.get('PVM3D_THIRD_PARTY', os.path.join(REPO, 'third_party'))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def cm(patch):
    return SimpleNamespace(patch_size=patch, network_arch_init_kwargs={}, configuration={})


def _ext():
    from pvm3d.nnunet_trainers import trainers_external
    return trainers_external


def _needs(path, *mods):
    if not os.path.exists(os.path.join(TP, path)):
        pytest.skip(f'{path} not fetched (scripts/cluster/fetch_baselines.sh)')
    for m in mods:
        pytest.importorskip(m)


def _forward_ok(net, n_in, n_out, patch):
    net = net.to(DEV).train()
    x = torch.randn(1, n_in, *patch, device=DEV)
    y = net(x)
    y = y[0] if isinstance(y, (list, tuple)) else y
    assert tuple(y.shape) == (1, n_out, *patch)
    y.float().mean().backward()
    return sum(p.numel() for p in net.parameters())


@pytest.mark.skipif(DEV != 'cuda', reason='LightM-UNet uses mamba_ssm (CUDA)')
def test_lightmunet():
    _needs('LightM-UNet', 'monai')
    net = _ext().nnUNetTrainerLightMUNet.build_network_architecture(None, cm([64, 64, 64]), 1, 3)
    _forward_ok(net, 1, 3, [64, 64, 64])


def test_slimunetr_and_divisibility_guard():
    _needs('Slim-UNETR')
    ext = _ext()
    net = ext.nnUNetTrainerSlimUNETR.build_network_architecture(None, cm([64, 96, 64]), 4, 3)
    _forward_ok(net, 4, 3, [64, 96, 64])
    with pytest.raises(ValueError, match='divisible by 32'):
        ext.nnUNetTrainerSlimUNETR.build_network_architecture(None, cm([40, 56, 40]), 1, 3)


def test_segformer3d():
    _needs('SegFormer3D', 'einops')
    net = _ext().nnUNetTrainerSegFormer3D.build_network_architecture(None, cm([64, 64, 64]), 1, 3)
    _forward_ok(net, 1, 3, [64, 64, 64])


@pytest.mark.skipif(DEV != 'cuda', reason='SegMamba uses its mamba fork (CUDA)')
def test_segmamba_nslices_follow_patch_depth():
    _needs('SegMamba', 'monai')
    try:
        from mamba_ssm import Mamba
        Mamba(d_model=8, bimamba_type='v3', nslices=2)
    except TypeError:
        pytest.skip('official mamba_ssm installed; SegMamba needs its fork (separate venv)')
    patch = [32, 64, 64]  # not 128^3: the hard-coded nslices would be wrong without the hook
    net = _ext().nnUNetTrainerSegMamba.build_network_architecture(None, cm(patch), 1, 3)
    _forward_ok(net, 1, 3, patch)
