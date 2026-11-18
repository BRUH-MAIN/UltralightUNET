"""nnU-Net v2 trainers for the external lightweight / Mamba baselines (roadmap §4, E6).

Every architecture runs inside the identical nnU-Net pipeline (preprocessing,
augmentation, patch sampling, 1000 x 250 iterations, sliding-window validation)
so that only the network -- and, following nnU-Net Revisited's "closest
configuration" rule, the optimizer recipe its authors used -- differs:

  model         code (fetched, not vendored)        recipe kept from the authors' repo
  LightM-UNet   LightM-UNet/lightm-unet/nnunetv2/nets  Adam 1e-4, wd 1e-5, eps 1e-5, poly 0.9, fp32
                (its own nnU-Net trainer)              training, no deep supervision
  SlimUNETR     Slim-UNETR/src/SlimUNETR              AdamW 1e-3, wd 0.05, warmup + cosine to 1e-7,
                                                      dropout 0.3, BraTS config widths
  SegFormer3D   SegFormer3D/architectures             AdamW 1e-4, wd 0.01, warmup + cosine to 6e-6
  SegMamba      SegMamba/model_segmamba               SGD 1e-2, wd 3e-5, momentum 0.99 (= nnU-Net)

Code location: $PVM3D_THIRD_PARTY or $REPO/third_party (scripts/cluster/fetch_baselines.sh).
Patch constraints: SlimUNETR/SegFormer3D need every patch dimension divisible by
32, SegMamba by 16 -> train with ``-p nnUNetPlans_div32`` / ``nnUNetPlans_div16``
(scripts/cluster/make_plans_variant.py). SegMamba additionally needs its own venv
(see fetch_baselines.sh).
"""

import contextlib
import importlib.util
import math
import os
import sys

import torch

# realpath: nnU-Net may import this file through the symlink scripts/cluster/install_trainers.py
# creates inside site-packages; the repo root must come from the real location.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as _nnunet_trainer_module  # noqa: E402
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402

from pvm3d.nnunet_glue import (FrequentCheckpointMixin, NoCompileMixin, NoDeepSupervisionMixin,  # noqa: E402
                               SmokeMixin)


def third_party_root():
    return os.environ.get('PVM3D_THIRD_PARTY', os.path.join(_REPO, 'third_party'))


def _load_file(alias, path, extra_path=None):
    """Import one third-party file under a private module name (LightM-UNet's
    network lives inside a package that is *also* called nnunetv2)."""
    if alias in sys.modules:
        return sys.modules[alias]
    if not os.path.isfile(path):
        raise FileNotFoundError(f'{path} missing -- run scripts/cluster/fetch_baselines.sh')
    if extra_path and extra_path not in sys.path:
        sys.path.insert(0, extra_path)
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _require_divisible(configuration_manager, div, name):
    patch = list(configuration_manager.patch_size)
    if any(p % div for p in patch):
        raise ValueError(f'{name} needs every patch dimension divisible by {div}, got {patch}. Train with '
                         f'-p nnUNetPlans_div{div} (scripts/cluster/make_plans_variant.py --divisible {div}).')
    return patch


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warm-up then cosine decay, stepped like nnU-Net's schedulers (step(epoch))."""
    def __init__(self, optimizer, base_lr, total_epochs, warmup_epochs, eta_min):
        self.base, self.T, self.W, self.eta_min = base_lr, total_epochs, max(1, warmup_epochs), eta_min
        self.ctr = 0
        super().__init__(optimizer, -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        e = current_step
        if e < self.W:
            lr = self.base * (e + 1) / self.W
        else:
            t = (e - self.W) / max(1, self.T - self.W)
            lr = self.eta_min + (self.base - self.eta_min) * (1 + math.cos(math.pi * min(t, 1.0))) / 2
        for g in self.optimizer.param_groups:
            g['lr'] = lr


class FP32TrainingMixin:
    """Train without autocast/GradScaler (LightM-UNet's own trainer does this).
    nnU-Net hard-codes `autocast(...)` in train_step/validation_step via
    `from torch import autocast`, so that module-level name is swapped for a no-op
    for the duration of each step. Final sliding-window validation is unchanged."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.grad_scaler = None

    @contextlib.contextmanager
    def _no_autocast(self):
        orig = _nnunet_trainer_module.autocast
        _nnunet_trainer_module.autocast = lambda *a, **k: contextlib.nullcontext()
        try:
            yield
        finally:
            _nnunet_trainer_module.autocast = orig

    def train_step(self, batch):
        with self._no_autocast():
            return super().train_step(batch)

    def validation_step(self, batch):
        with self._no_autocast():
            return super().validation_step(batch)


class _ExternalBase(FrequentCheckpointMixin, NoCompileMixin, NoDeepSupervisionMixin, nnUNetTrainer):
    pass


# ---------------------------------------------------------------- LightM-UNet
class nnUNetTrainerLightMUNet(FP32TrainingMixin, _ExternalBase):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.initial_lr = 1e-4
        self.weight_decay = 1e-5

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=False):
        mod = _load_file('_tp_lightmunet', os.path.join(third_party_root(), 'LightM-UNet', 'lightm-unet', 'nnunetv2',
                                                        'nets', 'LightMUNet.py'))
        return mod.LightMUNet(spatial_dims=len(configuration_manager.patch_size), init_filters=32,
                              in_channels=num_input_channels, out_channels=num_output_channels,
                              blocks_down=[1, 2, 2, 4], blocks_up=[1, 1, 1])

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.network.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay,
                               eps=1e-5)
        return opt, PolyLRScheduler(opt, self.initial_lr, self.num_epochs, exponent=0.9)


# ---------------------------------------------------------------- SlimUNETR
class nnUNetTrainerSlimUNETR(_ExternalBase):
    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=False):
        patch = _require_divisible(configuration_manager, 32, 'SlimUNETR')
        root = os.path.join(third_party_root(), 'Slim-UNETR')
        mod = _load_file('_tp_slimunetr', os.path.join(root, 'src', 'SlimUNETR', 'SlimUNETR.py'), extra_path=root)
        return mod.SlimUNETR(in_channels=num_input_channels, out_channels=num_output_channels, embed_dim=96,
                             embedding_dim=math.prod(p // 32 for p in patch),  # tokens after the 4,2,2,2 strides
                             channels=(24, 48, 60), blocks=(1, 2, 3, 2), heads=(1, 2, 4, 4), r=(4, 2, 2, 1),
                             dropout=0.3)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.network.parameters(), lr=1e-3, weight_decay=0.05)
        # authors: 5 warm-up epochs of 300 -> the same fraction of nnU-Net's 1000
        return opt, WarmupCosineLR(opt, 1e-3, self.num_epochs, round(self.num_epochs * 5 / 300), 1e-7)


# ---------------------------------------------------------------- SegFormer3D
class nnUNetTrainerSegFormer3D(_ExternalBase):
    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=False):
        _require_divisible(configuration_manager, 32, 'SegFormer3D')
        mod = _load_file('_tp_segformer3d', os.path.join(third_party_root(), 'SegFormer3D', 'architectures',
                                                         'segformer3d.py'))
        return mod.SegFormer3D(in_channels=num_input_channels, num_classes=num_output_channels)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.network.parameters(), lr=1e-4, weight_decay=0.01)
        # authors: 20 warm-up epochs of 800, cosine to 6e-6
        return opt, WarmupCosineLR(opt, 1e-4, self.num_epochs, round(self.num_epochs * 20 / 800), 6e-6)


# ---------------------------------------------------------------- SegMamba
class nnUNetTrainerSegMamba(_ExternalBase):
    """Needs the SegMamba venv (its mamba fork). Its encoder hard-codes
    nslices=[64,32,16,8] for 128^3 patches; a pre-hook on every MambaLayer sets
    nslices to the actual depth of the feature map instead, which is what the
    hard-coded values are for a 128^3 patch."""
    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=False):
        _require_divisible(configuration_manager, 16, 'SegMamba')
        root = os.path.join(third_party_root(), 'SegMamba')
        mod = _load_file('_tp_segmamba', os.path.join(root, 'model_segmamba', 'segmamba.py'))
        net = mod.SegMamba(in_chans=num_input_channels, out_chans=num_output_channels, depths=[2, 2, 2, 2],
                           feat_size=[48, 96, 192, 384])
        for m in net.modules():
            if type(m).__name__ == 'MambaLayer' and hasattr(getattr(m, 'mamba', None), 'nslices'):
                m.register_forward_pre_hook(lambda mod_, inp: setattr(mod_.mamba, 'nslices', inp[0].shape[2]))
        return net

    def configure_optimizers(self):
        opt = torch.optim.SGD(self.network.parameters(), lr=1e-2, weight_decay=3e-5, momentum=0.99, nesterov=True)
        return opt, PolyLRScheduler(opt, 1e-2, self.num_epochs)


# ---------------------------------------------------------------- smoke variants
class nnUNetTrainerLightMUNet_smoke(SmokeMixin, nnUNetTrainerLightMUNet):
    pass


class nnUNetTrainerSlimUNETR_smoke(SmokeMixin, nnUNetTrainerSlimUNETR):
    pass


class nnUNetTrainerSegFormer3D_smoke(SmokeMixin, nnUNetTrainerSegFormer3D):
    pass
