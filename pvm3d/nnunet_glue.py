"""Glue between nnU-Net v2 and the models used in the PVM-3D benchmark.

Kept outside pvm3d/nnunet_trainers/ on purpose: nnU-Net's external-trainer
lookup (nnUNet_extTrainer) imports *every* module in that folder, so only
trainer modules live there.
"""

import math
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from pvm3d.network import PVMUNet3D, pvm_widths


# --------------------------------------------------------------------------
# building PVM-3D from nnU-Net plans
# --------------------------------------------------------------------------
def plans_geometry(configuration_manager):
    ak = configuration_manager.network_arch_init_kwargs
    strides = [list(s) if isinstance(s, (list, tuple)) else [s] * 3 for s in ak['strides']]
    if len(strides[0]) != 3:
        raise ValueError('PVM-3D is a 3D network; use a 3d_* configuration, not '
                         f'{configuration_manager.configuration.get("data_identifier", "2d")}')
    ks = ak.get('kernel_sizes')
    ks = [list(k) if isinstance(k, (list, tuple)) else [k] * 3 for k in ks] if ks is not None else None
    return strides, ks


def build_pvm_from_plans(cfg: dict, configuration_manager, num_input_channels, num_output_channels,
                         enable_deep_supervision=True):
    strides, ks = plans_geometry(configuration_manager)
    n = len(strides)
    widths = cfg.get('widths') or pvm_widths(cfg['base_width'], n)
    return PVMUNet3D(
        input_channels=num_input_channels, num_classes=num_output_channels, widths=widths,
        strides=strides, kernel_sizes=ks if cfg.get('kernel_from_plans', True) else None,
        n_pvm_enc=min(cfg['n_pvm_enc'], n - 1), n_pvm_dec=min(cfg['n_pvm_dec'], n - 1),
        bridge=cfg['bridge'], deep_supervision=enable_deep_supervision, scan=cfg['scan'],
        n_branches=cfg['n_branches'], share_weights=cfg['share_weights'], mixer=cfg['mixer'],
        d_state=cfg['d_state'], expand=cfg['expand'], mamba_impl=cfg.get('mamba_impl', 'mamba_ssm'),
        # optional keys: absent from existing trainers' PVM dicts, so their recorded
        # pvm_config.json (and the resume guard) are unchanged
        mamba_init=cfg.get('mamba_init', 'ultralight'), inner_skip=cfg.get('inner_skip', True))


def unwrap(module):
    from torch._dynamo import OptimizedModule
    if hasattr(module, 'module'):  # DDP
        module = module.module
    if isinstance(module, OptimizedModule):
        module = module._orig_mod
    return module


def count_inference_params(net: nn.Module) -> int:
    """Parameters actually used at inference. Deep-supervision-only heads
    exist in the state dict (nnU-Net's predictor loads it strictly) but are
    never evaluated at test time, so they are excluded -- for PVMUNet3D
    (`ds_heads`) and dynamic_network_architectures U-Nets (`seg_layers[:-1]`)
    alike, so the two are counted on equal terms."""
    net = unwrap(net)
    total = sum(p.numel() for p in net.parameters())
    if isinstance(net, PVMUNet3D):
        total -= sum(p.numel() for p in net.ds_heads.parameters())
    dec = getattr(net, 'decoder', None)
    seg = getattr(dec, 'seg_layers', None)
    if seg is not None and len(seg) > 1:
        total -= sum(p.numel() for layer in seg[:-1] for p in layer.parameters())
    return total


# --------------------------------------------------------------------------
# parameter-matched plain nnU-Net (the "is a tiny CNN just as good?" control)
# --------------------------------------------------------------------------
def tiny_unet_features(target_params, configuration_manager, num_input_channels, num_output_channels,
                       max_base=32.0, step=0.125):
    """Per-stage widths for the parameter-matched nnU-Net: nnU-Net's own width
    *ratios* (e.g. 1:2:4:8:10:10) times a fractional base width, rounded per
    stage, with the base chosen to minimise |inference params - target|.
    Integer bases are too coarse (32% off on Hippocampus); 1/8 steps get
    within a few percent. Pure function of the plans -> inference rebuilds
    the identical network."""
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

    ak = deepcopy(configuration_manager.network_arch_init_kwargs)
    default = ak['features_per_stage']
    ratios = [f / default[0] for f in default]
    best, tried = None, set()
    for base in np.arange(1.0, max_base + step, step):
        feats = tuple(max(1, int(round(base * r))) for r in ratios)
        if feats in tried:
            continue
        tried.add(feats)
        net = get_network_from_plans(configuration_manager.network_arch_class_name,
                                     {**ak, 'features_per_stage': list(feats)},
                                     configuration_manager.network_arch_init_kwargs_req_import,
                                     num_input_channels, num_output_channels, allow_init=False,
                                     deep_supervision=False)
        p = count_inference_params(net)
        if best is None or abs(p - target_params) < abs(best[1] - target_params):
            best = (list(feats), p)
        if p > 2 * target_params:
            break
    return best[0]


def build_tiny_unet_matched(target_params, configuration_manager, num_input_channels, num_output_channels,
                            enable_deep_supervision=True):
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    feats = tiny_unet_features(target_params, configuration_manager, num_input_channels, num_output_channels)
    ak = deepcopy(configuration_manager.network_arch_init_kwargs)
    return get_network_from_plans(configuration_manager.network_arch_class_name, {**ak, 'features_per_stage': feats},
                                  configuration_manager.network_arch_init_kwargs_req_import,
                                  num_input_channels, num_output_channels, allow_init=True,
                                  deep_supervision=enable_deep_supervision)


# --------------------------------------------------------------------------
# trainer mixins (compose with any nnUNetTrainer subclass)
# --------------------------------------------------------------------------
class NoCompileMixin:
    """mamba_ssm's fused kernels are opaque to torch.compile; compiling around
    them buys nothing and has broken in the past (U-Mamba disables it too)."""
    def _do_i_compile(self):
        return False


class FrequentCheckpointMixin:
    """checkpoint_latest every 10 epochs instead of nnU-Net's 50. Changes nothing
    about training; bounds the work lost when a 72 h allocation ends mid-run
    (scripts/cluster/pack_runs.py resumes with --c)."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.save_every = 10


class PVMDeepSupervisionMixin:
    def set_deep_supervision_enabled(self, enabled: bool):
        unwrap(self.network).deep_supervision = enabled


class NoDeepSupervisionMixin:
    """For architectures without multi-scale outputs (external baselines) and
    the no-DS ablation."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # explicit signature: nnUNetTrainer.__init__ introspects it into my_init_kwargs
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    def set_deep_supervision_enabled(self, enabled: bool):
        m = unwrap(self.network)
        if hasattr(m, 'deep_supervision'):
            m.deep_supervision = False


class _ClosedFormCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """torch's CosineAnnealingLR, but stepped the way nnU-Net steps schedulers
    (``step(current_epoch)``). Closed form is periodic with period 2*T_max --
    exactly what configs/config_setting_3d.py's T_max=50 does over 300 epochs."""
    def __init__(self, optimizer, T_max, eta_min, base_lr):
        self.T_max, self.eta_min, self.base = T_max, eta_min, base_lr
        self.ctr = 0
        super().__init__(optimizer, -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1
        lr = self.eta_min + (self.base - self.eta_min) * (1 + math.cos(math.pi * current_step / self.T_max)) / 2
        for g in self.optimizer.param_groups:
            g['lr'] = lr


class AdamWCosine50Mixin:
    """Ablation A7: the 2D paper's optimizer + schedule (AdamW lr 1e-3, wd 1e-2,
    CosineAnnealingLR T_max=50, eta_min 1e-5) inside the nnU-Net pipeline, so
    only the optimizer/schedule differ from the nnU-Net-recipe run."""
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.network.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                                weight_decay=1e-2)
        return opt, _ClosedFormCosineLR(opt, T_max=50, eta_min=1e-5, base_lr=1e-3)


class SmokeMixin:
    """Two tiny epochs: exercises train, checkpoint, final sliding-window
    validation and export end-to-end in minutes."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # explicit signature: nnUNetTrainer.__init__ introspects it into my_init_kwargs
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2
        self.num_iterations_per_epoch = 5
        self.num_val_iterations_per_epoch = 2


class DataFractionMixin:
    """E7 data efficiency: train on a seeded FRACTION of the fold's training
    cases; validation set unchanged so scores stay comparable to the full run."""
    FRACTION = 1.0

    def do_split(self):
        tr, val = super().do_split()
        n = max(1, int(round(len(tr) * self.FRACTION)))
        rng = np.random.RandomState(12345 + (0 if self.fold == 'all' else int(self.fold)))
        tr = sorted(rng.choice(sorted(tr), n, replace=False).tolist())
        self.print_to_log_file(f'DataFraction {self.FRACTION}: training on {len(tr)} cases')
        return tr, val


# AMOS2022 modality. The id rule (MRI iff id >= 500) is NOT reliable: the official
# metadata (zenodo 7262581, labeled_data_meta_0000_0599.csv) lists amos_id 500
# twice -- once as a CT (Optima CT540), once as an MR (Ingenia). The modality is
# therefore read from the images: `convert_datasets.py verify` writes
# <nnUNet_raw>/Dataset219_*/modality.json ({case: "CT"|"MRI"}, CT = min intensity
# below -200 HU), and amos_is_mri() uses that map. The id rule is only a fallback.
AMOS_MRI_MIN_ID = 500
_AMOS_MODALITY = None


def _amos_modality_map():
    global _AMOS_MODALITY
    if _AMOS_MODALITY is None:
        import glob as _glob
        import json as _json
        import os as _os
        raw = _os.environ.get('nnUNet_raw', '')
        files = _glob.glob(_os.path.join(raw, 'Dataset219_*', 'modality.json')) if raw else []
        _AMOS_MODALITY = _json.load(open(files[0])) if files else {}
    return _AMOS_MODALITY


def amos_is_mri(case_id: str) -> bool:
    m = _amos_modality_map()
    if case_id in m:
        return m[case_id] == 'MRI'
    import warnings
    warnings.warn(f'no AMOS modality map entry for {case_id}; falling back to the id rule (ambiguous at 500) -- '
                  'run scripts/cluster/convert_datasets.py verify')
    return int(case_id.split('_')[-1]) >= AMOS_MRI_MIN_ID


class CTOnlyMixin:
    """E8 cross-modality: train on the fold's CT cases only; validation keeps CT
    and MRI so scripts/eval3d.py --modality-split reports both."""
    def do_split(self):
        tr, val = super().do_split()
        ct = [k for k in tr if not amos_is_mri(k)]
        self.print_to_log_file(f'CTOnly: {len(ct)} CT / {len(tr) - len(ct)} MRI cases in training split; '
                               f'using CT only')
        return ct, val
