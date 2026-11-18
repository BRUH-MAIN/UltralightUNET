"""nnU-Net v2 trainers for PVM-3D: model sizes, ablations, data-efficiency and
cross-modality variants. Every experiment in docs/3d_journal_roadmap.md that
trains a PVM-3D model is one class here, selected with ``-tr``:

    export nnUNet_extTrainer=$REPO/pvm3d/nnunet_trainers   # scripts/cluster/env.sh does this
    nnUNetv2_train 27 3d_fullres 0 -tr nnUNetTrainerPVM3D_S_dd

Everything except the network (and, for the recipe ablation, the optimizer)
is nnU-Net's default: 1000 epochs x 250 iterations, SGD+poly, deep
supervision, nnU-Net's preprocessing/augmentation/patch size/batch size and
sliding-window validation with mirroring.
"""

import os
import sys

# realpath: nnU-Net may import this file through the symlink scripts/cluster/install_trainers.py
# creates inside site-packages; the repo root must come from the real location.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402

from pvm3d.nnunet_glue import (AdamWCosine50Mixin, CTOnlyMixin, DataFractionMixin,  # noqa: E402
                               FrequentCheckpointMixin, NoCompileMixin, NoDeepSupervisionMixin, PVMDeepSupervisionMixin,
                               SmokeMixin, build_pvm_from_plans)


class nnUNetTrainerPVM3D(FrequentCheckpointMixin, NoCompileMixin, PVMDeepSupervisionMixin, nnUNetTrainer):
    """Base class: PVM-3D-S design with raster scan (the direct port of the 2D
    PVM layer). Subclasses only override entries of ``PVM`` (and, for the
    ablation family, ``NUM_EPOCHS``)."""
    NUM_EPOCHS = None  # None = nnU-Net's 1000
    PVM = dict(base_width=16, n_pvm_enc=3, n_pvm_dec=3, bridge='both', scan='raster', n_branches=4,
               share_weights=True, mixer='mamba', d_state=16, expand=2, kernel_from_plans=True,
               mamba_impl='mamba_ssm')

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.NUM_EPOCHS is not None:
            self.num_epochs = self.NUM_EPOCHS

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        return build_pvm_from_plans(cls.PVM, configuration_manager, num_input_channels, num_output_channels,
                                    enable_deep_supervision)

    def on_train_start(self):
        """Refuse to resume a run whose architecture config changed since it
        started. Scan-order changes keep every tensor shape, so a checkpoint
        would load silently into a different model without this check."""
        from batchgenerators.utilities.file_and_folder_operations import isfile, join, load_json, save_json
        cfg = {**{k: v for k, v in self.PVM.items() if k != 'mamba_impl'}, 'num_epochs': self.num_epochs}
        f = join(self.output_folder, 'pvm_config.json')
        if isfile(f) and load_json(f) != cfg:
            raise RuntimeError(f'{f} records {load_json(f)} but {type(self).__name__} now builds {cfg}. '
                               'The trainer definition changed after this run started -- use a new trainer name.')
        save_json(cfg, f)
        super().on_train_start()


def _variant(name, base, doc, num_epochs=None, **overrides):
    attrs = {'PVM': {**base.PVM, **overrides}, '__doc__': doc}
    if num_epochs is not None:
        attrs['NUM_EPOCHS'] = num_epochs
    cls = type(name, (base,), attrs)
    globals()[name] = cls
    return cls


# ---- model family (E2 width pilot, E4 main benchmark, E5 Pareto) ----------
_variant('nnUNetTrainerPVM3D_T', nnUNetTrainerPVM3D, 'PVM-3D-T: base width 8 (UltraLight 2D widths).', base_width=8)
_variant('nnUNetTrainerPVM3D_S', nnUNetTrainerPVM3D, 'PVM-3D-S: base width 16.', base_width=16)
_variant('nnUNetTrainerPVM3D_B', nnUNetTrainerPVM3D, 'PVM-3D-B: base width 32.', base_width=32)
_variant('nnUNetTrainerPVM3D_Legacy', nnUNetTrainerPVM3D,
         'Legacy design (models/UltraLight_VM_UNet_3D.py: widths 8.., PVM in the 2 deepest encoder stages '
         '+ deepest decoder block) on nnU-Net geometry/recipe -- E1: how much of the 4.4-pt Hippocampus '
         'gap the pipeline alone closes.', base_width=8, n_pvm_enc=2, n_pvm_dec=1)

# ---- E3 ablations (all relative to nnUNetTrainerPVM3D_S_ref) ---------------
# Every ablation, and the reference it is compared to, trains for
# ABLATION_EPOCHS. 1000 = nnU-Net's full schedule. If the S1 throughput
# measurement shows the budget cannot carry ~70 ablation runs at 1000 epochs,
# set this to 500 BEFORE S2 starts (on_train_start refuses to resume a run whose
# epoch count changed). Main-benchmark trainers are unaffected.
ABLATION_EPOCHS = 1000
S = globals()['nnUNetTrainerPVM3D_S']
_variant('nnUNetTrainerPVM3D_S_ref', S, 'Ablation reference: PVM-3D-S (raster) at ABLATION_EPOCHS.',
         num_epochs=ABLATION_EPOCHS)
S_REF = globals()['nnUNetTrainerPVM3D_S_ref']
_ABLATIONS = {
    # A4 scan order
    'dd': dict(scan='dd'), 'cross': dict(scan='cross'), 'bidir': dict(scan='bidir'), 'tri': dict(scan='tri'), 'random': dict(scan='random'),
    # A1 is Mamba load-bearing?
    'noMamba': dict(mixer='identity'), 'convMixer': dict(mixer='conv'),
    # A2 number of parallel branches
    'br1': dict(n_branches=1), 'br2': dict(n_branches=2), 'br8': dict(n_branches=8),
    # A3 shared vs independent Mamba weights
    'unshared': dict(share_weights=False),
    # A5 PVM placement
    'enc2dec1': dict(n_pvm_enc=2, n_pvm_dec=1), 'encAll': dict(n_pvm_enc=99, n_pvm_dec=99),
    # A6 attention bridge
    'bridgeNone': dict(bridge='none'), 'bridgeSpatial': dict(bridge='spatial'),
    'bridgeChannel': dict(bridge='channel'),
    # A8 SSM hyperparameters
    'dstate8': dict(d_state=8), 'dstate32': dict(d_state=32), 'expand1': dict(expand=1),
    # A9 why is Mamba bypassed? (prior bespoke-pipeline runs: knockout drop ~0)
    'nativeInit': dict(mamba_init='native'),      # Mamba keeps its own init (inherited init: ~50x weaker output)
    'noInnerSkip': dict(inner_skip=False),        # no skip_scale*x bypass: information must pass through Mamba
    'nativeInitNoInnerSkip': dict(mamba_init='native', inner_skip=False),
}
for _tag, _ov in _ABLATIONS.items():
    _variant(f'nnUNetTrainerPVM3D_S_{_tag}', S_REF, f'Ablation of PVM-3D-S: {_ov}', **_ov)


# A7 training recipe / deep supervision
class nnUNetTrainerPVM3D_S_noDS(NoDeepSupervisionMixin, S_REF):
    """A7: PVM-3D-S without deep supervision."""


class nnUNetTrainerPVM3D_S_AdamWCos50(AdamWCosine50Mixin, NoDeepSupervisionMixin, S_REF):
    """A7: the 2D paper's optimizer/schedule (AdamW, cosine T_max=50), no DS, nnU-Net data pipeline."""


# ---- Gate B: the design the main benchmark uses ---------------------------
# Set FINAL_DESIGN at the end of S2 to the winning ablation settings (e.g.
# dict(scan='dd')), then never edit it again: E4-E8 runs, the Pareto family and
# the parameter-matched tiny U-Nets all derive from it, and on_train_start
# refuses to resume any run whose config changed. Empty = PVM-3D-S defaults.
FINAL_DESIGN = dict()
for _size, _bw in (('T', 8), ('S', 16), ('B', 32)):
    _variant(f'nnUNetTrainerPVM3DFinal_{_size}', nnUNetTrainerPVM3D,
             f'PVM-3D-{_size}, final (Gate B) design: {FINAL_DESIGN or "defaults"}', base_width=_bw, **FINAL_DESIGN)
FS = globals()['nnUNetTrainerPVM3DFinal_S']

# ---- E7 data efficiency / E8 cross-modality (final design) ----------------
for _frac in (10, 25, 50):
    globals()[f'nnUNetTrainerPVM3DFinal_S_frac{_frac}'] = type(
        f'nnUNetTrainerPVM3DFinal_S_frac{_frac}', (DataFractionMixin, FS), {'FRACTION': _frac / 100})


class nnUNetTrainerPVM3DFinal_S_CTOnly(CTOnlyMixin, FS):
    """E8: AMOS CT-only training, zero-shot MRI evaluation."""


# ---- smoke tests -----------------------------------------------------------
class nnUNetTrainerPVM3D_S_smoke(SmokeMixin, S):
    """2 epochs x 5 iterations; for laptop / start-of-block sanity checks."""


class nnUNetTrainerPVM3D_S_dd_smoke(SmokeMixin, globals()['nnUNetTrainerPVM3D_S_dd']):
    """Smoke test of the direction-diverse scan path."""
