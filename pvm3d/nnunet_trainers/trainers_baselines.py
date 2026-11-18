"""nnU-Net v2 trainers for the baselines we train ourselves (roadmap §4).

* nnUNetTrainerTinyUNet_match{T,S,B}: nnU-Net's own plans-derived U-Net with
  per-stage widths scaled (nnU-Net's width ratios x a fractional base) so its
  inference-parameter count matches the final-design PVM-3D-{T,S,B} on the same plans. This is the control for
  "is a tiny CNN just as good?" -- the question nnU-Net Revisited's P1/P2
  pitfalls say reviewers will ask.
* Data-fraction (E7) and AMOS CT-only (E8) variants of nnU-Net (org.) and the
  tiny U-Net, so every model in those experiments sees the identical subset.

External architectures (LightM-UNet, SlimUNETR, SegFormer3D, SegMamba) are in
trainers_external.py.
"""

import os
import sys

# realpath: nnU-Net may import this file through the symlink scripts/cluster/install_trainers.py
# creates inside site-packages; the repo root must come from the real location.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer  # noqa: E402

from pvm3d.nnunet_glue import (CTOnlyMixin, DataFractionMixin, FrequentCheckpointMixin, SmokeMixin,  # noqa: E402
                               build_pvm_from_plans, build_tiny_unet_matched, count_inference_params)
from pvm3d.nnunet_trainers.trainers_pvm3d import (nnUNetTrainerPVM3DFinal_B,  # noqa: E402
                                                 nnUNetTrainerPVM3DFinal_S, nnUNetTrainerPVM3DFinal_T)


class _TinyUNetMatched(FrequentCheckpointMixin, nnUNetTrainer):
    MATCH = None  # a nnUNetTrainerPVM3D subclass

    @classmethod
    def build_network_architecture(cls, plans_manager, configuration_manager, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        ref = build_pvm_from_plans({**cls.MATCH.PVM, 'mamba_impl': 'pytorch'}, configuration_manager,
                                   num_input_channels, num_output_channels, enable_deep_supervision=False)
        return build_tiny_unet_matched(count_inference_params(ref), configuration_manager, num_input_channels,
                                       num_output_channels, enable_deep_supervision)


class nnUNetTrainerTinyUNet_matchT(_TinyUNetMatched):
    MATCH = nnUNetTrainerPVM3DFinal_T


class nnUNetTrainerTinyUNet_matchS(_TinyUNetMatched):
    MATCH = nnUNetTrainerPVM3DFinal_S


class nnUNetTrainerTinyUNet_matchB(_TinyUNetMatched):
    MATCH = nnUNetTrainerPVM3DFinal_B


class nnUNetTrainerTinyUNet_matchS_smoke(SmokeMixin, nnUNetTrainerTinyUNet_matchS):
    pass


for _frac in (10, 25, 50):
    for _base in (nnUNetTrainer, nnUNetTrainerTinyUNet_matchS):
        _name = f'{_base.__name__}_frac{_frac}'
        globals()[_name] = type(_name, (DataFractionMixin, _base), {'FRACTION': _frac / 100})


class nnUNetTrainer_CTOnly(CTOnlyMixin, nnUNetTrainer):
    pass


class nnUNetTrainerTinyUNet_matchS_CTOnly(CTOnlyMixin, nnUNetTrainerTinyUNet_matchS):
    pass
