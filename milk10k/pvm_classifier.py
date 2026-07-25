"""Lightweight Mamba classifier for MILK10k, built on the replicated PVM Layer.

The contribution of Phase 2. It reuses the *exact* ``PVMLayer`` (parallel Vision
Mamba) from the replicated UltraLight VM-UNet -- imported, not copied, from
``models.UltraLight_VM_UNet`` -- and re-purposes the paper's encoder, which was
only ever used for segmentation, as a classification backbone.

Three model modes, selected by ``modality``:

  'derm' / 'clin'  single-modality classifier: one PVM encoder + a linear head.
                   This is milestone 1 and the ablation baseline.
  'both'           the multimodal model: one encoder per view, fused, then the
                   head. Two fusion variants are provided (see ``fusion``); which
                   one wins is an empirical question settled after milestone 1.

The encoder mirrors UltraLight VM-UNet's contracting path exactly: three conv
stages then three PVM stages, halving resolution each step, so at 256x256 the PVM
layers see sequences of 256 / 64 / 64 tokens -- the same tiny scans the pure-
PyTorch Mamba was validated on.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from models.UltraLight_VM_UNet import PVMLayer  # reused from the replicated model


class PVMEncoder(nn.Module):
    """UltraLight VM-UNet's encoder, ending in a global-pooled feature vector."""

    def __init__(self, input_channels=3, c_list=(8, 16, 24, 32, 48, 64)):
        super().__init__()
        self.out_dim = c_list[5]
        self.encoder1 = nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1)
        self.encoder4 = PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        self.encoder5 = PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        self.encoder6 = PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.ebn5 = nn.GroupNorm(4, c_list[4])

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))   # H/2
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))  # H/4
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))  # H/8
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2))  # H/16
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2))  # H/32
        out = F.gelu(self.encoder6(out))                                 # H/32, c5
        return out                                                       # (B, c5, H/32, W/32)


class CrossModalMambaFusion(nn.Module):
    """Fuse two modality feature maps by running one shared PVM Layer over their
    concatenated token sequence, so each view's tokens can attend across to the
    other through the shared state-space scan. Output is pooled per modality and
    summed. Kept simple on purpose; upgraded only if it beats concat fusion."""

    def __init__(self, dim):
        super().__init__()
        self.pvm = PVMLayer(input_dim=dim, output_dim=dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, fa, fb):
        B, C, H, W = fa.shape
        # stack the two maps side by side along width -> one (B, C, H, 2W) field,
        # so the PVM scan crosses the modality boundary.
        joint = torch.cat([fa, fb], dim=3)
        joint = self.pvm(joint)
        pooled = joint.mean(dim=(2, 3))          # (B, C)
        return self.norm(pooled)


class MILK10kClassifier(nn.Module):
    def __init__(self, num_classes=11, modality='derm', input_channels=3,
                 c_list=(8, 16, 24, 32, 48, 64), fusion='cross_mamba'):
        super().__init__()
        self.modality = modality
        self.fusion = fusion
        dim = c_list[5]

        if modality in ('derm', 'clin'):
            self.encoder = PVMEncoder(input_channels, c_list)
            head_in = dim
        elif modality == 'both':
            self.encoder_derm = PVMEncoder(input_channels, c_list)
            self.encoder_clin = PVMEncoder(input_channels, c_list)
            if fusion == 'cross_mamba':
                self.fuse = CrossModalMambaFusion(dim)
                head_in = dim
            elif fusion == 'concat':
                head_in = dim * 2
            else:
                raise ValueError(f'unknown fusion {fusion!r}')
        else:
            raise ValueError(f'unknown modality {modality!r}')

        self.head = nn.Sequential(nn.LayerNorm(head_in), nn.Linear(head_in, num_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # same initialisation policy as UltraLight VM-UNet (see its _init_weights)
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """x: dict of modality -> (B, 3, H, W). Returns logits (B, num_classes)."""
        if self.modality in ('derm', 'clin'):
            feat = self.encoder(x[self.modality]).mean(dim=(2, 3))       # global avg pool
        else:
            fd = self.encoder_derm(x['derm'])
            fc = self.encoder_clin(x['clin'])
            if self.fusion == 'cross_mamba':
                feat = self.fuse(fd, fc)
            else:
                feat = torch.cat([fd.mean(dim=(2, 3)), fc.mean(dim=(2, 3))], dim=1)
        return self.head(feat)
