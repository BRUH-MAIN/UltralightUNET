"""PVM-3D: volumetric extension of UltraLight VM-UNet.

Promoted from the architecture-planning prototype at
``docs/3d_pvm_prototype/model3d.py`` -- see ``docs/3d_extension_plan.md`` for the
design rationale (why 4 downsampling stages instead of the 2D model's 5, why
``c_list`` has 5 entries instead of 6). This file differs from that prototype in
exactly the way ``models/UltraLight_VM_UNet.py`` differs from
``models/mamba_pytorch.py``: it imports the real ``mamba_ssm`` CUDA kernel
instead of the pure-PyTorch oracle, and it carries the batched-Mamba-call
performance patch (one call over the 4 concatenated branches, not four
sequential calls to the same module) that the 2D model documents in
``PVMLayer.forward``.

``PVMLayer`` itself is untouched from the 2D file, copied verbatim -- it
flattens ``x.shape[2:]`` generically, so the 4-way channel split + shared
Mamba works on a 3D token sequence exactly as it does on a 2D one.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F

try:  # timm >= 0.9
    from timm.layers import trunc_normal_
except ImportError:  # timm < 0.9, where timm.layers is not yet the canonical path
    from timm.models.layers import trunc_normal_

import mamba_ssm
from mamba_ssm import Mamba

MAMBA_BACKEND = f"mamba_ssm {mamba_ssm.__version__}"


class PVMLayer(nn.Module):
    """Verbatim copy of models/UltraLight_VM_UNet.py's PVMLayer. Dimension-agnostic:
    nothing here changes for 3D input."""
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)

        # batched (see models/UltraLight_VM_UNet.py:66 for the full rationale):
        # one Mamba call over the 4 branches stacked on the batch axis, instead
        # of 4 sequential calls to the same module.
        xs = torch.cat([x1, x2, x3, x4], dim=0)
        ys = self.mamba(xs)
        y1, y2, y3, y4 = torch.chunk(ys, 4, dim=0)

        x_mamba1 = y1 + self.skip_scale * x1
        x_mamba2 = y2 + self.skip_scale * x2
        x_mamba3 = y3 + self.skip_scale * x3
        x_mamba4 = y4 + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out


class Channel_Att_Bridge3D(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        self.split_att = split_att
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.att1 = nn.Linear(c_list_sum, c_list[0])
        self.att2 = nn.Linear(c_list_sum, c_list[1])
        self.att3 = nn.Linear(c_list_sum, c_list[2])
        self.att4 = nn.Linear(c_list_sum, c_list[3])
        self.sigmoid = nn.Sigmoid()

    def forward(self, t1, t2, t3, t4):
        att = torch.cat((self.avgpool(t1), self.avgpool(t2), self.avgpool(t3), self.avgpool(t4)), dim=1)
        att = self.get_all_att(att.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1))
        att1 = self.sigmoid(self.att1(att)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t1)
        att2 = self.sigmoid(self.att2(att)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t2)
        att3 = self.sigmoid(self.att3(att)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t3)
        att4 = self.sigmoid(self.att4(att)).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1).expand_as(t4)
        return att1, att2, att3, att4


class Spatial_Att_Bridge3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv3d = nn.Sequential(nn.Conv3d(2, 1, 7, stride=1, padding=9, dilation=3), nn.Sigmoid())

    def forward(self, t1, t2, t3, t4):
        out = []
        for t in (t1, t2, t3, t4):
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            out.append(self.shared_conv3d(att))
        return out


class SC_Att_Bridge3D(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()
        self.catt = Channel_Att_Bridge3D(c_list, split_att)
        self.satt = Spatial_Att_Bridge3D()

    def forward(self, t1, t2, t3, t4):
        r1, r2, r3, r4 = t1, t2, t3, t4
        s1, s2, s3, s4 = self.satt(t1, t2, t3, t4)
        t1, t2, t3, t4 = s1 * t1, s2 * t2, s3 * t3, s4 * t4
        r1_, r2_, r3_, r4_ = t1, t2, t3, t4
        t1, t2, t3, t4 = t1 + r1, t2 + r2, t3 + r3, t4 + r4
        c1, c2, c3, c4 = self.catt(t1, t2, t3, t4)
        t1, t2, t3, t4 = c1 * t1, c2 * t2, c3 * t3, c4 * t4
        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_


class UltraLight_VM_UNet_3D(nn.Module):
    """4 downsampling stages (2D model has 5), sized for small-ROI volumes such
    as 64^3 Hippocampus patches rather than BraTS-scale input. c_list has 5
    entries (3 plain-conv + 2 PVM: one pooled, one unpooled bottleneck) -- one
    fewer PVM stage than the 2D model's 3, matching the one fewer downsampling
    level. See docs/3d_extension_plan.md for why.
    """
    def __init__(self, num_classes=3, input_channels=1, c_list=(8, 16, 24, 32, 48),
                 split_att='fc', bridge=True):
        super().__init__()
        self.bridge = bridge

        self.encoder1 = nn.Sequential(nn.Conv3d(input_channels, c_list[0], 3, padding=1))
        self.encoder2 = nn.Sequential(nn.Conv3d(c_list[0], c_list[1], 3, padding=1))
        self.encoder3 = nn.Sequential(nn.Conv3d(c_list[1], c_list[2], 3, padding=1))
        self.encoder4 = nn.Sequential(PVMLayer(c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(PVMLayer(c_list[3], c_list[4]))  # bottleneck, unpooled

        if bridge:
            self.scab = SC_Att_Bridge3D(c_list, split_att)
            print('SC_Att_Bridge3D was used')

        self.decoder1 = nn.Sequential(PVMLayer(c_list[4], c_list[3]))
        self.decoder2 = nn.Sequential(nn.Conv3d(c_list[3], c_list[2], 3, padding=1))
        self.decoder3 = nn.Sequential(nn.Conv3d(c_list[2], c_list[1], 3, padding=1))
        self.decoder4 = nn.Sequential(nn.Conv3d(c_list[1], c_list[0], 3, padding=1))

        self.ebn1 = nn.GroupNorm(4, c_list[0])
        self.ebn2 = nn.GroupNorm(4, c_list[1])
        self.ebn3 = nn.GroupNorm(4, c_list[2])
        self.ebn4 = nn.GroupNorm(4, c_list[3])
        self.dbn1 = nn.GroupNorm(4, c_list[3])
        self.dbn2 = nn.GroupNorm(4, c_list[2])
        self.dbn3 = nn.GroupNorm(4, c_list[1])
        self.dbn4 = nn.GroupNorm(4, c_list[0])

        self.final = nn.Conv3d(c_list[0], num_classes, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
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
        out = F.gelu(F.max_pool3d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        out = F.gelu(F.max_pool3d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        out = F.gelu(F.max_pool3d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        out = F.gelu(F.max_pool3d(self.ebn4(self.encoder4(out)), 2, 2))
        t4 = out

        if self.bridge:
            t1, t2, t3, t4 = self.scab(t1, t2, t3, t4)

        out = F.gelu(self.encoder5(out))  # bottleneck PVM, no pool

        out4 = F.gelu(self.dbn1(self.decoder1(out)))
        out4 = torch.add(out4, t4)
        out3 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out4)), scale_factor=(2, 2, 2),
                                     mode='trilinear', align_corners=True))
        out3 = torch.add(out3, t3)
        out2 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out3)), scale_factor=(2, 2, 2),
                                     mode='trilinear', align_corners=True))
        out2 = torch.add(out2, t2)
        out1 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out2)), scale_factor=(2, 2, 2),
                                     mode='trilinear', align_corners=True))
        out1 = torch.add(out1, t1)
        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        return out0  # raw logits; softmax/CE applied outside, unlike the 2D model's sigmoid binary head
