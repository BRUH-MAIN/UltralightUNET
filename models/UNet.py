"""Plain 2D U-Net baseline (Ronneberger et al., 2015), for comparison against UltraLight VM-UNet.

The UltraLight paper's Table 1 reports a U-Net of 2.009 M params / 3.224 GFLOPs at 256x256. The
paper does not publish that U-Net's code, so this is a standard re-implementation sized to land in
the same range: 4 downsamplings, widths [16, 32, 64, 128, 256], two 3x3 conv-BN-ReLU per stage,
2x2 max-pool down, 2x2 transposed-conv up, skip connections by concatenation.

    this model        1.944 M params   3.467 GFLOPs   (thop, 256x256)
    paper's U-Net     2.009 M params   3.224 GFLOPs

i.e. -3.2% params, +7.5% FLOPs -- close, not identical. The bilinear-upsample variant was measured
too (1.813 M / 3.108 G) and is further off on params, so transposed conv is used.

Like UltraLight_VM_UNet.forward, returns sigmoid probabilities, because utils.BceDiceLoss feeds the
output straight into nn.BCELoss.
"""

import torch
import torch.nn as nn


def _double_conv(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, base=16, depth=4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.encoders = nn.ModuleList(
            [_double_conv(input_channels if i == 0 else ch[i - 1], ch[i]) for i in range(depth + 1)])
        self.pool = nn.MaxPool2d(2)
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(depth, 0, -1)])
        self.decoders = nn.ModuleList(
            [_double_conv(ch[i], ch[i - 1]) for i in range(depth, 0, -1)])
        self.head = nn.Conv2d(ch[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else self.pool(x))
            skips.append(x)
        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips[:-1])):
            x = dec(torch.cat([skip, up(x)], dim=1))
        return torch.sigmoid(self.head(x))
