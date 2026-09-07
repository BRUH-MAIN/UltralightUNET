"""Standard 3D U-Net, sized the way this task is actually trained in practice
(MONAI/nnU-Net-style channel progression 16-32-64-128-256, double-conv blocks,
4 downsampling stages -- matched in depth to the PVM-3D prototype for a fair
comparison), not the original 19M-param Cicek et al. sizing, which nobody
runs for a task this small.
"""
import torch
from torch import nn


class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.InstanceNorm3d(out_ch), nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.InstanceNorm3d(out_ch), nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class StandardUNet3D(nn.Module):
    def __init__(self, num_classes=3, input_channels=1, c_list=(16, 32, 64, 128, 256)):
        super().__init__()
        self.enc1 = DoubleConv3D(input_channels, c_list[0])
        self.enc2 = DoubleConv3D(c_list[0], c_list[1])
        self.enc3 = DoubleConv3D(c_list[1], c_list[2])
        self.enc4 = DoubleConv3D(c_list[2], c_list[3])
        self.bottleneck = DoubleConv3D(c_list[3], c_list[4])
        self.pool = nn.MaxPool3d(2)

        self.up4 = nn.ConvTranspose3d(c_list[4], c_list[3], 2, stride=2)
        self.dec4 = DoubleConv3D(c_list[3] * 2, c_list[3])
        self.up3 = nn.ConvTranspose3d(c_list[3], c_list[2], 2, stride=2)
        self.dec3 = DoubleConv3D(c_list[2] * 2, c_list[2])
        self.up2 = nn.ConvTranspose3d(c_list[2], c_list[1], 2, stride=2)
        self.dec2 = DoubleConv3D(c_list[1] * 2, c_list[1])
        self.up1 = nn.ConvTranspose3d(c_list[1], c_list[0], 2, stride=2)
        self.dec1 = DoubleConv3D(c_list[0] * 2, c_list[0])
        self.final = nn.Conv3d(c_list[0], num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)
