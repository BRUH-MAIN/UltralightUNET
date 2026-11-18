"""Shape / range / size checks for the plain U-Net baseline (models/UNet.py)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.UNet import UNet


def test_output_shape_and_range():
    model = UNet(num_classes=1, input_channels=3, base=16).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 256, 256))
    assert out.shape == (2, 1, 256, 256)
    # utils.BceDiceLoss feeds this straight into nn.BCELoss, so it must be a probability
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_param_count_close_to_papers_unet():
    # paper's Table 1 U-Net: 2.009 M params. Ours is documented at 1.944 M.
    n = sum(p.numel() for p in UNet(base=16).parameters())
    assert 1.90e6 < n < 2.05e6, n
