"""The batched PVMLayer.forward must match upstream's four-call form.

PVMLayer.forward stacks the four parallel branches onto the batch axis and makes
one Mamba call, where upstream makes four. That is only legitimate because Mamba
treats the batch dimension as independent. This asserts it empirically, at the
real layer shapes, for the whole model as well as for individual layers.

Any divergence here means the speed patch changed the model, not just its cost.
"""

import pytest
import torch

from models.UltraLight_VM_UNet import PVMLayer, UltraLight_VM_UNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (input_dim, output_dim, spatial) for the six PVM layers at a 256x256 input
LAYERS = [
    (24, 32, 32),   # encoder4, L=1024
    (32, 48, 16),   # encoder5, L=256
    (48, 64, 8),    # encoder6, L=64
    (64, 48, 8),    # decoder1, L=64
    (48, 32, 8),    # decoder2, L=64
    (32, 24, 16),   # decoder3, L=256
]


@pytest.mark.parametrize("in_dim,out_dim,hw", LAYERS,
                         ids=[f"{i}->{o}@{s}x{s}" for i, o, s in LAYERS])
def test_batched_matches_reference(in_dim, out_dim, hw):
    torch.manual_seed(0)
    layer = PVMLayer(input_dim=in_dim, output_dim=out_dim).to(DEVICE).eval()
    x = torch.randn(2, in_dim, hw, hw, device=DEVICE)

    with torch.no_grad():
        fast = layer(x)
        ref = layer._forward_reference(x)

    err = (fast - ref).abs().max().item()
    scale = max(ref.abs().max().item(), 1.0)
    # Not bitwise: a matmul over 4B rows can reduce in a different order than four
    # over B rows. Anything beyond fp32 round-off would mean a real change.
    assert err / scale < 1e-5, f"max|diff| = {err} (scale {scale})"


def test_gradients_match():
    """The batching must not change what the optimiser sees, either."""
    torch.manual_seed(0)
    layer = PVMLayer(input_dim=24, output_dim=32).to(DEVICE)
    x = torch.randn(2, 24, 32, 32, device=DEVICE)

    grads = []
    for fn in (layer.forward, layer._forward_reference):
        layer.zero_grad()
        fn(x).pow(2).sum().backward()
        grads.append({n: p.grad.detach().clone() for n, p in layer.named_parameters()})

    for name in grads[0]:
        err = (grads[0][name] - grads[1][name]).abs().max().item()
        scale = max(grads[1][name].abs().max().item(), 1.0)
        assert err / scale < 1e-4, f"grad {name}: max|diff| = {err} (scale {scale})"


def test_full_model_matches_reference():
    """End to end: swapping every PVMLayer back to the upstream form changes nothing."""
    torch.manual_seed(0)
    model = UltraLight_VM_UNet().to(DEVICE).eval()
    x = torch.randn(2, 3, 256, 256, device=DEVICE)

    with torch.no_grad():
        fast = model(x)
        for m in model.modules():
            if isinstance(m, PVMLayer):
                m.forward = m._forward_reference
        ref = model(x)

    err = (fast - ref).abs().max().item()
    assert err < 1e-5, f"full model max|diff| = {err}"
