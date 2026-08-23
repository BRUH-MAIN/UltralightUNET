"""Post-training INT8 dynamic quantization: model size, CPU latency, DSC.

Why CPU. mamba_ssm's fused kernel is CUDA-only with no CPU fallback, and
torch.quantization's dynamic-quantized kernels are CPU-only -- so there is no
single device this can run on using the production model as-is. This instead
builds the CPU-compatible model by swapping `mamba_ssm.Mamba` for
`models.mamba_pytorch.Mamba` (the pure-PyTorch oracle), using the exact same
monkey-patch `tests/test_mamba_equivalence.py` already uses to prove the two
backends are numerically equivalent -- and loads a mamba_ssm-trained
checkpoint straight into it, since their state_dicts share identical keys
(that equivalence is what that test verifies).

`torch.quantization.quantize_dynamic` targets `nn.Linear` (its best-supported
case, no calibration data needed) -- but only the ones reached through a
normal `forward()` call: `PVMLayer.proj` and `Channel_Att_Bridge.att1-5`
(split_att='fc'). Mamba's own in_proj/x_proj/dt_proj/out_proj are *excluded*:
`models/mamba_pytorch.py` reads `self.in_proj.weight` as a raw tensor for a
manual matmul rather than calling `self.in_proj(x)`, and a dynamic-quantized
`nn.Linear` exposes `.weight` as a method, not a tensor -- `quantize_dynamic`
on those raises `TypeError: unsupported operand type(s) for @` at the first
forward pass. Since those four projections hold most of each PVMLayer's
parameters, this is a real scope limit on the size reduction below, not a
cosmetic one -- see the summary. Conv2d layers stay fp32 throughout either
way; dynamic quantization support for Conv2d is far less standard.

Usage:
    python scripts/quantize_eval.py --dataset ISIC2017
    python scripts/quantize_eval.py --dataset ISIC2017 --n-latency 100
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from sklearn.metrics import confusion_matrix

from configs.config_setting import setting_config
from loader import isic_loader
from scripts.cross_eval import find_best_checkpoint


def build_cpu_model(model_cfg):
    """UltraLight_VM_UNet with PVMLayer's Mamba swapped for the pure-PyTorch
    oracle -- same monkey-patch as tests/test_mamba_equivalence.py's
    _build_model, so this is a tested substitution, not a new assumption."""
    import models.UltraLight_VM_UNet as model_module
    from models.mamba_pytorch import Mamba as MambaRef
    original = model_module.Mamba
    model_module.Mamba = MambaRef
    try:
        return model_module.UltraLight_VM_UNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            c_list=model_cfg['c_list'],
            split_att=model_cfg['split_att'],
            bridge=model_cfg['bridge'])
    finally:
        model_module.Mamba = original


def load_weights(model, ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    thop_keys = [k for k in state if k.endswith(('total_ops', 'total_params'))]
    for k in thop_keys:
        del state[k]
    model.load_state_dict(state)
    model.eval()
    return model


def state_dict_size_mb(model):
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def bench_latency(model, sample, warmup=5, n=50):
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        t0 = time.perf_counter()
        for _ in range(n):
            model(sample)
        return (time.perf_counter() - t0) / n * 1000  # ms/image


def evaluate_dsc(model, test_dataset, threshold=0.5, log_every=100):
    preds, gts = [], []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            img, msk = test_dataset[i]
            out = model(img.unsqueeze(0).float()).squeeze().numpy()
            preds.append(out.reshape(-1))
            gts.append(msk.squeeze().numpy().reshape(-1))
            if (i + 1) % log_every == 0 or i + 1 == len(test_dataset):
                print(f"  {i + 1}/{len(test_dataset)}", end="\r")
    print()
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    y_pre = (preds >= threshold).astype(int)
    y_true = (gts >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    dsc = float(2 * TP) / float(2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    return dsc, cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='ISIC2017')
    ap.add_argument('--n-latency', type=int, default=50, help='timed forward passes for latency')
    args = ap.parse_args()

    config = setting_config
    ckpt_path, run_dir = find_best_checkpoint(args.dataset)
    print(f"checkpoint: {os.path.relpath(ckpt_path, REPO_ROOT)}\n")

    data_path = os.path.join(REPO_ROOT, 'data', args.dataset) + os.sep
    test_dataset = isic_loader(path_Data=data_path, train=False, Test=True)
    sample = test_dataset[0][0].unsqueeze(0).float()

    print("=== fp32 (CPU) ===")
    model = build_cpu_model(config.model_config)
    load_weights(model, ckpt_path)
    fp32_size = state_dict_size_mb(model)
    fp32_latency = bench_latency(model, sample, n=args.n_latency)
    print(f"size: {fp32_size:.3f} MB   latency: {fp32_latency:.2f} ms/image")
    print("evaluating DSC on the full test set...")
    fp32_dsc, fp32_cm = evaluate_dsc(model, test_dataset)
    print(f"DSC: {fp32_dsc:.4f}\n")

    print("=== INT8 dynamic-quantized (Linear layers outside Mamba, CPU) ===")
    # See the module docstring: Mamba's own Linear projections are read via
    # raw .weight tensor access, incompatible with quantize_dynamic's
    # module-replacement mechanism, so they're named-excluded here.
    quantizable = {name for name, mod in model.named_modules()
                   if isinstance(mod, torch.nn.Linear) and '.mamba.' not in name}
    excluded_params = sum(p.numel() for name, p in model.named_parameters()
                          if '.mamba.' in name and ('in_proj' in name or 'out_proj' in name
                                                     or 'x_proj' in name or 'dt_proj' in name))
    print(f"quantizing {len(quantizable)} Linear layers ({', '.join(sorted(quantizable))})")
    print(f"excluded (Mamba internals, raw-tensor access): {excluded_params:,} params")
    q_model = torch.quantization.quantize_dynamic(model, quantizable, dtype=torch.qint8)
    q_size = state_dict_size_mb(q_model)
    q_latency = bench_latency(q_model, sample, n=args.n_latency)
    print(f"size: {q_size:.3f} MB   latency: {q_latency:.2f} ms/image")
    print("evaluating DSC on the full test set...")
    q_dsc, q_cm = evaluate_dsc(q_model, test_dataset)
    print(f"DSC: {q_dsc:.4f}\n")

    print("=== summary ===")
    print(f"{'':16s} {'size (MB)':>10s} {'latency (ms)':>13s} {'DSC':>8s}")
    print(f"{'fp32':16s} {fp32_size:10.3f} {fp32_latency:13.2f} {fp32_dsc:8.4f}")
    print(f"{'int8 dynamic':16s} {q_size:10.3f} {q_latency:13.2f} {q_dsc:8.4f}")
    print(f"\nsize reduction: {(1 - q_size/fp32_size)*100:.1f}%   "
         f"latency change: {(q_latency/fp32_latency - 1)*100:+.1f}%   "
         f"DSC delta: {q_dsc - fp32_dsc:+.4f}")


if __name__ == "__main__":
    main()
