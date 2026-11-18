"""Parameters and MACs for every benchmarked model, Mamba included.

thop -- what utils3d.cal_params_flops3d uses -- cannot see inside mamba_ssm:
the in/x/dt/out projections are called as raw matmuls (not nn.Linear modules)
and the depthwise conv1d + selective scan are fused CUDA kernels. For a
lightweight-model paper the efficiency column is a headline number, so it is
counted properly here:

  1. torch.utils.flop_counter.FlopCounterMode counts every conv/matmul/attention
     op at the aten level (FLOPs = 2 x MACs), with per-module attribution.
  2. For each Mamba module, the dispatcher's count is *replaced* by an analytic
     count of the whole block (same formula for both backends, so mamba_ssm and
     the pure-PyTorch oracle give identical totals):
        in_proj  B*L*d*2*Di      conv1d  B*L*Di*k      x_proj  B*L*Di*(R+2N)
        dt_proj  B*L*R*Di        out_proj B*L*Di*d
        selective scan 9*B*L*Di*N + 2*B*L*Di   (VMamba's flops_selective_scan_ref
        convention; counted as MAC-equivalents, as VMamba/fvcore do)
     where B is the batch the Mamba actually sees (x branches x scans for PVM).

Reported: inference params (DS-only heads excluded), GMACs per forward of one
patch (the plans' patch size, batch 1), and the share of GMACs inside Mamba.

Usage:
    source scripts/cluster/env.sh
    python scripts/flops3d.py -d 4 -tr nnUNetTrainerPVM3D_T nnUNetTrainerPVM3D_S nnUNetTrainer
    python scripts/flops3d.py --plans-file plans/Dataset220.json --dataset-json ... -tr ...
    python scripts/flops3d.py --legacy      # the 64^3 Hippocampus model, vs. thop
"""

import argparse
import json
import math
import os
import sys
import warnings

import torch

warnings.filterwarnings('ignore')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _is_mamba(m):
    return type(m).__name__ == 'Mamba' and hasattr(m, 'd_inner') and hasattr(m, 'd_state')


def mamba_block_macs(m, B, L):
    d, Di, N, k, R = m.d_model, m.d_inner, m.d_state, m.d_conv, m.dt_rank
    proj = B * L * (d * 2 * Di + Di * (R + 2 * N) + R * Di + Di * d)
    conv = B * L * Di * k
    scan = 9 * B * L * Di * N + 2 * B * L * Di
    return proj + conv + scan, conv + scan


@torch.no_grad()
def count_macs(net, x):
    from torch.utils.flop_counter import FlopCounterMode

    root = type(net).__name__
    calls = []
    hooks = []
    for name, m in net.named_modules():
        if _is_mamba(m):
            hooks.append(m.register_forward_hook(
                lambda mod, inp, out, name=name: calls.append((name, mod, inp[0].shape[0], inp[0].shape[1]))))
    try:
        with FlopCounterMode(display=False, depth=None) as fc:
            net(x)
    finally:
        for h in hooks:
            h.remove()
    counts = fc.get_flop_counts()
    total_flops = sum(counts['Global'].values())
    seen = set()
    mamba_dispatch_flops = 0
    mamba_macs = mamba_kernel_macs = 0
    for name, mod, B, L in calls:
        key = f'{root}.{name}'
        if key not in seen:  # module-level counts already sum over repeated calls
            mamba_dispatch_flops += sum(counts.get(key, {}).values())
            seen.add(key)
        a, b = mamba_block_macs(mod, B, L)
        mamba_macs += a
        mamba_kernel_macs += b
    other_macs = (total_flops - mamba_dispatch_flops) / 2
    return dict(gmacs=(other_macs + mamba_macs) / 1e9, gmacs_mamba=mamba_macs / 1e9,
                gmacs_mamba_kernels=mamba_kernel_macs / 1e9, n_mamba_calls=len(calls))


def thop_macs(net, x):
    import copy
    from thop import profile
    macs, _ = profile(copy.deepcopy(net), inputs=(x,), verbose=False)
    return macs / 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--dataset')
    ap.add_argument('-tr', '--trainers', nargs='+', default=['nnUNetTrainerPVM3D_S'])
    ap.add_argument('-p', '--plans', default='nnUNetPlans')
    ap.add_argument('-c', '--configuration', default='3d_fullres')
    ap.add_argument('--plans-file')
    ap.add_argument('--dataset-json')
    ap.add_argument('--patch', type=int, nargs=3, help='override the plans patch size')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--legacy', action='store_true', help='legacy 64^3 Hippocampus PVM-3D + StandardUNet3D')
    ap.add_argument('--thop', action='store_true', help='also report thop MACs for comparison')
    ap.add_argument('-o', '--out', help='append JSON lines here')
    args = ap.parse_args()

    impl = 'mamba_ssm' if args.device.startswith('cuda') else 'pytorch'
    rows = []
    if args.legacy:
        from pvm3d.network import PVMUNet3D, legacy_config
        from models.StandardUNet3D import StandardUNet3D
        from pvm3d.nnunet_glue import count_inference_params
        x = torch.randn(1, 1, 64, 64, 64, device=args.device)
        for label, net in (('PVM-3D legacy', PVMUNet3D(1, 3, mamba_impl=impl, **legacy_config())),
                           ('StandardUNet3D', StandardUNet3D(3, 1))):
            net = net.to(args.device).eval()
            r = dict(model=label, dataset='Hippocampus-64^3', patch=[64, 64, 64], params=count_inference_params(net),
                     **count_macs(net, x))
            r['gmacs_thop'] = thop_macs(net, x)
            rows.append(r)
    else:
        from pvm3d.zoo import build, describe, load_plans
        pm, cm, dj, n_in, n_out = load_plans(args.dataset, args.plans, args.configuration, args.plans_file,
                                             args.dataset_json)
        patch = args.patch or list(cm.patch_size)
        x = torch.randn(1, n_in, *patch, device=args.device)
        for tr in args.trainers:
            net = build(tr, pm, cm, n_in, n_out, deep_supervision=False, mamba_impl=impl).to(args.device).eval()
            r = dict(model=tr, dataset=pm.dataset_name, plans=pm.plans_name, patch=patch, n_in=n_in, n_out=n_out,
                     params=describe(net).params, **count_macs(net, x))
            if args.thop:
                r['gmacs_thop'] = thop_macs(net, x)
            rows.append(r)
            del net
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()

    for r in rows:
        extra = f"  thop {r['gmacs_thop']:.4f}" if 'gmacs_thop' in r else ''
        print(f"{r['model']:<40} {r['dataset']:<28} patch {tuple(r['patch'])}  params {r['params']:>11,}  "
              f"GMACs {r['gmacs']:.4f} (Mamba {r['gmacs_mamba']:.4f}, of which conv1d+scan "
              f"{r['gmacs_mamba_kernels']:.4f}){extra}")
    if args.out:
        with open(args.out, 'a') as fh:
            for r in rows:
                fh.write(json.dumps(r) + '\n')


if __name__ == '__main__':
    main()
