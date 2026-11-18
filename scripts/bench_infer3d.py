"""Inference / training-step efficiency for every benchmarked model (roadmap E10, Table 2).

Measures the architecture-dependent part of deployment: nnU-Net's own
sliding-window inference (nnUNetPredictor.predict_sliding_window_return_logits,
Gaussian weighting, tile step 0.5, optional mirroring TTA) over one
preprocessed-space volume of the dataset's median size, with the network built
by its trainer exactly as at test time (pvm3d/zoo.py). Weights are random --
latency and memory do not depend on them.

Per model it reports
  * latency  (mean/std over --reps after a warm-up)
  * peak GPU memory (torch max_memory_allocated) and the FLARE-style
    memory-time area (MB*s, sampled every 5 ms from a background thread)
  * optionally (--train-step) ms/iteration and peak memory of one nnU-Net-style
    training step (plans patch + batch size, autocast + GradScaler, deep
    supervision on) -- the "can I train this on X" column
Precision: GPU runs use nnU-Net's default autocast (fp16) unless --fp32; CPU
runs are fp32 and swap mamba_ssm for the pure-PyTorch oracle (equivalent per
tests/test_mamba_equivalence.py), the only way to run Mamba without CUDA.

Usage:
    source scripts/cluster/env.sh
    python scripts/bench_infer3d.py -d 4 -tr nnUNetTrainerPVM3D_S nnUNetTrainer --reps 10 -o bench.jsonl
    python scripts/bench_infer3d.py -d 4 -tr nnUNetTrainerPVM3D_S --device cpu --threads 8
    python scripts/bench_infer3d.py -d 220 -tr ... --train-step --no-tta
"""

import argparse
import contextlib
import io
import json
import os
import sys
import threading
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


class MemSampler(threading.Thread):
    def __init__(self, dt=0.005):
        super().__init__(daemon=True)
        self.dt, self.samples, self.stop_evt = dt, [], threading.Event()

    def run(self):
        t0 = time.perf_counter()
        while not self.stop_evt.is_set():
            self.samples.append((time.perf_counter() - t0, torch.cuda.memory_allocated() / 2 ** 20))
            time.sleep(self.dt)

    def area_mb_s(self):
        if len(self.samples) < 2:
            return float('nan')
        t, m = np.array(self.samples).T
        return float(np.trapezoid(m, t) if hasattr(np, 'trapezoid') else np.trapz(m, t))


@contextlib.contextmanager
def maybe_disable_autocast(disable):
    """nnUNetPredictor hard-codes `torch.autocast(..., enabled=True)` on CUDA;
    for --fp32 runs it is replaced by a no-op for the duration of the call."""
    if not disable:
        yield
        return
    orig = torch.autocast
    torch.autocast = lambda *a, **k: contextlib.nullcontext()
    try:
        yield
    finally:
        torch.autocast = orig


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def bench_inference(net, pm, cm, dj, trainer, n_in, image_size, device, tta, reps, fp32):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=tta,
                           perform_everything_on_device=device.type == 'cuda', device=device, verbose=False,
                           allow_tqdm=False)
    pred.manual_initialization(net, pm, cm, None, dj, trainer, (0, 1, 2))
    x = torch.randn(n_in, *image_size)
    out = {}
    buf = io.StringIO()
    with maybe_disable_autocast(fp32), contextlib.redirect_stdout(buf), torch.no_grad():
        pred.predict_sliding_window_return_logits(x)  # warm-up (cudnn autotune, kernel load)
        times, peaks, areas = [], [], []
        for _ in range(reps):
            sampler = None
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                sampler = MemSampler()
                sampler.start()
            sync(device)
            t0 = time.perf_counter()
            pred.predict_sliding_window_return_logits(x)
            sync(device)
            times.append(time.perf_counter() - t0)
            if sampler is not None:
                sampler.stop_evt.set()
                sampler.join()
                peaks.append(torch.cuda.max_memory_allocated() / 2 ** 20)
                areas.append(sampler.area_mb_s())
    out['latency_s_mean'] = float(np.mean(times))
    out['latency_s_std'] = float(np.std(times))
    if peaks:
        out['infer_peak_mem_mb'] = float(np.max(peaks))
        out['infer_mem_time_mb_s'] = float(np.mean(areas))
    out['oom_fallback_to_cpu'] = 'unsuccessful' in buf.getvalue()
    out['n_tiles_input_shape'] = list(image_size)
    return out


def bench_train_step(trainer, pm, cm, n_in, n_out, device, reps):
    """One nnU-Net-shaped training iteration: plans batch/patch, autocast,
    GradScaler, deep supervision with nnU-Net's DS weights, CE loss."""
    import torch.nn.functional as F
    from pvm3d.zoo import build
    net = build(trainer, pm, cm, n_in, n_out, deep_supervision=True).to(device).train()
    bs, patch = cm.batch_size, list(cm.patch_size)
    opt = torch.optim.SGD(net.parameters(), lr=1e-2, momentum=0.99, nesterov=True)
    scaler = torch.amp.GradScaler('cuda')
    x = torch.randn(bs, n_in, *patch, device=device)
    y = torch.randint(0, n_out, (bs, *patch), device=device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            outs = net(x)
            outs = outs if isinstance(outs, (list, tuple)) else [outs]
            w = np.array([1 / 2 ** i for i in range(len(outs))])
            w[-1] = 0 if len(outs) > 1 else w[-1]
            w = w / w.sum()
            loss = 0
            for wi, o in zip(w, outs):
                if wi == 0:
                    continue
                yt = F.interpolate(y[:, None].float(), size=o.shape[2:], mode='nearest')[:, 0].long()
                loss = loss + wi * F.cross_entropy(o.float(), yt)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(reps):
        step()
    torch.cuda.synchronize()
    return dict(train_ms_per_iter=(time.perf_counter() - t0) / reps * 1000,
                train_peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20, train_batch_size=bs,
                train_patch=patch)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--dataset')
    ap.add_argument('-tr', '--trainers', nargs='+', required=True)
    ap.add_argument('-p', '--plans', default='nnUNetPlans')
    ap.add_argument('-c', '--configuration', default='3d_fullres')
    ap.add_argument('--plans-file')
    ap.add_argument('--dataset-json')
    ap.add_argument('--image-size', type=int, nargs=3, help='preprocessed volume size (default: plans median)')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--threads', type=int, help='CPU threads (torch.set_num_threads)')
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--no-tta', action='store_true', help='disable mirroring TTA (nnU-Net default is on)')
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--train-step', action='store_true')
    ap.add_argument('-o', '--out', help='append JSON lines here')
    args = ap.parse_args()

    from pvm3d.zoo import build, describe, env_info, load_plans
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    impl = 'mamba_ssm' if device.type == 'cuda' else 'pytorch'
    pm, cm, dj, n_in, n_out = load_plans(args.dataset, args.plans, args.configuration, args.plans_file,
                                         args.dataset_json)
    image_size = args.image_size or [int(round(s)) for s in cm.median_image_size_in_voxels]

    for tr in args.trainers:
        net = build(tr, pm, cm, n_in, n_out, deep_supervision=False, mamba_impl=impl)
        row = dict(model=tr, dataset=pm.dataset_name, plans=pm.plans_name, configuration=args.configuration,
                   patch=list(cm.patch_size), params=describe(net).params, tta=not args.no_tta,
                   precision='fp32' if (args.fp32 or device.type == 'cpu') else 'fp16-autocast',
                   **env_info(device))
        row.update(bench_inference(net, pm, cm, dj, tr, n_in, image_size, device, not args.no_tta, args.reps,
                                   args.fp32))
        del net
        if args.train_step and device.type == 'cuda':
            torch.cuda.empty_cache()
            row.update(bench_train_step(tr, pm, cm, n_in, n_out, device, max(3, args.reps)))
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        mem = f"  peak {row['infer_peak_mem_mb']:.0f} MB" if 'infer_peak_mem_mb' in row else ''
        trn = (f"  train {row['train_ms_per_iter']:.0f} ms/it, {row['train_peak_mem_mb']:.0f} MB"
               if 'train_ms_per_iter' in row else '')
        flag = '  [OOM -> CPU fallback!]' if row['oom_fallback_to_cpu'] else ''
        print(f"{tr:<40} {row['precision']:<13} tta={row['tta']!s:<5} {image_size}: "
              f"{row['latency_s_mean']:.3f} +- {row['latency_s_std']:.3f} s{mem}{trn}{flag}")
        if args.out:
            with open(args.out, 'a') as fh:
                fh.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
