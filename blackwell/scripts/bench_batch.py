"""Measure training throughput and VRAM against batch size, on this GPU.

Why this exists: UltraLight VM-UNet is 0.049M parameters, so VRAM is never the
limit -- a batch of 8 uses under 1 GB of a 15 GB T4. The cost is dominated by
kernel-launch overhead, because the selective scan issues ~176 sequential
launches per forward *regardless of batch size*. A larger batch therefore
amortises a fixed cost over more images, which is a real speedup up to the point
where the GPU finally becomes compute- or bandwidth-bound.

Where that point falls is hardware specific, so measure rather than guess:

    python scripts/bench_batch.py

IMPORTANT: config.batch_size = 8 is a hyperparameter from the paper ("a batch
size of 8"). Raising it changes the number of optimiser steps per epoch -- 157 at
bs=8 versus 39 at bs=32 -- and so changes the training trajectory and the result.
It is a legitimate lever for iterating on your own variants, provided you hold it
constant across everything you compare, but it is not free, and it does not
belong in a run whose purpose is to reproduce the paper's numbers.

Validation and test batch size are a different matter: they affect only speed,
not the model. See the note at the bottom of the output.
"""

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.UltraLight_VM_UNet import UltraLight_VM_UNet

N_TRAIN = 1250
N_VAL = 150


def bench(bs, size=256, warmup=3):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = UltraLight_VM_UNet().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(bs, 3, size, size).cuda()
    y = torch.rand(bs, 1, size, size).cuda()

    def step():
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy(model(x), y).backward()
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    n = max(3, min(12, 128 // bs))
    t = time.time()
    for _ in range(n):
        step()
    torch.cuda.synchronize()
    s = (time.time() - t) / n
    vram = torch.cuda.max_memory_allocated() / 1e9
    del model, opt, x, y
    return s, vram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[8, 16, 32, 64, 128, 256])
    args = ap.parse_args()

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{name}  ({total:.1f} GB)\n")
    print(f'{"batch":>6s} {"s/iter":>8s} {"img/s":>8s} {"steps/ep":>9s} '
          f'{"train ep":>9s} {"VRAM GB":>8s} {"% of GPU":>9s} {"vs bs=8":>8s}')
    print("-" * 74)

    base = None
    best = (0, None)
    for bs in args.batches:
        try:
            s, vram = bench(bs)
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:6d}      OOM")
            torch.cuda.empty_cache()
            break
        ips = bs / s
        base = base or ips
        if ips > best[0]:
            best = (ips, bs)
        print(f"{bs:6d} {s:8.3f} {ips:8.1f} {N_TRAIN/bs:9.0f} "
              f"{N_TRAIN/bs*s:8.1f}s {vram:8.2f} {vram/total*100:8.1f}% {ips/base:7.2f}x")

    print(f"\npeak throughput at batch {best[1]} ({best[0]:.1f} img/s, "
          f"{best[0]/base:.2f}x over the paper's batch of 8)")
    print("\nA falling img/s at large batch usually means VRAM spilling to host")
    print("memory, not a property of the model -- check the % of GPU column.")
    print("\nReminder: train batch 8 is from the paper. Validation batch size is")
    print(f"free to change, but must divide {N_VAL} exactly -- the val loader uses")
    print("drop_last=True, so a non-divisor silently discards images.")


if __name__ == "__main__":
    main()
