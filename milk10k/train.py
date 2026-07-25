"""Train the MILK10k lightweight Mamba classifier.

Run from the repo root so the shared components resolve:

    python -m milk10k.train

Reuses the Phase-1 optimiser/scheduler/seed/logger helpers from `utils.py` and
the `PVMLayer` from `models/`. The training loop and metrics are classification-
specific (`milk10k/engine.py`). Blackwell-safe throughout (GPU preflight,
weights_only=False, timm-robust import via the shared model).
"""

import copy
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils import set_seed, get_logger, get_optimizer, get_scheduler  # reused from Phase 1
from milk10k.configs.config import config
from milk10k.loader import MILK10kDataset
from milk10k.pvm_classifier import MILK10kClassifier
from milk10k.engine import (train_one_epoch, evaluate, log_metrics,
                            build_criterion, compute_class_weights)

MODALITIES = {'derm': ('derm',), 'clin': ('clin',), 'both': ('derm', 'clin')}


def count_params_flops(model, config, logger):
    """Params + a thop FLOP estimate, dict-input aware and non-polluting.

    thop registers buffers on the model it profiles, so we profile a deepcopy --
    the same fix applied in blackwell/utils.py. The model takes a dict, so we hand
    thop a one-sample dict for the active modality.
    """
    total = sum(p.numel() for p in model.parameters())
    dev = next(model.parameters()).device
    mods = MODALITIES[config.modality]
    x = {m: torch.randn(1, 3, config.input_size, config.input_size, device=dev) for m in mods}
    flops = float('nan')
    try:
        from thop import profile
        flops, _ = profile(copy.deepcopy(model), inputs=(x,), verbose=False)
    except Exception as e:
        print(f'  (thop FLOP estimate skipped: {e})')
    msg = f'params: {total} ({total/1e6:.4f}M), GFLOPs: {flops/1e9:.4f}'
    print(msg)
    logger.info(msg)
    return total


def preflight(logger):
    assert torch.cuda.is_available(), 'no CUDA device visible'
    cc = torch.cuda.get_device_capability(0)
    info = (f'device: {torch.cuda.get_device_name(0)} (sm_{cc[0]}{cc[1]}), '
            f'torch {torch.__version__}, built for {torch.cuda.get_arch_list()}')
    print(info)
    logger.info(info)
    try:
        (torch.zeros(8, 8, device='cuda') @ torch.zeros(8, 8, device='cuda')).cpu()
    except RuntimeError as e:
        raise SystemExit(
            f'torch {torch.__version__} cannot execute on sm_{cc[0]}{cc[1]} '
            f'(built for {torch.cuda.get_arch_list()}).\nBlackwell needs torch>=2.7/cu128.\n{e}')


def main(config):
    run = f'milk10k_{config.modality}_{datetime.now():%Y%m%d_%H%M%S}'
    work_dir = os.path.join(config.work_dir + run)
    os.makedirs(os.path.join(work_dir, 'checkpoints'), exist_ok=True)
    logger = get_logger('train', os.path.join(work_dir, 'log'))
    logger.info(f'run {run}; modality={config.modality}; loss={config.loss}')

    set_seed(config.seed)
    preflight(logger)

    manifest = pd.read_csv(config.manifest)
    mods = MODALITIES[config.modality]
    mk = lambda split, train: MILK10kDataset(manifest, config.images_dir, split,
                                             modalities=mods, input_size=config.input_size, train=train)
    train_loader = DataLoader(mk('train', True), batch_size=config.batch_size, shuffle=True,
                              pin_memory=True, num_workers=config.num_workers, drop_last=True)
    val_loader = DataLoader(mk('val', False), batch_size=config.batch_size, shuffle=False,
                            pin_memory=True, num_workers=config.num_workers)
    test_loader = DataLoader(mk('test', False), batch_size=config.batch_size, shuffle=False,
                             pin_memory=True, num_workers=config.num_workers)

    model = MILK10kClassifier(num_classes=config.num_classes, modality=config.modality,
                              input_channels=config.input_channels, c_list=tuple(config.c_list),
                              fusion=config.fusion).cuda()
    count_params_flops(model, config, logger)

    weights = compute_class_weights(manifest, config, next(model.parameters()).device)
    criterion = build_criterion(config, class_weights=weights)
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    best_f1, best_epoch = -1.0, 0
    for epoch in range(1, config.epochs + 1):
        train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, logger, config)
        if epoch % config.val_interval == 0:
            _, metrics = evaluate(val_loader, model, criterion, config, split='val')
            f1 = log_metrics(metrics, epoch, logger, split='val')
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                torch.save(model.state_dict(), os.path.join(work_dir, 'checkpoints', 'best.pth'))
        torch.save({'epoch': epoch, 'best_f1': best_f1, 'best_epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()},
                   os.path.join(work_dir, 'checkpoints', 'latest.pth'))

    # final test on the best checkpoint
    best_path = os.path.join(work_dir, 'checkpoints', 'best.pth')
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=False))
        _, metrics = evaluate(test_loader, model, criterion, config, split='test')
        log_metrics(metrics, best_epoch, logger, split='test')
        metrics['best_val_epoch'] = best_epoch
        metrics['best_val_macro_f1'] = best_f1
        with open(os.path.join(work_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f'\nbest val macro-F1 {best_f1:.4f} @ epoch {best_epoch}')
        print(f'TEST macro-F1 {metrics["macro_f1"]:.4f}  -> {work_dir}/test_metrics.json')


if __name__ == '__main__':
    main(config)
