"""Evaluate a trained MILK10k classifier, with a per-skin-tone fairness breakdown.

    python -m milk10k.test --weights milk10k/results/<run>/checkpoints/best.pth --modality derm

Reports overall macro-F1 / per-class F1 (as in training) plus macro-F1 stratified
by skin-tone class (0-5). MILK10k is one of the few skin datasets carrying skin
tone, so this is a cheap, high-value fairness analysis -- with the honest caveat
that the rarest tones have too few test lesions to read much into.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from milk10k.configs.config import config
from milk10k.loader import MILK10kDataset
from milk10k.pvm_classifier import MILK10kClassifier
from milk10k.engine import build_criterion, evaluate, compute_class_weights

MODALITIES = {'derm': ('derm',), 'clin': ('clin',), 'both': ('derm', 'clin')}


@torch.no_grad()
def collect(loader, model, device):
    model.eval()
    ys, ps, tones = [], [], []
    for batch in loader:
        x = {k: v.to(device).float() for k, v in batch.items() if k in ('derm', 'clin')}
        logits = model(x)
        ps.append(logits.argmax(1).cpu().numpy())
        ys.append(batch['label'].numpy())
        tones.append(np.asarray(batch['skin_tone']))
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(tones)


def fairness_by_skin_tone(y, pred, tone, num_classes):
    print('\nper-skin-tone macro-F1 (0=very dark .. 5=very light):')
    print(f'{"tone":>5s} {"lesions":>8s} {"macro-F1":>9s}   note')
    rows = {}
    for t in sorted(set(tone.tolist())):
        mask = tone == t
        n = int(mask.sum())
        f1 = f1_score(y[mask], pred[mask], labels=list(range(num_classes)),
                      average='macro', zero_division=0)
        note = 'too few to read' if n < 30 else ''
        print(f'{t:5d} {n:8d} {f1:9.4f}   {note}')
        rows[int(t)] = {'n': n, 'macro_f1': float(f1)}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--modality', default=config.modality, choices=list(MODALITIES))
    ap.add_argument('--fusion', default=config.fusion)
    args = ap.parse_args()
    config.modality = args.modality
    config.fusion = args.fusion

    manifest = pd.read_csv(config.manifest)
    ds = MILK10kDataset(manifest, config.images_dir, 'test', modalities=MODALITIES[args.modality],
                        input_size=config.input_size, train=False)
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = MILK10kClassifier(num_classes=config.num_classes, modality=args.modality,
                              c_list=tuple(config.c_list), fusion=args.fusion).cuda()
    state = torch.load(args.weights, map_location='cpu', weights_only=False)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    for k in [k for k in state if k.endswith(('total_ops', 'total_params'))]:
        del state[k]                       # strip any thop buffers, as in blackwell/test.py
    model.load_state_dict(state)

    device = next(model.parameters()).device
    weights = compute_class_weights(manifest, config, device)
    _, metrics = evaluate(loader, model, build_criterion(config, weights), config, split='test')
    print(f'\nTEST  macro-F1 {metrics["macro_f1"]:.4f}  bal-acc {metrics["balanced_acc"]:.4f}  '
          f'acc {metrics["accuracy"]:.4f}')
    print('per-class F1:', {k: round(v, 3) for k, v in metrics['per_class_f1'].items()})

    y, pred, tone = collect(loader, model, device)
    metrics['fairness_skin_tone'] = fairness_by_skin_tone(y, pred, tone, config.num_classes)

    out = os.path.join(os.path.dirname(os.path.dirname(args.weights)), 'test_metrics_standalone.json')
    with open(out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
