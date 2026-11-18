"""Does a trained PVM-3D model actually use its Mamba? (inference-time knockout)

For each fold's final checkpoint this predicts the fold's validation cases
twice with nnU-Net's own predictor -- once as trained, once with every
PVMLayer3D's Mamba output replaced by zeros (the layer's skip path
``skip_scale * x`` and everything else untouched) -- and reports the per-case
DSC drop. It also records, per PVM layer, the ratio
||Mamba output|| / ||skip_scale * x|| on the as-trained pass.

This reproduces the diagnostic of the earlier bespoke-pipeline study
(pvm3D, results/diagnostics/mamba_knockout), which found drops of ~0 DSC and
near-zero Mamba/skip ratios in deeper residual models, i.e. the network had
learned to route around Mamba. It complements -- it does not replace -- the
retraining ablations A1 (``_S_noMamba``, ``_S_convMixer``): a knockout shows
whether a trained network *depends* on Mamba; A1 shows whether training *with*
Mamba buys anything.

Usage:
    source scripts/cluster/env.sh <DATA_ROOT>
    python scripts/mamba_knockout.py -d 4 -tr nnUNetTrainerPVM3D_S -f 0 1 2 3 4
    -> <trainer folder>/mamba_knockout.csv (per case) and mamba_knockout_layers.csv
"""

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import warnings

import numpy as np
import torch

warnings.filterwarnings('ignore')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from eval3d import _eval_case, regions_from_dataset_json  # noqa: E402
from pvm3d.network import PVMLayer3D  # noqa: E402


def pvm_layers(net):
    return [(name, m) for name, m in net.named_modules()
            if isinstance(m, PVMLayer3D) and m.mixer == 'mamba']


def predict(pred, cases, raw_images, n_channels, ending, out_dir):
    lists = [[os.path.join(raw_images, f'{c}_{k:04d}{ending}') for k in range(n_channels)] for c in cases]
    pred.predict_from_files(lists, [os.path.join(out_dir, c) for c in cases], save_probabilities=False,
                            overwrite=True, num_processes_preprocessing=2, num_processes_segmentation_export=2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--dataset', required=True)
    ap.add_argument('-tr', '--trainer', required=True)
    ap.add_argument('-p', '--plans', default='nnUNetPlans')
    ap.add_argument('-c', '--configuration', default='3d_fullres')
    ap.add_argument('-f', '--folds', nargs='+', default=['0', '1', '2', '3', '4'])
    ap.add_argument('--checkpoint', default='checkpoint_final.pth')
    ap.add_argument('--max-cases', type=int, default=None, help='subsample validation cases per fold (speed)')
    ap.add_argument('--no-tta', action='store_true')
    args = ap.parse_args()

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    ds = maybe_convert_to_dataset_name(args.dataset)
    dj = json.load(open(os.path.join(nnUNet_preprocessed, ds, 'dataset.json')))
    ending = dj.get('file_ending', '.nii.gz')
    n_ch = len(dj.get('channel_names', dj.get('modality', {'0': ''})))
    regions = regions_from_dataset_json(dj)
    splits = json.load(open(os.path.join(nnUNet_preprocessed, ds, 'splits_final.json')))
    tdir = os.path.join(nnUNet_results, ds, f'{args.trainer}__{args.plans}__{args.configuration}')
    gt_dir = os.path.join(nnUNet_preprocessed, ds, 'gt_segmentations')
    raw_images = os.path.join(nnUNet_raw, ds, 'imagesTr')

    case_rows, layer_rows = [], []
    for f in args.folds:
        if not os.path.isfile(os.path.join(tdir, f'fold_{f}', args.checkpoint)):
            print(f'[skip] fold {f}: no {args.checkpoint}')
            continue
        cases = sorted(splits[int(f)]['val'])
        if args.max_cases:
            cases = cases[:args.max_cases]
        pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=not args.no_tta,
                               perform_everything_on_device=True, device=torch.device('cuda'), verbose=False,
                               allow_tqdm=False)
        pred.initialize_from_trained_model_folder(tdir, use_folds=(int(f),), checkpoint_name=args.checkpoint)
        layers = pvm_layers(pred.network)
        if not layers:
            sys.exit(f'{args.trainer} has no Mamba PVM layers to knock out')

        # pass 1: as trained, recording ||Mamba out|| / ||skip_scale * x|| per layer
        ratios = {name: [] for name, _ in layers}
        hooks = []
        for name, layer in layers:
            for mb in layer.mamba:
                skip = (lambda layer=layer: layer.skip_scale.abs()) if layer.inner_skip else (lambda: 1.0)
                hooks.append(mb.register_forward_hook(
                    lambda m, inp, out, name=name, skip=skip: ratios[name].append(
                        (out.float().norm() / (skip() * inp[0].float()).norm().clamp_min(1e-12)).item())))
        tmp = tempfile.mkdtemp(prefix='knockout_')
        try:
            predict(pred, cases, raw_images, n_ch, ending, os.path.join(tmp, 'trained'))
            for h in hooks:
                h.remove()
            # pass 2: Mamba output zeroed in every PVM layer
            hooks = [mb.register_forward_hook(lambda m, inp, out: torch.zeros_like(out))
                     for _, layer in layers for mb in layer.mamba]
            predict(pred, cases, raw_images, n_ch, ending, os.path.join(tmp, 'knockout'))
            for h in hooks:
                h.remove()
            for c in cases:
                gt = os.path.join(gt_dir, c + ending)
                a = _eval_case((c, os.path.join(tmp, 'trained', c + ending), gt, regions, 2.0, {}))
                b = _eval_case((c, os.path.join(tmp, 'knockout', c + ending), gt, regions, 2.0, {}))
                for ra, rb in zip(a, b):
                    case_rows.append(dict(fold=f, case=c, region=ra['region'], dsc=ra['dsc'], dsc_knockout=rb['dsc'],
                                          drop=ra['dsc'] - rb['dsc'] if not np.isnan(ra['dsc']) else np.nan))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        for name, layer in layers:
            mb = layer.mamba[0]
            # memory horizon of the slowest SSM mode from the learned, input-independent part
            # of the step size: 1 / (softplus(dt_bias) * min|A|) tokens. Inherited UltraLight
            # init: ~1.4 tokens; mamba_ssm's own init: ~60 (median). Did training lengthen it?
            dt = torch.nn.functional.softplus(mb.dt_proj.bias.detach().float())
            mem = 1.0 / (dt * torch.exp(mb.A_log.detach().float()).min(dim=1).values)
            layer_rows.append(dict(fold=f, layer=name, memory_tokens_median=float(mem.median()),
                                   memory_tokens_max=float(mem.max()),
                                   skip_scale=float(layer.skip_scale.item()) if layer.inner_skip else float('nan'),
                                   mamba_to_skip_ratio=float(np.mean(ratios[name])) if ratios[name] else np.nan))
        fr = [r for r in case_rows if r['fold'] == f]
        print(f'fold {f}: n={len(cases)}  DSC {np.nanmean([r["dsc"] for r in fr]):.4f} -> knockout '
              f'{np.nanmean([r["dsc_knockout"] for r in fr]):.4f}  (drop {np.nanmean([r["drop"] for r in fr]):+.4f})  '
              'ratios ' + ', '.join(f'{r["mamba_to_skip_ratio"]:.2f}' for r in layer_rows if r['fold'] == f))

    if not case_rows:
        sys.exit('no fold had a checkpoint')
    for path, rows in ((os.path.join(tdir, 'mamba_knockout.csv'), case_rows),
                       (os.path.join(tdir, 'mamba_knockout_layers.csv'), layer_rows)):
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    drops = np.array([r['drop'] for r in case_rows], float)
    drops = drops[~np.isnan(drops)]
    from scipy.stats import wilcoxon
    p = wilcoxon(drops).pvalue if np.any(drops != 0) else 1.0
    print(f'all folds: mean drop {drops.mean():+.4f} (median {np.median(drops):+.4f}, n={drops.size} case-regions, '
          f'Wilcoxon p={p:.3g}) -> {tdir}/mamba_knockout*.csv')


if __name__ == '__main__':
    main()
