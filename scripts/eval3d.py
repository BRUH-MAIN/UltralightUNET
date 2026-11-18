"""Per-case DSC / NSD / HD95 for nnU-Net cross-validation predictions.

Reads each fold's ``validation/`` predictions (written by nnUNetv2_train's
final validation) against ``$nnUNet_preprocessed/<Dataset>/gt_segmentations``,
in physical units from the image headers, and writes one CSV row per
(case, label-or-region). That CSV is the input to scripts/stats3d.py.

Conventions (stated in the paper's methods section):
  * labels / regions come from dataset.json in its order; region-based datasets
    (BraTS: WT/TC/ET, KiTS: kidney+masses/masses/tumor) are evaluated as regions,
    i.e. a voxel counts for every region whose label set contains it.
  * GT empty and prediction empty -> DSC/NSD/HD95 = NaN (excluded from means,
    as nnU-Net's own summary.json does for DSC); exactly one of them empty ->
    DSC = NSD = 0, HD95 = NaN (count reported separately).
  * NSD: fraction of surface voxels of each mask within `tol` mm of the other
    mask's surface (Nikolov et al.; voxel-surface approximation as in MONAI's
    compute_surface_dice). Default tol = 2 mm, nnU-Net Revisited's setting.
  * HD95: max of the two directed 95th-percentile surface distances (MONAI).
  * Surface = mask minus its 6-connected erosion; distances via exact EDT with
    voxel spacing, computed on the union bounding box (exact, much faster).

DSC is cross-checked against nnU-Net's summary.json for every fold
(--check-nnunet, default on): a mismatch means this script and nnU-Net
disagree about labels/regions and the run should not be trusted.

Usage:
    source scripts/cluster/env.sh
    python scripts/eval3d.py -d 4 -tr nnUNetTrainerPVM3D_S -f 0 1 2 3 4
    python scripts/eval3d.py -d 219 -tr nnUNetTrainerPVM3D_S_CTOnly -f 0 --modality-split amos
"""

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

FIELDS = ['dataset', 'trainer', 'plans', 'configuration', 'fold', 'case', 'region', 'modality',
          'dsc', 'nsd', 'hd95', 'gt_voxels', 'pred_voxels', 'gt_ml', 'pred_ml']


def _surface(mask):
    er = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(3, 1), border_value=0)
    return mask & ~er


def mask_metrics(gt, pred, spacing, tol_mm):
    g, p = int(gt.sum()), int(pred.sum())
    if g == 0 and p == 0:
        return np.nan, np.nan, np.nan
    inter = int(np.logical_and(gt, pred).sum())
    dsc = 2.0 * inter / (g + p)
    if g == 0 or p == 0:
        return dsc, 0.0, np.nan
    union = gt | pred
    idx = np.argwhere(union)
    lo = np.maximum(idx.min(0) - 1, 0)
    hi = np.minimum(idx.max(0) + 2, gt.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    sg, sp = _surface(gt[sl]), _surface(pred[sl])
    d_to_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    d_to_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d_p2g, d_g2p = d_to_g[sp], d_to_p[sg]
    nsd = ((d_p2g <= tol_mm).sum() + (d_g2p <= tol_mm).sum()) / (d_p2g.size + d_g2p.size)
    hd95 = max(np.percentile(d_p2g, 95), np.percentile(d_g2p, 95))
    return dsc, float(nsd), float(hd95)


def regions_from_dataset_json(dj):
    out = []
    for name, val in dj['labels'].items():
        if name.lower() in ('background', 'ignore'):
            continue
        vals = tuple(val) if isinstance(val, (list, tuple)) else (int(val),)
        if vals == (0,):
            continue
        out.append((name, vals))
    return out


def _read_seg(path):
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    seg, props = SimpleITKIO().read_seg(path)
    return seg[0], np.asarray(props['spacing'], dtype=np.float64)


def _eval_case(args):
    case, pred_path, gt_path, regions, tol, meta = args
    gt, spacing = _read_seg(gt_path)
    pred, _ = _read_seg(pred_path)
    if gt.shape != pred.shape:
        raise RuntimeError(f'{case}: gt {gt.shape} vs pred {pred.shape}')
    vox_ml = float(np.prod(spacing)) / 1000.0
    rows = []
    for name, vals in regions:
        g = np.isin(gt, vals)
        p = np.isin(pred, vals)
        dsc, nsd, hd95 = mask_metrics(g, p, spacing, tol)
        rows.append({**meta, 'case': case, 'region': name, 'dsc': dsc, 'nsd': nsd, 'hd95': hd95,
                     'gt_voxels': int(g.sum()), 'pred_voxels': int(p.sum()),
                     'gt_ml': g.sum() * vox_ml, 'pred_ml': p.sum() * vox_ml})
    return rows


def modality_of(case, scheme):
    if scheme == 'amos':
        from pvm3d.nnunet_glue import amos_is_mri
        return 'MRI' if amos_is_mri(case) else 'CT'
    return ''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--dataset', required=True, help='nnU-Net dataset id or name')
    ap.add_argument('-tr', '--trainer', required=True)
    ap.add_argument('-p', '--plans', default='nnUNetPlans')
    ap.add_argument('-c', '--configuration', default='3d_fullres')
    ap.add_argument('-f', '--folds', nargs='+', default=['0', '1', '2', '3', '4'])
    ap.add_argument('--tol', type=float, default=2.0, help='NSD tolerance in mm (default 2, as nnU-Net Revisited)')
    ap.add_argument('--pred-subdir', default='validation',
                    help="per-fold folder holding predictions (e.g. a robustness run's output)")
    ap.add_argument('--modality-split', choices=['amos'], default=None)
    ap.add_argument('-o', '--out', default=None, help='CSV path (default: <trainer folder>/eval3d_<tol>mm.csv)')
    ap.add_argument('-np', '--num-processes', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--no-check-nnunet', action='store_true')
    args = ap.parse_args()

    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    ds = maybe_convert_to_dataset_name(args.dataset)
    dj = json.load(open(os.path.join(nnUNet_preprocessed, ds, 'dataset.json')))
    ending = dj.get('file_ending', '.nii.gz')
    regions = regions_from_dataset_json(dj)
    gt_dir = os.path.join(nnUNet_preprocessed, ds, 'gt_segmentations')
    tdir = os.path.join(nnUNet_results, ds, f'{args.trainer}__{args.plans}__{args.configuration}')
    out = args.out or os.path.join(tdir, f'eval3d_{args.tol:g}mm.csv')

    jobs = []
    for f in args.folds:
        pdir = os.path.join(tdir, f'fold_{f}', args.pred_subdir)
        if not os.path.isdir(pdir):
            print(f'[skip] fold {f}: no {pdir}')
            continue
        cases = sorted(x[:-len(ending)] for x in os.listdir(pdir) if x.endswith(ending))
        for c in cases:
            meta = dict(dataset=ds, trainer=args.trainer, plans=args.plans, configuration=args.configuration,
                        fold=f, modality=modality_of(c, args.modality_split))
            jobs.append((c, os.path.join(pdir, c + ending), os.path.join(gt_dir, c + ending), regions, args.tol, meta))
    if not jobs:
        sys.exit('nothing to evaluate')

    with Pool(args.num_processes) as pool:
        rows = [r for rs in pool.imap(_eval_case, jobs, chunksize=1) for r in rs]

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # per-fold summary + cross-check against nnU-Net's own Dice
    bad = False
    for f in sorted({r['fold'] for r in rows}, key=str):
        fr = [r for r in rows if r['fold'] == f]
        per_region = {name: np.nanmean([r['dsc'] for r in fr if r['region'] == name]) for name, _ in regions}
        fg = float(np.nanmean(list(per_region.values())))
        nsd = float(np.nanmean([np.nanmean([r['nsd'] for r in fr if r['region'] == n]) for n, _ in regions]))
        msg = f'fold {f}: n={len({r["case"] for r in fr})}  DSC {fg:.4f}  NSD@{args.tol:g}mm {nsd:.4f}  ' + \
              '  '.join(f'{k}={v:.4f}' for k, v in per_region.items())
        summ = os.path.join(tdir, f'fold_{f}', args.pred_subdir, 'summary.json')
        if not args.no_check_nnunet and os.path.isfile(summ):
            ref = json.load(open(summ))['foreground_mean']['Dice']
            ok = abs(ref - fg) < 1e-4
            bad |= not ok
            msg += f'   [nnU-Net summary.json Dice {ref:.4f}: {"OK" if ok else "MISMATCH"}]'
        print(msg)
    n_one_empty = sum(1 for r in rows if (r['gt_voxels'] == 0) != (r['pred_voxels'] == 0))
    print(f'{len(rows)} rows ({n_one_empty} with exactly one of gt/pred empty) -> {out}')
    if bad:
        sys.exit('DSC disagrees with nnU-Net summary.json -- check label/region handling before using these numbers')


if __name__ == '__main__':
    main()
