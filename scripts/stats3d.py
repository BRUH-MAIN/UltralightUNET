"""Aggregate eval3d.py CSVs into the paper's tables, tests and Pareto figure.

Inputs: per-case CSVs from scripts/eval3d.py (any number of datasets x
trainers), optionally the published nnU-Net Revisited table and efficiency
JSON lines from scripts/flops3d.py / scripts/bench_infer3d.py.

Outputs (in --out):
  summary.csv        dataset x model: DSC / NSD / HD95 mean +- SD over folds
                     (fold score = mean over regions of the per-region mean over
                     cases -- the definition of nnU-Net's summary.json
                     foreground_mean, so it is comparable to Revisited's table)
  paired.csv         --ref vs every other model we trained, per dataset and
                     metric: n paired cases, mean difference (ref - other) with
                     a paired-bootstrap 95% CI, Wilcoxon signed-rank p, Holm-
                     corrected p (family = the comparisons within one dataset
                     and metric)
  ranks.csv          bootstrap rank stability per dataset (cases resampled,
                     ranks recomputed; mean rank and 95% interval) + mean rank
                     across datasets
  table_main.md/.tex DSC table, our models + reused published rows (marked †)
  pareto.png/.pdf    DSC vs inference params, one panel per dataset
                     (needs --efficiency)

Usage:
    python scripts/stats3d.py --csv "$nnUNet_results"/*/*/eval3d_2mm.csv \\
        --ref nnUNetTrainerPVM3D_S --reference-table pvm3d/reference/nnunet_revisited_dsc.csv \\
        --efficiency results/efficiency/flops.jsonl --out results/paper
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRETTY = {
    'nnUNetTrainerPVM3D_T': 'PVM-3D-T', 'nnUNetTrainerPVM3D_S': 'PVM-3D-S', 'nnUNetTrainerPVM3D_B': 'PVM-3D-B',
    'nnUNetTrainerPVM3D_Legacy': 'PVM-3D (legacy design)', 'nnUNetTrainer': 'nnU-Net (org.)',
    'nnUNetTrainerTinyUNet_matchT': 'Tiny nnU-Net (≈T)', 'nnUNetTrainerTinyUNet_matchS': 'Tiny nnU-Net (≈S)',
    'nnUNetTrainerTinyUNet_matchB': 'Tiny nnU-Net (≈B)', 'nnUNetTrainerLightMUNet': 'LightM-UNet',
    'nnUNetTrainerSlimUNETR': 'SlimUNETR', 'nnUNetTrainerSegFormer3D': 'SegFormer3D',
    'nnUNetTrainerSegMamba': 'SegMamba',
}
# substring of the nnU-Net dataset name -> column in the Revisited table
DATASET_KEYS = [('ACDC', 'ACDC'), ('KiTS', 'KiTS'), ('AMOS', 'AMOS'), ('Amos', 'AMOS'), ('BraTS', 'BraTS'),
                ('Liver', 'LiTS'), ('LiTS', 'LiTS'), ('BTCV', 'BTCV'), ('Hippocampus', 'Hippocampus')]
METRICS = (('dsc', True), ('nsd', True), ('hd95', False))  # (name, higher is better)

# reference categorical palette, first three slots only: a scatter puts every
# pair of series side by side, and only slots 1-3 clear the all-pairs CVD floor
FAMILY_COLORS = {'PVM-3D (ours)': '#2a78d6', 'Lightweight baselines': '#eb6834', 'Full-size baselines': '#1baf7a'}
FULL_SIZE = ('nnUNetTrainer', 'nnUNetTrainerSegMamba', 'nnUNetTrainerResEncM')
SHORT = {'nnUNetTrainerPVM3D_T': 'T', 'nnUNetTrainerPVM3D_S': 'S', 'nnUNetTrainerPVM3D_B': 'B',
         'nnUNetTrainer': 'nnU-Net', 'nnUNetTrainerTinyUNet_matchT': 'Tiny-T', 'nnUNetTrainerTinyUNet_matchS': 'Tiny-S',
         'nnUNetTrainerTinyUNet_matchB': 'Tiny-B', 'nnUNetTrainerLightMUNet': 'LightM', 'nnUNetTrainerSlimUNETR': 'Slim',
         'nnUNetTrainerSegFormer3D': 'SegF3D', 'nnUNetTrainerSegMamba': 'SegMamba'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def pretty(m):
    return PRETTY.get(m, m.replace('nnUNetTrainer', '') or m)


def dataset_key(ds):
    for sub, key in DATASET_KEYS:
        if sub in ds:
            return key
    return ds


def family(model):
    if model.startswith('nnUNetTrainerPVM3D'):
        return 'PVM-3D (ours)'
    if model in FULL_SIZE:
        return 'Full-size baselines'
    return 'Lightweight baselines'


# ---------------------------------------------------------------- aggregation
def fold_scores(df):
    per_region = df.groupby(['dataset', 'trainer', 'fold', 'region'])[['dsc', 'nsd', 'hd95']].mean()
    per_fold = per_region.groupby(['dataset', 'trainer', 'fold']).mean()
    agg = per_fold.groupby(['dataset', 'trainer']).agg(['mean', 'std', 'count'])
    agg.columns = [f'{m}_{s}' for m, s in agg.columns]
    return agg.reset_index()


def per_case(df):
    return df.groupby(['dataset', 'trainer', 'fold', 'case'])[['dsc', 'nsd', 'hd95']].mean().reset_index()


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def paired_tests(pc, ref, n_boot, seed):
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(seed)
    out = []
    for ds, g in pc.groupby('dataset'):
        if ref not in set(g.trainer):
            continue
        a = g[g.trainer == ref].set_index(['fold', 'case'])
        for metric, _ in METRICS:
            rows = []
            for other in sorted(set(g.trainer) - {ref}):
                b = g[g.trainer == other].set_index(['fold', 'case'])
                j = a[[metric]].join(b[[metric]], lsuffix='_ref', rsuffix='_oth', how='inner').dropna()
                if len(j) < 5:
                    continue
                d = (j[f'{metric}_ref'] - j[f'{metric}_oth']).to_numpy()
                boots = rng.choice(d, size=(n_boot, d.size), replace=True).mean(1)
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                rows.append(dict(dataset=ds, metric=metric, ref=ref, other=other, n=len(d), mean_diff=d.mean(),
                                 ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5), p=p))
            if rows:
                for r, padj in zip(rows, holm([r['p'] for r in rows])):
                    r['p_holm'] = padj
                    r['significant_0.05'] = padj < 0.05
                out += rows
    return pd.DataFrame(out)


def rank_stability(pc, n_boot, seed, metric='dsc'):
    rng = np.random.default_rng(seed)
    rows = []
    for ds, g in pc.groupby('dataset'):
        wide = g.pivot_table(index=['fold', 'case'], columns='trainer', values=metric).dropna()
        if wide.shape[1] < 2 or len(wide) < 5:
            continue
        vals = wide.to_numpy()
        idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
        means = vals[idx].mean(1)                              # (n_boot, n_models)
        ranks = (-means).argsort(1).argsort(1) + 1             # 1 = best
        for k, m in enumerate(wide.columns):
            rows.append(dict(dataset=ds, model=m, mean_rank=ranks[:, k].mean(),
                             rank_lo=np.percentile(ranks[:, k], 2.5), rank_hi=np.percentile(ranks[:, k], 97.5),
                             n_cases=len(vals)))
    r = pd.DataFrame(rows)
    if len(r):
        overall = r.groupby('model').mean_rank.mean().rename('mean_rank_across_datasets')
        r = r.merge(overall, on='model')
    return r


# ---------------------------------------------------------------- tables
def main_table(summary, reference):
    ours = summary.assign(key=summary.dataset.map(dataset_key))
    cols = sorted(ours.key.unique(), key=lambda k: [x[1] for x in DATASET_KEYS].index(k)
                  if k in [x[1] for x in DATASET_KEYS] else 99)
    rows = []
    for m, g in ours.groupby('trainer'):
        row = {'model': pretty(m), 'source': 'this work'}
        for _, r in g.iterrows():
            row[r.key] = (100 * r.dsc_mean, 100 * (r.dsc_std if r.dsc_count > 1 else np.nan), int(r.dsc_count))
        rows.append(row)
    if reference is not None:
        for _, r in reference.iterrows():
            row = {'model': r.model + ' †', 'source': 'nnU-Net Revisited'}
            for c in cols:
                if c in reference.columns and not pd.isna(r[c]):
                    row[c] = (float(r[c]), np.nan, 5)
            rows.append(row)
    return pd.DataFrame(rows), cols


def fmt_cell(v, best, tex):
    if not isinstance(v, tuple):
        return '–'
    mean, sd, n = v
    s = f'{mean:.2f}' + (f' ± {sd:.2f}' if not np.isnan(sd) else '') + ('' if n == 5 else f' ({n}f)')
    if best:
        s = f'\\textbf{{{s}}}' if tex else f'**{s}**'
    return s.replace('±', '$\\pm$') if tex else s


def write_tables(tab, cols, out):
    best = {c: max((v[0] for v in tab.get(c, pd.Series(dtype=object)) if isinstance(v, tuple)), default=None)
            for c in cols}
    md = ['| Model | ' + ' | '.join(cols) + ' |', '|---|' + '---|' * len(cols)]
    tex = ['\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\toprule',
           'Model & ' + ' & '.join(cols) + ' \\\\', '\\midrule']
    for _, r in tab.iterrows():
        cells = [r.get(c) for c in cols]
        md.append(f'| {r.model} | ' + ' | '.join(fmt_cell(v, isinstance(v, tuple) and v[0] == best[c], False)
                                                for v, c in zip(cells, cols)) + ' |')
        tex.append(r.model.replace('†', '$^\\dagger$') + ' & ' +
                   ' & '.join(fmt_cell(v, isinstance(v, tuple) and v[0] == best[c], True)
                              for v, c in zip(cells, cols)) + ' \\\\')
    tex += ['\\bottomrule', '\\end{tabular}']
    note = ('\n† 5-fold CV DSC from nnU-Net Revisited (Isensee et al., MICCAI 2024), same framework and default '
            'splits; no per-case predictions, so excluded from paired tests. (nf) = only n folds complete.\n')
    open(os.path.join(out, 'table_main.md'), 'w').write('\n'.join(md) + '\n' + note)
    open(os.path.join(out, 'table_main.tex'), 'w').write('\n'.join(tex) + '\n')


# ---------------------------------------------------------------- figure
def pareto(summary, eff, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    e = eff.groupby(['dataset', 'model']).params.first().reset_index().rename(columns={'model': 'trainer'})
    d = summary.merge(e, on=['dataset', 'trainer'])
    if d.empty:
        print('[pareto] no overlap between efficiency rows and evaluated models; skipped')
        return
    dss = sorted(d.dataset.unique())
    n = len(dss)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.titlecolor': INK})
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, ds in zip(axes.flat, dss):
        g = d[d.dataset == ds]
        for fam, color in FAMILY_COLORS.items():
            h = g[g.trainer.map(family) == fam]
            if len(h):
                ax.errorbar(h.params, 100 * h.dsc_mean, yerr=100 * h.dsc_std.fillna(0), fmt='none', ecolor=color,
                            elinewidth=1, capsize=0, alpha=0.6, zorder=2)
                ax.scatter(h.params, 100 * h.dsc_mean, s=40, color=color, edgecolors='white', linewidths=2,
                           zorder=3, label=fam)
        # short direct label on every point (the legend carries family). Placed right of
        # the point unless the label's extent would cross a neighbour's error bar, then
        # left, then below. Extents in log10-decades: ~0.065/char at 7.5 pt in a 3.6" panel.
        xs = np.log10(g.params.to_numpy())
        ys = 100 * g.dsc_mean.to_numpy()
        sds = 100 * g.dsc_std.fillna(0).to_numpy()
        pad_y = 0.04 * (np.ptp(ys + sds) + np.ptp(ys - sds) + 1e-9)
        xmin = xs.min() - 0.15
        for i, r in enumerate(g.itertuples()):
            text = SHORT.get(r.trainer, pretty(r.trainer))
            width = 0.09 + 0.065 * len(text)

            def hits(lo, hi):
                return any(lo < xs[j] < hi and ys[j] - sds[j] - pad_y <= ys[i] + pad_y <= ys[j] + sds[j] + pad_y
                           for j in range(len(xs)) if j != i)
            if not hits(xs[i], xs[i] + width + 0.05):
                kw = dict(xytext=(6, 3), ha='left', va='bottom')
            elif xs[i] - width > xmin and not hits(xs[i] - width - 0.05, xs[i]):
                kw = dict(xytext=(-6, 3), ha='right', va='bottom')
            else:
                kw = dict(xytext=(5, -4), ha='left', va='top')
            ax.annotate(text, (r.params, ys[i]), textcoords='offset points', fontsize=7.5, color=INK2, **kw)
        ax.set_xscale('log')
        ax.set_title(dataset_key(ds), fontsize=10, loc='left')
        ax.set_xlabel('inference parameters (log)')
        ax.set_ylabel('DSC (%)')
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    for ax in axes.flat[1:n]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in lbls:
                handles.append(h)
                lbls.append(l)
    fig.legend(handles, lbls, loc='lower center', ncol=len(lbls), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(out, f'pareto.{ext}'), dpi=200, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', nargs='+', required=True, help='eval3d CSVs (globs ok)')
    ap.add_argument('--ref', default='nnUNetTrainerPVM3D_S', help='model the paired tests compare against')
    ap.add_argument('--reference-table', default=os.path.join(REPO, 'pvm3d', 'reference', 'nnunet_revisited_dsc.csv'))
    ap.add_argument('--efficiency', nargs='*', default=[], help='flops3d / bench_infer3d JSON lines')
    ap.add_argument('--n-boot', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--out', default=os.path.join(REPO, 'results', 'paper'))
    args = ap.parse_args()

    files = sorted({f for pat in args.csv for f in glob.glob(pat)})
    if not files:
        sys.exit('no CSVs matched')
    df = pd.concat([pd.read_csv(f, dtype={'fold': str}) for f in files], ignore_index=True)
    os.makedirs(args.out, exist_ok=True)

    summary = fold_scores(df)
    summary.to_csv(os.path.join(args.out, 'summary.csv'), index=False)
    pc = per_case(df)
    paired = paired_tests(pc, args.ref, args.n_boot, args.seed)
    paired.to_csv(os.path.join(args.out, 'paired.csv'), index=False)
    ranks = rank_stability(pc, min(args.n_boot, 2000), args.seed)
    ranks.to_csv(os.path.join(args.out, 'ranks.csv'), index=False)

    reference = pd.read_csv(args.reference_table) if args.reference_table and os.path.isfile(args.reference_table) \
        else None
    tab, cols = main_table(summary, reference)
    write_tables(tab, cols, args.out)

    if args.efficiency:
        eff = pd.DataFrame([json.loads(l) for f in args.efficiency for l in open(f) if l.strip()])
        pareto(summary, eff, args.out)

    print(f'{len(files)} CSVs, {len(df)} rows -> {args.out}')
    with pd.option_context('display.width', 160, 'display.max_columns', 20):
        print(summary[['dataset', 'trainer', 'dsc_mean', 'dsc_std', 'dsc_count', 'nsd_mean', 'hd95_mean']]
              .to_string(index=False))
        if len(paired):
            print(paired[paired.metric == 'dsc'][['dataset', 'other', 'n', 'mean_diff', 'ci_lo', 'ci_hi', 'p_holm']]
                  .to_string(index=False))


if __name__ == '__main__':
    main()
