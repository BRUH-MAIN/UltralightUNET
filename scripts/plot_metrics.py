"""Plot per-epoch training/validation metrics and final test results from a run's log.

Every metric this repo computes (loss, DSC, IoU, accuracy, sensitivity,
specificity, the test confusion matrix, and now per-image test DSC) is only ever
written as plain-text lines in <work_dir>/log/*.log -- there is no CSV/JSON
history file. This parses that text with the same regex-over-concatenated-log-text
technique notebooks/train_test.ipynb's cell 19 already uses for its paper-vs-ours
table, and turns it into PNGs.

Handles both old and new logs transparently: engine.py used to log the full
DSC/IoU/accuracy/sensitivity/specificity set only every config.val_interval
epochs and never logged per-image test DSC; it now logs the former every epoch
and does log the latter. Runs against an old log still work -- the sparse/missing
data is rendered honestly (markers instead of an interpolated line; a skip
message instead of a crash) rather than papered over.

Usage:
    python scripts/plot_metrics.py [--work-dir results/<run>/]

With no --work-dir, uses the most recently modified results/*/ directory. Can
also be imported from a notebook: `from scripts.plot_metrics import generate_all`.
"""

import argparse
import ast
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')  # PATCH: headless backend, matches utils.py's save_imgs setup.
                        # Must be set before pyplot is imported anywhere in this
                        # process; idempotent if utils.py already set it (e.g. a
                        # notebook that did `from utils import *` first).
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- palette: the dataviz skill's validated default categorical order (light-mode
# steps -- these PNGs are flat files with no theme toggle, so there's no dark
# variant to carry). Assigned by entity, fixed per chart, never cycled.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRID = '#e1e0d9'
GOOD = '#0ca30c'
# sequential blue, steps 100/200/350/450/600, for the confusion-matrix heatmap
SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#5598e7', '#2a78d6', '#184f95']

# PATCH: kept in sync by hand with notebooks/train_test.ipynb cell 19, which has
# its own copy of this dict and the same precision-recovery identity below. Not
# shared code -- cell 19 isn't practically importable from a script, and its
# logic is already validated against results/COMPARISON.md; duplicating six
# numbers is a smaller risk than touching working, load-bearing logic.
PAPER = {'DSC': 0.9091, 'IoU': 0.8334, 'ACC': 0.9646,
         'SE': 0.9053, 'SP': 0.9790, 'Prec': 0.9481}

ITER_RE = re.compile(r'train: epoch (\d+), iter:(\d+), loss: ([\d.]+), lr: ([\d.eE+-]+)')
EPOCH_SUMMARY_RE = re.compile(r'train epoch: (\d+) done, mean loss: ([\d.]+)')
VAL_RE = re.compile(
    r'val epoch: (\d+), loss: ([\d.]+)'
    r'(?:,\s*miou: ([\d.]+), f1_or_dsc: ([\d.]+), accuracy: ([\d.]+),'
    r'\s*specificity: ([\d.]+), sensitivity: ([\d.]+),'
    r'\s*confusion_matrix: \[\[.*?\]\])?',
    re.S)
TEST_RE = re.compile(
    r'test of best model, loss: ([\d.]+),\s*miou: ([\d.]+), f1_or_dsc: ([\d.]+), accuracy: ([\d.]+),'
    r'\s*specificity: ([\d.]+), sensitivity: ([\d.]+),'
    r'\s*confusion_matrix: (\[\[.*?\]\])',
    re.S)
PER_IMAGE_DICE_RE = re.compile(r'test image (\d+): dice: ([\d.]+)')


# ---------------------------------------------------------------- discovery --

def find_latest_run(results_dir=None):
    """Most recently modified results/*/ dir -- generalizes the glob+getmtime
    pattern notebooks/train_test.ipynb cell 16 already uses for one dataset's
    run-name prefix to any run under results/."""
    results_dir = results_dir or os.path.join(_HERE, 'results')
    runs = sorted((p for p in glob.glob(os.path.join(results_dir, '*')) if os.path.isdir(p)),
                  key=os.path.getmtime)
    if not runs:
        raise SystemExit(f'no run directories found under {results_dir}')
    return runs[-1]


def read_log_text(work_dir):
    logs = sorted(glob.glob(os.path.join(work_dir, 'log', '*.log')), key=os.path.getmtime)
    if not logs:
        raise SystemExit(f'no log files found under {os.path.join(work_dir, "log")}')
    return ''.join(open(p, encoding='utf-8', errors='replace').read() for p in logs)


# ------------------------------------------------------------------ parsing --

def parse_train_iter_lines(text):
    return [(int(e), int(it), float(l), float(lr)) for e, it, l, lr in ITER_RE.findall(text)]


def parse_train_epoch_summary(text):
    return {int(e): float(l) for e, l in EPOCH_SUMMARY_RE.findall(text)}


def derive_train_loss_per_epoch(text):
    """Prefers the 'train epoch: N done' summary line. Falls back, for logs that
    predate it, to each epoch's LAST print_interval-gated iter line -- a running
    mean over the most iterations seen in that epoch, the closest available
    proxy for the true epoch mean."""
    summary = parse_train_epoch_summary(text)
    if summary:
        return dict(sorted(summary.items()))
    last_per_epoch = {}
    for e, _it, loss, _lr in parse_train_iter_lines(text):
        last_per_epoch[e] = loss
    return dict(sorted(last_per_epoch.items()))


def parse_lr_per_epoch(text):
    """lr is constant within an epoch (scheduler.step() runs once, after the
    loop), so the first iter line seen per epoch is exact, not an approximation."""
    first_per_epoch = {}
    for e, _it, _loss, lr in parse_train_iter_lines(text):
        first_per_epoch.setdefault(e, lr)
    return dict(sorted(first_per_epoch.items()))


def parse_val_blocks(text):
    """One dict per 'val epoch' line, sorted by epoch. Metric fields (all but
    epoch/loss) are None on old logs' loss-only lines -- val_one_epoch used to
    log those only every config.val_interval epochs."""
    out = []
    for m in VAL_RE.finditer(text):
        e, loss, miou, dsc, acc, sp, se = m.groups()
        out.append({
            'epoch': int(e), 'loss': float(loss),
            'miou': float(miou) if miou is not None else None,
            'dsc': float(dsc) if dsc is not None else None,
            'acc': float(acc) if acc is not None else None,
            'sp': float(sp) if sp is not None else None,
            'se': float(se) if se is not None else None,
        })
    out.sort(key=lambda d: d['epoch'])
    return out


def _parse_confusion(cm_str):
    (tn, fp), (fn, tp) = ast.literal_eval(re.sub(r'\s+', ',', cm_str).replace('[,', '['))
    return tn, fp, fn, tp


def parse_test_summary(text):
    """The final 'test of best model' block (last match, matching notebook cell
    19's hits[-1] -- if test.py was re-run against this work_dir, use the most
    recent result), or None if the run hasn't been tested yet."""
    matches = TEST_RE.findall(text)
    if not matches:
        return None
    loss, miou, dsc, acc, sp, se, cm = matches[-1]
    tn, fp, fn, tp = _parse_confusion(cm)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return {
        'loss': float(loss), 'miou': float(miou), 'dsc': float(dsc),
        'acc': float(acc), 'sp': float(sp), 'se': float(se), 'prec': prec,
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp,
    }


def parse_per_image_test_dice(text):
    return sorted(((int(i), float(d)) for i, d in PER_IMAGE_DICE_RE.findall(text)),
                  key=lambda x: x[0])


# ----------------------------------------------------------------- plotting --

def _style_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(INK_MUTED)
    ax.spines['bottom'].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)


def _save(fig, out_dir, name):
    fig.tight_layout()
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_loss_curve(text, out_dir):
    train = derive_train_loss_per_epoch(text)
    val_blocks = parse_val_blocks(text)
    if not train and not val_blocks:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    if train:
        epochs, losses = zip(*sorted(train.items()))
        ax.plot(epochs, losses, color=BLUE, linewidth=2, label='train loss')
    if val_blocks:
        epochs = [b['epoch'] for b in val_blocks]
        losses = [b['loss'] for b in val_blocks]
        ax.plot(epochs, losses, color=ORANGE, linewidth=2, label='val loss')
        best_i = min(range(len(losses)), key=lambda i: losses[i])
        ax.axvline(epochs[best_i], color=INK_MUTED, linewidth=1, linestyle='--')
        ax.annotate(f'best epoch {epochs[best_i]}\nval loss {losses[best_i]:.4f}',
                    xy=(epochs[best_i], losses[best_i]), xytext=(8, 8),
                    textcoords='offset points', color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Training & validation loss')
    ax.legend(frameon=False)
    _style_axes(ax)
    return _save(fig, out_dir, 'loss_curve.png')


def plot_lr_curve(text, out_dir):
    lr = parse_lr_per_epoch(text)
    if not lr:
        return None
    epochs, lrs = zip(*sorted(lr.items()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, lrs, color=BLUE, linewidth=2)
    ax.set_xlabel('epoch')
    ax.set_ylabel('learning rate')
    ax.set_title('Learning rate schedule')
    _style_axes(ax)
    return _save(fig, out_dir, 'lr_curve.png')


def plot_val_metrics(text, out_dir):
    blocks = [b for b in parse_val_blocks(text) if b['dsc'] is not None]
    if not blocks:
        print('val_metrics.png: skipped -- no epochs with the full metric set in '
              'this log (pre-dates the always-log-full-metrics change in engine.py)')
        return None
    epochs = [b['epoch'] for b in blocks]
    # Old logs only have one full-metric point every ~30 epochs -- connecting
    # those with a line would imply data between samples that doesn't exist.
    dense = len(epochs) > 1 and all(b - a == 1 for a, b in zip(epochs, epochs[1:]))
    series = [('DSC', 'dsc', BLUE), ('IoU', 'miou', ORANGE), ('Accuracy', 'acc', AQUA),
              ('Sensitivity', 'se', YELLOW), ('Specificity', 'sp', MAGENTA)]
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, key, color in series:
        ys = [b[key] for b in blocks]
        if dense:
            ax.plot(epochs, ys, color=color, linewidth=2, label=label)
        else:
            ax.scatter(epochs, ys, color=color, s=36, label=label, zorder=3)
    ax.set_xlabel('epoch')
    ax.set_ylabel('score')
    ax.set_ylim(0, 1)
    ax.set_title('Validation metrics')
    ax.legend(frameon=False, ncol=3, loc='lower right')
    _style_axes(ax)
    return _save(fig, out_dir, 'val_metrics.png')


def plot_confusion_matrix(test, out_dir):
    if test is None:
        print('confusion_matrix.png: skipped -- no test results in this log yet')
        return None
    tn, fp, fn, tp = test['TN'], test['FP'], test['FN'], test['TP']
    mat = [[tn, fp], [fn, tp]]
    total = tn + fp + fn + tp
    vmax = max(max(row) for row in mat)
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
    cell_labels = [['TN', 'FP'], ['FN', 'TP']]
    for r in range(2):
        for c in range(2):
            val = mat[r][c]
            pct = val / total * 100
            color = 'white' if val > vmax * 0.55 else INK
            ax.text(c, r, f'{cell_labels[r][c]}\n{val:,}\n({pct:.1f}%)',
                    ha='center', va='center', color=color, fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['pred negative', 'pred positive'])
    ax.set_yticks([0, 1]); ax.set_yticklabels(['actual negative', 'actual positive'])
    ax.set_title('Test confusion matrix (pooled, all pixels)')
    return _save(fig, out_dir, 'confusion_matrix.png')


def plot_paper_vs_ours(test, out_dir):
    if test is None:
        print('paper_vs_ours.png: skipped -- no test results in this log yet')
        return None
    ours = {'DSC': test['dsc'], 'IoU': test['miou'], 'ACC': test['acc'],
            'SE': test['se'], 'SP': test['sp'], 'Prec': test['prec']}
    keys = ['DSC', 'IoU', 'SE', 'SP', 'ACC', 'Prec']
    x = list(range(len(keys)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    # ±0.01 tolerance band around the paper's DSC -- the primary metric this
    # replication is judged against (see results/COMPARISON.md).
    dsc_i = keys.index('DSC')
    ax.fill_between([dsc_i - 0.45, dsc_i + 0.45],
                     PAPER['DSC'] - 0.01, PAPER['DSC'] + 0.01, color=GOOD, alpha=0.15)
    ax.bar([i - width / 2 for i in x], [PAPER[k] for k in keys], width, color=BLUE, label='paper')
    ax.bar([i + width / 2 for i in x], [ours[k] for k in keys], width, color=ORANGE, label='ours')
    for i, k in enumerate(keys):
        delta = ours[k] - PAPER[k]
        ax.annotate(f'{delta:+.3f}', xy=(i, max(PAPER[k], ours[k]) + 0.015),
                    ha='center', fontsize=8, color=INK_SECONDARY)
    ax.set_xticks(x); ax.set_xticklabels(keys)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('score')
    ax.set_title('Final test metrics: paper vs. ours')
    ax.legend(frameon=False)
    _style_axes(ax)
    return _save(fig, out_dir, 'paper_vs_ours.png')


def plot_test_dice_distribution(text, out_dir):
    per_image = parse_per_image_test_dice(text)
    if not per_image:
        print('test_dice_distribution.png: skipped -- this log has no per-image '
              'test dice lines (pre-dates the per-image logging change in engine.py)')
        return None
    dices = [d for _, d in per_image]
    mean_d = sum(dices) / len(dices)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(dices, bins=30, color=BLUE, edgecolor='white', linewidth=0.5)
    ax.axvline(mean_d, color=INK_MUTED, linewidth=1.5, linestyle='--')
    ax.annotate(f'mean {mean_d:.4f}', xy=(mean_d, ax.get_ylim()[1] * 0.95),
                xytext=(6, 0), textcoords='offset points',
                color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel('per-image DSC')
    ax.set_ylabel('count')
    ax.set_title(f'Test-set per-image DSC distribution (n={len(dices)})')
    _style_axes(ax)
    return _save(fig, out_dir, 'test_dice_distribution.png')


# --------------------------------------------------------------------- CLI --

def generate_all(work_dir):
    """Parse <work_dir>'s log and write every plot to <work_dir>/plots/.
    Returns the list of PNG paths written (some plots may be skipped, with a
    printed message, if the log lacks the data they need)."""
    work_dir = work_dir.rstrip('/\\')
    text = read_log_text(work_dir)
    test = parse_test_summary(text)
    out_dir = os.path.join(work_dir, 'plots')
    os.makedirs(out_dir, exist_ok=True)

    plots = [
        plot_loss_curve(text, out_dir),
        plot_lr_curve(text, out_dir),
        plot_val_metrics(text, out_dir),
        plot_confusion_matrix(test, out_dir),
        plot_paper_vs_ours(test, out_dir),
        plot_test_dice_distribution(text, out_dir),
    ]
    return [p for p in plots if p]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--work-dir', help='a results/<run>/ dir '
                     '(default: the most recently modified results/*/ dir)')
    args = ap.parse_args()
    work_dir = args.work_dir or find_latest_run()
    print(f'reading log from: {work_dir}')
    paths = generate_all(work_dir)
    if paths:
        print('\nwrote:')
        for p in paths:
            print(f'  {p}')
    else:
        print('\nno plots written -- log had no parseable data')


if __name__ == '__main__':
    main()
