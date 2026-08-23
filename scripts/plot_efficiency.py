"""Plot the Week 4 efficiency-push Pareto curve: DSC vs. parameters, for the
channel-width variants trained by scripts/train_width_variant.py on ISIC2017.

Data is hardcoded from the completed runs (results/COMPARISON.md has the
full per-run detail) rather than parsed from logs -- this is a one-off
summary figure for three specific finished experiments, not a general log
parser the way scripts/plot_metrics.py is.

Usage:
    python scripts/plot_efficiency.py
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

from scripts.plot_metrics import BLUE, ORANGE, AQUA, INK, INK_SECONDARY, INK_MUTED, GRID, _style_axes, _save

# name, params, thop GFLOPs, DSC, color
POINTS = [
    ("baseline (paper c_list)", 49457, 0.0602, 0.8993, BLUE),
    ("half",                    14633, 0.0204, 0.9010, ORANGE),
    ("quarter",                  5581, 0.0141, 0.8976, AQUA),
]
NOISE_BAND = 0.003  # run-to-run noise floor established in COMPARISON.md's "Run 4" note


def main():
    out_dir = os.path.join(_HERE, 'results', 'efficiency')
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, xkey, xlabel in ((axes[0], 1, 'parameters'), (axes[1], 2, 'GFLOPs')):
        xs = [p[xkey] for p in POINTS]
        ys = [p[3] for p in POINTS]
        baseline_dsc = POINTS[0][3]
        ax.axhspan(baseline_dsc - NOISE_BAND, baseline_dsc + NOISE_BAND,
                  color=INK_MUTED, alpha=0.12, zorder=0,
                  label=f'±{NOISE_BAND} run-to-run noise floor')
        for name, params, gflops, dsc, color in POINTS:
            x = params if xkey == 1 else gflops
            ax.scatter([x], [dsc], color=color, s=90, zorder=3, edgecolor=INK, linewidth=0.8)
            ax.annotate(name, (x, dsc), textcoords='offset points', xytext=(8, 6),
                       fontsize=9, color=INK_SECONDARY)
        ax.plot(xs, ys, color=INK_MUTED, linewidth=1, linestyle='--', zorder=1)
        ax.set_xscale('log')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('DSC' if xkey == 1 else '')
        ax.set_title(f'DSC vs. {xlabel} (log scale)', fontsize=11, color=INK)
        _style_axes(ax)

    axes[0].legend(fontsize=8, frameon=False, loc='lower right')
    fig.suptitle('Channel-width scaling on ISIC2017: DSC barely moves down to ~9x fewer params',
                fontsize=11, color=INK)
    path = _save(fig, out_dir, 'width_scaling_pareto.png')
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
