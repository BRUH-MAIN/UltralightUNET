"""scripts/stats3d.py: aggregation definitions and paired-test behaviour on
synthetic data with a planted effect."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from stats3d import fold_scores, holm, paired_tests, per_case, rank_stability  # noqa: E402


def synth(effects, n_cases=40, seed=0, regions=('a', 'b')):
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.6, 0.95, size=(5, n_cases, len(regions)))
    rows = []
    for model, eff in effects.items():
        noise = rng.normal(0, 0.01, size=base.shape)
        vals = np.clip(base + eff + noise, 0, 1)
        for f in range(5):
            for c in range(n_cases):
                for k, r in enumerate(regions):
                    rows.append(dict(dataset='Dataset027_ACDC', trainer=model, fold=str(f), case=f'c{f}_{c}', region=r,
                                     dsc=vals[f, c, k], nsd=vals[f, c, k], hd95=10 * (1 - vals[f, c, k])))
    return pd.DataFrame(rows)


def test_holm_matches_definition():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_fold_score_is_mean_over_regions_of_mean_over_cases():
    df = pd.DataFrame(dict(dataset='d', trainer='m', fold='0', case=['x', 'x', 'y'], region=['a', 'b', 'a'],
                           dsc=[1.0, 0.0, 0.5], nsd=0.0, hd95=0.0))
    s = fold_scores(df)
    # region a: mean(1.0, 0.5)=0.75, region b: 0.0 -> 0.375 (nnU-Net foreground_mean convention)
    assert s.dsc_mean.iloc[0] == pytest.approx(0.375)


def test_paired_tests_find_planted_effect_and_not_a_null_one():
    df = synth({'ref': 0.0, 'worse': -0.02, 'same': 0.0})
    p = paired_tests(per_case(df), 'ref', n_boot=2000, seed=0)
    d = p[p.metric == 'dsc'].set_index('other')
    assert d.loc['worse', 'significant_0.05'] and d.loc['worse', 'ci_lo'] > 0
    assert not d.loc['same', 'significant_0.05']
    assert d.loc['same', 'ci_lo'] < 0 < d.loc['same', 'ci_hi']


def test_rank_stability_orders_models():
    df = synth({'best': 0.03, 'mid': 0.0, 'worst': -0.03})
    r = rank_stability(per_case(df), n_boot=500, seed=0).set_index('model')
    assert r.loc['best', 'mean_rank'] < r.loc['mid', 'mean_rank'] < r.loc['worst', 'mean_rank']
