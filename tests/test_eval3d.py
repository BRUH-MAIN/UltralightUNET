"""scripts/eval3d.py metric definitions.

Two kinds of check: closed-form geometry (where DSC/NSD/HD95 are known by
hand), and agreement with nnU-Net's own evaluator on non-trivial predictions,
for both label- and region-based datasets -- the same cross-check eval3d.py
runs against every fold's summary.json.
"""

import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'scripts'))

from eval3d import _eval_case, mask_metrics, regions_from_dataset_json  # noqa: E402


def cube(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = True
    return m


def test_identical_masks():
    m = cube((20, 20, 20), (5, 5, 5), (15, 15, 15))
    assert mask_metrics(m, m, (1, 1, 1), 2.0) == (1.0, 1.0, 0.0)


def test_shift_one_voxel_respects_spacing():
    a = cube((20, 20, 20), (5, 5, 5), (15, 15, 15))
    b = cube((20, 20, 20), (5, 5, 6), (15, 15, 16))      # shifted 1 voxel along the last axis
    dsc, nsd2, hd = mask_metrics(a, b, (1.0, 1.0, 2.0), 2.0)  # that axis is 2 mm
    assert dsc == pytest.approx(2 * 900 / 2000)
    assert nsd2 == pytest.approx(1.0)                    # every surface point within 2 mm
    assert hd == pytest.approx(2.0)
    _, nsd1, _ = mask_metrics(a, b, (1.0, 1.0, 2.0), 1.0)
    assert nsd1 < 1.0                                    # the shifted faces are 2 mm off


def test_empty_conventions():
    z = np.zeros((8, 8, 8), bool)
    m = cube((8, 8, 8), (2, 2, 2), (5, 5, 5))
    assert all(np.isnan(v) for v in mask_metrics(z, z, (1, 1, 1), 2.0))
    dsc, nsd, hd = mask_metrics(m, z, (1, 1, 1), 2.0)
    assert dsc == 0.0 and nsd == 0.0 and np.isnan(hd)


def test_regions_from_dataset_json():
    dj = {'labels': {'background': 0, 'whole tumor': [1, 2, 3], 'tumor core': [2, 3], 'enhancing': 3}}
    assert regions_from_dataset_json(dj) == [('whole tumor', (1, 2, 3)), ('tumor core', (2, 3)),
                                             ('enhancing', (3,))]


@pytest.mark.parametrize('use_regions', [False, True])
def test_dsc_matches_nnunet_evaluator(tmp_path, use_regions):
    sitk = pytest.importorskip('SimpleITK')
    from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    from scipy import ndimage

    rng = np.random.default_rng(0)
    ref_dir, pred_dir = tmp_path / 'ref', tmp_path / 'pred'
    ref_dir.mkdir()
    pred_dir.mkdir()
    for i in range(4):
        gt = np.zeros((24, 32, 28), np.uint8)
        gt[4:18, 6:26, 5:22] = 1
        gt[8:14, 10:20, 9:18] = 2
        gt[10:12, 13:17, 12:15] = 3
        pred = gt.copy()
        pred[ndimage.binary_dilation(gt == 2, iterations=1 + i % 2) & (gt == 1)] = 2  # over-segment class 2
        pred[rng.random(gt.shape) < 0.02] = 0                                        # salt noise
        for arr, d in ((gt, ref_dir), (pred, pred_dir)):
            img = sitk.GetImageFromArray(arr)
            img.SetSpacing((0.8, 0.8, 3.0))  # anisotropic, as for ACDC
            sitk.WriteImage(img, str(d / f'case_{i}.nii.gz'))

    regions = [(1, 2, 3), (2, 3), (3,)] if use_regions else [1, 2, 3]
    ref = compute_metrics_on_folder(str(ref_dir), str(pred_dir), None, SimpleITKIO(), '.nii.gz', regions,
                                    num_processes=1)
    named = [(str(r), tuple(r) if isinstance(r, tuple) else (r,)) for r in regions]
    ours = []
    for i in range(4):
        ours += _eval_case((f'case_{i}', str(pred_dir / f'case_{i}.nii.gz'), str(ref_dir / f'case_{i}.nii.gz'),
                            named, 2.0, {}))
    for (name, _), r in zip(named, regions):
        mine = np.nanmean([row['dsc'] for row in ours if row['region'] == name])
        theirs = ref['mean'][r]['Dice']  # nnU-Net keys by the label int / region tuple itself
        assert mine == pytest.approx(theirs, abs=1e-6), (name, mine, theirs)
        assert all(0 <= row['nsd'] <= 1 and row['hd95'] >= 0 for row in ours)
