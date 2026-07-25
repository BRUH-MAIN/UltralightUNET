"""Correctness checks for the MILK10k pipeline.

Run from the repo root:  python -m pytest milk10k/tests -q

The split-integrity tests are the most important: they encode the two lessons
from Phase 1 -- a biased/leaky split silently wrecks the result, so it is asserted,
not trusted.
"""

import os

import numpy as np
import pandas as pd
import pytest
import torch

from milk10k.configs.config import config, CLASSES
from milk10k.pvm_classifier import MILK10kClassifier
from milk10k.engine import evaluate, build_criterion
from models.UltraLight_VM_UNet import PVMLayer

MANIFEST = config.manifest


# ---- split integrity (needs only the manifest, not the images) ----

@pytest.fixture(scope='module')
def manifest():
    if not os.path.exists(MANIFEST):
        pytest.skip('manifest not built yet (run prepare_milk10k.py)')
    return pd.read_csv(MANIFEST)


def test_no_lesion_in_two_splits(manifest):
    assert manifest['lesion_id'].is_unique, 'a lesion appears more than once'
    # every lesion has exactly one split label
    assert manifest['split'].isin(['train', 'val', 'test']).all()


def test_every_class_in_every_split(manifest):
    tab = pd.crosstab(manifest['label'], manifest['split'])
    tab = tab.reindex(index=CLASSES, columns=['train', 'val', 'test'], fill_value=0)
    absent = tab[(tab == 0).any(axis=1)]
    assert len(absent) == 0, f'macro-F1 undefined -- classes absent from a split:\n{absent}'


def test_split_sizes_reasonable(manifest):
    frac = manifest['split'].value_counts(normalize=True)
    assert 0.65 <= frac['train'] <= 0.75
    assert 0.10 <= frac['val'] <= 0.20
    assert 0.10 <= frac['test'] <= 0.20


def test_labels_in_range(manifest):
    assert manifest['label_idx'].between(0, len(CLASSES) - 1).all()


# ---- model ----

@pytest.mark.parametrize('modality,fusion', [
    ('derm', 'concat'), ('clin', 'concat'), ('both', 'concat'), ('both', 'cross_mamba')])
def test_model_forward_and_reuses_pvmlayer(modality, fusion):
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = MILK10kClassifier(num_classes=11, modality=modality, fusion=fusion).to(dev)
    # the classifier must genuinely reuse the replicated PVMLayer
    assert any(isinstance(mod, PVMLayer) for mod in m.modules())
    x = {k: torch.randn(2, 3, 256, 256, device=dev)
         for k in (('derm', 'clin') if modality == 'both' else (modality,))}
    out = m(x)
    assert out.shape == (2, 11)
    assert sum(p.numel() for p in m.parameters()) < 1_000_000  # genuinely lightweight


def test_lightweight_budget():
    """The single-modality classifier stays well under the 0.049M segmentation model."""
    m = MILK10kClassifier(num_classes=11, modality='derm')
    assert sum(p.numel() for p in m.parameters()) < 49_457


# ---- metric plumbing (synthetic, no images) ----

def test_macro_f1_perfect_and_zero():
    from sklearn.metrics import f1_score
    y = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    assert f1_score(y, y, average='macro') == 1.0
    wrong = (y + 1) % 4
    assert f1_score(y, wrong, average='macro', zero_division=0) < 0.5
