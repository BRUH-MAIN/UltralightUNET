"""Build the MILK10k lesion-level manifest with a stratified, seeded split.

Produces one row per lesion (not per image), joining:
  - the diagnosis label from MILK10k_Training_GroundTruth.csv (one-hot, 11 classes)
  - the clinical + dermatoscopic image ids and metadata from ..._Metadata.csv

and assigns each lesion to train / val / test.

Two hard requirements, both learned the hard way in Phase 1:

  1. **Lesion-level split.** Each lesion has two images (clinical + dermatoscopic).
     Splitting by image would leak a lesion's two views across train and test.
     We split by lesion_id so both views always stay on the same side.

  2. **Class-stratified.** The dataset is extremely imbalanced (BCC 48% down to
     MAL_OTH at 9 lesions, a 280x ratio). macro-F1 is undefined for a class with
     no test samples, so every class must appear in every split. Stratification
     guarantees it; the rarest class (9 lesions) lands ~5/2/2.

The output manifest is small (~5,240 rows) and is committed to git, so the split
is version-controlled and reproducible on any machine. The images themselves come
straight from ISIC (milk10k/scripts/download_milk10k.py); they are not re-hosted.
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# canonical class order == GroundTruth column order
CLASSES = ['AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF', 'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC']
SPLIT_SEED = 42          # matches config.seed and the Phase-1 convention
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def build_records(root):
    gt = pd.read_csv(os.path.join(root, 'MILK10k_Training_GroundTruth.csv'))
    meta = pd.read_csv(os.path.join(root, 'MILK10k_Training_Metadata.csv'))

    assert list(gt.columns[1:]) == CLASSES, 'unexpected GroundTruth columns'
    assert np.allclose(gt[CLASSES].sum(axis=1), 1.0), 'labels are not clean one-hot'

    gt = gt.copy()
    gt['label'] = np.array(CLASSES)[gt[CLASSES].values.argmax(1)]
    gt['label_idx'] = gt[CLASSES].values.argmax(1)

    # one metadata row per (lesion, modality); pivot to one row per lesion
    is_derm = meta['image_type'].str.contains('dermoscopic', case=False)
    clin = meta[~is_derm].set_index('lesion_id')
    derm = meta[is_derm].set_index('lesion_id')

    rec = gt[['lesion_id', 'label', 'label_idx']].set_index('lesion_id')
    rec['clinical_isic_id'] = clin['isic_id']
    rec['derm_isic_id'] = derm['isic_id']
    # lesion-level metadata (same across the two images); take from the derm row,
    # fall back to clinical where missing
    for col in ['age_approx', 'sex', 'site', 'skin_tone_class']:
        rec[col] = derm[col].combine_first(clin[col])

    rec = rec.reset_index()
    assert rec['clinical_isic_id'].notna().all(), 'some lesion is missing a clinical image'
    assert rec['derm_isic_id'].notna().all(), 'some lesion is missing a dermatoscopic image'
    return rec


def split_records(rec):
    y = rec['label_idx'].values
    idx = np.arange(len(rec))
    # 70 / 15 / 15, stratified by class, seeded. Two-stage: first hold out
    # (val+test), then halve that into val and test -- both stratified.
    train_idx, temp_idx = train_test_split(
        idx, test_size=VAL_FRAC + TEST_FRAC, stratify=y, random_state=SPLIT_SEED)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=y[temp_idx], random_state=SPLIT_SEED)

    split = np.empty(len(rec), dtype=object)
    split[train_idx] = 'train'
    split[val_idx] = 'val'
    split[test_idx] = 'test'
    rec = rec.copy()
    rec['split'] = split
    return rec


def check(rec):
    # (1) no lesion in more than one split -- guaranteed by construction, asserted anyway
    assert rec['lesion_id'].is_unique, 'duplicate lesion_id'
    # (2) every class present in every split, or macro-F1 breaks
    tab = pd.crosstab(rec['label'], rec['split'])
    tab = tab.reindex(index=CLASSES, columns=['train', 'val', 'test'], fill_value=0)
    print('\nclass x split (lesion counts):')
    print(tab.to_string())
    missing = tab[(tab == 0).any(axis=1)]
    if len(missing):
        raise SystemExit(f'\nclasses absent from a split (macro-F1 undefined):\n{missing}')
    print('\nsplit totals:', dict(rec['split'].value_counts()))
    print('every class present in every split: OK')

    # informational: the balanced class weights the loss will use, from TRAIN only.
    # weight_i = N_train / (n_classes * n_i) -- standard inverse-frequency, mean ~1.
    tr = rec[rec.split == 'train']['label_idx'].value_counts().reindex(range(len(CLASSES))).fillna(0)
    w = tr.sum() / (len(CLASSES) * tr.clip(lower=1))
    print('\ninverse-frequency class weights (train):')
    for i, c in enumerate(CLASSES):
        print(f'  {c:8s} n={int(tr[i]):4d}  weight={w[i]:.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='milk10k/data')
    # tracked in git (defines the reproducible split); kept out of the ignored data/ dir
    ap.add_argument('--out', default='milk10k/milk10k_manifest.csv')
    args = ap.parse_args()

    rec = build_records(args.root)
    print(f'built {len(rec)} lesion records')
    rec = split_records(rec)
    check(rec)

    cols = ['lesion_id', 'split', 'label', 'label_idx', 'clinical_isic_id', 'derm_isic_id',
            'skin_tone_class', 'age_approx', 'sex', 'site']
    rec[cols].to_csv(args.out, index=False)
    print(f'\nwrote {args.out}  ({os.path.getsize(args.out)/1e3:.0f} KB, {len(rec)} rows)')


if __name__ == '__main__':
    main()
