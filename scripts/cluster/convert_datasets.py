"""Convert the seven benchmark datasets to nnU-Net format, plan/preprocess, verify.

Uses nnU-Net's own conversion code wherever it exists, so case identifiers --
and therefore nnU-Net's seeded default 5-fold split -- are the ones the nnU-Net
Revisited baselines were trained on (pvm3d/reference/README.md). Where nnU-Net's
script hard-codes its authors' paths (BraTS21) or has no script (BTCV), the
conversion is reproduced here with identifiers that keep the source ordering.

    source scripts/cluster/env.sh /scratch/$USER/pvm3d
    python scripts/cluster/convert_datasets.py msd    --src /data/Task04_Hippocampus   # -> Dataset004
    python scripts/cluster/convert_datasets.py msd    --src /data/Task03_Liver         # LiTS -> Dataset003
    python scripts/cluster/convert_datasets.py acdc   --src /data/ACDC/database        # -> Dataset027
    python scripts/cluster/convert_datasets.py kits23 --src /data/kits23/dataset       # -> Dataset220
    python scripts/cluster/convert_datasets.py amos   --src /data/amos22               # -> Dataset219
    python scripts/cluster/convert_datasets.py brats21 --src /data/BraTS2021_Training_Data  # -> Dataset137
    python scripts/cluster/convert_datasets.py btcv   --src /data/BTCV/RawData/Training     # -> Dataset017
    python scripts/cluster/convert_datasets.py plan   -d 4 27 220 219 3 137 17 -np 16
    python scripts/cluster/convert_datasets.py verify
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

EXPECTED = {  # dataset id -> (name fragment, labelled training cases)
    4: ('Hippocampus', 260), 27: ('ACDC', 200), 220: ('KiTS2023', 489), 219: ('AMOS2022', 360),
    3: ('Liver', 131), 137: ('BraTS2021', 1251), 17: ('BTCV', 30),
}

BTCV_LABELS = {'background': 0, 'spleen': 1, 'right kidney': 2, 'left kidney': 3, 'gallbladder': 4, 'esophagus': 5,
               'liver': 6, 'stomach': 7, 'aorta': 8, 'inferior vena cava': 9, 'portal and splenic vein': 10,
               'pancreas': 11, 'right adrenal gland': 12, 'left adrenal gland': 13}


def run(cmd):
    print('+', ' '.join(cmd))
    subprocess.run(cmd, check=True)


def conv_brats21(src):
    from nnunetv2.dataset_conversion.Dataset137_BraTS21 import copy_BraTS_segmentation_and_convert_labels_to_nnUNet
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    from nnunetv2.paths import nnUNet_raw
    out = os.path.join(nnUNet_raw, 'Dataset137_BraTS2021')
    imgs, lbls = os.path.join(out, 'imagesTr'), os.path.join(out, 'labelsTr')
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(lbls, exist_ok=True)
    cases = sorted(d for d in os.listdir(src) if d.startswith('BraTS') and os.path.isdir(os.path.join(src, d)))
    for c in cases:  # identical to nnunetv2/dataset_conversion/Dataset137_BraTS21.py __main__
        for k, mod in enumerate(('t1', 't1ce', 't2', 'flair')):
            shutil.copy(os.path.join(src, c, f'{c}_{mod}.nii.gz'), os.path.join(imgs, f'{c}_{k:04d}.nii.gz'))
        copy_BraTS_segmentation_and_convert_labels_to_nnUNet(os.path.join(src, c, f'{c}_seg.nii.gz'),
                                                             os.path.join(lbls, f'{c}.nii.gz'))
    generate_dataset_json(out, channel_names={0: 'T1', 1: 'T1ce', 2: 'T2', 3: 'Flair'},
                          labels={'background': 0, 'whole tumor': (1, 2, 3), 'tumor core': (2, 3),
                                  'enhancing tumor': (3,)},
                          num_training_cases=len(cases), file_ending='.nii.gz', regions_class_order=(1, 2, 3),
                          license='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          reference='see https://www.synapse.org/#!Synapse:syn25829067/wiki/610863',
                          dataset_release='1.0')
    print(f'BraTS2021: {len(cases)} cases -> {out}')


def conv_btcv(src):
    """BTCV RawData/Training/{img/imgXXXX.nii.gz,label/labelXXXX.nii.gz}. Identifiers
    btcv_XXXX keep the source numbering, hence the source order nnU-Net's split
    is computed over."""
    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    from nnunetv2.paths import nnUNet_raw
    out = os.path.join(nnUNet_raw, 'Dataset017_BTCV')
    imgs, lbls = os.path.join(out, 'imagesTr'), os.path.join(out, 'labelsTr')
    os.makedirs(imgs, exist_ok=True)
    os.makedirs(lbls, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, 'img', 'img*.nii.gz')))
    for f in files:
        num = os.path.basename(f)[3:7]
        shutil.copy(f, os.path.join(imgs, f'btcv_{num}_0000.nii.gz'))
        shutil.copy(os.path.join(src, 'label', f'label{num}.nii.gz'), os.path.join(lbls, f'btcv_{num}.nii.gz'))
    generate_dataset_json(out, channel_names={0: 'CT'}, labels=BTCV_LABELS, num_training_cases=len(files),
                          file_ending='.nii.gz', dataset_name='BTCV',
                          reference='https://www.synapse.org/#!Synapse:syn3193805')
    print(f'BTCV: {len(files)} cases -> {out}')


def amos_modality_map(ds_dir, n_proc=8):
    """{case: 'CT'|'MRI'} from image intensities (CT is in Hounsfield units, far
    below 0; MR is not). Written to <ds>/modality.json, read by amos_is_mri()."""
    from multiprocessing import Pool
    out = os.path.join(ds_dir, 'modality.json')
    if os.path.isfile(out):
        return json.load(open(out))
    imgs = sorted(glob.glob(os.path.join(ds_dir, 'imagesTr', '*_0000.nii.gz')))
    with Pool(n_proc) as pool:
        mins = pool.map(_image_min, imgs)
    m = {os.path.basename(f)[:-len('_0000.nii.gz')]: ('CT' if v < -200 else 'MRI') for f, v in zip(imgs, mins)}
    json.dump(m, open(out, 'w'), indent=0, sort_keys=True)
    return m


def _image_min(path):
    import SimpleITK as sitk
    return float(sitk.GetArrayViewFromImage(sitk.ReadImage(path)).min())


def verify(n_proc=8):
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
    ok = True
    for did, (frag, n_exp) in EXPECTED.items():
        cands = glob.glob(os.path.join(nnUNet_raw, f'Dataset{did:03d}_*'))
        if not cands:
            print(f'Dataset{did:03d} ({frag}): not converted yet')
            continue
        ds = cands[0]
        labels = [f for f in os.listdir(os.path.join(ds, 'labelsTr')) if f.endswith('.nii.gz')]
        n = len(labels)
        msg = f'{os.path.basename(ds)}: {n} labelled cases (expected {n_exp})'
        good = n == n_exp
        if did == 219:
            mod = amos_modality_map(ds, n_proc)
            n_mri = sum(mod.get(f[:-7]) == 'MRI' for f in labels)
            by_id = {f[:-7] for f in labels if int(f[:-7].split('_')[-1]) >= 500}
            disagree = sorted(c for c in (f[:-7] for f in labels) if (mod.get(c) == 'MRI') != (c in by_id))
            msg += (f'; CT {n - n_mri} / MRI {n_mri} from image intensities (expected 300 / 60); '
                    f'id>=500 rule disagrees on {disagree or "no cases"}')
            good &= (n - n_mri, n_mri) == (300, 60)
        pre = os.path.join(nnUNet_preprocessed, os.path.basename(ds))
        msg += '; plans ' + ('present' if os.path.isfile(os.path.join(pre, 'nnUNetPlans.json')) else 'MISSING')
        sp = os.path.join(pre, 'splits_final.json')
        if os.path.isfile(sp):
            folds = json.load(open(sp))
            msg += f'; splits_final.json: {len(folds)} folds, val sizes {[len(f["val"]) for f in folds]}'
        print(('OK   ' if good else 'FAIL ') + msg)
        ok &= good
    if not ok:
        sys.exit('verification failed')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('what', choices=['msd', 'acdc', 'kits23', 'amos', 'brats21', 'btcv', 'plan', 'verify'])
    ap.add_argument('--src')
    ap.add_argument('-d', nargs='+', type=int, help='dataset ids for plan')
    ap.add_argument('-np', type=int, default=8)
    args = ap.parse_args()
    for v in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        if not os.environ.get(v):
            sys.exit(f'{v} not set -- source scripts/cluster/env.sh first')
    py = sys.executable
    if args.what == 'msd':
        run(['nnUNetv2_convert_MSD_dataset', '-i', args.src, '-np', str(args.np)])
    elif args.what == 'acdc':
        run([py, '-m', 'nnunetv2.dataset_conversion.Dataset027_ACDC', '-i', args.src])
    elif args.what == 'kits23':
        run([py, '-m', 'nnunetv2.dataset_conversion.Dataset220_KiTS2023', args.src])
    elif args.what == 'amos':
        run([py, '-m', 'nnunetv2.dataset_conversion.Dataset219_Amos2022_task2', args.src])
    elif args.what == 'brats21':
        conv_brats21(args.src)
    elif args.what == 'btcv':
        conv_btcv(args.src)
    elif args.what == 'plan':
        run(['nnUNetv2_plan_and_preprocess', '-d', *map(str, args.d), '-c', '3d_fullres', '-np', str(args.np),
             '--verify_dataset_integrity'])
    elif args.what == 'verify':
        verify(args.np)


if __name__ == '__main__':
    main()
