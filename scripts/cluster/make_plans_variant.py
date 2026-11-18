"""Write a plans variant whose 3d_fullres patch size is divisible by N.

Several external baselines hard-wire their total downsampling (SlimUNETR and
SegFormer3D: 32; SegMamba: 16), while nnU-Net plans a patch only divisible by
its *own* network's strides (ACDC: 20x256x224). Following nnU-Net Revisited's
"closest configuration" rule, those baselines train on a copy of the plans with
each patch dimension rounded *up* to the next multiple of N; everything else --
spacing, normalization, batch size, the preprocessed data itself
(same data_identifier) -- is unchanged, so no re-preprocessing is needed.

    python scripts/cluster/make_plans_variant.py -d 27 --divisible 32     # -> nnUNetPlans_div32.json
    nnUNetv2_train 27 3d_fullres 0 -tr nnUNetTrainerSlimUNETR -p nnUNetPlans_div32
"""

import argparse
import json
import math
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--datasets', nargs='+', required=True)
    ap.add_argument('--divisible', type=int, required=True)
    ap.add_argument('--source-plans', default='nnUNetPlans')
    ap.add_argument('-c', '--configuration', default='3d_fullres')
    args = ap.parse_args()

    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
    name = f'{args.source_plans}_div{args.divisible}'
    for d in args.datasets:
        ds = maybe_convert_to_dataset_name(d)
        src = os.path.join(nnUNet_preprocessed, ds, args.source_plans + '.json')
        if not os.path.isfile(src):
            sys.exit(f'missing {src}: plan/preprocess {ds} first')
        plans = json.load(open(src))
        cfg = plans['configurations'][args.configuration]
        old = cfg['patch_size']
        cfg['patch_size'] = [int(math.ceil(p / args.divisible) * args.divisible) for p in old]
        plans['plans_name'] = name
        # keep only the configuration we changed (and anything it inherits from)
        plans['configurations'] = {k: v for k, v in plans['configurations'].items()
                                   if k == args.configuration or k == cfg.get('inherits_from')}
        dst = os.path.join(nnUNet_preprocessed, ds, name + '.json')
        json.dump(plans, open(dst, 'w'), indent=2)
        print(f'{ds}: patch {old} -> {cfg["patch_size"]}  ({dst})')


if __name__ == '__main__':
    main()
