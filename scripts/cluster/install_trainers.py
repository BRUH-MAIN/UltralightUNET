"""Register pvm3d/nnunet_trainers/ inside the installed nnunetv2 package.

Why not just nnUNet_extTrainer: nnU-Net 2.8.1's external-trainer lookup
(nnunetv2/utilities/find_class_by_name.py: temporarily_cleanup_imports_from_path)
iterates ``sys.modules.items()`` while inspecting module attributes, which can
trigger lazy imports and raise "dictionary changed size during iteration".
nnUNetv2_train happens to survive because it looks the trainer up once, early;
a second lookup in the same process (e.g. our benchmarking scripts) reliably
crashes. A symlink under nnunetv2/training/nnUNetTrainer/variants/ makes the
*primary* lookup find our trainers, and that path does no sys.modules cleanup.

Idempotent. Run once per environment (scripts/cluster/setup_env.sh calls it):

    python scripts/cluster/install_trainers.py           # create + verify
    python scripts/cluster/install_trainers.py --remove
"""

import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, 'pvm3d', 'nnunet_trainers')
CHECK = ['nnUNetTrainerPVM3D_S', 'nnUNetTrainerPVM3D_S_dd', 'nnUNetTrainerTinyUNet_matchS', 'nnUNetTrainer_frac10']


def link_path():
    import nnunetv2
    return os.path.join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer', 'variants', 'pvm3d')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--remove', action='store_true')
    args = ap.parse_args()
    dst = link_path()
    if args.remove:
        if os.path.islink(dst):
            os.unlink(dst)
            print(f'removed {dst}')
        return
    if os.path.islink(dst):
        if os.path.realpath(dst) != os.path.realpath(SRC):
            sys.exit(f'{dst} points to {os.path.realpath(dst)}, not {SRC}; remove it first (--remove)')
    elif os.path.exists(dst):
        sys.exit(f'{dst} exists and is not a symlink; refusing to touch it')
    else:
        os.symlink(SRC, dst)
        print(f'linked {dst} -> {SRC}')

    env = {k: v for k, v in os.environ.items() if k != 'nnUNet_extTrainer'}  # prove the primary path works
    code = ('from nnunetv2.utilities.find_objects import recursive_find_trainer_class_by_name as f\n'
            f'for n in {CHECK!r}:\n'
            '    assert f(n).__name__ == n, n\n'
            'print("OK")\n')
    r = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, cwd='/')
    if 'OK' not in r.stdout:
        sys.exit('trainer discovery failed:\n' + r.stdout[-3000:] + r.stderr[-3000:])
    print(f'verified: {", ".join(CHECK)} resolve without nnUNet_extTrainer')


if __name__ == '__main__':
    main()
