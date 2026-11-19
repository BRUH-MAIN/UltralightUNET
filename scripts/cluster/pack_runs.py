"""Pack many nnU-Net trainings onto GPU allocations, resumably (roadmap §7).

Lightweight models leave most of a 48/80 GB GPU idle and are bound by the CPU
data-augmentation workers, so several runs share one GPU. This keeps a fixed
"capacity" of concurrent runs busy, resumes interrupted runs with --c, stops
launching new runs before the allocation's end, and reports progress.

Several SLURM jobs may work through the SAME run file at once (one per GPU):
each run is claimed with a lock file in its nnU-Net output folder
(``.pack_lock``, created atomically, heart-beaten every poll); a lock nobody has
touched for 15 min belongs to a dead job and is taken over.

Run file (one run per line; '#' comments; fold may be a range or list):

    # dataset  config      folds  trainer                  [plans=..] [weight=..] [venv=..]
    27         3d_fullres  0-4    nnUNetTrainerPVM3D_S
    27         3d_fullres  0      nnUNetTrainer            weight=2
    220        3d_fullres  0,2    nnUNetTrainerPVM3D_S_dd

weight = capacity units a run occupies (heavy baselines like nnU-Net (org.) or
SegMamba saturate the GPU on their own; give them weight >= 2).
venv   = run with that environment's nnUNetv2_train (SegMamba needs its own venv).

    source scripts/cluster/env.sh <DATA_ROOT>
    python scripts/cluster/pack_runs.py runs/S1.txt --capacity 4 --slurm-auto     # inside an sbatch job
    python scripts/cluster/pack_runs.py runs/S1.txt --capacity 4 --launch-window-h 66 --hard-stop-h 71.5
    python scripts/cluster/pack_runs.py runs/S1.txt --status
    python scripts/cluster/pack_runs.py runs/S1.txt --dry-run

State per run is read from nnU-Net's own output folder, so the packer itself is
stateless: resubmitting the job just continues where things are.
"""

import argparse
import glob
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field

POLL_S = 30
LOCK_NAME = '.pack_lock'
STALE_S = 15 * 60
OWNER = f'{socket.gethostname()}:{os.getpid()}:job{os.environ.get("SLURM_JOB_ID", "-")}'


@dataclass
class Run:
    dataset: str
    config: str
    fold: str
    trainer: str
    plans: str = 'nnUNetPlans'
    weight: int = 1
    venv: str = ''
    attempts: int = 0
    proc: object = field(default=None, repr=False)
    gpu: str = ''
    started: float = 0.0

    @property
    def key(self):
        return f'{self.dataset}_{self.trainer}_{self.plans}_{self.config}_f{self.fold}'

    def out_dir(self):
        from nnunetv2.paths import nnUNet_results
        from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
        ds = maybe_convert_to_dataset_name(self.dataset)
        return os.path.join(nnUNet_results, ds, f'{self.trainer}__{self.plans}__{self.config}', f'fold_{self.fold}')

    def has_data(self):
        """Dataset converted and planned with this run's plans? Runs on datasets that
        are not ready yet are skipped (state 'nodata') and picked up by a later job."""
        from nnunetv2.paths import nnUNet_preprocessed
        from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
        try:
            ds = maybe_convert_to_dataset_name(self.dataset)
        except RuntimeError:
            return False
        return os.path.isfile(os.path.join(nnUNet_preprocessed, ds, self.plans + '.json'))

    def state(self):
        if not self.has_data():
            return 'nodata'
        d = self.out_dir()
        if os.path.isfile(os.path.join(d, 'checkpoint_final.pth')) and \
                os.path.isfile(os.path.join(d, 'validation', 'summary.json')):
            return 'done'
        if any(os.path.isfile(os.path.join(d, f)) for f in ('checkpoint_latest.pth', 'checkpoint_final.pth')):
            return 'partial'
        return 'new'


# ---------------------------------------------------------------- run locks
def lock_path(run):
    return os.path.join(run.out_dir(), LOCK_NAME)


def lock_owner(run):
    """Owner string of a live lock, or None if unlocked / stale."""
    p = lock_path(run)
    try:
        if time.time() - os.path.getmtime(p) > STALE_S:
            return None
        return open(p).read().strip() or '?'
    except FileNotFoundError:
        return None


def try_lock(run):
    p = lock_path(run)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    for _ in range(3):
        try:
            fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.write(fd, OWNER.encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                stale = time.time() - os.path.getmtime(p) > STALE_S
            except FileNotFoundError:
                continue
            if not stale:
                return False
            # take over a dead job's lock: rename is atomic, so only one claimant wins
            grave = f'{p}.stale.{OWNER.replace(":", "_")}'
            try:
                os.rename(p, grave)
                os.remove(grave)
            except FileNotFoundError:
                pass
    return False


def heartbeat(run):
    try:
        os.utime(lock_path(run), None)
    except FileNotFoundError:  # someone judged us stale; reclaim
        try_lock(run)


def release(run):
    p = lock_path(run)
    try:
        if open(p).read().strip() == OWNER:
            os.remove(p)
    except FileNotFoundError:
        pass


def slurm_hours_left():
    """Remaining wall time of the enclosing SLURM job, from `squeue -o %L`."""
    job = os.environ.get('SLURM_JOB_ID')
    if not job:
        return None
    out = subprocess.run(['squeue', '-h', '-j', job, '-o', '%L'], capture_output=True, text=True).stdout.strip()
    return parse_slurm_time(out)


def parse_slurm_time(s):
    """'1-02:03:04' | '02:03:04' | '03:04' | 'UNLIMITED' -> hours (None if unlimited/unknown)."""
    if not s or s.upper() in ('UNLIMITED', 'INVALID', 'NOT_SET'):
        return None
    days = 0
    if '-' in s:
        d, s = s.split('-', 1)
        days = int(d)
    parts = [int(x) for x in s.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 24 + h + m / 60 + sec / 3600


def parse_folds(s):
    if s == 'all':
        return ['all']
    out = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            out += [str(i) for i in range(int(a), int(b) + 1)]
        else:
            out.append(part)
    return out


def parse_runfile(path):
    runs = []
    for ln, line in enumerate(open(path), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        tok = line.split()
        if len(tok) < 4:
            sys.exit(f'{path}:{ln}: need "dataset config folds trainer [key=val ...]"')
        kv = dict(t.split('=', 1) for t in tok[4:])
        for f in parse_folds(tok[2]):
            runs.append(Run(tok[0], tok[1], f, tok[3], plans=kv.get('plans', 'nnUNetPlans'),
                            weight=int(kv.get('weight', 1)), venv=os.path.expandvars(kv.get('venv', ''))))
    return runs


def progress(run):
    """(current epoch, mean recent epoch seconds, last fold Dice) from nnU-Net's log."""
    logs = sorted(glob.glob(os.path.join(run.out_dir(), 'training_log_*.txt')))
    epoch, times = None, []
    for lg in logs[-2:]:
        for line in open(lg, errors='ignore'):
            m = re.search(r'Epoch (\d+)\s*$', line)
            if m:
                epoch = int(m.group(1))
            m = re.search(r'Epoch time: ([\d.]+) s', line)
            if m:
                times.append(float(m.group(1)))
    dice = None
    summ = os.path.join(run.out_dir(), 'validation', 'summary.json')
    if os.path.isfile(summ):
        dice = json.load(open(summ))['foreground_mean']['Dice']
    return epoch, (sum(times[-20:]) / len(times[-20:]) if times else None), dice


def status(runs):
    print(f'{"run":<78} {"state":<8} {"epoch":>6} {"s/ep":>6} {"ETA h":>6} {"Dice":>7}')
    for r in runs:
        st = r.state()
        if st == 'nodata':
            print(f'{r.key:<78} {st:<8}')
            continue
        if st != 'done' and lock_owner(r):
            st = 'running'
        ep, spe, dice = progress(r)
        eta = (1000 - (ep or 0)) * spe / 3600 if spe and st != 'done' else None
        print(f'{r.key:<78} {st:<8} {ep if ep is not None else "":>6} {f"{spe:.0f}" if spe else "":>6} '
              f'{f"{eta:.1f}" if eta else "":>6} {f"{dice:.4f}" if dice is not None else "":>7}')
    n_done = sum(r.state() == 'done' for r in runs)  # noqa: re-read, runs may have finished meanwhile
    print(f'{n_done}/{len(runs)} done')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runfile')
    ap.add_argument('--capacity', type=int, default=4, help='concurrent capacity units per GPU')
    ap.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'), help='comma-separated GPU ids')
    ap.add_argument('--cores', type=int, default=len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                    else os.cpu_count())
    ap.add_argument('--launch-window-h', type=float, default=66.0,
                    help='do not start new runs after this many hours (leave time to reach a checkpoint)')
    ap.add_argument('--hard-stop-h', type=float, default=71.5, help='terminate everything at this point')
    ap.add_argument('--slurm-auto', action='store_true',
                    help='inside an sbatch job: set the hard stop 6 min and the launch window 21 min before the '
                         "job's time limit (from squeue)")
    ap.add_argument('--max-attempts', type=int, default=2, help='attempts per run in this session')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()

    for v in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        if not os.environ.get(v):
            sys.exit(f'{v} is not set -- source scripts/cluster/env.sh first')
    runs = parse_runfile(args.runfile)
    if args.status:
        status(runs)
        return

    if args.slurm_auto:
        left = slurm_hours_left()
        if left is None:
            sys.exit('--slurm-auto: not inside a SLURM job with a time limit')
        args.hard_stop_h = max(0.05, left - 0.1)
        args.launch_window_h = max(0.0, left - 0.35)
        print(f'SLURM job {os.environ["SLURM_JOB_ID"]}: {left:.2f} h left -> launch until {args.launch_window_h:.2f} h, '
              f'hard stop at {args.hard_stop_h:.2f} h')

    gpus = [g for g in args.gpus.split(',') if g != '']
    total_cap = args.capacity * len(gpus)
    da_per_unit = max(2, (args.cores - 2) // total_cap)
    from nnunetv2.paths import nnUNet_results
    logdir = os.path.join(nnUNet_results, '_pack_logs')
    os.makedirs(logdir, exist_ok=True)

    states = {r.key: r.state() for r in runs}
    queue = [r for r in runs if states[r.key] not in ('done', 'nodata')]
    n_nodata = sum(st == 'nodata' for st in states.values())
    print(f'{len(runs)} runs, {sum(st == "done" for st in states.values())} already done, {len(queue)} to go, '
          f'{n_nodata} waiting for their dataset/plans; capacity {args.capacity} x {len(gpus)} GPU(s), '
          f'{da_per_unit} DA workers per unit')
    if args.dry_run:
        for r in queue:
            owner = lock_owner(r)
            print(f'  would run: {r.key}  [{r.state()}] weight={r.weight}'
                  + (f'  (currently claimed by {owner})' if owner else ''))
        return

    t0 = time.time()
    running = []
    started_keys = set()
    stop = {'flag': False}

    def on_signal(signum, frame):
        stop['flag'] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def load(g):
        return sum(r.weight for r in running if r.gpu == g)

    while (queue or running) and not stop['flag']:
        hours = (time.time() - t0) / 3600
        if hours >= args.hard_stop_h:
            print(f'[{hours:.2f} h] hard stop reached')
            break
        for r in list(running):
            rc = r.proc.poll()
            if rc is None:
                continue
            running.remove(r)
            release(r)
            st = r.state()
            print(f'[{hours:.2f} h] finished {r.key}: rc={rc}, state={st}, '
                  f'{(time.time() - r.started) / 3600:.2f} h')
            if st != 'done' and r.attempts < args.max_attempts:
                queue.insert(0, r)  # retry (resumes with --c if a checkpoint exists)
        for r in running:
            heartbeat(r)
        queue = [r for r in queue if r.state() != 'done']  # possibly finished by another job
        # datasets converted/planned while this job runs: pick their runs up now
        waiting = [r for r in runs if r not in queue and r not in running and r.attempts == 0
                   and r.key not in started_keys and r.state() not in ('done', 'nodata')]
        for r in waiting:
            print(f'[{hours:.2f} h] dataset ready: queued {r.key}')
        queue += waiting
        if hours < args.launch_window_h:
            launched = True
            while launched:
                launched = False
                for r in list(queue):
                    g = min(gpus, key=load)
                    if load(g) + r.weight > args.capacity:
                        continue
                    if not try_lock(r):
                        continue  # another job is training it; revisit next poll
                    st = r.state()
                    if st == 'done':
                        release(r)
                        queue.remove(r)
                        continue
                    exe = os.path.join(r.venv, 'bin', 'nnUNetv2_train') if r.venv else 'nnUNetv2_train'
                    cmd = [exe, r.dataset, r.config, r.fold, '-tr', r.trainer, '-p', r.plans]
                    if st == 'partial':
                        cmd.append('--c')
                    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': g,
                           'nnUNet_n_proc_DA': str(da_per_unit * r.weight)}
                    log = open(os.path.join(logdir, r.key + '.log'), 'a')
                    log.write(f'\n===== {time.ctime()} {" ".join(cmd)} (GPU {g}) =====\n')
                    log.flush()
                    r.proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                              start_new_session=True)
                    r.gpu, r.started, r.attempts = g, time.time(), r.attempts + 1
                    started_keys.add(r.key)
                    queue.remove(r)
                    running.append(r)
                    print(f'[{hours:.2f} h] started {r.key} on GPU {g} ({st}{", resume" if st == "partial" else ""})')
                    launched = True
                    break
        time.sleep(POLL_S)

    if running:
        print(f'terminating {len(running)} run(s); they resume from checkpoint_latest next session')
        for r in running:
            try:
                os.killpg(r.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for r in running:
            try:
                r.proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(r.proc.pid, signal.SIGKILL)
            release(r)
    status(runs)


if __name__ == '__main__':
    main()
