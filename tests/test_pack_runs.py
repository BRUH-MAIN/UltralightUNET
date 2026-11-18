"""scripts/cluster/pack_runs.py: run-file parsing, SLURM time parsing and the
per-run lock that lets several SLURM jobs share one run file safely."""

import os
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'scripts', 'cluster'))

import pack_runs as pr  # noqa: E402


@pytest.fixture
def run(tmp_path, monkeypatch):
    r = pr.Run('4', '3d_fullres', '0', 'nnUNetTrainerPVM3D_S')
    monkeypatch.setattr(pr.Run, 'out_dir', lambda self: str(tmp_path / self.key))
    return r


def test_parse_slurm_time():
    assert pr.parse_slurm_time('1-02:30:00') == pytest.approx(26.5)
    assert pr.parse_slurm_time('02:00:36') == pytest.approx(2.01)
    assert pr.parse_slurm_time('45:00') == pytest.approx(0.75)
    assert pr.parse_slurm_time('UNLIMITED') is None
    assert pr.parse_slurm_time('') is None


def test_runfile_expands_folds_and_options(tmp_path, monkeypatch):
    monkeypatch.setenv('PVM3D_REPO', '/repo')
    f = tmp_path / 'r.txt'
    f.write_text('# c\n27 3d_fullres 0-2 nnUNetTrainer weight=2\n'
                 '220 3d_fullres 1,3 nnUNetTrainerSegMamba plans=nnUNetPlans_div16 venv=$PVM3D_REPO/.v  # tail\n')
    runs = pr.parse_runfile(str(f))
    assert [(r.dataset, r.fold, r.weight) for r in runs][:3] == [('27', '0', 2), ('27', '1', 2), ('27', '2', 2)]
    assert runs[3].plans == 'nnUNetPlans_div16' and runs[3].venv == '/repo/.v' and runs[4].fold == '3'


def test_lock_is_exclusive_and_released_by_owner_only(run, monkeypatch):
    assert pr.try_lock(run)
    assert pr.lock_owner(run) == pr.OWNER
    monkeypatch.setattr(pr, 'OWNER', 'otherhost:1:job9')   # a second packer
    assert not pr.try_lock(run)                            # live lock -> refused
    pr.release(run)                                        # not its lock -> no-op
    assert os.path.exists(pr.lock_path(run))


def test_stale_lock_is_taken_over(run, monkeypatch):
    assert pr.try_lock(run)
    old = time.time() - pr.STALE_S - 60
    os.utime(pr.lock_path(run), (old, old))               # owner died, no heartbeat
    assert pr.lock_owner(run) is None
    monkeypatch.setattr(pr, 'OWNER', 'otherhost:1:job9')
    assert pr.try_lock(run)
    assert open(pr.lock_path(run)).read() == 'otherhost:1:job9'
    assert not [f for f in os.listdir(os.path.dirname(pr.lock_path(run))) if '.stale.' in f]


def test_heartbeat_keeps_lock_fresh(run):
    assert pr.try_lock(run)
    old = time.time() - pr.STALE_S + 30
    os.utime(pr.lock_path(run), (old, old))
    pr.heartbeat(run)
    assert time.time() - os.path.getmtime(pr.lock_path(run)) < 5


def test_amos_modality_prefers_image_derived_map(tmp_path, monkeypatch):
    """amos_id 500 is listed as both CT and MR in the AMOS metadata, so the id rule
    cannot be trusted at the boundary; the map written by `verify` must win."""
    import json
    sys.path.insert(0, REPO)
    import pvm3d.nnunet_glue as g
    ds = tmp_path / 'Dataset219_AMOS2022_postChallenge_task2'
    ds.mkdir()
    (ds / 'modality.json').write_text(json.dumps({'amos_0500': 'CT', 'amos_0507': 'MRI', 'amos_0042': 'CT'}))
    monkeypatch.setenv('nnUNet_raw', str(tmp_path))
    monkeypatch.setattr(g, '_AMOS_MODALITY', None)
    assert not g.amos_is_mri('amos_0500') and g.amos_is_mri('amos_0507') and not g.amos_is_mri('amos_0042')
    with pytest.warns(UserWarning, match='id rule'):
        assert g.amos_is_mri('amos_0555')   # unmapped -> fallback, loudly
