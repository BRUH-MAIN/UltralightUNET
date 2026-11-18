"""Build any benchmarked network exactly as its nnU-Net trainer would.

Used by scripts/flops3d.py and scripts/bench_infer3d.py so that params, FLOPs
and latency are measured on the very module training/inference uses -- built
from the dataset's own plans (patch size, strides, kernel sizes, channels,
classes/regions), not from a hand-written approximation.
"""

import json
import os
from types import SimpleNamespace

import torch

from pvm3d.nnunet_glue import build_pvm_from_plans, count_inference_params


def load_plans(dataset=None, plans='nnUNetPlans', configuration='3d_fullres', plans_file=None, dataset_json=None):
    """Return (plans_manager, configuration_manager, dataset_json, n_in, n_out).

    Either a dataset id/name (reads $nnUNet_preprocessed/<ds>/) or explicit
    plans/dataset.json paths (e.g. plans copied back from the cluster)."""
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    if plans_file is None:
        from nnunetv2.paths import nnUNet_preprocessed
        from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
        base = os.path.join(nnUNet_preprocessed, maybe_convert_to_dataset_name(str(dataset)))
        plans_file = os.path.join(base, plans + '.json')
        dataset_json = dataset_json or os.path.join(base, 'dataset.json')
    pm = PlansManager(plans_file)
    cm = pm.get_configuration(configuration)
    dj = json.load(open(dataset_json))
    n_in = determine_num_input_channels(pm, cm, dj)
    n_out = pm.get_label_manager(dj).num_segmentation_heads
    return pm, cm, dj, n_in, n_out


_OUR_TRAINER_MODULES = ('pvm3d.nnunet_trainers.trainers_pvm3d', 'pvm3d.nnunet_trainers.trainers_baselines',
                        'pvm3d.nnunet_trainers.trainers_external')


def find_trainer(name):
    """Our trainers by direct import; nnU-Net's built-ins via its primary
    (package) search. Deliberately avoids nnU-Net's nnUNet_extTrainer path,
    which can crash on repeated lookups (see scripts/cluster/install_trainers.py)."""
    import importlib
    for modname in _OUR_TRAINER_MODULES:
        try:
            mod = importlib.import_module(modname)
        except ModuleNotFoundError as e:
            if e.name == modname:
                continue
            raise
        if hasattr(mod, name):
            return getattr(mod, name)
    from os.path import join
    import nnunetv2
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    cls = recursive_find_python_class(join(nnunetv2.__path__[0], 'training', 'nnUNetTrainer'), name,
                                      'nnunetv2.training.nnUNetTrainer', nnunetv2.__path__[0])
    if cls is None:
        raise RuntimeError(f'trainer {name} not found')
    return cls


def build(trainer_name, pm, cm, n_in, n_out, deep_supervision=False, mamba_impl=None):
    cls = find_trainer(trainer_name)
    if hasattr(cls, 'PVM'):
        cfg = dict(cls.PVM)
        if mamba_impl:
            cfg['mamba_impl'] = mamba_impl
        return build_pvm_from_plans(cfg, cm, n_in, n_out, deep_supervision)
    return cls.build_network_architecture(pm, cm, n_in, n_out, deep_supervision)


def describe(net):
    return SimpleNamespace(params=count_inference_params(net),
                           params_total=sum(p.numel() for p in net.parameters()))


def env_info(device):
    info = dict(torch=torch.__version__, device=str(device))
    if torch.cuda.is_available() and str(device).startswith('cuda'):
        info['gpu'] = torch.cuda.get_device_name(0)
        info['cuda'] = torch.version.cuda
    else:
        import platform
        info['cpu'] = platform.processor() or platform.machine()
        info['threads'] = torch.get_num_threads()
    return info
