"""PVM-3D: volumetric UltraLight VM-UNet, packaged for nnU-Net v2.

See docs/3d_journal_roadmap.md for what this package is for and how the
experiments use it. Layout:

    pvm3d/network.py           PVMUNet3D + PVMLayer3D (the model family)
    pvm3d/nnunet_trainers/     nnU-Net trainer classes -- point the
                               nnUNet_extTrainer env var at this folder
    pvm3d/reference/           published baseline numbers we compare against
"""
