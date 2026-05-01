"""
sam2
====
Core MedSAM2 package.

On import, Hydra is initialised against this module's ``configs/`` directory
so that ``compose(config_name=...)`` works from any working directory.
"""

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

if not GlobalHydra.instance().is_initialized():
    initialize_config_module("sam2", version_base="1.2")
