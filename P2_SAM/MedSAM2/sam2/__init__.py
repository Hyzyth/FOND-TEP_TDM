"""
sam2
====
Core MedSAM2 package.

On import, Hydra is initialised against this module's ``configs/`` directory
so that ``compose(config_name=...)`` works from any working directory.

Thread safety: ``initialize_config_module`` is called under a lock so that
concurrent DataLoader worker processes that import this package do not race
on the global Hydra singleton.
"""

import threading

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

_hydra_init_lock = threading.Lock()


def _ensure_hydra_initialized() -> None:
    """Initialise Hydra exactly once, even under concurrent imports."""
    with _hydra_init_lock:
        if not GlobalHydra.instance().is_initialized():
            initialize_config_module("sam2", version_base="1.2")


_ensure_hydra_initialized()