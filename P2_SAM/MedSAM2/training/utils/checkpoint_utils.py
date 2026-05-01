"""
training/utils/checkpoint_utils.py
=====================================
Utilities for loading, filtering, and verifying model checkpoints.
"""

import contextlib
import fnmatch
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr
from torch.jit._script import RecursiveScriptModule


def unix_pattern_to_parameter_names(
    constraints: List[str], all_parameter_names: Sequence[str]
) -> Set[str]:
    """Return the union of all parameter names matching any constraint pattern.

    Parameters
    ----------
    constraints : list[str]
        Unix-style wildcard patterns.
    all_parameter_names : sequence[str]
        Full list of parameter names to filter.

    Returns
    -------
    set[str]
    """
    result = []
    for pattern in constraints:
        matching = set(fnmatch.filter(all_parameter_names, pattern))
        assert len(matching) > 0, f"Pattern '{pattern}' matched no parameters."
        result.append(matching)
    return set.union(*result)


def filter_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Return only the state-dict entries whose keys match *patterns*."""
    if not patterns:
        return {}
    included = unix_pattern_to_parameter_names(patterns, list(state_dict.keys()))
    return {k: state_dict[k] for k in included}


def exclude_params_matching_unix_pattern(
    patterns: List[str], state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Return a state-dict with entries matching *patterns* removed."""
    if not patterns:
        return state_dict
    excluded = unix_pattern_to_parameter_names(patterns, list(state_dict.keys()))
    return {k: v for k, v in state_dict.items() if k not in excluded}


def _get_state_dict_summary(state_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    keys = sorted(state_dict.keys())
    return np.array([state_dict[k].sum().item() for k in keys])


def assert_skipped_parameters_are_frozen(model: nn.Module, patterns: List[str]) -> None:
    """Raise if any parameter matching *patterns* requires gradients."""
    if not patterns:
        return
    frozen_sd = filter_params_matching_unix_pattern(patterns, model.state_dict())
    non_frozen = {
        n for n, p in model.named_parameters()
        if n in frozen_sd and p.requires_grad
    }
    if non_frozen:
        raise ValueError(
            f"Parameters in skip_saving_parameters must be frozen: {non_frozen}"
        )


@contextlib.contextmanager
def with_check_parameter_frozen(model: nn.Module, patterns: List[str], disabled: bool = True):
    """Context manager that asserts frozen parameters are not updated."""
    if not patterns or disabled:
        yield
        return
    frozen_sd = filter_params_matching_unix_pattern(patterns, model.state_dict())
    before = _get_state_dict_summary(frozen_sd)
    yield
    frozen_sd = filter_params_matching_unix_pattern(patterns, model.state_dict())
    after = _get_state_dict_summary(frozen_sd)
    if not np.allclose(before, after, atol=1e-6):
        raise ValueError(
            "model_weight_initializer modified parameters listed in "
            "skip_saving_parameters. Either initialise them from within the model "
            "definition or set initialize_after_preemption=True."
        )


class CkptExcludeKernel:
    """Remove state-dict keys matching *key_pattern* during checkpoint loading."""

    def __init__(self, key_pattern: List[str]) -> None:
        self.key_pattern = key_pattern

    def __call__(self, state_dict: Dict) -> Dict:
        if not self.key_pattern:
            return state_dict
        excluded = unix_pattern_to_parameter_names(self.key_pattern, state_dict.keys())
        return {k: v for k, v in state_dict.items() if k not in excluded}


def get_state_dict(checkpoint: Any, ckpt_state_dict_keys: Tuple[str, ...]) -> Dict:
    """Navigate nested checkpoint dict to extract the model state dict."""
    sd = checkpoint
    for key in ckpt_state_dict_keys:
        if isinstance(sd, RecursiveScriptModule):
            return sd.state_dict()
        sd = sd[key]
    return sd


def load_checkpoint_and_apply_kernels(
    checkpoint_path: str,
    checkpoint_kernels: Optional[List[Callable]] = None,
    ckpt_state_dict_keys: Tuple[str, ...] = ("state_dict",),
    map_location: str = "cpu",
) -> Dict:
    """Load a checkpoint from disk and apply optional processing kernels.

    Parameters
    ----------
    checkpoint_path : str
    checkpoint_kernels : list[callable], optional
        Applied in order to the raw state dict (e.g. key exclusion).
    ckpt_state_dict_keys : tuple[str]
        Key path to the model weights within the checkpoint dict.
    map_location : str

    Returns
    -------
    dict  processed state dict
    """
    assert g_pathmgr.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    with g_pathmgr.open(checkpoint_path, "rb") as f:
        checkpoint = torch.load(f, map_location=map_location)
    state_dict = get_state_dict(checkpoint, ckpt_state_dict_keys)
    if checkpoint_kernels:
        for kernel in checkpoint_kernels:
            state_dict = kernel(state_dict=state_dict)
    logging.debug("Loaded state dict keys: %s", list(state_dict.keys())[:10])
    return state_dict


def check_load_state_dict_errors(
    missing_keys: List[str],
    unexpected_keys: List[str],
    strict: bool,
    ignore_missing_keys: Optional[List[str]] = None,
    ignore_unexpected_keys: Optional[List[str]] = None,
) -> None:
    """Raise or warn based on missing / unexpected keys after ``load_state_dict``."""
    if ignore_missing_keys:
        ignored = unix_pattern_to_parameter_names(ignore_missing_keys, missing_keys)
        missing_keys = [k for k in missing_keys if k not in ignored]
    if ignore_unexpected_keys:
        ignored_unexpected = unix_pattern_to_parameter_names(ignore_unexpected_keys, unexpected_keys)
        unexpected_keys = [k for k in unexpected_keys if k not in ignored_unexpected]

    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected: {unexpected_keys}."
    if missing_keys:
        err += f" Missing: {missing_keys}."
    if unexpected_keys or missing_keys:
        logging.warning(err)
        if unexpected_keys or strict:
            raise KeyError(err)


def load_state_dict_into_model(
    state_dict: Dict,
    model: nn.Module,
    strict: bool = True,
    ignore_missing_keys: Optional[List[str]] = None,
    ignore_unexpected_keys: Optional[List[str]] = None,
    checkpoint_kernels: Optional[List[Callable]] = None,
) -> nn.Module:
    """Load *state_dict* into *model*, with optional key filtering.

    Parameters
    ----------
    state_dict : dict
    model : nn.Module
    strict : bool   raise on unexpected keys when True
    ignore_missing_keys : list[str], optional  unix patterns to ignore
    ignore_unexpected_keys : list[str], optional
    checkpoint_kernels : list[callable], optional  pre-processing kernels

    Returns
    -------
    nn.Module  the model with weights loaded
    """
    if checkpoint_kernels:
        for kernel in checkpoint_kernels:
            state_dict = kernel(state_dict=state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    check_load_state_dict_errors(
        missing, unexpected, strict=strict,
        ignore_missing_keys=ignore_missing_keys,
        ignore_unexpected_keys=ignore_unexpected_keys,
    )
    return model
