"""
training/optimizer.py
======================
Optimizer construction utilities including layer-decay parameter groups,
gradient clipping, and learning-rate scheduling wrappers.
"""

import fnmatch
import inspect
import itertools
import logging
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Type

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer wrapper
# ──────────────────────────────────────────────────────────────────────────────

class Optimizer:
    """Wraps a PyTorch optimizer with per-param-group schedulers.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
    schedulers : list[dict] or None
        One dict per param-group mapping option name → scheduler callable.
    """

    def __init__(self, optimizer, schedulers=None) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._validate_optimizer_schedulers()
        self.step_schedulers(0.0, 0)

    def _validate_optimizer_schedulers(self):
        if self.schedulers is None:
            return
        for set_of_schedulers in self.schedulers:
            for option in set_of_schedulers:
                assert option in self.optimizer.defaults, (
                    f"Optimizer option '{option}' not found in {self.optimizer}."
                )

    def step_schedulers(self, where: float, step: int) -> None:
        """Update all param-group hyperparameters to the value at *where*."""
        if self.schedulers is None:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            for option, scheduler in self.schedulers[i].items():
                sig = inspect.signature(scheduler.__call__).parameters
                if "step" in sig or (hasattr(scheduler, "scheduler") and
                                      "step" in inspect.signature(scheduler.scheduler.__call__).parameters):
                    param_group[option] = scheduler(step=step, where=where)
                else:
                    param_group[option] = scheduler(where)

    def step(self, where: float, step: int, closure=None):
        self.step_schedulers(where, step)
        return self.optimizer.step(closure)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Parameter group utilities
# ──────────────────────────────────────────────────────────────────────────────

def unix_param_pattern_to_parameter_names(
    filter_param_names: Optional[List[str]],
    parameter_names: Dict[str, Tensor],
) -> Set[str]:
    if filter_param_names is None:
        return set()
    result = []
    for pattern in filter_param_names:
        matching = set(fnmatch.filter(parameter_names, pattern))
        assert len(matching) >= 1, f"Pattern '{pattern}' matched no parameters."
        result.append(matching)
    return set.union(*result)


def get_module_cls_to_param_names(
    model: nn.Module, param_allowlist: Optional[Set[str]] = None
) -> Dict[Type, Set[str]]:
    """Map each module class to the set of parameter names it directly owns."""
    mapping: Dict[Type, Set[str]] = {}
    for module_name, module in model.named_modules():
        cls = type(module)
        mapping.setdefault(cls, set())
        for pname, _ in module.named_parameters(recurse=False):
            full = f"{module_name}.{pname}" if module_name else pname
            if param_allowlist is None or full in param_allowlist:
                mapping[cls].add(full)
    return mapping


def unix_module_cls_pattern_to_parameter_names(
    filter_module_cls_names: Optional[List[str]],
    module_cls_to_param_names: Dict[Type, Set[str]],
) -> Set[str]:
    if not filter_module_cls_names:
        return set()
    result = []
    for cls_name in filter_module_cls_names:
        cls = hydra.utils.get_class(cls_name)
        assert cls in module_cls_to_param_names, f"Class '{cls_name}' not found in model."
        result.append(module_cls_to_param_names[cls])
    return set.union(*result)


def _unix_pattern_to_parameter_names(scheduler_cfg, parameter_names, module_cls_to_param_names):
    if "param_names" not in scheduler_cfg and "module_cls_names" not in scheduler_cfg:
        return None
    return unix_param_pattern_to_parameter_names(
        scheduler_cfg.get("param_names"), parameter_names
    ) | unix_module_cls_pattern_to_parameter_names(
        scheduler_cfg.get("module_cls_names"), module_cls_to_param_names
    )


def set_default_parameters(scheduler_cfgs: List[DictConfig], all_parameter_names: Set[str]) -> None:
    constraints = [c.parameter_names for c in scheduler_cfgs if c.parameter_names is not None]
    default_params = all_parameter_names - set.union(*constraints) if constraints else set(all_parameter_names)
    default_count = 0
    for cfg in scheduler_cfgs:
        if cfg.parameter_names is None:
            cfg.parameter_names = default_params
            default_count += 1
    assert default_count <= 1, "Only one scheduler per option can be the default."
    if default_count == 0:
        scheduler_cfgs.append({"parameter_names": default_params})


def name_constraints_to_parameters(
    param_constraints: List[Set[str]], named_parameters: Dict[str, Tensor]
) -> List[nn.Parameter]:
    matching = set.intersection(*param_constraints)
    return [v for n, v in named_parameters.items() if n in matching]


def map_scheduler_cfgs_to_param_groups(
    all_scheduler_cfgs, named_parameters
) -> Tuple[List[Dict], List[Dict]]:
    schedulers, param_groups = [], []
    for cfgs in itertools.product(*all_scheduler_cfgs):
        constraints = [c["parameter_names"] for c in cfgs]
        params = name_constraints_to_parameters(constraints, named_parameters)
        if not params:
            continue
        schedulers.append({c["option"]: c["scheduler"] for c in cfgs if "option" in c})
        param_groups.append({"params": params})
    return schedulers, param_groups


def validate_param_group_params(param_groups: List[Dict], model: nn.Module):
    model_params = {p for _, p in model.named_parameters()}
    group_params = [set(pg["params"]) for pg in param_groups]
    for p1, p2 in itertools.permutations(group_params, 2):
        assert p1.isdisjoint(p2), "Param groups must be disjoint."
    assert set.union(*group_params) == model_params, "Param groups must cover all model parameters."


def construct_optimizer(
    model: nn.Module,
    optimizer_conf,
    options_conf=None,
    param_group_modifiers_conf=None,
    param_allowlist: Optional[Set[str]] = None,
    validate_param_groups: bool = True,
) -> Optimizer:
    """Build an :class:`Optimizer` with per-layer learning-rate decay.

    Parameters
    ----------
    model : nn.Module
    optimizer_conf : partial torch optimizer (Hydra config)
    options_conf : Hydra config for lr / weight_decay schedulers
    param_group_modifiers_conf : list of modifier callables (e.g. layer decay)
    param_allowlist : set of parameter names to optimise (None = all)
    validate_param_groups : bool

    Returns
    -------
    Optimizer
    """
    if param_allowlist is None:
        param_allowlist = {n for n, _ in model.named_parameters()}
    named_parameters = {n: p for n, p in model.named_parameters() if n in param_allowlist}

    if not options_conf:
        opt = hydra.utils.instantiate(optimizer_conf, named_parameters.values())
        return Optimizer(opt)

    all_parameter_names = set(named_parameters.keys())
    module_cls_to_param_names = get_module_cls_to_param_names(model, param_allowlist)

    scheduler_cfgs_per_option = hydra.utils.instantiate(options_conf)
    all_scheduler_cfgs = []
    for option, scheduler_cfgs in scheduler_cfgs_per_option.items():
        for cfg in scheduler_cfgs:
            cfg.option = option
            cfg.parameter_names = _unix_pattern_to_parameter_names(cfg, all_parameter_names, module_cls_to_param_names)
        set_default_parameters(scheduler_cfgs, all_parameter_names)
        all_scheduler_cfgs.append(scheduler_cfgs)

    if param_group_modifiers_conf:
        for modifier_conf in param_group_modifiers_conf:
            modifier = hydra.utils.instantiate(modifier_conf)
            all_scheduler_cfgs = modifier(scheduler_cfgs=all_scheduler_cfgs, model=model)

    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(all_scheduler_cfgs, named_parameters)
    if validate_param_groups:
        validate_param_group_params(param_groups, model)
    opt = hydra.utils.instantiate(optimizer_conf, param_groups)
    return Optimizer(opt, schedulers)


# ──────────────────────────────────────────────────────────────────────────────
# Gradient clipping
# ──────────────────────────────────────────────────────────────────────────────

class GradientClipper:
    """Clip gradients by global norm.

    Parameters
    ----------
    max_norm : float or None  no clipping when None
    norm_type : int
    """

    def __init__(self, max_norm: float = 1.0, norm_type: int = 2) -> None:
        self.max_norm = float(max_norm) if max_norm is not None else None
        self.norm_type = norm_type

    def __call__(self, model: nn.Module):
        if self.max_norm is None:
            return
        nn.utils.clip_grad_norm_(model.parameters(), self.max_norm, self.norm_type)


# ──────────────────────────────────────────────────────────────────────────────
# Layer-decay modifier
# ──────────────────────────────────────────────────────────────────────────────

class ValueScaler:
    """Scale a scheduler's output by a constant multiplier."""

    def __init__(self, scheduler, mult_val: float) -> None:
        self.scheduler = scheduler
        self.mult_val = mult_val

    def __call__(self, *args, **kwargs):
        return self.scheduler(*args, **kwargs) * self.mult_val


def _rgetattr(obj, rattrs: Optional[str] = None):
    if rattrs is None:
        return obj
    for attr in rattrs.split("."):
        obj = getattr(obj, attr)
    return obj


def layer_decay_param_modifier(
    scheduler_cfgs: List[List[Dict]],
    model: nn.Module,
    layer_decay_value: float,
    layer_decay_min: Optional[float] = None,
    apply_to: Optional[str] = None,
    overrides: List[Dict] = (),
) -> List[List[Dict]]:
    """Apply per-layer learning-rate decay to the lr scheduler configs.

    Parameters
    ----------
    scheduler_cfgs : list[list[dict]]
    model : nn.Module  must expose ``get_layer_id`` and ``get_num_layers``
    layer_decay_value : float  decay factor per layer
    layer_decay_min : float or None  minimum decay value
    apply_to : str or None  dotted attribute path to sub-module
    overrides : list[dict]  per-parameter pattern overrides

    Returns
    -------
    list[list[dict]]  modified scheduler configs
    """
    sub_model = _rgetattr(model, apply_to)
    num_layers = sub_model.get_num_layers() + 1
    layer_decays = [layer_decay_value ** (num_layers - i) for i in range(num_layers + 1)]
    if layer_decay_min is not None:
        layer_decays = [max(v, layer_decay_min) for v in layer_decays]

    final_cfgs = []
    for cfg_group in scheduler_cfgs:
        curr = []
        for cfg in cfg_group:
            if cfg.get("option") != "lr":
                curr.append(cfg)
                continue
            param_names = sorted(cfg["parameter_names"])
            layer_groups: Dict[Any, dict] = {}
            for pname in param_names:
                layer_id = num_layers
                scale = layer_decays[layer_id]
                if apply_to and pname.startswith(apply_to):
                    layer_id = sub_model.get_layer_id(pname)
                    scale = layer_decays[layer_id]
                    for override in overrides:
                        if fnmatch.fnmatchcase(pname, override["pattern"]):
                            scale = float(override["value"])
                            layer_id = override["pattern"]
                            break
                if layer_id not in layer_groups:
                    layer_groups[layer_id] = {
                        "option": "lr",
                        "scheduler": ValueScaler(cfg["scheduler"], scale),
                        "parameter_names": {pname},
                    }
                else:
                    layer_groups[layer_id]["parameter_names"].add(pname)
            curr.extend(layer_groups.values())
        final_cfgs.append(curr)
    return final_cfgs
