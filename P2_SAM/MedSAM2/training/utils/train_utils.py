"""
training/utils/train_utils.py
================================
Miscellaneous training utilities: seed setting, distributed backend setup,
progress meters, checkpoint discovery, and OmegaConf resolvers.
"""

import logging
import math
import os
import random
import re
from datetime import timedelta
from typing import Optional

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributed as dist
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf


def multiply_all(*args):
    return np.prod(np.array(args)).item()


def collect_dict_keys(config):
    """Recursively collect all ``dict_key`` values from a dataset config."""
    val_keys = []
    if "_target_" in config and re.match(r".*collate_fn.*", config["_target_"]):
        val_keys.append(config["dict_key"])
    else:
        for v in config.values():
            if isinstance(v, type(config)):
                val_keys.extend(collect_dict_keys(v))
            elif isinstance(v, omegaconf.listconfig.ListConfig):
                for item in v:
                    if isinstance(item, type(config)):
                        val_keys.extend(collect_dict_keys(item))
    return val_keys


class Phase:
    TRAIN = "train"
    VAL = "val"


def register_omegaconf_resolvers() -> None:
    """Register custom OmegaConf resolvers used in YAML configs."""
    OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
    OmegaConf.register_new_resolver("get_class", hydra.utils.get_class)
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.register_new_resolver("times", multiply_all)
    OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
    OmegaConf.register_new_resolver("pow", lambda x, y: x ** y)
    OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
    OmegaConf.register_new_resolver("range", lambda x: list(range(x)))
    OmegaConf.register_new_resolver("int", lambda x: int(x))
    OmegaConf.register_new_resolver("ceil_int", lambda x: int(math.ceil(x)))
    OmegaConf.register_new_resolver("merge", lambda *x: OmegaConf.merge(*x))


def setup_distributed_backend(backend: str, timeout_mins: int) -> int:
    """Initialise torch.distributed and return the global rank."""
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
    logging.info("Initialising torch.distributed (timeout=%d min)", timeout_mins)
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_mins))
    return dist.get_rank()


def get_machine_local_and_dist_rank():
    local_rank = int(os.environ.get("LOCAL_RANK"))
    dist_rank  = int(os.environ.get("RANK"))
    return local_rank, dist_rank


def set_seeds(seed_value: int, max_epochs: int, dist_rank: int) -> None:
    """Set Python / NumPy / PyTorch seeds for reproducibility."""
    seed = (seed_value + dist_rank) * max_epochs
    logging.info("SEED: %d", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def makedir(dir_path: str) -> bool:
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        return True
    except Exception:
        logging.info("Error creating directory: %s", dir_path)
        return False


def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_amp_type(amp_type: Optional[str] = None):
    if amp_type is None:  return None
    assert amp_type in ("bfloat16", "float16")
    return torch.bfloat16 if amp_type == "bfloat16" else torch.float16


def log_env_variables() -> None:
    env_str = "\n".join(f"{k}={v}" for k, v in sorted(os.environ.items()))
    logging.info("ENV VARIABLES\n%s", env_str)


def human_readable_time(time_seconds: float) -> str:
    t = int(time_seconds)
    minutes, seconds = divmod(t, 60)
    hours, minutes   = divmod(minutes, 60)
    days, hours      = divmod(hours, 24)
    return f"{days:02d}d {hours:02d}h {minutes:02d}m"


def get_resume_checkpoint(save_dir: str) -> Optional[str]:
    """Return the path to ``checkpoint.pt`` in *save_dir*, or ``None``."""
    ckpt = os.path.join(save_dir, "checkpoint.pt")
    return ckpt if g_pathmgr.isfile(ckpt) else None


# ──────────────────────────────────────────────────────────────────────────────
# Progress meters
# ──────────────────────────────────────────────────────────────────────────────

class AverageMeter:
    """Computes and stores running average and current value."""

    def __init__(self, name: str, device, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmt = self.fmt.lstrip(":")
        val_str = f"{self.val:{fmt}}" if fmt else str(self.val)
        avg_str = f"{self.avg:{fmt}}" if fmt else str(self.avg)
        return f"{self.name}: {val_str} ({avg_str})"


class MemMeter:
    """Tracks peak GPU memory usage per iteration."""

    def __init__(self, name: str, device, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = self.avg = self.peak = self.sum = self.count = 0

    def update(self, n: int = 1, reset_peak_usage: bool = True):
        self.val = torch.cuda.max_memory_allocated() // 1e9
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count
        self.peak = max(self.peak, self.val)
        if reset_peak_usage:
            torch.cuda.reset_peak_memory_stats()

    def __str__(self):
        return f"{self.name}: {self.val:.2f} ({self.avg:.2f}/{self.peak:.2f})"


class DurationMeter:
    def __init__(self, name: str, device, fmt: str = ":f") -> None:
        self.name = name
        self.device = device
        self.fmt = fmt
        self.val = 0

    def reset(self): self.val = 0
    def update(self, val): self.val = val
    def add(self, val): self.val += val
    def __str__(self): return f"{self.name}: {human_readable_time(self.val)}"


class ProgressMeter:
    def __init__(self, num_batches, meters, real_meters, prefix="") -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.real_meters = real_meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(m) for m in self.meters]
        entries += [
            " | ".join(f"{os.path.join(name, subname)}: {val:.4f}"
                       for subname, val in meter.compute().items())
            for name, meter in self.real_meters.items()
        ]
        logging.info(" | ".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        nd = len(str(num_batches))
        return "[{:" + str(nd) + "d}/" + f"{num_batches}]"
