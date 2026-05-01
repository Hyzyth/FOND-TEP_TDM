"""
training/utils/distributed.py
================================
Distributed training utilities wrapping ``torch.distributed``.

All public functions degrade gracefully when called outside a distributed
context (world_size == 1), so the same code runs on single-GPU debug runs
and multi-node SLURM jobs without modification.

Preserved in full from the original MedSAM2 / Meta implementation.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import functools
import io
import logging
import os
import random
import tempfile
import time
from typing import Any, Callable, List, Tuple

import torch
import torch.autograd as autograd
import torch.distributed as dist


# Default GPU index used when DDP is managed externally.
_cuda_device_index: int = 0
_CPU_DEVICE_INDEX = -1   # sentinel: use CPU
_PRIMARY_RANK = 0


# ──────────────────────────────────────────────────────────────────────────────
# Process-group helpers
# ──────────────────────────────────────────────────────────────────────────────

@functools.lru_cache()
def _get_global_gloo_group():
    """Return a cached gloo process group spanning all ranks.

    When the main backend is NCCL the gloo group is created with a 12-hour
    timeout to prevent spurious time-outs during slow evaluation.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(
            backend="gloo",
            timeout=datetime.timedelta(seconds=43200),
        )
    return dist.group.WORLD


def is_dist_avail_and_initialized() -> bool:
    """Return True when torch.distributed is available and initialised."""
    return dist.is_available() and dist.is_initialized()


def is_distributed_training_run() -> bool:
    """Return True when running with more than one process."""
    return (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    )


def is_main_process() -> bool:
    """Return True on rank 0 (or when not distributed)."""
    return get_rank() == 0


def is_primary() -> bool:
    """Alias for :func:`is_main_process`."""
    return get_rank() == _PRIMARY_RANK


def get_rank() -> int:
    """Return the global rank of the current process (0 when not distributed)."""
    return (
        torch.distributed.get_rank()
        if is_dist_avail_and_initialized()
        else 0
    )


def get_primary_rank() -> int:
    return _PRIMARY_RANK


def get_world_size() -> int:
    """Return the total number of processes (1 when not distributed)."""
    return (
        torch.distributed.get_world_size()
        if is_dist_avail_and_initialized()
        else 1
    )


def barrier() -> None:
    """Synchronise all processes; no-op when not distributed."""
    if is_dist_avail_and_initialized():
        torch.distributed.barrier()


# ──────────────────────────────────────────────────────────────────────────────
# Device index management
# ──────────────────────────────────────────────────────────────────────────────

def set_cuda_device_index(idx: int) -> None:
    global _cuda_device_index
    _cuda_device_index = idx
    torch.cuda.set_device(_cuda_device_index)


def set_cpu_device() -> None:
    global _cuda_device_index
    _cuda_device_index = _CPU_DEVICE_INDEX


def get_cuda_device_index() -> int:
    return _cuda_device_index


# ──────────────────────────────────────────────────────────────────────────────
# Tensor device conversion helpers
# ──────────────────────────────────────────────────────────────────────────────

def convert_to_distributed_tensor(tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """Move *tensor* to CUDA if the NCCL backend requires it.

    Returns ``(tensor, orig_device)`` where ``orig_device`` is ``'cpu'``
    or ``'gpu'``, allowing the caller to restore the original device.
    """
    orig_device = "cpu" if not tensor.is_cuda else "gpu"
    if (
        torch.distributed.is_available()
        and torch.distributed.get_backend() == torch.distributed.Backend.NCCL
        and not tensor.is_cuda
    ):
        tensor = tensor.cuda()
    return tensor, orig_device


def convert_to_normal_tensor(tensor: torch.Tensor, orig_device: str) -> torch.Tensor:
    """Move *tensor* back to CPU if it was originally on CPU."""
    if tensor.is_cuda and orig_device == "cpu":
        tensor = tensor.cpu()
    return tensor


# ──────────────────────────────────────────────────────────────────────────────
# All-reduce operations
# ──────────────────────────────────────────────────────────────────────────────

def all_reduce_op(
    tensor: torch.Tensor,
    op: torch.distributed.ReduceOp,
    after_op_func: Callable[[torch.Tensor], torch.Tensor] = None,
) -> torch.Tensor:
    """Apply *op* all-reduce, handling device placement automatically."""
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.all_reduce(tensor, op)
        if after_op_func is not None:
            tensor = after_op_func(tensor)
        tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce then divide by world size (mean across processes)."""
    return all_reduce_op(
        tensor,
        torch.distributed.ReduceOp.SUM,
        lambda t: t / torch.distributed.get_world_size(),
    )


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce with SUM operation."""
    return all_reduce_op(tensor, torch.distributed.ReduceOp.SUM)


def all_reduce_min(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce with MIN operation."""
    return all_reduce_op(tensor, torch.distributed.ReduceOp.MIN)


def all_reduce_max(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce with MAX operation.

    Used by :func:`~training.trainer.Trainer._log_sync_data_times` to
    synchronise per-rank data-loading times across GPUs.
    """
    return all_reduce_op(tensor, torch.distributed.ReduceOp.MAX)


# ──────────────────────────────────────────────────────────────────────────────
# Broadcast
# ──────────────────────────────────────────────────────────────────────────────

def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast *tensor* from rank *src* to all other ranks."""
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        torch.distributed.broadcast(tensor, src)
        tensor = convert_to_normal_tensor(tensor, orig_device)
    return tensor


def broadcast_object(obj: Any, src: int = _PRIMARY_RANK, use_disk: bool = True) -> Any:
    """Broadcast any serialisable Python object from rank *src* to all ranks.

    Parameters
    ----------
    obj : picklable object
    src : int  source rank (default: primary / rank 0)
    use_disk : bool
        When True the receiving ranks write to a temp file before
        deserialising, reducing peak CPU memory at the cost of a disk write.
    """
    if get_rank() == src:
        buffer = io.BytesIO()
        torch.save(obj, buffer)
        data_view = buffer.getbuffer()
        length_tensor = broadcast(torch.LongTensor([len(data_view)]), src=src)
        data_tensor = broadcast(torch.ByteTensor(data_view), src=src)
    else:
        length_tensor = broadcast(torch.LongTensor([0]), src=src)
        data_tensor = torch.empty([length_tensor.item()], dtype=torch.uint8)
        data_tensor = broadcast(data_tensor, src=src)
        if use_disk:
            with tempfile.TemporaryFile("r+b") as f:
                f.write(data_tensor.numpy())
                del data_tensor
                f.seek(0)
                obj = torch.load(f)
        else:
            obj = torch.load(io.BytesIO(data_tensor.numpy()))
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# All-gather operations
# ──────────────────────────────────────────────────────────────────────────────

def gather_tensors_from_all(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather *tensor* from every rank; returns a list of per-rank tensors."""
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    if is_distributed_training_run():
        tensor, orig_device = convert_to_distributed_tensor(tensor)
        gathered = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, tensor)
        gathered = [convert_to_normal_tensor(t, orig_device) for t in gathered]
    else:
        gathered = [tensor]
    return gathered


def gather_from_all(tensor: torch.Tensor) -> torch.Tensor:
    """Gather *tensor* from all ranks and concatenate along dim 0."""
    return torch.cat(gather_tensors_from_all(tensor), dim=0)


def all_gather_tensor(tensor: torch.Tensor, world_size=None) -> List[torch.Tensor]:
    """All-gather a contiguous tensor across all ranks."""
    if world_size is None:
        world_size = get_world_size()
    assert tensor.is_contiguous(), f"{tensor.shape} is not contiguous!"
    tensor, orig_device = convert_to_distributed_tensor(tensor)
    tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_all, tensor, async_op=False)
    return [convert_to_normal_tensor(t, orig_device) for t in tensor_all]


def all_gather_batch(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """All-gather a list of tensors, returning one concatenated tensor per input."""
    world_size = get_world_size()
    if world_size == 1:
        return tensors
    return [torch.cat(all_gather_tensor(t, world_size), dim=0) for t in tensors]


def all_gather_batch_with_grad(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """All-gather while keeping the autograd graph connected (via :class:`GatherLayer`)."""
    world_size = get_world_size()
    if world_size == 1:
        return tensors
    return [torch.cat(GatherLayer.apply(t), dim=0) for t in tensors]


def all_gather(data, force_cpu: bool = False, force_filesys: bool = False,
               filesys_save_dir=None) -> list:
    """All-gather arbitrary picklable *data* across all ranks.

    Transport mode is controlled by environment variables or keyword arguments:

    - ``MDETR_FILESYS_REDUCE_RANK_0_ONLY=1``: filesystem gather, rank-0 only.
    - ``MDETR_FILESYS_REDUCE=1`` / *force_filesys*: filesystem gather, all ranks.
    - ``MDETR_CPU_REDUCE=1`` / *force_cpu*: use gloo/CPU for the gather.
    - default: NCCL/GPU gather.

    Parameters
    ----------
    data : picklable object
    force_cpu : bool
    force_filesys : bool
    filesys_save_dir : str or None
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    if os.getenv("MDETR_FILESYS_REDUCE_RANK_0_ONLY") == "1":
        return all_gather_via_filesys(data, filesys_save_dir, gather_to_rank_0_only=True)
    if os.getenv("MDETR_FILESYS_REDUCE") == "1" or force_filesys:
        return all_gather_via_filesys(data, filesys_save_dir)

    cpu_group = None
    if os.getenv("MDETR_CPU_REDUCE") == "1" or force_cpu:
        cpu_group = _get_global_gloo_group()

    buffer = io.BytesIO()
    torch.save(data, buffer)
    data_view = buffer.getbuffer()
    device = "cuda" if cpu_group is None else "cpu"
    tensor = torch.ByteTensor(data_view).to(device)

    local_size = torch.tensor([tensor.numel()], device=device, dtype=torch.long)
    size_list = [torch.tensor([0], device=device, dtype=torch.long) for _ in range(world_size)]
    if cpu_group is None:
        dist.all_gather(size_list, local_size)
    else:
        print("gathering on cpu")
        dist.all_gather(size_list, local_size, group=cpu_group)
    size_list = [int(s.item()) for s in size_list]
    max_size = max(size_list)
    local_size = int(local_size.item())

    tensor_list = [torch.empty((max_size,), dtype=torch.uint8, device=device) for _ in size_list]
    if local_size != max_size:
        padding = torch.empty((max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    if cpu_group is None:
        dist.all_gather(tensor_list, tensor)
    else:
        dist.all_gather(tensor_list, tensor, group=cpu_group)

    data_list = []
    for size, t in zip(size_list, tensor_list):
        t = torch.split(t, [size, max_size - size], dim=0)[0]
        data_list.append(torch.load(io.BytesIO(t.cpu().numpy())))
    return data_list


def all_gather_via_filesys(data, filesys_save_dir=None, gather_to_rank_0_only=False) -> list:
    """All-gather arbitrary picklable data via shared filesystem.

    Parameters
    ----------
    data : picklable object
    filesys_save_dir : str or None
        Directory for temp files. Falls back to ``$EXP_DIR`` then module dir.
    gather_to_rank_0_only : bool
        If True only rank 0 loads the full gathered list.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    print("gathering via files")
    cpu_group = _get_global_gloo_group()

    save_dir = filesys_save_dir or os.environ.get("EXP_DIR") or os.path.dirname(__file__)
    save_dir = os.path.join(save_dir, "all_gather_via_filesys")
    if is_main_process():
        os.makedirs(save_dir, exist_ok=True)

    timestamp = int(time.time()) if is_main_process() else 0
    salt = random.randint(0, 2**31 - 1) if is_main_process() else 0
    ts_salt = torch.tensor([timestamp, salt], dtype=torch.long)
    dist.all_reduce(ts_salt, group=cpu_group)
    timestamp, salt = ts_salt.tolist()

    rank_save = get_rank()
    fname = f"data_to_gather_{timestamp}_{salt}_{rank_save}.pkl"
    fpath = os.path.join(save_dir, fname)
    assert not os.path.exists(fpath), f"{fpath} already exists"
    torch.save(data, fpath)
    dist.barrier(group=cpu_group)

    data_list = []
    if rank_save == 0 or not gather_to_rank_0_only:
        for r in range(world_size):
            load_path = os.path.join(save_dir, f"data_to_gather_{timestamp}_{salt}_{r}.pkl")
            assert os.path.exists(load_path), f"Cannot read {load_path}"
            data_list.append(torch.load(load_path))
    dist.barrier(group=cpu_group)
    os.remove(fpath)
    return data_list


# ──────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ──────────────────────────────────────────────────────────────────────────────

class GatherLayer(autograd.Function):
    """All-gather with autograd support (gradients flow back through the gather).

    Unlike ``torch.distributed.all_gather`` which cuts the gradient graph,
    this keeps it connected — useful for contrastive / metric learning losses.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_grads = torch.stack(grads)
        dist.all_reduce(all_grads)
        return all_grads[dist.get_rank()]


def unwrap_ddp_if_wrapped(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module if *model* is wrapped in DDP."""
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


def init_distributed_data_parallel_model(
    model: torch.nn.Module,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = True,
    bucket_cap_mb: int = 25,
) -> torch.nn.parallel.DistributedDataParallel:
    """Wrap *model* in DDP using the module-level device index."""
    global _cuda_device_index
    if _cuda_device_index == _CPU_DEVICE_INDEX:
        return torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=broadcast_buffers,
            find_unused_parameters=find_unused_parameters,
            bucket_cap_mb=bucket_cap_mb,
        )
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[_cuda_device_index],
        output_device=_cuda_device_index,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        bucket_cap_mb=bucket_cap_mb,
    )


def create_new_process_group(group_size: int):
    """Create sub-groups of *group_size* GPUs; return the one the current rank belongs to.

    ``world_size`` must be divisible by ``group_size``.
    Every rank must call this function to participate in all new_group calls.
    """
    assert group_size > 0
    world_size = torch.distributed.get_world_size()
    if world_size <= 8 and group_size > world_size:
        logging.warning(
            "group_size=%d > world_size=%d; capping to world_size.",
            group_size, world_size,
        )
        group_size = world_size
    assert world_size >= group_size and world_size % group_size == 0

    group = None
    for group_num in range(world_size // group_size):
        group_ids = range(group_num * group_size, (group_num + 1) * group_size)
        cur_group = torch.distributed.new_group(ranks=list(group_ids))
        if torch.distributed.get_rank() // group_size == group_num:
            group = cur_group
    assert group is not None
    return group
