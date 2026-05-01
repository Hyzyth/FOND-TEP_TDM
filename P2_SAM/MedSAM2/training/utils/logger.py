"""
training/utils/logger.py
==========================
TensorBoard logger wrapper used by :class:`~training.trainer.Trainer`.
"""

import atexit
import functools
import logging
import sys
import uuid
from typing import Any, Dict, Optional, Union

from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from numpy import ndarray
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from training.utils.distributed import get_rank
from training.utils.train_utils import makedir

Scalar = Union[Tensor, ndarray, int, float]


def make_tensorboard_logger(log_dir: str, **writer_kwargs: Any):
    """Create a :class:`TensorBoardLogger` for *log_dir*."""
    makedir(log_dir)
    return TensorBoardLogger(path=log_dir, summary_writer_method=SummaryWriter, **writer_kwargs)


class TensorBoardWriterWrapper:
    """Thin wrapper around ``SummaryWriter`` that only logs on rank 0."""

    def __init__(self, path: str, *args, filename_suffix=None,
                 summary_writer_method=SummaryWriter, **kwargs) -> None:
        self._writer: Optional[SummaryWriter] = None
        self._rank = get_rank()
        self._path = path
        if self._rank == 0:
            logging.info("TensorBoard log dir: %s", path)
            self._writer = summary_writer_method(
                log_dir=path, *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        atexit.register(self.close)

    @property
    def writer(self): return self._writer
    @property
    def path(self): return self._path

    def flush(self):
        if self._writer: self._writer.flush()

    def close(self):
        if self._writer:
            self._writer.close()
            self._writer = None


class TensorBoardLogger(TensorBoardWriterWrapper):
    """Logger that writes scalars to TensorBoard."""

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        for k, v in payload.items():
            self.log(k, v, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_hparams(self, hparams: Dict[str, Scalar], meters: Dict[str, Scalar]) -> None:
        if self._writer:
            self._writer.add_hparams(hparams, meters)


class Logger:
    """Facade that dispatches to TensorBoard (and future backends)."""

    def __init__(self, logging_conf) -> None:
        tb_config = logging_conf.tensorboard_writer
        tb_should_log = tb_config and tb_config.pop("should_log", True)
        self.tb_logger = instantiate(tb_config) if tb_should_log else None

    def log_dict(self, payload: Dict[str, Scalar], step: int) -> None:
        if self.tb_logger: self.tb_logger.log_dict(payload, step)

    def log(self, name: str, data: Scalar, step: int) -> None:
        if self.tb_logger: self.tb_logger.log(name, data, step)

    def log_hparams(self, hparams, meters) -> None:
        if self.tb_logger: self.tb_logger.log_hparams(hparams, meters)


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = g_pathmgr.open(filename, mode="a", buffering=10 * 1024)
    atexit.register(io.close)
    return io


def setup_logging(name, output_dir=None, rank=0,
                  log_level_primary="INFO", log_level_secondary="ERROR"):
    """Configure a named logger with console and optional file handlers."""
    log_filename = None
    if output_dir:
        makedir(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"
    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)
    fmt = logging.Formatter("%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s")
    for h in logger.handlers[:]:
        logger.removeHandler(h)
    logger.root.handlers = []
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    ch.setLevel(log_level_primary if rank == 0 else log_level_secondary)
    logger.addHandler(ch)
    if log_filename and rank == 0:
        fh = logging.StreamHandler(_cached_log_stream(log_filename))
        fh.setLevel(log_level_primary)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logging.root = logger
