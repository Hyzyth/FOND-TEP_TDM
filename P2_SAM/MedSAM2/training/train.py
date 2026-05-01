"""
training/train.py
==================
CLI entry point for launching MedSAM2 fine-tuning.

Supports both local multi-GPU runs and SLURM cluster submission via
``submitit``.  The training configuration is resolved by Hydra from the YAML
file passed with ``-c``.

Usage (local, 2 GPUs)
---------------------
python training/train.py \\
    -c sam2/configs/sam2.1_hiera_tiny_hecktor.yaml \\
    --output-path /data/ethan/MedSAM2/exp_log/hecktor_finetune \\
    --use-cluster 0 \\
    --num-gpus 2

Usage (SLURM)
-------------
python training/train.py \\
    -c sam2/configs/sam2.1_hiera_tiny_hecktor.yaml \\
    --output-path /data/ethan/MedSAM2/exp_log/hecktor_finetune \\
    --use-cluster 1 \\
    --num-gpus 4 \\
    --num-nodes 2
"""

import logging
import os
import random
import sys
import traceback
from argparse import ArgumentParser

import submitit
import torch
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr
from omegaconf import OmegaConf

from training.utils.train_utils import makedir, register_omegaconf_resolvers

os.environ["HYDRA_FULL_ERROR"] = "1"


def single_proc_run(
    local_rank: int,
    main_port: int,
    cfg,
    world_size: int,
    node_rank: int,
    master_addr: str,
) -> None:
    """Single-GPU training process entry point.

    Parameters
    ----------
    local_rank : int
    main_port : int
    cfg : OmegaConf config
    world_size : int
    node_rank : int
    master_addr : str
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(main_port)
    os.environ["RANK"] = str(node_rank * cfg.launcher.gpus_per_node + local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        register_omegaconf_resolvers()
    except Exception as e:
        logging.info("OmegaConf resolver already registered: %s", e)
    trainer = instantiate(cfg.trainer, _recursive_=False)
    trainer.run()


def single_node_runner(
    cfg,
    main_port: int,
    node_rank: int = 0,
    master_addr: str = "localhost",
) -> None:
    """Spawn one process per GPU on a single node.

    Parameters
    ----------
    cfg : OmegaConf config
    main_port : int
    node_rank : int
    master_addr : str
    """
    num_proc = cfg.launcher.gpus_per_node
    world_size = cfg.launcher.gpus_per_node * cfg.launcher.num_nodes
    torch.multiprocessing.set_start_method("spawn")
    if num_proc == 1:
        # Direct call (no spawn) makes debugging with breakpoints easier.
        single_proc_run(0, main_port, cfg, world_size, node_rank, master_addr)
    else:
        torch.multiprocessing.start_processes(
            single_proc_run,
            args=(main_port, cfg, world_size, node_rank, master_addr),
            nprocs=num_proc,
            start_method="spawn",
        )


def _format_exception(e: Exception, limit: int = 20) -> str:
    tb = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{tb}"


class SubmititRunner(submitit.helpers.Checkpointable):
    """Wrapper for ``submitit``-based SLURM submission."""

    def __init__(self, port: int, cfg) -> None:
        self.cfg = cfg
        self.port = port

    def __call__(self):
        job_env = submitit.JobEnvironment()
        os.environ["MASTER_ADDR"] = job_env.hostnames[0]
        os.environ["MASTER_PORT"] = str(self.port)
        os.environ["RANK"] = str(job_env.global_rank)
        os.environ["LOCAL_RANK"] = str(job_env.local_rank)
        os.environ["WORLD_SIZE"] = str(job_env.num_tasks)
        register_omegaconf_resolvers()
        cfg_resolved = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=False))
        try:
            instantiate(cfg_resolved.trainer, _recursive_=False).run()
        except Exception as e:
            logging.error(_format_exception(e))
            raise


def main(args, cfg) -> None:
    """Parse launcher settings and either train locally or submit to SLURM."""
    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(os.getcwd(), "sam2_logs", args.config)

    logging.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    makedir(cfg.launcher.experiment_log_dir)

    # Write resolved configs to disk for reproducibility.
    with g_pathmgr.open(os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Apply CLI overrides.
    if args.num_gpus is not None:
        cfg.launcher.gpus_per_node = args.num_gpus
    if args.num_nodes is not None:
        cfg.launcher.num_nodes = args.num_nodes

    submitit_conf = cfg.get("submitit", None)
    assert submitit_conf is not None, "Missing [submitit] block in config."

    use_cluster = args.use_cluster if args.use_cluster is not None else submitit_conf.use_cluster

    if use_cluster:
        _submit_slurm(args, cfg, submitit_conf)
    else:
        master_addr = args.master_addr or "localhost"
        main_port = args.main_port or random.randint(*submitit_conf.port_range)
        node_rank = int(os.environ.get("SLURM_PROCID", 0))
        single_node_runner(cfg, main_port, node_rank=node_rank, master_addr=master_addr)


def _submit_slurm(args, cfg, submitit_conf):
    """Submit the training job to a SLURM cluster via submitit."""
    submitit_dir = os.path.join(cfg.launcher.experiment_log_dir, "submitit_logs")
    executor = submitit.AutoExecutor(folder=submitit_dir)
    job_kwargs = {
        "timeout_min": 60 * submitit_conf.timeout_hour,
        "name": getattr(submitit_conf, "name", args.config),
        "slurm_partition": submitit_conf.get("partition", None),
        "gpus_per_node": cfg.launcher.gpus_per_node,
        "tasks_per_node": cfg.launcher.gpus_per_node,
        "cpus_per_task": submitit_conf.cpus_per_task,
        "nodes": cfg.launcher.num_nodes,
    }
    executor.update_parameters(**{k: v for k, v in job_kwargs.items() if v is not None})
    main_port = random.randint(*submitit_conf.port_range)
    job = executor.submit(SubmititRunner(main_port, cfg))
    logging.info("Submitit job ID: %s", job.job_id)


if __name__ == "__main__":
    initialize_config_module("sam2", version_base="1.2")

    parser = ArgumentParser(description="Train MedSAM2 on HECKTOR or other medical datasets.")
    parser.add_argument("-c", "--config", required=True, type=str,
                        help="Hydra config name (e.g. sam2/configs/sam2.1_hiera_tiny_hecktor.yaml).")
    parser.add_argument("--use-cluster", type=int, default=None,
                        help="0 = local, 1 = SLURM cluster.")
    parser.add_argument("--partition", type=str, default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--num-nodes", type=int, default=None)
    parser.add_argument("--master-addr", type=str, default=None)
    parser.add_argument("--main-port", type=int, default=None)
    parser.add_argument("--dataset-path", type=str, default=None,
                        help="Overrides cfg.dataset.train_folder.")
    parser.add_argument("--output-path", type=str, default=None,
                        help="Overrides cfg.launcher.experiment_log_dir.")

    args = parser.parse_args()
    args.use_cluster = bool(args.use_cluster) if args.use_cluster is not None else None

    register_omegaconf_resolvers()
    cfg = compose(config_name=args.config)

    if args.dataset_path is not None:
        cfg.dataset.train_folder = args.dataset_path
    if args.output_path is not None:
        cfg.launcher.experiment_log_dir = args.output_path

    main(args, cfg)
