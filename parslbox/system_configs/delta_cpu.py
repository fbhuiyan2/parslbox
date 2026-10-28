import os
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class DeltaCpuConfig(SystemConfig):
    """
    Configuration for NCSA Delta CPU nodes.

    Delta CPU node specifications:
    - 2x AMD EPYC 7763 (Milan), 128 cores per node, SMT off
    - 256 GB DRAM
    - SLURM partitions `cpu`, `cpu-interactive`, `cpu-preempt`
    - Nodes are shared by default; for an exclusive node request every
      consumable resource or use `--exclusive --mem=0`

    Notes:
    - `--account` is required (see the `accounts` command).
    - Default memory is 1000 MB per core.
    - MPI codes are launched with `srun` inside the batch job.

    Status: written from the site documentation; not yet run on the system.
    Defaults are meant to be runnable, then tweaked via config.yaml / --sched-opts.
    Sub-node allocation: see the 'SLURM sub-node allocations' caveat in
    resource_manager/README.md before relying on it.
    """

    SYSTEM_NAME = 'delta-cpu'
    CORES_PER_NODE = 128   # 2x 64-core AMD EPYC 7763; SMT off
    GPUS_PER_NODE = 0
    SCHEDULER = "SLURM"
    MPI_CMD_TO_USE = "srun"
    MPI_BACKEND = "srun"
    MAX_WORKERS_PER_NODE = 128
    WORKER_CPU_AFFINITY = None
    GPU_TYPE = None
    DRAM_PER_NODE = 256          # GB

    def __init__(self):
        super().__init__()

    def detect_resources(self) -> tuple[int, int]:
        """
        Detect allocated nodes for a SLURM job on Delta CPU nodes.

        Returns:
            tuple[int, int]: (nodes, 0) -- no GPUs on CPU nodes.
        """
        nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        return nodes, 0

    def get_config(self, run_dir: Path, retries: int = 0,
                   max_workers: Optional[int] = None) -> Config:
        """
        Generate Parsl configuration for Delta CPU nodes.

        Args:
            run_dir: Path for Parsl's run directory.
            retries: Number of retries for failed Parsl apps.
            max_workers: Optional override for total workers across all nodes.
                         If None, uses MAX_WORKERS_PER_NODE * nodes.
                         If provided, will be capped at MAX_WORKERS_PER_NODE * nodes.

        Returns:
            Config: A Parsl configuration object.
        """
        nodes, _ = self.detect_resources()

        if max_workers is not None:
            max_workers_per_node = min(max_workers, self.MAX_WORKERS_PER_NODE * nodes)
        else:
            max_workers_per_node = self.MAX_WORKERS_PER_NODE * nodes

        cores_per_worker = self.CORES_PER_NODE / max_workers_per_node

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_delta_cpu",
                    heartbeat_period=120,
                    heartbeat_threshold=300,
                    worker_debug=True,
                    available_accelerators=0,
                    max_workers_per_node=max_workers_per_node,
                    cores_per_worker=cores_per_worker,
                    prefetch_capacity=0,
                    provider=LocalProvider(
                        init_blocks=1,
                        max_blocks=1,
                        launcher=SimpleLauncher(),
                    ),
                )
            ],
            run_dir=str(run_dir),
            retries=retries,
        )

    def get_default_mpi_config_yaml(self) -> dict:
        return {
            "backend": self.MPI_BACKEND,
            "use_gpu_wrapper": False,
            "cpu_bind_method": "cores",
            "use_hostlist": True
        }
