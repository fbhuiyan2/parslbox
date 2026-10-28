import os
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class Bridges2CpuConfig(SystemConfig):
    """
    Configuration for PSC Bridges-2 Regular Memory (RM) nodes.

    Bridges-2 RM node specifications:
    - 2x AMD EPYC 7742, 128 cores per node
    - 256 GB DRAM on 488 nodes, 512 GB on 16
    - SLURM partitions: `RM` (whole nodes, SLURM's default), `RM-512` (not
      shared), `RM-shared` (part of one node, up to 64 cores); `pbx sbatch`
      always needs --queue

    Notes:
    - `-A` is optional (the default allocation is used).
    - Extreme-memory (`EM`, 96 cores) nodes are not covered.
    - The docs load a compiler plus a matching MPI module (OpenMPI, MVAPICH2
      or Intel MPI) but show no launch line; if srun does not work with it,
      set `backend: openmpi` in config.yaml.

    Status: written from the site documentation; not yet run on the system.
    Defaults are meant to be runnable, then tweaked via config.yaml / --sched-opts.
    Sub-node allocation: see the 'SLURM sub-node allocations' caveat in
    resource_manager/README.md before relying on it.
    """

    SYSTEM_NAME = 'bridges2-cpu'
    CORES_PER_NODE = 128   # 2x 64-core AMD EPYC 7742 (RM nodes); SMT not stated in the docs
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
        Detect allocated nodes for a SLURM job on Bridges-2 CPU nodes.

        Returns:
            tuple[int, int]: (nodes, 0) -- no GPUs on CPU nodes.
        """
        nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        return nodes, 0

    def get_config(self, run_dir: Path, retries: int = 0,
                   max_workers: Optional[int] = None) -> Config:
        """
        Generate Parsl configuration for Bridges-2 CPU nodes.

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
                    label="htex_bridges2_cpu",
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
