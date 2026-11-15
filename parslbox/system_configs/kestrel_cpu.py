import os
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class KestrelCpuConfig(SystemConfig):
    """
    Configuration for NLR (formerly NREL) Kestrel CPU nodes.

    Kestrel CPU node specifications:
    - 2x Intel Xeon Sapphire Rapids, 52 cores each (104 cores per node)
    - 256 GB DDR5 (64 nodes have 1 TB; ~240 GB usable on a standard node)
    - HPE Slingshot 11 interconnect
    - SLURM; the `shared` partition allows sub-node jobs (billed by the larger
      of the core, memory or GPU share)

    Notes:
    - `--account` is required (marked "Required" in the site's example sbatch
      scripts).
    - Outside `shared` and `debug`, SLURM routes a job to a partition from its
      walltime and memory request when none is named. `pbx sbatch` always
      writes `--partition`, so pass --queue (e.g. standard).
    - The site's Performance Recommendations page prefers Cray MPICH over
      OpenMPI and launches with `srun`.

    Status: written from the site documentation; not yet run on the system.
    Defaults are meant to be runnable, then tweaked via config.yaml / --sched-opts.
    Sub-node allocation: see the 'SLURM sub-node allocations' caveat in
    resource_manager/README.md before relying on it.
    """

    SYSTEM_NAME = 'kestrel-cpu'
    CORES_PER_NODE = 104   # 2x 52-core Intel Sapphire Rapids; SMT not stated in the docs
    GPUS_PER_NODE = 0
    SCHEDULER = "SLURM"
    MPI_CMD_TO_USE = "srun"
    MPI_BACKEND = "srun"
    MAX_WORKERS_PER_NODE = 104
    WORKER_CPU_AFFINITY = None
    GPU_TYPE = None
    DRAM_PER_NODE = 256          # GB

    def __init__(self):
        super().__init__()

    def detect_resources(self) -> tuple[int, int]:
        """
        Detect allocated nodes for a SLURM job on Kestrel CPU nodes.

        Returns:
            tuple[int, int]: (nodes, 0) -- no GPUs on CPU nodes.
        """
        nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        return nodes, 0

    def get_config(self, run_dir: Path, retries: int = 0,
                   max_workers: Optional[int] = None) -> Config:
        """
        Generate Parsl configuration for Kestrel CPU nodes.

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
                    label="htex_kestrel_cpu",
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
