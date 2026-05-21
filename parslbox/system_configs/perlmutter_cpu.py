import os
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class PerlmutterCpuConfig(SystemConfig):
    """
    Configuration for NERSC Perlmutter CPU nodes.

    Perlmutter CPU node specifications:
    - 2x AMD EPYC 7763 (Milan) CPUs (128 physical cores, 256 logical with SMT)
    - 512 GB DDR4 DRAM
    - 4 NUMA domains per socket (NPS=4), 8 NUMA domains total
    - 1x HPE Slingshot 11 NIC
    - PCIe 4.0 NIC-CPU connection
    """

    SYSTEM_NAME = 'perlmutter-cpu'
    CORES_PER_NODE = 128         # 128 physical cores (SMT disabled for MPI compatibility)
    GPUS_PER_NODE = 0
    SCHEDULER = "SLURM"
    MPI_CMD_TO_USE = "srun"
    MPI_BACKEND = "srun"
    MAX_WORKERS_PER_NODE = 128
    WORKER_CPU_AFFINITY = None
    GPU_TYPE = None
    DRAM_PER_NODE = 512          # 512 GB DDR4

    def __init__(self):
        super().__init__()

    def detect_resources(self) -> tuple[int, int]:
        """
        Detect allocated nodes for a SLURM job on Perlmutter CPU partition.

        Returns:
            tuple[int, int]: (nodes, 0) — no GPUs on CPU nodes.
        """
        nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        return nodes, 0

    def get_config(self, run_dir: Path, retries: int = 0,
                   max_workers: Optional[int] = None) -> Config:
        """
        Generate Parsl configuration for Perlmutter CPU nodes.

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
                    label="htex_perlmutter_cpu",
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

    def get_default_sched_opts(self) -> str:
        return "#SBATCH --constraint=cpu"

    def get_default_mpi_config_yaml(self) -> dict:
        return {
            "backend": self.MPI_BACKEND,
            "use_gpu_wrapper": False,
            "cpu_bind_method": "cores",
            "use_hostlist": True
        }
