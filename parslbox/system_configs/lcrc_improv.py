import os
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class LcrcImprovConfig(SystemConfig):
    """
    Configuration class for the LCRC Improv supercomputer.
    
    Improv specifications:
    - 825 dual-socket compute nodes
    - 128 cores per node (2 AMD 7713 64-core CPUs @ 2.0 GHz)
    - 256 GB DDR4 memory (standard nodes)
    - 68 nodes with 6TB NVMe SSD (12 of those with 1024 GB DDR4)
    - Nvidia/Mellanox HDR200 interconnect
    - PBS Pro scheduler
    - CPU-only system (no GPUs)
    """
    
    # System specifications
    SYSTEM_NAME = 'lcrc-improv'
    CORES_PER_NODE = 128  # 2 AMD 7713 64-core CPUs
    GPUS_PER_NODE = 0     # CPU-only system
    SCHEDULER = "PBS"
    MPI_CMD_TO_USE = "mpirun"  # Legacy
    MPI_BACKEND = "openmpi"  # OpenMPI on Improv
    MAX_WORKERS_PER_NODE = 128  # Can use all cores as workers
    WORKER_CPU_AFFINITY = None  # Allow system to handle CPU affinity
    GPU_TYPE = None  # No GPUs
    
    def __init__(self):
        """Initialize LCRC Improv configuration with validation."""
        super().__init__()
    
    def detect_resources(self) -> tuple[int, int]:
        """
        Detects the number of nodes for a PBS job on Improv.

        Since Improv is a CPU-only system, this returns 0 for GPU count.

        Returns:
            tuple[int, int]: A tuple of (nodes, total_gpus)
                            total_gpus is always 0 for Improv
        """
        # --- Get node count from PBS ---
        node_file = os.environ.get("PBS_NODEFILE")
        if node_file and os.path.exists(node_file):
            with open(node_file, 'r') as f:
                # Use a set to count unique nodes
                nodes = len(set(f.read().strip().splitlines()))
        else:
            raise FileNotFoundError(
                f"Node file 'PBS_NODEFILE' not found. "
                "LCRC Improv config expects a node list file from PBS."
            )

        # CPU-only system, no GPUs
        return nodes, 0

    def get_config(self, run_dir: Path, retries: int = 0, max_workers: Optional[int] = None) -> Config:
        """
        Generates a Parsl configuration for the LCRC Improv supercomputer.

        This config is designed for execution via a PBS batch job on Improv.
        Since Improv is a CPU-only system, workers are allocated based on
        available CPU cores rather than GPUs.

        Args:
            run_dir (Path): The path for Parsl's run directory.
            retries (int): The number of retries for failed Parsl apps.
            max_workers (Optional[int]): Optional override for total workers across all nodes.
                                        If None, uses MAX_WORKERS_PER_NODE (default behavior).
                                        If provided, will be capped at MAX_WORKERS_PER_NODE.

        Returns:
            Config: A Parsl configuration object.
        """
        nodes, _ = self.detect_resources()

        # Ensure nodes is at least 1 to prevent division by zero
        if nodes == 0:
            nodes = 1

        # For CPU-only system, max workers is based on cores
        if max_workers is not None:
            max_workers_per_node = min(max_workers, self.MAX_WORKERS_PER_NODE)
        else:
            max_workers_per_node = self.MAX_WORKERS_PER_NODE

        # Calculate how many physical cores each worker gets
        # Default: 1 core per worker for maximum parallelism
        cores_per_worker = self.CORES_PER_NODE / max_workers_per_node

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_lcrc_improv",
                    heartbeat_period=120,
                    heartbeat_threshold=300,
                    worker_debug=True,
                    available_accelerators=0,  # No GPUs
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
        """LCRC Improv MPI defaults for config generation."""
        return {
            "backend": self.MPI_BACKEND,
        }
