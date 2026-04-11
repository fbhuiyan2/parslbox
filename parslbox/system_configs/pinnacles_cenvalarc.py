import os
import subprocess
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class PinnaclesCenvalarcConfig(SystemConfig):
    """
    Configuration for UC Merced Pinnacles cluster (CENVALARC partition).

    CENVALARC partitions:
    - cenvalarc.compute: 12 CPU-only nodes (2× Intel 32-Core Xeon Gold 6530, 256 GB RAM)
    - cenvalarc.bigmem:   4 CPU-only nodes (2× Intel 32-Core Xeon Gold 6530, 1 TB RAM)
    - cenvalarc.gpu:      8 GPU nodes      (2× Intel 32-Core Xeon Gold 6530, 256 GB RAM,
                                             2× NVIDIA L40S or 2× NVIDIA H200 NVL)

    All nodes: 64 cores (2× 32-core). GPU nodes: 2 NVIDIA GPUs (L40S or H200 NVL).
    Single config handles both CPU and GPU partitions via nvidia-smi detection.
    """

    # System specifications
    SYSTEM_NAME = 'pinnacles-cenvalarc'
    CORES_PER_NODE = 64          # 2× Intel 32-Core Xeon Gold 6530
    GPUS_PER_NODE = 2            # 2× NVIDIA GPUs (L40S or H200 NVL)
    SCHEDULER = "SLURM"
    MPI_CMD_TO_USE = "srun"
    MPI_BACKEND = "srun"
    MAX_WORKERS_PER_NODE = 2
    WORKER_CPU_AFFINITY = None
    GPU_TYPE = 'cuda'

    def __init__(self):
        """Initialize Pinnacles CENVALARC configuration."""
        super().__init__()

    def detect_resources(self) -> tuple[int, int]:
        """
        Detect allocated nodes and GPUs for a SLURM job on Pinnacles.

        Node count comes from SLURM_JOB_NUM_NODES. GPU count is detected
        via nvidia-smi on the local node and multiplied by the node count
        (all GPU nodes are homogeneous with 2× L40S).

        On CPU partitions (cenvalarc.compute, cenvalarc.bigmem), nvidia-smi
        will fail or report 0 GPUs, resulting in CPU-only mode.

        Returns:
            tuple[int, int]: (nodes, total_gpus)
        """
        # --- Get node count from SLURM ---
        nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

        # --- Get per-node GPU count via nvidia-smi ---
        gpus_per_node = 0
        try:
            result = subprocess.run(
                ['nvidia-smi', '-L'],
                capture_output=True, text=True, check=True
            )
            gpus_per_node = len([
                line for line in result.stdout.strip().split('\n')
                if line.strip()
            ])
        except (subprocess.CalledProcessError, FileNotFoundError):
            # No GPUs available (CPU partition) — this is expected
            gpus_per_node = 0

        total_gpus = nodes * gpus_per_node
        return nodes, total_gpus

    def get_config(self, run_dir: Path, retries: int = 0,
                   max_workers: Optional[int] = None) -> Config:
        """
        Generate Parsl configuration for Pinnacles CENVALARC.

        Dynamically adjusts workers based on detected resources:
        - GPU partition: workers = GPUs per node (2)
        - CPU partition: workers = 2 (sensible default for CPU jobs)

        Args:
            run_dir: Path for Parsl's run directory.
            retries: Number of retries for failed Parsl apps.
            max_workers: Optional override for workers per node.

        Returns:
            Config: A Parsl configuration object.
        """
        nodes, total_gpus = self.detect_resources()

        if nodes == 0:
            nodes = 1

        detected_gpus_per_node = total_gpus // nodes if nodes > 0 else 0

        # Determine workers per node based on detected resources
        if detected_gpus_per_node > 0:
            # GPU partition: one worker per GPU
            workers_per_node = detected_gpus_per_node
        else:
            # CPU partition: default to 2 workers
            workers_per_node = 2

        # Apply max_workers override if provided
        if max_workers is not None:
            workers_per_node = min(max_workers, workers_per_node)

        cores_per_worker = self.CORES_PER_NODE / workers_per_node

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_pinnacles_cenvalarc",
                    heartbeat_period=120,
                    heartbeat_threshold=300,
                    worker_debug=True,
                    available_accelerators=0,
                    max_workers_per_node=workers_per_node,
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
        """Pinnacles CENVALARC MPI defaults for config generation."""
        return {
            "backend": self.MPI_BACKEND,
            "use_gpu_wrapper": True,
            "cpu_bind_method": "depth",
            "use_hostlist": True
        }
