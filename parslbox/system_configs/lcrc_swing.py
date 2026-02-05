import os
import subprocess
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class LcrcSwingConfig(SystemConfig):
    """
    Configuration class for the LCRC Swing supercomputer.
    
    Swing specifications:
    - 128 cores per node (2 AMD EPYC 7742 64-core CPUs)
    - 8 NVIDIA A100 GPUs per node (40GB on gpu1-4,6; 80GB on gpu5)
    - PBS Pro scheduler
    - Sub-node GPU allocation supported (1, 2, 4, or 8 GPUs per job)
    - 1TB DDR4 memory per node (2TB on gpu5)
    """
    
    # System specifications
    SYSTEM_NAME = 'lcrc-swing'
    CORES_PER_NODE = 128  # 2 AMD EPYC 7742 64-core CPUs
    GPUS_PER_NODE = 8     # 8 NVIDIA A100 GPUs per node
    SCHEDULER = "PBS"
    MPI_CMD_TO_USE = "mpirun"
    MAX_WORKERS_PER_NODE = 8
    WORKER_CPU_AFFINITY = None  # Allow system to handle CPU affinity
    GPU_TYPE = 'cuda'
    
    def __init__(self):
        """Initialize LCRC Swing configuration with validation."""
        super().__init__()
    
    def detect_resources(self) -> tuple[int, int]:
        """
        Detects the number of nodes and total GPUs for a PBS job on Swing.

        Swing supports sub-node GPU allocation, so this function uses `nvidia-smi -L`
        to get the exact count of GPUs visible to the job. If that fails, it falls
        back to estimating the GPU count based on the number of nodes in PBS_NODEFILE.

        Returns:
            tuple[int, int]: A tuple of (nodes, total_gpus)
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
                "LCRC Swing config expects a node list file from PBS."
            )

        # --- Get GPU count using nvidia-smi ---
        # Swing allows sub-node GPU allocation, so we need to detect actual GPUs
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True)
            # Count non-empty lines in the output
            detected_gpu_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If nvidia-smi fails, fallback to node-based estimation
            detected_gpu_count = 0

        # --- Determine final GPU count ---
        if detected_gpu_count > 0:
            return nodes, detected_gpu_count
        else:
            # Fallback: assume a fixed number of GPUs per node
            total_gpus = nodes * self.GPUS_PER_NODE
            return nodes, total_gpus

    def get_config(self, run_dir: Path, retries: int = 0, max_workers: Optional[int] = None) -> Config:
        """
        Generates a Parsl configuration for the LCRC Swing supercomputer.

        This config is designed for execution via a PBS batch job on Swing.
        It supports sub-node GPU allocation and uses nvidia-smi to detect
        the actual number of GPUs available to the job.

        Args:
            run_dir (Path): The path for Parsl's run directory.
            retries (int): The number of retries for failed Parsl apps.
            max_workers (Optional[int]): Optional override for total workers across all nodes.
                                        If None, uses detected GPU count (default behavior).
                                        If provided, will be capped at detected GPU count.

        Returns:
            Config: A Parsl configuration object.
        """
        nodes, total_gpus = self.detect_resources()

        # Ensure nodes is at least 1 to prevent division by zero
        if nodes == 0:
            nodes = 1

        detected_gpus_per_node = total_gpus // nodes
        
        # For sub-node allocation, max workers is based on actual detected GPUs
        # not the theoretical maximum per node
        if max_workers is not None:
            max_workers_per_node = min(max_workers, detected_gpus_per_node)
        else:
            max_workers_per_node = detected_gpus_per_node  # One worker per detected GPU

        # Calculate how many physical cores each worker (mapped to a GPU) gets
        # Swing allocates 1/8th of node resources per GPU
        cores_per_worker = self.CORES_PER_NODE / max_workers_per_node

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_lcrc_swing",
                    heartbeat_period=120,
                    heartbeat_threshold=300,
                    worker_debug=True,
                    available_accelerators=0,  # Managed externally
                    max_workers_per_node=max_workers_per_node,
                    cores_per_worker=cores_per_worker,
                    prefetch_capacity=0,  # Recommended for GPU workloads
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
