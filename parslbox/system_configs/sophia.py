import os
import subprocess
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class SophiaConfig(SystemConfig):
    """
    Configuration class for the ALCF Sophia supercomputer.
    
    Sophia specifications:
    - 128 cores per node (2 AMD Rome 64-core CPUs)
    - 8 NVIDIA A100 GPUs per node (DGX A100)
    - PBS scheduler
    """
    
    # System specifications
    SYSTEM_NAME = 'sophia'
    CORES_PER_NODE = 128
    GPUS_PER_NODE = 8
    SCHEDULER = "PBS"
    MPI_CMD_TO_USE = "mpirun"  # Legacy
    MPI_BACKEND = "openmpi"  # OpenMPI on Sophia
    MAX_WORKERS_PER_NODE = 8
    WORKER_CPU_AFFINITY = None
    GPU_TYPE = 'cuda'
    
    def __init__(self):
        """Initialize Sophia configuration with validation."""
        super().__init__()
    
    def detect_resources(self) -> tuple[int, int]:
        """
        Detects the number of nodes and total GPUs for a PBS job on Sophia.

        This function first attempts to use `nvidia-smi -L` to get an exact count
        of GPUs visible to the job. If that fails, it falls back to estimating
        the GPU count based on the number of nodes in PBS_NODEFILE.

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
                f"Node file 'PBS_NODEFILE' not found."
                "Sophia config expects a node list file from PBS."
            )

        # --- Get GPU count using nvidia-smi ---
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
            # Fallback: assume a fixed number of GPUs per node on Sophia
            total_gpus = nodes * self.GPUS_PER_NODE
            return nodes, total_gpus

    def get_config(self, run_dir: Path, retries: int = 0, max_workers: Optional[int] = None) -> Config:
        """
        Generates a Parsl configuration for the ALCF Sophia supercomputer.

        This config is designed for multi-node execution via a PBS batch job.
        It uses the MpiExecLauncher to place one worker per GPU.

        Args:
            run_dir (Path): The path for Parsl's run directory.
            retries (int): The number of retries for failed Parsl apps.
            max_workers (Optional[int]): Optional override for total workers across all nodes.
                                        If None, uses MAX_WORKERS_PER_NODE * nodes (default behavior).
                                        If provided, will be capped at MAX_WORKERS_PER_NODE * nodes.

        Returns:
            Config: A Parsl configuration object.
        """
        nodes, total_gpus = self.detect_resources()

        # Ensure nodes is at least 1 to prevent division by zero
        if nodes == 0:
            nodes = 1
        
        detected_gpus_per_node = total_gpus // nodes
        
        # Use provided max_workers or fall back to default calculation
        # Cap at system maximum to prevent oversubscription
        if max_workers is not None:
            max_workers_per_node = min(max_workers, detected_gpus_per_node) # because Sophia allows sub-node GPU allocation #self.MAX_WORKERS_PER_NODE * nodes)
        else:
            max_workers_per_node = detected_gpus_per_node #self.MAX_WORKERS_PER_NODE * nodes    # Because LocalProvider does not launch workers on compute nodes.
                                                                        # It only launches workers on the first node where the Parsl manager is running.

        # Calculate how many physical cores each worker (mapped to a GPU) gets
        cores_per_worker = self.CORES_PER_NODE / max_workers_per_node      # cores to be assigned to each worker. Oversubscription is possible
                                                                            # by setting cores_per_worker < 1.0.

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_sophia",
                    heartbeat_period=120,
                    heartbeat_threshold=300,
                    worker_debug=True,
                    available_accelerators=0, #total_gpus,
                    max_workers_per_node=max_workers_per_node,
                    cores_per_worker=cores_per_worker,
                    #cpu_affinity=self.WORKER_CPU_AFFINITY,
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
        """Polaris default scheduler directives."""
        return "#PBS -l filesystems=home:eagle"
