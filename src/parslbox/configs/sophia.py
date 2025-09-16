import os
import subprocess
from pathlib import Path
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher

# --- Sophia Specific Details ---
# Each Sophia node has 2 AMD Rome 64-core CPUs.
CORES_PER_NODE = 128
# Each Sophia DGX A100 node has 8 GPUs. This is used as a fallback.
GPUS_PER_NODE_FALLBACK = 8


def _detect_sophia_resources() -> tuple[int, int]:
    """
    Detects the number of nodes and total GPUs for a PBS job on Sophia.

    This function first attempts to use `nvidia-smi -L` to get an exact count
    of GPUs visible to the job. If that fails, it falls back to estimating
    the GPU count based on the number of nodes in PBS_NODEFILE.

    Returns:
        A tuple of (nodes, total_gpus).
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
        gpu_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If nvidia-smi fails, fallback to node-based estimation
        gpu_count = 0

    # --- Determine final GPU count ---
    if gpu_count > 0:
        return nodes, gpu_count
    else:
        # Fallback: assume a fixed number of GPUs per node on Sophia
        total_gpus = nodes * GPUS_PER_NODE_FALLBACK
        return nodes, total_gpus


def get_config(run_dir: Path, retries: int = 0) -> Config:
    """
    Generates a Parsl configuration for the ALCF Sophia supercomputer.

    This config is designed for multi-node execution via a PBS batch job.
    It uses the MpiExecLauncher to place one worker per GPU.

    Args:
        run_dir (Path): The path for Parsl's run directory.
        retries (int): The number of retries for failed Parsl apps.

    Returns:
        Config: A Parsl configuration object.
    """
    nodes, total_gpus = _detect_sophia_resources()

    # Ensure nodes is at least 1 to prevent division by zero
    if nodes == 0:
        nodes = 1
    
    gpus_per_node = total_gpus // nodes
    cores_per_worker = CORES_PER_NODE // gpus_per_node

    return Config(
        executors=[
            HighThroughputExecutor(
                label="htex_sophia",
                heartbeat_period=60,
                heartbeat_threshold=120,
                worker_debug=True,
                available_accelerators=total_gpus,
                max_workers_per_node=gpus_per_node,
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
