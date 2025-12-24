import os
from pathlib import Path
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base import SystemConfig


class CruxConfig(SystemConfig):
    """
    Configuration class for the Crux supercomputer.
    
    Crux specifications:
    - 128 cores per node (2 AMD EPYC 7742 Rome 64-core CPUs)
    - 0 GPUs per node (CPU-only system)
    - PBS scheduler
    - 256 GB DDR4 memory per node
    - 8 NUMA domains per node (4 per CPU)
    """
    
    # System specifications
    SYSTEM_NAME = 'crux'
    CORES_PER_NODE = 256    #128 --> 256 with hyperthreading
    GPUS_PER_NODE = 0  # CPU-only system
    SCHEDULER = "PBS"
    MPI_CMD_TO_USE = "mpiexec"
    MAX_WORKERS_PER_NODE = 4    # There are 8 NUMA domains so 8 can be assigned - Although PBX resource_manager does not implement NUMA domains
                                # But that is an overkill and spwans too many workers when using 100s of nodes, so, 2 or 4 is better
                                # 4 will allow node_occupancy down to 0.25
    WORKER_CPU_AFFINITY = "list:0-15,128-143:16-31,144-159:32-47,160-175:48-63,176-191:64-79,192-207:80-95,208-223:96-111,224-239:112-127,240-255"
    GPU_TYPE = None  # CPU-only system
    
    def __init__(self):
        """Initialize Crux configuration with validation."""
        super().__init__()
    
    def detect_resources(self) -> tuple[int, int]:
        """
        Detects the number of nodes and total GPUs allocated for a PBS job on Crux.
        
        On Crux, users are allocated full nodes. This function reads the PBS_NODEFILE
        to determine the number of allocated nodes. Since Crux is a CPU-only system,
        total GPUs will always be 0.

        Returns:
            tuple[int, int]: A tuple of (nodes, total_gpus)
        """
        node_file = os.environ.get("PBS_NODEFILE")
        
        if node_file and os.path.exists(node_file):
            with open(node_file, 'r') as f:
                # Each line in the nodefile corresponds to a unique node
                nodes = len(set(f.read().strip().splitlines()))
        else:
            raise FileNotFoundError(
                f"Node file 'PBS_NODEFILE' not found. "
                "Crux config expects a node list file from PBS."
                )
        
        # Crux is a CPU-only system, so total GPUs is always 0
        total_gpus = 0
        return nodes, total_gpus

    def get_config(self, run_dir: Path, retries: int = 0) -> Config:
        """
        Generates a Parsl configuration for the Crux supercomputer.

        This config is designed for multi-node CPU-only execution via a PBS batch job.
        It optimizes for the NUMA architecture with 8 NUMA domains per node.
        
        Crux NUMA topology:
        - CPU 0: NUMA 0-3 (cores 0-63, 128-191)
        - CPU 1: NUMA 4-7 (cores 64-127, 192-255)

        Args:
            run_dir (Path): The path for Parsl's run directory.
            retries (int): The number of retries for failed Parsl apps.

        Returns:
            Config: A Parsl configuration object.
        """
        nodes, total_gpus = self.detect_resources()
        
        # For CPU-only workloads, we can use multiple workers per node

        max_workers_per_node=self.MAX_WORKERS_PER_NODE*nodes    # Because LocalProvider does not launch workers on compute nodes.
                                                                # It only launches workers on the first node where the Parsl manager is running.
        
        cores_per_worker = self.CORES_PER_NODE / max_workers_per_node      # cores to be assigned to each worker. Oversubscription is possible
                                                                            # by setting cores_per_worker < 1.0.
        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_crux",
                    heartbeat_period=120,
                    heartbeat_threshold=300,
                    worker_debug=True,
                    # No GPUs available on Crux
                    available_accelerators=0,
                    # Use configurable max workers per node
                    max_workers_per_node=max_workers_per_node,
                    # 32 cores per worker (one NUMA domain worth)
                    cores_per_worker=cores_per_worker,
                    # Use configurable CPU affinity for Parsl workers
                    #cpu_affinity=self.WORKER_CPU_AFFINITY,
                    prefetch_capacity=0,  # Good for CPU workloads
                    provider=LocalProvider(
                        init_blocks=1,
                        max_blocks=1,
                        launcher=SimpleLauncher(),
                    ),
                )
            ],
            run_dir=str(run_dir), # run_dir must be a string
            retries=retries,
        )
