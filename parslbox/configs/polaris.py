import os
from pathlib import Path
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.configs.base import SystemConfig


class PolarisConfig(SystemConfig):
    """
    Configuration class for the ALCF Polaris supercomputer.
    
    Polaris specifications:
    - 32 physical cores per node (64 with hyperthreading)
    - 4 NVIDIA A100 GPUs per node
    - PBS scheduler
    """
    
    # System specifications
    CORES_PER_NODE = 32
    GPUS_PER_NODE = 4
    SCHEDULER = "PBS"
    
    def detect_resources(self) -> tuple[int, int]:
        """
        Detects the number of nodes and total GPUs allocated for a PBS job on Polaris.
        
        On Polaris, users are allocated full nodes. This function reads the PBS_NODEFILE
        to determine the number of allocated nodes and calculates total GPUs based on
        the fixed number of GPUs per node.

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
                "Polaris config expects a node list file from PBS."
                )
        total_gpus = nodes * self.GPUS_PER_NODE
        return nodes, total_gpus

    def get_config(self, run_dir: Path, retries: int = 0) -> Config:
        """
        Generates a Parsl configuration for the ALCF Polaris supercomputer.

        This config is designed for multi-node execution via a PBS batch job.
        It uses the MpiExecLauncher to place one worker per GPU, ensuring
        optimal resource utilization.

        Args:
            run_dir (Path): The path for Parsl's run directory.
            retries (int): The number of retries for failed Parsl apps.

        Returns:
            Config: A Parsl configuration object.
        """
        nodes, total_gpus = self.detect_resources()
        
        # Calculate how many physical cores each worker (mapped to a GPU) gets
        cores_per_worker = self.CORES_PER_NODE // self.GPUS_PER_NODE

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_polaris",
                    heartbeat_period=60,
                    heartbeat_threshold=120,
                    worker_debug=True,
                    # Tell the executor how many total GPUs are available
                    available_accelerators=total_gpus,
                    # One worker per GPU on each node
                    max_workers_per_node=self.GPUS_PER_NODE,
                    # Assign a balanced number of cores to each worker
                    cores_per_worker=cores_per_worker,
                    # This CPU affinity string is optimized for Polaris's architecture,
                    # ensuring each worker's threads are bound to cores physically
                    # close to the GPU it's managing.
                    cpu_affinity="list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39",
                    prefetch_capacity=0,  # Recommended for GPU workloads
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


# Backward compatibility: create instance and expose original function
_polaris_config = PolarisConfig()

def get_config(run_dir: Path, retries: int = 0) -> Config:
    """
    Backward compatibility function for the original get_config interface.
    """
    return _polaris_config.get_config(run_dir, retries)
