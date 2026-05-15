import os
import subprocess
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class PerlmutterGpuConfig(SystemConfig):
    """
    Configuration for NERSC Perlmutter GPU nodes.

    Perlmutter GPU node specifications:
    - 1x AMD EPYC 7763 (Milan) CPU (64 physical cores, 128 logical with SMT)
    - 4x NVIDIA A100 GPUs (40 GB HBM2e; 80 GB HBM2e with gpu&hbm80g constraint)
    - 256 GB DDR4 DRAM
    - PCIe 4.0 GPU-CPU and NIC-CPU connections
    - 4x HPE Slingshot 11 NICs
    - 4x NVLink connections between each pair of GPUs
    """

    SYSTEM_NAME = 'perlmutter-gpu'
    CORES_PER_NODE = 128         # 64 physical cores, 128 logical with SMT
    GPUS_PER_NODE = 4            # 4x NVIDIA A100
    SCHEDULER = "SLURM"
    MPI_CMD_TO_USE = "srun"
    MPI_BACKEND = "srun"
    MAX_WORKERS_PER_NODE = 4     # One worker per GPU
    WORKER_CPU_AFFINITY = None
    GPU_TYPE = 'cuda'
    DRAM_PER_NODE = 256          # 256 GB DDR4

    def __init__(self):
        super().__init__()

    def detect_resources(self) -> tuple[int, int]:
        """
        Detect allocated nodes and GPUs for a SLURM job on Perlmutter.

        Node count comes from SLURM_JOB_NUM_NODES. GPU count is detected
        via nvidia-smi on the local node and multiplied by the node count
        (all GPU nodes have 4x A100).

        Returns:
            tuple[int, int]: (nodes, total_gpus)
        """
        # --- Get node count from SLURM ---
        nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

        if nodes > 1:
            # Multi-node: nvidia-smi only sees local GPUs, so use the known
            # per-node count to compute total across all nodes.
            total_gpus = nodes * self.GPUS_PER_NODE
            return nodes, total_gpus
        else:
            # Single node: use nvidia-smi to detect actual visible GPUs,
            # which may differ from GPUS_PER_NODE in sub-node allocations.
            try:
                result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True)
                # Each line of nvidia-smi -L output corresponds to one GPU
                detected_gpu_count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
            except (subprocess.CalledProcessError, FileNotFoundError):
                detected_gpu_count = 0

            if detected_gpu_count > 0:
                return nodes, detected_gpu_count
            else:
                # Fallback: assume full node allocation
                total_gpus = nodes * self.GPUS_PER_NODE
                return nodes, total_gpus

    def get_config(self, run_dir: Path, retries: int = 0,
                   max_workers: Optional[int] = None) -> Config:
        """
        Generate Parsl configuration for Perlmutter GPU nodes.

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

        if detected_gpus_per_node > 0:
            workers_per_node = detected_gpus_per_node
        else:
            workers_per_node = 4

        if max_workers is not None:
            workers_per_node = min(max_workers, workers_per_node)

        cores_per_worker = self.CORES_PER_NODE / workers_per_node

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_perlmutter_gpu",
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

    def get_default_sched_opts(self) -> str:
        """
        Perlmutter GPU default scheduler directives.

        Users can override to 'gpu&hbm80g' via config.yaml sched_opts
        or CLI --sched-opts for 80 GB HBM A100 nodes.
        """
        return "#SBATCH -C gpu"

    def get_default_mpi_config_yaml(self) -> dict:
        return {
            "backend": self.MPI_BACKEND,
            "use_gpu_wrapper": False,
            "cpu_bind_method": "cores",
            "use_hostlist": True
        }
