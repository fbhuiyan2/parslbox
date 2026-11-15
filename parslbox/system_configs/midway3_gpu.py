import os
import subprocess
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SimpleLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class Midway3GpuConfig(SystemConfig):
    """
    Configuration for UChicago RCC Midway3 GPU nodes (`gpu` partition).

    Midway3 `gpu` node specifications:
    - 2x Intel Xeon Gold 6248R, 48 cores per node, 4 GPUs per node
    - The GPU model varies by node: 5x V100 (192 GB), 5x Quadro RTX 6000
      (192 GB), 1x A100 (384 GB)
    - SLURM; GPUs are shared, `--gres=gpu:N` with N from 1 to 4

    Notes:
    - Pick a model with `--constraint=<feature>`: the docs show
      `--constraint=v100` and `--constraint=rtx6000`, and list `a100` as a GPU
      type in the partition table. Pass it with
      --sched-opts '#SBATCH --constraint=v100'. Check the exact feature names
      with `sinfo -o "%20P %f"`.
    - The config ships no GPU count: pass it with --sched-opts
      '#SBATCH --gres=gpu:N'. One core is assigned by default; request
      more with --ntasks for mixed CPU-GPU codes.

    Status: written from the site documentation; not yet run on the system.
    Defaults are meant to be runnable, then tweaked via config.yaml / --sched-opts.
    Sub-node allocation: see the 'SLURM sub-node allocations' caveat in
    resource_manager/README.md before relying on it.
    """

    SYSTEM_NAME = 'midway3-gpu'
    CORES_PER_NODE = 48          # 2x 24-core Intel Xeon Gold 6248R
    GPUS_PER_NODE = 4             # 4 GPUs; model depends on node, see below
    SCHEDULER = "SLURM"
    MPI_CMD_TO_USE = "srun"
    MPI_BACKEND = "srun"
    MAX_WORKERS_PER_NODE = 4     # One worker per GPU
    WORKER_CPU_AFFINITY = None
    GPU_TYPE = 'cuda'
    DRAM_PER_NODE = 192          # GB

    def __init__(self):
        super().__init__()

    def detect_resources(self) -> tuple[int, int]:
        """
        Detect allocated nodes and GPUs for a SLURM job on Midway3 GPU nodes.

        Node count comes from SLURM_JOB_NUM_NODES. On a single node the GPU
        count is read from nvidia-smi, which only lists the GPUs the job was
        given, so a sub-node allocation is counted correctly. On several
        nodes nvidia-smi only sees the local one, so the known per-node count
        is used (multi-node jobs hold whole nodes).

        Returns:
            tuple[int, int]: (nodes, total_gpus)
        """
        nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

        if nodes > 1:
            total_gpus = nodes * self.GPUS_PER_NODE
            return nodes, total_gpus

        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True)
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
        Generate Parsl configuration for Midway3 GPU nodes.

        One worker per allocated GPU. The count comes from detect_resources()
        rather than GPUS_PER_NODE * nodes, so a sub-node allocation is not
        given workers for GPUs it does not hold, and is capped at
        GPUS_PER_NODE * nodes; on whole nodes that is exactly perlmutter-gpu's
        rule.

        Args:
            run_dir: Path for Parsl's run directory.
            retries: Number of retries for failed Parsl apps.
            max_workers: Optional override for total workers across all nodes.
                         If None, uses one worker per allocated GPU.
                         If provided, will be capped at the same limit.

        Returns:
            Config: A Parsl configuration object.
        """
        nodes, total_gpus = self.detect_resources()

        # Never more than the whole-node count perlmutter-gpu would use, in
        # case nvidia-smi lists more devices than GPUS_PER_NODE (MIG slices,
        # a larger node variant).
        worker_cap = min(total_gpus, self.MAX_WORKERS_PER_NODE * nodes)

        if max_workers is not None:
            max_workers_per_node = min(max_workers, worker_cap)
        else:
            max_workers_per_node = worker_cap

        cores_per_worker = self.CORES_PER_NODE / max_workers_per_node

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_midway3_gpu",
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
