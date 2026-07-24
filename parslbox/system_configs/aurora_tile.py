import os
from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import MpiExecLauncher
from parslbox.system_configs.base_sysconf import SystemConfig


class AuroraTileConfig(SystemConfig):
    """
    Configuration class for the ALCF Aurora supercomputer (tile mode).
    
    Aurora specifications (tile mode):
    - 104 physical cores per node (208 with hyperthreading)
    - 6 Intel Data Center Max 1550 Series GPUs per node
    - Each GPU has 2 tiles (stacks) for a total of 12 tiles per node
    - This config treats each tile as an independent GPU unit
    - PBS scheduler
    """
    
    # System specifications
    SYSTEM_NAME = 'aurora-tile'
    CORES_PER_NODE = 208  # 104 physical cores with hyperthreading [4 CPU sockets reserved for system services]
    EXCLUDE_CORES = [0, 104, 52, 156]   # aurora reserves these cores for system services
    GPUS_PER_NODE = 12    # 6 physical GPUs × 2 tiles each = 12 tile units
    SCHEDULER = "PBS"
    MPI_CMD_TO_USE = "mpiexec"  # Legacy
    MPI_BACKEND = "mpich"  # MPICH on Aurora
    MAX_WORKERS_PER_NODE = 12  # One worker per tile
    # Aurora is set up weird, even though there are 17 cores per tile, worker_cpu_affinity has 16 cores. check aurora system design to learn why
    WORKER_CPU_AFFINITY = "list:1-8,105-112:9-16,113-120:17-24,121-128:25-32,129-136:33-40,137-144:41-48,145-152:53-60,157-164:61-68,165-172:69-76,173-180:77-84,181-188:85-92,189-196:93-100,197-204"
    GPU_TYPE = 'intel'
    
    def __init__(self):
        """Initialize Aurora tile configuration with validation."""
        super().__init__()
    
    def detect_resources(self) -> tuple[int, int]:
        """
        Detects the number of nodes and total GPU tiles allocated for a PBS job on Aurora.
        
        On Aurora, users are allocated full nodes. This function reads the PBS_NODEFILE
        to determine the number of allocated nodes and calculates total tiles based on
        the fixed number of tiles per node (12 tiles = 6 GPUs × 2 tiles each).

        Returns:
            tuple[int, int]: A tuple of (nodes, total_tiles)
        """
        node_file = os.environ.get("PBS_NODEFILE")
        
        if node_file and os.path.exists(node_file):
            with open(node_file, 'r') as f:
                # Each line in the nodefile corresponds to a unique node
                nodes = len(set(f.read().strip().splitlines()))
        else:
            raise FileNotFoundError(
                f"Node file 'PBS_NODEFILE' not found. "
                "Aurora config expects a node list file from PBS."
                )
        total_tiles = nodes * self.GPUS_PER_NODE
        return nodes, total_tiles

    def get_config(self, run_dir: Path, retries: int = 0, max_workers: Optional[int] = None) -> Config:
        """
        Generates a Parsl configuration for the ALCF Aurora supercomputer (tile mode).

        This config is designed for multi-node execution via a PBS batch job. It uses
        the MpiExecLauncher to place one Parsl manager per compute node (via `mpiexec`
        with `--ppn 1`), and each manager spawns its workers locally on that node. This
        distributes workers across the allocation instead of concentrating them on the
        head node, so it scales past the head-node RAM ceiling that a SimpleLauncher
        hits above ~10k workers.

        Args:
            run_dir (Path): The path for Parsl's run directory.
            retries (int): The number of retries for failed Parsl apps.
            max_workers (Optional[int]): Optional override for total workers across all nodes.
                                        If None, uses MAX_WORKERS_PER_NODE per node (default behavior).
                                        If provided, the per-node worker count is capped at
                                        MAX_WORKERS_PER_NODE.

        Returns:
            Config: A Parsl configuration object.
        """
        nodes, _ = self.detect_resources()

        # Per-node worker count (NOT x nodes) — mpiexec launches one manager
        # per compute node, and each manager spawns up to this many workers
        # locally on its node.
        if max_workers is not None:
            max_workers_per_node = min(max_workers // nodes, self.MAX_WORKERS_PER_NODE)
        else:
            max_workers_per_node = self.MAX_WORKERS_PER_NODE

        # Cores assigned to each worker (mapped to a GPU tile). Oversubscription is
        # possible by setting cores_per_worker < 1.0.
        cores_per_worker = self.CORES_PER_NODE / max(1, max_workers_per_node)

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_aurora_tile",
                    heartbeat_period=120,
                    heartbeat_threshold=300,
                    worker_debug=True,
                    available_accelerators=0,
                    max_workers_per_node=max_workers_per_node,
                    cores_per_worker=cores_per_worker,
                    #cpu_affinity=self.WORKER_CPU_AFFINITY,
                    prefetch_capacity=0,  # Recommended for GPU workloads
                    provider=LocalProvider(
                        init_blocks=1,
                        max_blocks=1,
                        min_blocks=1,
                        nodes_per_block=nodes,
                        # ALCF-recommended args (Aurora docs):
                        #   bind_cmd="--cpu-bind" — MPICH/PALS syntax (default
                        #     "--bind-to" is OpenMPI-only and is rejected by PALS)
                        #   overrides="--ppn 1" — exactly one manager rank per
                        #     compute node; manager then spawns the
                        #     max_workers_per_node workers locally
                        launcher=MpiExecLauncher(
                            bind_cmd="--cpu-bind",
                            overrides="--ppn 1",
                        ),
                    ),
                )
            ],
            run_dir=str(run_dir), # run_dir must be a string
            retries=retries,
        )
    
    def get_default_sched_opts(self) -> str:
        """Aurora Tile default scheduler directives."""
        return "#PBS -l filesystems=home:flare"

    def get_default_mpi_config_yaml(self) -> dict:
        """Aurora Tile MPI defaults for config generation."""
        return {
            "backend": self.MPI_BACKEND,
            "use_gpu_wrapper": True,
            "cpu_bind_method": "rankfile",
            "use_hostlist": True
        }
