"""
Perlmutter GPU config with SrunLauncher.

Variant of PerlmutterGpuConfig that places one Parsl manager per compute
node via `srun`, with workers spawned locally on each compute node. Use
this config in place of `perlmutter-gpu` for runs where the head node
would OOM under the default SimpleLauncher.

SrunLauncher uses `--overlap` so the outer srun (managing managers) can
coexist with PBX's inner srun calls (running scientific apps per job).
"""

from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import SrunLauncher

from parslbox.system_configs.perlmutter_gpu import PerlmutterGpuConfig


class PerlmutterGpuSrunConfig(PerlmutterGpuConfig):
    """Same hardware as perlmutter-gpu; one manager per compute node via srun."""

    SYSTEM_NAME = "perlmutter-gpu-srun"

    def get_config(self, run_dir: Path, retries: int = 0,
                   max_workers: Optional[int] = None) -> Config:
        nodes, _ = self.detect_resources()

        # Per-node worker count (NOT x nodes) — srun launches one manager
        # per compute node, and each manager spawns up to this many workers
        # locally on its node.
        if max_workers is not None:
            max_workers_per_node = min(max_workers // nodes,
                                       self.MAX_WORKERS_PER_NODE)
        else:
            max_workers_per_node = self.MAX_WORKERS_PER_NODE

        cores_per_worker = self.CORES_PER_NODE / max(1, max_workers_per_node)

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_perlmutter_gpu_srun",
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
                        min_blocks=1,
                        nodes_per_block=nodes,
                        launcher=SrunLauncher(
                            overrides='--overlap --ntasks-per-node=1 --cpus-per-task=1'
                        ),
                    ),
                )
            ],
            run_dir=str(run_dir),
            retries=retries,
        )
