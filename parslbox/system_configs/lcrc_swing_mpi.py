"""
LCRC Swing config with MpiRunLauncher.

Variant of LcrcSwingConfig that places one Parsl manager per compute node
via `mpirun` (OpenMPI), with workers spawned locally on each compute node.
Use this config in place of `lcrc-swing` for runs where the head node
would OOM under the default SimpleLauncher (per-worker RSS x total workers
exceeds head-node RAM). Empirically, switch to this variant above ~10k workers.
"""

from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import MpiRunLauncher

from parslbox.system_configs.lcrc_swing import LcrcSwingConfig


class LcrcSwingMpiConfig(LcrcSwingConfig):
    """Same hardware as lcrc-swing; one manager per compute node via OpenMPI mpirun."""

    SYSTEM_NAME = "lcrc-swing-mpi"

    def get_config(self, run_dir: Path, retries: int = 0,
                   max_workers: Optional[int] = None) -> Config:
        nodes, total_gpus = self.detect_resources()
        if nodes == 0:
            nodes = 1
        detected_gpus_per_node = total_gpus // nodes

        # Per-node worker count (NOT x nodes) — mpirun launches one manager
        # per compute node, and each manager spawns up to this many workers
        # locally on its node.
        if max_workers is not None:
            max_workers_per_node = min(max_workers // nodes,
                                       detected_gpus_per_node)
        else:
            max_workers_per_node = detected_gpus_per_node

        cores_per_worker = self.CORES_PER_NODE / max(1, max_workers_per_node)

        return Config(
            executors=[
                HighThroughputExecutor(
                    label="htex_lcrc_swing_mpi",
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
                        # OpenMPI mpirun args for one-manager-per-node:
                        #   --bind-to none — manager floats across node CPUs
                        #     (avoids pinning conflict with inner mpirun ranks
                        #     placed by pbx's CPU mapping)
                        #   -N 1 — exactly one manager rank per compute node;
                        #     manager then spawns max_workers_per_node workers
                        #     locally
                        # OpenMPI on PBS with TM integration reads PBS_NODEFILE
                        # automatically; no explicit --hostfile needed.
                        launcher=MpiRunLauncher(
                            overrides="--bind-to none -N 1",
                        ),
                    ),
                )
            ],
            run_dir=str(run_dir),
            retries=retries,
        )
