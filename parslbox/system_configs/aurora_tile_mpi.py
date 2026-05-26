"""
Aurora tile-mode config with MpiExecLauncher.

Variant of AuroraTileConfig that places one Parsl manager per compute node
via `mpiexec`, with workers spawned locally on each compute node. Use this
config in place of `aurora-tile` for runs where the head node would OOM
under the default SimpleLauncher (per-worker RSS x total workers exceeds
head-node RAM). Empirically, switch to this variant above ~10k workers.
"""

from pathlib import Path
from typing import Optional
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.providers import LocalProvider
from parsl.launchers import MpiExecLauncher

from parslbox.system_configs.aurora_tile import AuroraTileConfig


class AuroraTileMpiConfig(AuroraTileConfig):
    """Same hardware as aurora-tile; one manager per compute node via mpiexec."""

    SYSTEM_NAME = "aurora-tile-mpi"

    def get_config(self, run_dir: Path, retries: int = 0,
                   max_workers: Optional[int] = None) -> Config:
        nodes, _ = self.detect_resources()

        # Per-node worker count (NOT x nodes) — mpiexec launches one manager
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
                    label="htex_aurora_tile_mpi",
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
            run_dir=str(run_dir),
            retries=retries,
        )
