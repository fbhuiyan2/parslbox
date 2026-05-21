"""
Regression tests for the multi-GPU-per-rank case.

Background: parslbox historically assumed 1 GPU per rank. When Python apps
were switched to USES_MPI=True with ranks_per_node=1, a single rank can
own multiple GPUs (e.g., all 4 Polaris GPUs). Code paths assuming
gpu_ids[0] silently dropped GPUs past the first. These tests pin down the
fix and prevent regression.

Covers:
  - ResourceAssignment.get_env_vars sets CUDA_VISIBLE_DEVICES / ZE_AFFINITY_MASK
    correctly for both raw and tile-mode systems
  - srun --gpu-bind logic uses mask_gpu (not map_gpu) for multi-GPU per rank
"""

from parslbox.resource_manager.models import ResourceAssignment, JobResourceSpec


def _make_single_node_assignment(gpu_ids):
    """Single-node assignment with one rank owning the given GPU IDs."""
    assignment = ResourceAssignment(
        job_id=1,
        node_ids=["node-0"],
        hostnames=["test-host"],
        node_occupancy=1.0,
    )
    assignment.assign_resources_to_rank(
        rank=0, node_idx=0,
        gpu_ids=gpu_ids,
        cpu_ids=list(range(8 * len(gpu_ids))),
    )
    return assignment


class TestGetEnvVarsMultiGPU:
    def test_cuda_env_vars_includes_all_gpus(self):
        a = _make_single_node_assignment([0, 1, 2, 3])
        env = a.get_env_vars()
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"

    def test_ze_mask_full_mode_is_raw_ids(self):
        a = _make_single_node_assignment([0, 1, 5])
        env = a.get_env_vars(tile_mode=False)
        assert env["ZE_AFFINITY_MASK"] == "0,1,5"

    def test_ze_mask_tile_mode_uses_phys_tile_format(self):
        # Tile IDs 0..3 → "0.0,0.1,1.0,1.1"
        a = _make_single_node_assignment([0, 1, 2, 3])
        env = a.get_env_vars(tile_mode=True)
        assert env["ZE_AFFINITY_MASK"] == "0.0,0.1,1.0,1.1"

    def test_srun_backend_skips_gpu_env_vars(self):
        a = _make_single_node_assignment([0, 1, 2, 3])
        env = a.get_env_vars(mpi_backend="srun")
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert "ZE_AFFINITY_MASK" not in env


class TestSrunGpuBindMultiGPU:
    """The srun backend must use mask_gpu (not map_gpu) when any rank has >1 GPU."""

    def test_single_rank_multi_gpu_uses_mask_gpu(self):
        from parslbox.resource_manager.mpi_command_builder import MPICommandBuilder
        from parslbox.resource_manager.mpi_config import MPIConfig, MPIBackend
        from parslbox.system_configs.polaris import PolarisConfig

        a = _make_single_node_assignment([0, 1, 2, 3])
        builder = MPICommandBuilder(
            mpi_config=MPIConfig(backend=MPIBackend.SRUN),
            system_config=PolarisConfig(),
        )
        flags = builder._build_srun_gpu_bind(a)
        # All 4 GPUs in one rank → mask_gpu with hex 0xf
        assert flags == ["--gpu-bind=mask_gpu:0xf"]

    def test_one_gpu_per_rank_uses_map_gpu(self):
        from parslbox.resource_manager.mpi_command_builder import MPICommandBuilder
        from parslbox.resource_manager.mpi_config import MPIConfig, MPIBackend
        from parslbox.system_configs.polaris import PolarisConfig

        # Two ranks with one GPU each
        a = ResourceAssignment(
            job_id=2,
            node_ids=["node-0"],
            hostnames=["test-host"],
            node_occupancy=1.0,
        )
        a.assign_resources_to_rank(rank=0, node_idx=0, gpu_ids=[0], cpu_ids=[0, 1])
        a.assign_resources_to_rank(rank=1, node_idx=0, gpu_ids=[1], cpu_ids=[2, 3])

        builder = MPICommandBuilder(
            mpi_config=MPIConfig(backend=MPIBackend.SRUN),
            system_config=PolarisConfig(),
        )
        flags = builder._build_srun_gpu_bind(a)
        assert flags == ["--gpu-bind=map_gpu:0,1"]
