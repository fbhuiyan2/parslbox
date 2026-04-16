#!/usr/bin/env python3
"""
Comprehensive Test Suite for ParslBox Resource Manager

Tests all job types, resource allocation, tracking, cleanup, and MPI command generation.
Covers affinity scenarios and edge cases.

This test file uses the current MPI command builder (MPICommandBuilder / build_mpi_command)
and MPIConfig/MPIBackend for MPI command generation tests.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from typing import Dict, List

# Add parslbox to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parslbox.resource_manager import (
    ResourceManager,
    JobResourceSpec,
    ResourceAssignment,
    InsufficientResources,
    InvalidResourceSpec,
    create_job_resource_spec
)
from parslbox.resource_manager.cpu_affinity import CPUAffinityManager
from parslbox.resource_manager.mpi_command_builder import (
    MPICommandBuilder,
    build_mpi_command
)
from parslbox.resource_manager.mpi_config import MPIConfig, MPIBackend
from parslbox.resource_manager.helpers.mpi_launcher_helpers import (
    generate_openmpi_rankfile,
    generate_mpich_rankfile
)


class MockSystemConfig:
    """Mock system configuration for testing."""

    def __init__(self, cores_per_node=32, gpus_per_node=4, num_nodes=2,
                 worker_cpu_affinity=None, mpi_cmd="mpirun", exclude_cores=None,
                 gpu_type="nvidia"):
        self.CORES_PER_NODE = cores_per_node
        self.GPUS_PER_NODE = gpus_per_node
        self.SCHEDULER = "PBS"
        self.MPI_CMD_TO_USE = mpi_cmd
        self.WORKER_CPU_AFFINITY = worker_cpu_affinity
        self.EXCLUDE_CORES = exclude_cores
        self.GPU_TYPE = gpu_type
        self._num_nodes = num_nodes
        self._total_gpus = num_nodes * gpus_per_node

    def detect_resources(self):
        """Mock resource detection."""
        return self._num_nodes, self._total_gpus

    def _get_node_hostnames(self, num_nodes):
        """Mock hostname generation."""
        return [f"test-node-{i:02d}" for i in range(num_nodes)]


def _backend_for_mpi_cmd(mpi_cmd: str) -> MPIBackend:
    """Map MPI command string to MPIBackend enum."""
    mapping = {
        "mpirun": MPIBackend.OPENMPI,
        "mpiexec": MPIBackend.MPICH,
        "srun": MPIBackend.SRUN,
    }
    return mapping.get(mpi_cmd, MPIBackend.OPENMPI)


@pytest.fixture
def mock_config():
    """Standard mock configuration."""
    return MockSystemConfig()


@pytest.fixture
def mock_config_with_affinity():
    """Mock configuration with CPU affinity."""
    # 4 GPUs, each with 8 cores in affinity groups
    affinity = "list:0-7:8-15:16-23:24-31"
    return MockSystemConfig(worker_cpu_affinity=affinity)


@pytest.fixture
def mock_config_partial_affinity():
    """Mock configuration with partial CPU affinity (fewer cores per group)."""
    # 4 GPUs, but only 4 cores per group (insufficient for 8 cores per GPU)
    affinity = "list:0-3:8-11:16-19:24-27"
    return MockSystemConfig(worker_cpu_affinity=affinity)


@pytest.fixture
def resource_manager(mock_config):
    """Create resource manager with standard config."""
    return ResourceManager(mock_config, job_tracker=None)


@pytest.fixture
def resource_manager_with_affinity(mock_config_with_affinity):
    """Create resource manager with affinity config."""
    return ResourceManager(mock_config_with_affinity, job_tracker=None)


@pytest.fixture
def resource_manager_partial_affinity(mock_config_partial_affinity):
    """Create resource manager with partial affinity config."""
    return ResourceManager(mock_config_partial_affinity, job_tracker=None)


class TestJobClassification:
    """Test job type classification logic."""

    def test_subnode_cpu_job_classification(self, resource_manager):
        """Test sub-node CPU job classification."""
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}
        spec = create_job_resource_spec(job)

        assert resource_manager._is_subnode_job(spec) == True
        assert spec.is_gpu_job() == False
        assert spec.is_multinode_job() == False

    def test_subnode_gpu_job_classification(self, resource_manager):
        """Test sub-node GPU job classification."""
        job = {'job_id': 2, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}
        spec = create_job_resource_spec(job)

        assert resource_manager._is_subnode_job(spec) == True
        assert spec.is_gpu_job() == True
        assert spec.is_multinode_job() == False

    def test_fullnode_cpu_job_classification(self, resource_manager):
        """Test full-node CPU job classification."""
        job = {'job_id': 3, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        spec = create_job_resource_spec(job)

        assert resource_manager._is_subnode_job(spec) == False
        assert spec.is_gpu_job() == False
        assert spec.is_multinode_job() == False

    def test_fullnode_gpu_job_classification(self, resource_manager):
        """Test full-node GPU job classification."""
        job = {'job_id': 4, 'num_nodes': 1, 'ngpus': 4, 'node_occupancy': 1.0}
        spec = create_job_resource_spec(job)

        assert resource_manager._is_subnode_job(spec) == False
        assert spec.is_gpu_job() == True
        assert spec.is_multinode_job() == False

    def test_multinode_cpu_job_classification(self, resource_manager):
        """Test multi-node CPU job classification."""
        job = {'job_id': 5, 'num_nodes': 2, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        spec = create_job_resource_spec(job)

        assert resource_manager._is_subnode_job(spec) == False
        assert spec.is_gpu_job() == False
        assert spec.is_multinode_job() == True

    def test_multinode_gpu_job_classification(self, resource_manager):
        """Test multi-node GPU job classification."""
        job = {'job_id': 6, 'num_nodes': 2, 'ngpus': 8, 'node_occupancy': 1.0}
        spec = create_job_resource_spec(job)

        assert resource_manager._is_subnode_job(spec) == False
        assert spec.is_gpu_job() == True  # Multi-node GPU jobs are still GPU jobs
        assert spec.is_multinode_job() == True


class TestSubnodeJobs:
    """Test sub-node job allocation and tracking."""

    def test_subnode_cpu_job_allocation(self, resource_manager):
        """Test sub-node CPU job resource allocation."""
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}

        assignment = resource_manager.assign_resources(job)

        # Verify assignment structure
        assert assignment.job_id == 1
        assert len(assignment.node_ids) == 1
        assert len(assignment.hostnames) == 1
        assert assignment.node_occupancy == 0.5

        # Verify per-rank assignments
        assert len(assignment.get_all_ranks()) == 2

        # Check CPU core assignments
        rank0_cores = assignment.get_cpu_assignments_for_rank(0)
        rank1_cores = assignment.get_cpu_assignments_for_rank(1)

        assert len(rank0_cores) > 0
        assert len(rank1_cores) > 0
        assert len(rank0_cores) + len(rank1_cores) == 16  # 0.5 * 32 cores

        # Verify no GPU assignments
        assert assignment.get_gpu_assignments_for_rank(0) == []
        assert assignment.get_gpu_assignments_for_rank(1) == []

        # Verify node state
        node = resource_manager.nodes[0]
        assert 1 in node.assigned_jobs
        assert len(node.available_core_ids) == 16  # 32 - 16 used
        assert node.cpu_occupancy == 0.5

    def test_subnode_gpu_job_allocation(self, resource_manager):
        """Test sub-node GPU job resource allocation."""
        job = {'job_id': 2, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}

        assignment = resource_manager.assign_resources(job)

        # Verify assignment structure
        assert assignment.job_id == 2
        assert len(assignment.node_ids) == 1
        assert assignment.is_single_node() == True

        # Verify per-rank assignments (1 rank per GPU)
        assert len(assignment.get_all_ranks()) == 2

        # Check GPU assignments
        rank0_gpus = assignment.get_gpu_assignments_for_rank(0)
        rank1_gpus = assignment.get_gpu_assignments_for_rank(1)

        assert len(rank0_gpus) == 1
        assert len(rank1_gpus) == 1
        assert rank0_gpus[0] != rank1_gpus[0]  # Different GPUs
        assert set(rank0_gpus + rank1_gpus) == {0, 1}  # GPUs 0 and 1

        # Check CPU core assignments (8 cores per GPU)
        rank0_cores = assignment.get_cpu_assignments_for_rank(0)
        rank1_cores = assignment.get_cpu_assignments_for_rank(1)

        assert len(rank0_cores) == 8  # 32/4 = 8 cores per GPU
        assert len(rank1_cores) == 8
        assert set(rank0_cores).isdisjoint(set(rank1_cores))  # No overlap

        # Verify node state
        node = resource_manager.nodes[0]
        assert 2 in node.assigned_jobs
        assert len(node.available_gpu_ids) == 2  # 4 - 2 used
        assert len(node.available_core_ids) == 16  # 32 - 16 used

    def test_subnode_job_sharing(self, resource_manager):
        """Test that multiple sub-node jobs can share a node."""
        # First job: 0.25 occupancy
        job1 = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.25, 'ranks_per_node': 1}
        assignment1 = resource_manager.assign_resources(job1)

        # Second job: 0.5 occupancy
        job2 = {'job_id': 2, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}
        assignment2 = resource_manager.assign_resources(job2)

        # Both should be on the same node
        assert assignment1.hostnames[0] == assignment2.hostnames[0]

        # Verify node occupancy
        node = resource_manager.nodes[0]
        assert node.cpu_occupancy == 0.75  # 0.25 + 0.5
        assert len(node.assigned_jobs) == 2

        # Third job: 0.3 occupancy (should go to second node since first is at 0.75)
        job3 = {'job_id': 3, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.3, 'ranks_per_node': 1}
        assignment3 = resource_manager.assign_resources(job3)

        # Should be on a different node
        assert assignment3.hostnames[0] != assignment1.hostnames[0]

        # Fourth job: 0.8 occupancy (should fail - would exceed 1.0 on both nodes)
        job4 = {'job_id': 4, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.8, 'ranks_per_node': 1}

        with pytest.raises(InsufficientResources):
            resource_manager.assign_resources(job4)


class TestFullnodeJobs:
    """Test full-node job allocation and tracking."""

    def test_fullnode_cpu_job_allocation(self, resource_manager):
        """Test full-node CPU job resource allocation."""
        job = {'job_id': 3, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}

        assignment = resource_manager.assign_resources(job)

        # Verify assignment structure
        assert assignment.job_id == 3
        assert len(assignment.node_ids) == 1
        assert assignment.node_occupancy == 1.0

        # Verify ranks with per-rank CPU core distribution
        assert len(assignment.get_all_ranks()) == 4

        # Full-node CPU jobs distribute all cores across ranks
        all_cores = []
        for rank in range(4):
            rank_cores = assignment.get_cpu_assignments_for_rank(rank)
            assert len(rank_cores) == 8  # 32 cores / 4 ranks = 8 per rank
            all_cores.extend(rank_cores)
            assert assignment.get_gpu_assignments_for_rank(rank) == []

        assert len(set(all_cores)) == 32  # All cores assigned, no overlap

        # Verify node state
        node = resource_manager.nodes[0]
        assert 3 in node.assigned_jobs
        assert node.cpu_occupancy == 1.0
        assert node.is_completely_free() == False

    def test_fullnode_gpu_job_allocation(self, resource_manager):
        """Test full-node GPU job resource allocation."""
        job = {'job_id': 4, 'num_nodes': 1, 'ngpus': 4, 'node_occupancy': 1.0}

        assignment = resource_manager.assign_resources(job)

        # Verify assignment structure
        assert assignment.job_id == 4
        assert len(assignment.node_ids) == 1

        # Verify per-rank assignments (1 rank per GPU, even for full-node)
        assert len(assignment.get_all_ranks()) == 4

        # Check GPU assignments
        all_gpus = []
        for rank in range(4):
            rank_gpus = assignment.get_gpu_assignments_for_rank(rank)
            assert len(rank_gpus) == 1
            all_gpus.extend(rank_gpus)

        assert set(all_gpus) == {0, 1, 2, 3}  # All 4 GPUs assigned

        # Check CPU core assignments
        all_cores = []
        for rank in range(4):
            rank_cores = assignment.get_cpu_assignments_for_rank(rank)
            assert len(rank_cores) == 8  # 32/4 = 8 cores per GPU
            all_cores.extend(rank_cores)

        assert len(set(all_cores)) == 32  # All cores assigned, no overlap

        # Verify node state
        node = resource_manager.nodes[0]
        assert 4 in node.assigned_jobs
        assert len(node.available_gpu_ids) == 0  # All GPUs used
        assert len(node.available_core_ids) == 0  # All cores used
        assert node.cpu_occupancy == 1.0

    def test_multinode_cpu_job_allocation(self, resource_manager):
        """Test multi-node CPU job resource allocation."""
        job = {'job_id': 5, 'num_nodes': 2, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}

        assignment = resource_manager.assign_resources(job)

        # Verify assignment structure
        assert assignment.job_id == 5
        assert len(assignment.node_ids) == 2
        assert len(assignment.hostnames) == 2
        assert assignment.is_single_node() == False

        # Verify total ranks with per-rank CPU distribution
        assert len(assignment.get_all_ranks()) == 8  # 2 nodes * 4 ranks

        # Multi-node CPU jobs distribute all cores across ranks per node
        all_cores = []
        for rank in range(8):
            rank_cores = assignment.get_cpu_assignments_for_rank(rank)
            assert len(rank_cores) == 8  # 32 cores / 4 ranks per node = 8 per rank
            all_cores.extend(rank_cores)
            assert assignment.get_gpu_assignments_for_rank(rank) == []

        assert len(all_cores) == 64  # 32 cores * 2 nodes (core IDs repeat per node)

        # Verify node states
        for i in range(2):
            node = resource_manager.nodes[i]
            assert 5 in node.assigned_jobs
            assert node.cpu_occupancy == 1.0
            assert node.is_completely_free() == False

    def test_multinode_gpu_job_allocation(self, resource_manager):
        """Test multi-node GPU job resource allocation."""
        job = {'job_id': 6, 'num_nodes': 2, 'ngpus': 8, 'node_occupancy': 1.0}

        assignment = resource_manager.assign_resources(job)

        # Verify assignment structure
        assert assignment.job_id == 6
        assert len(assignment.node_ids) == 2

        # Verify total ranks (1 rank per GPU)
        assert len(assignment.get_all_ranks()) == 8  # 2 nodes * 4 GPUs

        # Check GPU assignments across nodes
        all_gpus = []
        for rank in range(8):
            rank_gpus = assignment.get_gpu_assignments_for_rank(rank)
            assert len(rank_gpus) == 1
            all_gpus.extend(rank_gpus)

        # Each node should have GPUs 0,1,2,3
        node0_ranks = assignment.get_ranks_for_node(0)
        node1_ranks = assignment.get_ranks_for_node(1)

        assert len(node0_ranks) == 4
        assert len(node1_ranks) == 4

        # Verify node states
        for i in range(2):
            node = resource_manager.nodes[i]
            assert 6 in node.assigned_jobs
            assert len(node.available_gpu_ids) == 0  # All GPUs used
            assert len(node.available_core_ids) == 0  # All cores used
            assert node.cpu_occupancy == 1.0


class TestAffinityScenarios:
    """Test different CPU affinity scenarios."""

    def test_gpu_job_with_full_affinity(self, resource_manager_with_affinity):
        """Test GPU job allocation with full affinity available."""
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}

        assignment = resource_manager_with_affinity.assign_resources(job)

        # Check that affinity cores are used
        rank0_cores = assignment.get_cpu_assignments_for_rank(0)
        rank1_cores = assignment.get_cpu_assignments_for_rank(1)

        # With affinity "list:0-7:8-15:16-23:24-31", GPU 0 should get cores 0-7, GPU 1 should get cores 8-15
        rank0_gpu = assignment.get_gpu_assignments_for_rank(0)[0]
        rank1_gpu = assignment.get_gpu_assignments_for_rank(1)[0]

        if rank0_gpu == 0:
            assert set(rank0_cores) == set(range(0, 8))
        elif rank0_gpu == 1:
            assert set(rank0_cores) == set(range(8, 16))

        if rank1_gpu == 0:
            assert set(rank1_cores) == set(range(0, 8))
        elif rank1_gpu == 1:
            assert set(rank1_cores) == set(range(8, 16))

    def test_gpu_job_with_partial_affinity(self, resource_manager_partial_affinity):
        """Test GPU job allocation with partial affinity (fallback to non-affinity cores)."""
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}

        assignment = resource_manager_partial_affinity.assign_resources(job)

        # Each rank should still get 8 cores, but may include non-affinity cores
        rank0_cores = assignment.get_cpu_assignments_for_rank(0)
        rank1_cores = assignment.get_cpu_assignments_for_rank(1)

        assert len(rank0_cores) == 8
        assert len(rank1_cores) == 8
        assert set(rank0_cores).isdisjoint(set(rank1_cores))  # No overlap

    def test_gpu_job_without_affinity(self, resource_manager):
        """Test GPU job allocation without affinity configuration."""
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}

        assignment = resource_manager.assign_resources(job)

        # Should still work, just without affinity optimization
        rank0_cores = assignment.get_cpu_assignments_for_rank(0)
        rank1_cores = assignment.get_cpu_assignments_for_rank(1)

        assert len(rank0_cores) == 8
        assert len(rank1_cores) == 8
        assert set(rank0_cores).isdisjoint(set(rank1_cores))


class TestResourceCleanup:
    """Test resource cleanup and tracking."""

    def test_subnode_job_cleanup(self, resource_manager):
        """Test cleanup of sub-node job resources."""
        # Allocate resources
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}
        assignment = resource_manager.assign_resources(job)

        # Verify initial state
        node = resource_manager.nodes[0]
        initial_available_cores = len(node.available_core_ids)
        assert node.cpu_occupancy == 0.5
        assert 1 in node.assigned_jobs

        # Free resources
        resource_manager.free_resources(1)

        # Verify cleanup
        assert node.cpu_occupancy == 0.0
        assert 1 not in node.assigned_jobs
        assert len(node.available_core_ids) == 32  # All cores available again
        assert 1 not in resource_manager.job_assignments

    def test_gpu_job_cleanup(self, resource_manager):
        """Test cleanup of GPU job resources."""
        # Allocate resources
        job = {'job_id': 2, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}
        assignment = resource_manager.assign_resources(job)

        # Verify initial state
        node = resource_manager.nodes[0]
        assert len(node.available_gpu_ids) == 2  # 4 - 2 used
        assert len(node.available_core_ids) == 16  # 32 - 16 used
        assert 2 in node.assigned_jobs

        # Free resources
        resource_manager.free_resources(2)

        # Verify cleanup
        assert len(node.available_gpu_ids) == 4  # All GPUs available
        assert len(node.available_core_ids) == 32  # All cores available
        assert 2 not in node.assigned_jobs
        assert node.cpu_occupancy == 0.0

    def test_fullnode_job_cleanup(self, resource_manager):
        """Test cleanup of full-node job resources."""
        # Allocate resources
        job = {'job_id': 3, 'num_nodes': 1, 'ngpus': 4, 'node_occupancy': 1.0}
        assignment = resource_manager.assign_resources(job)

        # Verify initial state
        node = resource_manager.nodes[0]
        assert len(node.available_gpu_ids) == 0
        assert len(node.available_core_ids) == 0
        assert node.cpu_occupancy == 1.0
        assert not node.is_completely_free()

        # Free resources
        resource_manager.free_resources(3)

        # Verify cleanup
        assert len(node.available_gpu_ids) == 4
        assert len(node.available_core_ids) == 32
        assert node.cpu_occupancy == 0.0
        assert node.is_completely_free()

    def test_multinode_job_cleanup(self, resource_manager):
        """Test cleanup of multi-node job resources."""
        # Allocate resources
        job = {'job_id': 4, 'num_nodes': 2, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = resource_manager.assign_resources(job)

        # Verify initial state
        for i in range(2):
            node = resource_manager.nodes[i]
            assert node.cpu_occupancy == 1.0
            assert not node.is_completely_free()
            assert 4 in node.assigned_jobs

        # Free resources
        resource_manager.free_resources(4)

        # Verify cleanup
        for i in range(2):
            node = resource_manager.nodes[i]
            assert node.cpu_occupancy == 0.0
            assert node.is_completely_free()
            assert 4 not in node.assigned_jobs


class TestMPICommandGeneration:
    """Test MPI command generation using the current MPICommandBuilder / build_mpi_command.

    Uses MPIConfig and MPIBackend for all MPI command generation tests.
    """

    def test_subnode_cpu_openmpi_command(self, resource_manager):
        """Test OpenMPI command generation for sub-node CPU job with rankfile binding."""
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        # Test job type detection
        job_type = spec.detect_job_type(resource_manager.system_config)
        assert job_type == "subnode_cpu"

        # Generate MPI commands using current builder
        mpi_config = MPIConfig(
            backend=MPIBackend.OPENMPI,
            use_hostlist=True,
            cpu_bind_method="rankfile",
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        # Should use rankfile for sub-node jobs
        mpirun_cmd = commands["PBX_MPI_PREFIX"]
        assert "PBX_MPIRUN_PREFIX" in commands
        assert commands["PBX_MPIRUN_PREFIX"] == mpirun_cmd
        assert "rankfile" in mpirun_cmd
        assert "-np 2" in mpirun_cmd
        assert "test-node-00" in mpirun_cmd

    def test_subnode_gpu_openmpi_command(self, resource_manager):
        """Test OpenMPI command generation for sub-node GPU job with rankfile and wrapper."""
        job = {'job_id': 2, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        # Test job type detection
        job_type = spec.detect_job_type(resource_manager.system_config)
        assert job_type == "subnode_gpu"

        # Generate MPI commands with GPU wrapper enabled
        mpi_config = MPIConfig(
            backend=MPIBackend.OPENMPI,
            use_hostlist=True,
            cpu_bind_method="rankfile",
            use_gpu_wrapper=True,
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        # Should use rankfile + wrapper for GPU jobs
        mpirun_cmd = commands["PBX_MPI_PREFIX"]
        assert "rankfile" in mpirun_cmd
        assert "-np 2" in mpirun_cmd
        assert "gpu_wrapper" in mpirun_cmd

    def test_fullnode_cpu_openmpi_depth_command(self, resource_manager):
        """Test OpenMPI command generation for full-node CPU job with depth binding."""
        job = {'job_id': 3, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        # Test job type detection
        job_type = spec.detect_job_type(resource_manager.system_config)
        assert job_type == "fullnode_cpu"

        # Generate MPI commands using depth binding (equivalent to PE binding in deprecated builder)
        mpi_config = MPIConfig(
            backend=MPIBackend.OPENMPI,
            use_hostlist=True,
            cpu_bind_method="depth",
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        # Should use PE binding for full-node CPU jobs
        mpirun_cmd = commands["PBX_MPIRUN_PREFIX"]
        assert "--map-by core:PE=" in mpirun_cmd
        assert "--bind-to core" in mpirun_cmd
        assert "-np 4" in mpirun_cmd

    def test_fullnode_gpu_openmpi_command(self, resource_manager):
        """Test OpenMPI command generation for full-node GPU job."""
        job = {'job_id': 4, 'num_nodes': 1, 'ngpus': 4, 'node_occupancy': 1.0}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        # Test job type detection
        job_type = spec.detect_job_type(resource_manager.system_config)
        assert job_type == "fullnode_gpu"

        # Generate MPI commands with rankfile binding and GPU wrapper
        mpi_config = MPIConfig(
            backend=MPIBackend.OPENMPI,
            use_hostlist=True,
            cpu_bind_method="rankfile",
            use_gpu_wrapper=True,
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        # Should use rankfile + wrapper for GPU jobs (even full-node)
        mpirun_cmd = commands["PBX_MPIRUN_PREFIX"]
        assert "rankfile" in mpirun_cmd
        assert "-np 4" in mpirun_cmd
        assert "gpu_wrapper" in mpirun_cmd

    def test_multinode_cpu_openmpi_depth_command(self, resource_manager):
        """Test OpenMPI command generation for multi-node CPU job with depth binding."""
        job = {'job_id': 5, 'num_nodes': 2, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        # Test job type detection
        job_type = spec.detect_job_type(resource_manager.system_config)
        assert job_type == "fullnode_cpu"

        # Generate MPI commands
        mpi_config = MPIConfig(
            backend=MPIBackend.OPENMPI,
            use_hostlist=True,
            cpu_bind_method="depth",
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        # Should use PE binding for multi-node CPU jobs
        mpirun_cmd = commands["PBX_MPIRUN_PREFIX"]
        assert "--map-by core:PE=" in mpirun_cmd
        assert "--bind-to core" in mpirun_cmd
        assert "-np 8" in mpirun_cmd
        assert "-H" in mpirun_cmd
        assert "test-node-00" in mpirun_cmd
        assert "test-node-01" in mpirun_cmd

    def test_multinode_gpu_openmpi_command(self, resource_manager):
        """Test OpenMPI command generation for multi-node GPU job."""
        job = {'job_id': 6, 'num_nodes': 2, 'ngpus': 8, 'node_occupancy': 1.0}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        # Test job type detection
        job_type = spec.detect_job_type(resource_manager.system_config)
        assert job_type == "fullnode_gpu"

        # Generate MPI commands
        mpi_config = MPIConfig(
            backend=MPIBackend.OPENMPI,
            use_hostlist=True,
            cpu_bind_method="rankfile",
            use_gpu_wrapper=True,
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        # Should use rankfile + wrapper for multi-node GPU jobs
        mpirun_cmd = commands["PBX_MPIRUN_PREFIX"]
        assert "rankfile" in mpirun_cmd
        assert "-np 8" in mpirun_cmd
        assert "gpu_wrapper" in mpirun_cmd

    def test_mpich_subnode_cpu_list_binding(self):
        """Test MPICH command generation for sub-node CPU job with list binding."""
        config = MockSystemConfig(mpi_cmd="mpiexec")
        rm = ResourceManager(config, job_tracker=None)

        # Subnode CPU job
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.MPICH,
            cpu_bind_method="list",
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        mpiexec_cmd = commands["PBX_MPIEXEC_PREFIX"]
        assert commands["PBX_MPI_PREFIX"] == mpiexec_cmd
        assert "--cpu-bind list:" in mpiexec_cmd
        assert "-n 2" in mpiexec_cmd

    def test_mpich_fullnode_cpu_depth_binding(self):
        """Test MPICH command generation for full-node CPU job with depth binding."""
        config = MockSystemConfig(mpi_cmd="mpiexec")
        rm = ResourceManager(config, job_tracker=None)

        # Full-node CPU job
        job = {'job_id': 2, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.MPICH,
            cpu_bind_method="depth",
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        mpiexec_cmd = commands["PBX_MPIEXEC_PREFIX"]
        assert "--ppn 4" in mpiexec_cmd
        assert "--depth 8" in mpiexec_cmd
        assert "--cpu-bind depth" in mpiexec_cmd

    def test_srun_minimal_command(self):
        """Test SRUN command generation with minimal configuration."""
        config = MockSystemConfig(mpi_cmd="srun")
        rm = ResourceManager(config, job_tracker=None)

        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(backend=MPIBackend.SRUN)
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        srun_cmd = commands["PBX_SRUN_PREFIX"]
        assert commands["PBX_MPI_PREFIX"] == srun_cmd
        assert "-n 4" in srun_cmd
        assert "--ntasks-per-node 4" in srun_cmd

    def test_srun_with_depth_binding(self):
        """Test SRUN command generation with depth binding."""
        config = MockSystemConfig(mpi_cmd="srun")
        rm = ResourceManager(config, job_tracker=None)

        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.SRUN,
            cpu_bind_method="depth",
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        srun_cmd = commands["PBX_SRUN_PREFIX"]
        assert "--cpus-per-task 8" in srun_cmd
        assert "--cpu-bind=cores" in srun_cmd

    def test_subnode_gpu_srun_depth_binding(self):
        """Test depth binding for sub-node GPU job uses per-GPU core share, not total cores.

        Regression test: a 1-GPU job on a 4-GPU/32-core node must get
        cores_per_rank = 32 // 4 = 8, not 32 // 1 = 32.
        """
        config = MockSystemConfig(mpi_cmd="srun")
        rm = ResourceManager(config, job_tracker=None)

        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 1, 'node_occupancy': 1.0}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.SRUN,
            cpu_bind_method="depth",
            use_gpu_wrapper=True,
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        srun_cmd = commands["PBX_SRUN_PREFIX"]
        # 32 cores / 4 GPUs = 8 cores per rank
        assert "--cpus-per-task 8" in srun_cmd
        assert "--cpu-bind=cores" in srun_cmd

    def test_subnode_gpu_mpich_depth_binding(self):
        """Test MPICH depth binding for sub-node GPU job uses per-GPU core share.

        Regression test: a 2-GPU job on a 4-GPU/32-core node must get
        cores_per_rank = 32 // 4 = 8, not 32 // 2 = 16.
        """
        config = MockSystemConfig(mpi_cmd="mpiexec")
        rm = ResourceManager(config, job_tracker=None)

        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.MPICH,
            cpu_bind_method="depth",
            use_gpu_wrapper=True,
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        mpiexec_cmd = commands["PBX_MPIEXEC_PREFIX"]
        # 32 cores / 4 GPUs = 8 cores per rank
        assert "--depth 8" in mpiexec_cmd
        assert "--cpu-bind depth" in mpiexec_cmd

    def test_subnode_gpu_openmpi_depth_binding(self, resource_manager):
        """Test OpenMPI depth binding for sub-node GPU job uses per-GPU core share.

        Regression test: a 1-GPU job on a 4-GPU/32-core node must get
        PE = 32 // 4 = 8, not 32 // 1 = 32.
        """
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 1, 'node_occupancy': 1.0}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.OPENMPI,
            cpu_bind_method="depth",
            use_gpu_wrapper=True,
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        mpirun_cmd = commands["PBX_MPI_PREFIX"]
        # 32 cores / 4 GPUs = 8 cores per rank
        assert "--map-by core:PE=8" in mpirun_cmd
        assert "--bind-to core" in mpirun_cmd

    def test_subnode_gpu_depth_binding_with_excluded_cores(self):
        """Test depth binding for sub-node GPU job accounts for excluded cores.

        On a 32-core/4-GPU node with cores [0,1,30,31] excluded,
        effective_cores = 28, so cores_per_rank = 28 // 4 = 7.
        """
        config = MockSystemConfig(
            cores_per_node=32, gpus_per_node=4, mpi_cmd="srun",
            exclude_cores=[0, 1, 30, 31],
        )
        rm = ResourceManager(config, job_tracker=None)

        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 1, 'node_occupancy': 1.0}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.SRUN,
            cpu_bind_method="depth",
            use_gpu_wrapper=True,
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        srun_cmd = commands["PBX_SRUN_PREFIX"]
        # (32 - 4 excluded) / 4 GPUs = 7 cores per rank
        assert "--cpus-per-task 7" in srun_cmd
        assert "--cpu-bind=cores" in srun_cmd

    def test_gpu_wrapper_disabled_via_config(self, resource_manager):
        """Test that gpu-wrapper can be disabled via MPIConfig.disable."""
        job = {'job_id': 10, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.OPENMPI,
            cpu_bind_method="rankfile",
            use_gpu_wrapper=True,
            disable=["gpu-wrapper"],
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        mpirun_cmd = commands["PBX_MPI_PREFIX"]
        assert "gpu_wrapper" not in mpirun_cmd

    def test_add_custom_flags(self, resource_manager):
        """Test adding custom MPI flags via MPIConfig.add."""
        job = {'job_id': 11, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.OPENMPI,
            add=["--mca btl ^openib", "--verbose"],
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        mpirun_cmd = commands["PBX_MPI_PREFIX"]
        assert "--mca" in mpirun_cmd
        assert "btl" in mpirun_cmd
        assert "^openib" in mpirun_cmd
        assert "--verbose" in mpirun_cmd

    def test_disable_and_add_combined(self):
        """Test both disable and add working together."""
        config = MockSystemConfig()
        rm = ResourceManager(config, job_tracker=None)

        job = {'job_id': 12, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.MPICH,
            use_hostlist=True,
            cpu_bind_method="rankfile",
            disable=["-hosts", "rankfile"],
            add=["--verbose"],
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        cmd = commands["PBX_MPI_PREFIX"]
        assert "-hosts" not in cmd
        assert "rankfile" not in cmd
        assert "--verbose" in cmd

    def test_add_with_template_substitution(self, resource_manager):
        """Test add rules with template substitution ({hostlist})."""
        job = {'job_id': 13, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.MPICH,
            add=["-hosts {hostlist}"],
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        cmd = commands["PBX_MPI_PREFIX"]
        assert "-hosts test-node-00" in cmd

    def test_disable_core_arg_openmpi_map_by(self, resource_manager):
        """Test that disable can remove --map-by from core mpi_args (OpenMPI).

        Regression test for the bug where disable only applied to mpi_extra,
        not core mpi_args. This is the OpenMPI 4.1.x vs 5.x workaround case:
        disable the 5.x --map-by syntax and add the 4.1.x --rankfile syntax.
        """
        job = {'job_id': 20, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.OPENMPI,
            cpu_bind_method="none",  # triggers --map-by ppr:N:node in core args
            disable=["--map-by"],
            add=["--rankfile /tmp/my_rankfile"],
        )
        commands = build_mpi_command(mpi_config, resource_manager.system_config, assignment, spec)

        cmd = commands["PBX_MPI_PREFIX"]
        assert "--map-by" not in cmd
        assert "--rankfile /tmp/my_rankfile" in cmd
        assert "-np 4" in cmd  # core arg that should remain

    def test_disable_core_arg_srun_ntasks_per_node(self):
        """Test that disable can remove --ntasks-per-node from core mpi_args (srun)."""
        config = MockSystemConfig(mpi_cmd="srun")
        rm = ResourceManager(config, job_tracker=None)

        job = {'job_id': 21, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.SRUN,
            disable=["--ntasks-per-node"],
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        cmd = commands["PBX_SRUN_PREFIX"]
        assert "--ntasks-per-node" not in cmd
        assert "-n 4" in cmd  # core arg that should remain

    def test_disable_core_arg_mpich_ppn(self):
        """Test that disable can remove --ppn from core mpi_args (MPICH)."""
        config = MockSystemConfig(mpi_cmd="mpiexec")
        rm = ResourceManager(config, job_tracker=None)

        job = {'job_id': 22, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.MPICH,
            disable=["--ppn"],
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        cmd = commands["PBX_MPIEXEC_PREFIX"]
        assert "--ppn" not in cmd
        assert "-n 4" in cmd  # core arg that should remain

    def test_add_flags_srun(self):
        """Test adding custom flags via add for srun backend."""
        config = MockSystemConfig(mpi_cmd="srun")
        rm = ResourceManager(config, job_tracker=None)

        job = {'job_id': 23, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.SRUN,
            add=["--exclusive", "--mem=64G"],
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        cmd = commands["PBX_SRUN_PREFIX"]
        assert "--exclusive" in cmd
        assert "--mem=64G" in cmd

    def test_disable_and_add_combined_srun(self):
        """Test disable + add working together for srun backend."""
        config = MockSystemConfig(mpi_cmd="srun")
        rm = ResourceManager(config, job_tracker=None)

        job = {'job_id': 24, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = rm.assign_resources(job)
        spec = create_job_resource_spec(job)

        mpi_config = MPIConfig(
            backend=MPIBackend.SRUN,
            disable=["--ntasks-per-node"],
            add=["--exclusive"],
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        cmd = commands["PBX_SRUN_PREFIX"]
        assert "--ntasks-per-node" not in cmd
        assert "--exclusive" in cmd
        assert "-n 4" in cmd


class TestRankfileGeneration:
    """Test rankfile generation for OpenMPI and MPICH."""

    def test_openmpi_rankfile_generation(self, resource_manager):
        """Test OpenMPI rankfile generation."""
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        rankfile_path = generate_openmpi_rankfile(assignment, resource_manager.system_config, spec)

        # Read and verify rankfile content
        with open(rankfile_path, 'r') as f:
            content = f.read()

        assert "rank 0=test-node-00 slot=" in content
        assert "rank 1=test-node-00 slot=" in content

        # Cleanup
        os.unlink(rankfile_path)

    def test_mpich_rankfile_generation(self, resource_manager):
        """Test MPICH rankfile generation."""
        job = {'job_id': 2, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}
        assignment = resource_manager.assign_resources(job)
        spec = create_job_resource_spec(job)

        rankfile_path = generate_mpich_rankfile(assignment, resource_manager.system_config, spec)

        # Read and verify rankfile content
        with open(rankfile_path, 'r') as f:
            content = f.read()

        lines = content.strip().split('\n')
        assert len(lines) == 2  # 2 ranks

        # Each line should have: rank node_idx cpu_cores (GPU assignment handled by wrapper)
        for i, line in enumerate(lines):
            parts = line.split()
            assert parts[0] == str(i)  # rank
            assert parts[1] == "0"     # node index
            assert len(parts) >= 3     # cpu cores
            # Note: GPU assignment is handled by wrapper script, not rankfile

        # Cleanup
        os.unlink(rankfile_path)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_insufficient_resources_cpu(self, resource_manager):
        """Test insufficient CPU resources error."""
        # Fill up the first node
        job1 = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        resource_manager.assign_resources(job1)

        # Fill up the second node
        job2 = {'job_id': 2, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        resource_manager.assign_resources(job2)

        # Try to allocate another job (should fail)
        job3 = {'job_id': 3, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.1, 'ranks_per_node': 1}

        with pytest.raises(InsufficientResources):
            resource_manager.assign_resources(job3)

    def test_insufficient_resources_gpu(self, resource_manager):
        """Test insufficient GPU resources error."""
        # Use all GPUs on first node
        job1 = {'job_id': 1, 'num_nodes': 1, 'ngpus': 4, 'node_occupancy': 1.0}
        resource_manager.assign_resources(job1)

        # Use all GPUs on second node
        job2 = {'job_id': 2, 'num_nodes': 1, 'ngpus': 4, 'node_occupancy': 1.0}
        resource_manager.assign_resources(job2)

        # Try to allocate another GPU job (should fail)
        job3 = {'job_id': 3, 'num_nodes': 1, 'ngpus': 1, 'node_occupancy': 1.0}

        with pytest.raises(InsufficientResources):
            resource_manager.assign_resources(job3)

    def test_insufficient_nodes_multinode(self, resource_manager):
        """Test insufficient nodes for multi-node job."""
        # Try to allocate a 3-node job (only 2 nodes available)
        job = {'job_id': 1, 'num_nodes': 3, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}

        with pytest.raises(InvalidResourceSpec):
            resource_manager.assign_resources(job)

    def test_invalid_resource_spec(self, resource_manager):
        """Test invalid resource specifications."""
        # Too many GPUs per node
        job1 = {'job_id': 1, 'num_nodes': 1, 'ngpus': 8, 'node_occupancy': 1.0}

        with pytest.raises(InvalidResourceSpec):
            resource_manager.assign_resources(job1)

    def test_job_not_found_cleanup(self, resource_manager):
        """Test cleanup of non-existent job."""
        from parslbox.resource_manager.exceptions import JobNotFound

        with pytest.raises(JobNotFound):
            resource_manager.free_resources(999)

    def test_duplicate_job_assignment(self, resource_manager):
        """Test assigning resources to same job twice."""
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}
        resource_manager.assign_resources(job)

        # Try to assign again
        with pytest.raises(ValueError, match="already has resources assigned"):
            resource_manager.assign_resources(job)


class TestExcludeCores:
    """Test EXCLUDE_CORES functionality."""

    def test_exclude_cores_basic(self):
        """Test basic exclude cores functionality."""
        # Create config with excluded cores
        config = MockSystemConfig(cores_per_node=32, exclude_cores=[0, 1, 30, 31])
        rm = ResourceManager(config, job_tracker=None)

        # Check that nodes have correct available cores
        node = rm.nodes[0]
        assert len(node.available_core_ids) == 28  # 32 - 4 excluded
        assert 0 not in node.available_core_ids
        assert 1 not in node.available_core_ids
        assert 30 not in node.available_core_ids
        assert 31 not in node.available_core_ids
        assert 2 in node.available_core_ids
        assert 29 in node.available_core_ids

    def test_exclude_cores_cpu_job_allocation(self):
        """Test CPU job allocation with excluded cores."""
        config = MockSystemConfig(cores_per_node=32, exclude_cores=[0, 1, 30, 31])
        rm = ResourceManager(config, job_tracker=None)

        # Allocate a sub-node CPU job
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}
        assignment = rm.assign_resources(job)

        # Should get 14 cores (0.5 * 28 available cores)
        rank0_cores = assignment.get_cpu_assignments_for_rank(0)
        rank1_cores = assignment.get_cpu_assignments_for_rank(1)
        total_assigned = len(rank0_cores) + len(rank1_cores)
        assert total_assigned == 14

        # Verify excluded cores are not assigned
        all_assigned_cores = rank0_cores + rank1_cores
        assert 0 not in all_assigned_cores
        assert 1 not in all_assigned_cores
        assert 30 not in all_assigned_cores
        assert 31 not in all_assigned_cores

        # Verify node state
        node = rm.nodes[0]
        assert len(node.available_core_ids) == 14  # 28 - 14 used

    def test_exclude_cores_gpu_job_allocation(self):
        """Test GPU job allocation with excluded cores."""
        config = MockSystemConfig(cores_per_node=32, gpus_per_node=4, exclude_cores=[0, 1, 30, 31])
        rm = ResourceManager(config, job_tracker=None)

        # Allocate a GPU job
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}
        assignment = rm.assign_resources(job)

        # Should get 7 cores per GPU (28 available / 4 GPUs = 7 cores per GPU)
        rank0_cores = assignment.get_cpu_assignments_for_rank(0)
        rank1_cores = assignment.get_cpu_assignments_for_rank(1)
        assert len(rank0_cores) == 7
        assert len(rank1_cores) == 7

        # Verify excluded cores are not assigned
        all_assigned_cores = rank0_cores + rank1_cores
        assert 0 not in all_assigned_cores
        assert 1 not in all_assigned_cores
        assert 30 not in all_assigned_cores
        assert 31 not in all_assigned_cores

    def test_exclude_cores_cleanup(self):
        """Test resource cleanup with excluded cores."""
        config = MockSystemConfig(cores_per_node=32, exclude_cores=[0, 1, 30, 31])
        rm = ResourceManager(config, job_tracker=None)

        # Allocate and then free a job
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}
        assignment = rm.assign_resources(job)

        # Free the job
        rm.free_resources(1)

        # Verify excluded cores are still excluded after cleanup
        node = rm.nodes[0]
        assert len(node.available_core_ids) == 28  # Back to 28 available
        assert 0 not in node.available_core_ids
        assert 1 not in node.available_core_ids
        assert 30 not in node.available_core_ids
        assert 31 not in node.available_core_ids

    def test_exclude_cores_multinode_cleanup(self):
        """Test multinode job cleanup with excluded cores."""
        config = MockSystemConfig(cores_per_node=32, exclude_cores=[0, 1, 30, 31])
        rm = ResourceManager(config, job_tracker=None)

        # Allocate a multinode job
        job = {'job_id': 1, 'num_nodes': 2, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 4}
        assignment = rm.assign_resources(job)

        # Free the job
        rm.free_resources(1)

        # Verify excluded cores are still excluded on both nodes
        for node in rm.nodes:
            assert len(node.available_core_ids) == 28
            assert 0 not in node.available_core_ids
            assert 1 not in node.available_core_ids
            assert 30 not in node.available_core_ids
            assert 31 not in node.available_core_ids

    def test_exclude_cores_invalid_cores(self):
        """Test handling of invalid excluded cores."""
        # Test with some invalid core IDs (outside valid range)
        config = MockSystemConfig(cores_per_node=32, exclude_cores=[0, 1, 35, 40, 30, 31])
        rm = ResourceManager(config, job_tracker=None)

        # Should only exclude valid cores (0, 1, 30, 31)
        node = rm.nodes[0]
        assert len(node.available_core_ids) == 28  # 32 - 4 valid excluded cores
        assert 0 not in node.available_core_ids
        assert 1 not in node.available_core_ids
        assert 30 not in node.available_core_ids
        assert 31 not in node.available_core_ids
        # Invalid cores (35, 40) should be ignored

    def test_exclude_cores_none(self):
        """Test that None exclude_cores works (no exclusion)."""
        config = MockSystemConfig(cores_per_node=32, exclude_cores=None)
        rm = ResourceManager(config, job_tracker=None)

        # Should have all cores available
        node = rm.nodes[0]
        assert len(node.available_core_ids) == 32
        assert set(node.available_core_ids) == set(range(32))

    def test_exclude_cores_empty_list(self):
        """Test that empty exclude_cores list works (no exclusion)."""
        config = MockSystemConfig(cores_per_node=32, exclude_cores=[])
        rm = ResourceManager(config, job_tracker=None)

        # Should have all cores available
        node = rm.nodes[0]
        assert len(node.available_core_ids) == 32
        assert set(node.available_core_ids) == set(range(32))

    def test_exclude_cores_with_affinity(self):
        """Test exclude cores with CPU affinity."""
        # Exclude cores 0,1 and use affinity that includes some excluded cores
        affinity = "list:0-7:8-15:16-23:24-31"  # GPU 0 gets cores 0-7 (includes excluded 0,1)
        config = MockSystemConfig(cores_per_node=32, gpus_per_node=4,
                                 worker_cpu_affinity=affinity, exclude_cores=[0, 1, 30, 31])
        rm = ResourceManager(config, job_tracker=None)

        # Allocate a GPU job
        job = {'job_id': 1, 'num_nodes': 1, 'ngpus': 1, 'node_occupancy': 1.0}
        assignment = rm.assign_resources(job)

        # Should get cores but excluded cores should not be assigned
        rank0_cores = assignment.get_cpu_assignments_for_rank(0)
        assert len(rank0_cores) == 7  # 28 available / 4 GPUs = 7 cores per GPU
        assert 0 not in rank0_cores
        assert 1 not in rank0_cores
        assert 30 not in rank0_cores
        assert 31 not in rank0_cores


class TestResourceStatus:
    """Test resource status and monitoring."""

    def test_initial_resource_status(self, resource_manager):
        """Test initial resource status."""
        status = resource_manager.get_resource_status()

        assert status['total_nodes'] == 2
        assert status['free_nodes'] == 2
        assert status['used_nodes'] == 0
        assert status['total_gpus'] == 8
        assert status['available_gpus'] == 8
        assert status['used_gpus'] == 0
        assert status['total_cpu_capacity'] == 2.0
        assert status['used_cpu_capacity'] == 0.0
        assert status['active_jobs'] == 0
        assert status['backlogged_jobs'] == 0

    def test_resource_status_with_jobs(self, resource_manager):
        """Test resource status with active jobs."""
        # Allocate some resources
        job1 = {'job_id': 1, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}  # Sub-node GPU job
        resource_manager.assign_resources(job1)

        job2 = {'job_id': 2, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.5, 'ranks_per_node': 2}  # Sub-node CPU job (shares node)
        resource_manager.assign_resources(job2)

        status = resource_manager.get_resource_status()

        assert status['used_nodes'] == 1  # Both jobs share the same node
        assert status['free_nodes'] == 1  # One node is still free
        assert status['used_gpus'] == 2   # Job1 uses 2 GPUs
        assert status['available_gpus'] == 6  # 8 total - 2 used = 6 available
        assert status['active_jobs'] == 2
        assert status['used_cpu_capacity'] > 0


def test_comprehensive_workflow():
    """Test a comprehensive workflow with multiple job types."""
    config = MockSystemConfig(cores_per_node=32, gpus_per_node=4, num_nodes=3)
    rm = ResourceManager(config, job_tracker=None)

    # 1. Sub-node CPU job
    job1 = {'job_id': 1, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 0.25, 'ranks_per_node': 2}
    assignment1 = rm.assign_resources(job1)
    assert assignment1.node_occupancy == 0.25

    # 2. Sub-node GPU job (shares node with job1)
    job2 = {'job_id': 2, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}
    assignment2 = rm.assign_resources(job2)
    assert assignment2.hostnames[0] == assignment1.hostnames[0]  # Same node

    # 3. Full-node CPU job
    job3 = {'job_id': 3, 'num_nodes': 1, 'ngpus': 0, 'node_occupancy': 1.0, 'ranks_per_node': 8}
    assignment3 = rm.assign_resources(job3)
    assert assignment3.hostnames[0] != assignment1.hostnames[0]  # Different node

    # 4. Multi-node GPU job
    job4 = {'job_id': 4, 'num_nodes': 2, 'ngpus': 8, 'node_occupancy': 1.0}

    # Should fail - not enough free nodes
    with pytest.raises(InsufficientResources):
        rm.assign_resources(job4)

    # 5. Free some resources
    rm.free_resources(3)  # Free the full-node job

    # 6. Now multi-node job should work
    assignment4 = rm.assign_resources(job4)
    assert len(assignment4.node_ids) == 2
    assert assignment4.get_total_gpus() == 8

    # 7. Verify final status
    status = rm.get_resource_status()
    assert status['active_jobs'] == 3  # jobs 1, 2, 4
    assert status['used_nodes'] == 3   # All nodes used


def test_mpi_command_generation_workflow():
    """Test MPI command generation across different backends in a realistic workflow."""
    config = MockSystemConfig(cores_per_node=32, gpus_per_node=4, num_nodes=2)
    rm = ResourceManager(config, job_tracker=None)

    # Allocate a GPU job
    job = {'job_id': 100, 'num_nodes': 1, 'ngpus': 2, 'node_occupancy': 1.0}
    assignment = rm.assign_resources(job)
    spec = create_job_resource_spec(job)

    # Test across all three backends
    backends = [
        (MPIBackend.OPENMPI, "PBX_MPIRUN_PREFIX"),
        (MPIBackend.MPICH, "PBX_MPIEXEC_PREFIX"),
        (MPIBackend.SRUN, "PBX_SRUN_PREFIX"),
    ]

    for backend, expected_key in backends:
        mpi_config = MPIConfig(
            backend=backend,
            use_gpu_wrapper=True,
            cpu_bind_method="rankfile" if backend != MPIBackend.SRUN else "list",
        )
        commands = build_mpi_command(mpi_config, config, assignment, spec)

        # Verify both PBX_MPI_PREFIX and backend-specific key exist
        assert "PBX_MPI_PREFIX" in commands
        assert expected_key in commands
        assert commands["PBX_MPI_PREFIX"] == commands[expected_key]

        # Verify command contains basic rank info
        cmd = commands["PBX_MPI_PREFIX"]
        assert "2" in cmd  # total ranks

        # Verify GPU wrapper is in the command
        assert "gpu_wrapper" in cmd


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
