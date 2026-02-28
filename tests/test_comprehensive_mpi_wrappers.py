"""
Comprehensive tests for MPI wrapper and rankfile generation.

Tests cover:
- Subnode, fullnode, and multinode jobs
- CUDA and Intel GPU systems
- Aurora tile and full GPU modes
- Rankfile generation for different scenarios
- Wrapper generation for different MPI implementations
"""

import pytest
import os
import tempfile
from unittest.mock import Mock
from parslbox.resource_manager.helpers.mpi_launcher_helpers import (
    generate_openmpi_rankfile,
    generate_mpiexec_rankfile,
    generate_openmpi_gpu_wrapper,
    generate_mpiexec_gpu_wrapper,
    mpiexec_cuda_gpu_wrapper,
    mpiexec_intel_gpu_wrapper,
    openmpi_cuda_gpu_wrapper,
    openmpi_intel_gpu_wrapper
)
from parslbox.system_configs.aurora_tile import AuroraTileConfig
from parslbox.system_configs.aurora_gpu import AuroraGpuConfig
from parslbox.system_configs.polaris import PolarisConfig
from parslbox.resource_manager.models import ResourceAssignment, JobResourceSpec


class TestRankfileGeneration:
    """Test rankfile generation for different job types."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.polaris_config = PolarisConfig()
        self.aurora_tile_config = AuroraTileConfig()
        self.aurora_gpu_config = AuroraGpuConfig()
    
    def create_assignment(self, job_type, num_nodes=1, ranks_per_node=2, gpus_per_rank=1):
        """Create a resource assignment for testing."""
        assignment = ResourceAssignment(
            job_id=12345,
            node_ids=[f"node-{i}" for i in range(num_nodes)],
            hostnames=[f"test-node-{i:02d}" for i in range(num_nodes)],
            node_occupancy=1.0 if job_type in ["fullnode", "multinode"] else 0.5
        )
        
        global_rank = 0
        for node_idx in range(num_nodes):
            for local_rank in range(ranks_per_node):
                cpu_cores = [global_rank * 4 + i for i in range(4)]  # 4 cores per rank
                gpu_ids = [global_rank % 4] if gpus_per_rank > 0 else []  # GPU assignment
                
                assignment.assign_resources_to_rank(
                    rank=global_rank,
                    node_idx=node_idx,
                    gpu_ids=gpu_ids,
                    cpu_ids=cpu_cores
                )
                global_rank += 1
        
        return assignment
    
    def test_openmpi_rankfile_subnode_cpu(self):
        """Test OpenMPI rankfile generation for subnode CPU job."""
        assignment = self.create_assignment("subnode", num_nodes=1, ranks_per_node=2, gpus_per_rank=0)
        job_spec = JobResourceSpec(job_id=12345, ngpus=0, num_nodes=1, ranks_per_node=2)
        
        rankfile_path = generate_openmpi_rankfile(assignment, self.polaris_config, job_spec)
        
        with open(rankfile_path, 'r') as f:
            content = f.read()
        
        # Check rankfile format: rank <global_rank>=<hostname> slot=<cpu_cores>
        assert "rank 0=test-node-00 slot=0,1,2,3" in content
        assert "rank 1=test-node-00 slot=4,5,6,7" in content
        
        os.unlink(rankfile_path)
    
    def test_openmpi_rankfile_multinode_gpu(self):
        """Test OpenMPI rankfile generation for multinode GPU job."""
        assignment = self.create_assignment("multinode", num_nodes=2, ranks_per_node=2, gpus_per_rank=1)
        job_spec = JobResourceSpec(job_id=12345, ngpus=4, num_nodes=2, ranks_per_node=2)
        
        rankfile_path = generate_openmpi_rankfile(assignment, self.polaris_config, job_spec)
        
        with open(rankfile_path, 'r') as f:
            content = f.read()
        
        # Check multinode rankfile
        assert "rank 0=test-node-00 slot=0,1,2,3" in content
        assert "rank 1=test-node-00 slot=4,5,6,7" in content
        assert "rank 2=test-node-01 slot=8,9,10,11" in content
        assert "rank 3=test-node-01 slot=12,13,14,15" in content
        
        os.unlink(rankfile_path)
    
    def test_mpiexec_rankfile_subnode_gpu(self):
        """Test MPICH rankfile generation for subnode GPU job."""
        assignment = self.create_assignment("subnode", num_nodes=1, ranks_per_node=2, gpus_per_rank=1)
        job_spec = JobResourceSpec(job_id=12345, ngpus=2, num_nodes=1, ranks_per_node=2)
        
        rankfile_path = generate_mpiexec_rankfile(assignment, self.aurora_tile_config, job_spec)
        
        with open(rankfile_path, 'r') as f:
            content = f.read()
        
        # Check MPICH rankfile format: <rank> <host_index> <cpu_cores>
        assert "0 0 0,1,2,3" in content
        assert "1 0 4,5,6,7" in content
        
        os.unlink(rankfile_path)
    
    def test_mpiexec_rankfile_multinode_cpu(self):
        """Test MPICH rankfile generation for multinode CPU job."""
        assignment = self.create_assignment("multinode", num_nodes=2, ranks_per_node=3, gpus_per_rank=0)
        job_spec = JobResourceSpec(job_id=12345, ngpus=0, num_nodes=2, ranks_per_node=3)
        
        rankfile_path = generate_mpiexec_rankfile(assignment, self.polaris_config, job_spec)
        
        with open(rankfile_path, 'r') as f:
            content = f.read()
        
        # Check multinode CPU rankfile
        lines = content.strip().split('\n')
        assert len(lines) == 6  # 2 nodes × 3 ranks per node
        assert "0 0 0,1,2,3" in content  # rank 0 on node 0
        assert "3 1 12,13,14,15" in content  # rank 3 on node 1
        
        os.unlink(rankfile_path)


class TestCUDAWrapperGeneration:
    """Test CUDA GPU wrapper generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.polaris_config = PolarisConfig()
    
    def create_gpu_assignment(self, num_nodes=1, gpus_per_node=2):
        """Create a GPU assignment for testing."""
        assignment = ResourceAssignment(
            job_id=54321,
            node_ids=[f"node-{i}" for i in range(num_nodes)],
            hostnames=[f"polaris-node-{i:02d}" for i in range(num_nodes)],
            node_occupancy=1.0
        )
        
        global_rank = 0
        for node_idx in range(num_nodes):
            for gpu_idx in range(gpus_per_node):
                cpu_cores = [global_rank * 8 + i for i in range(8)]  # 8 cores per GPU
                gpu_ids = [gpu_idx]  # GPU ID relative to node
                
                assignment.assign_resources_to_rank(
                    rank=global_rank,
                    node_idx=node_idx,
                    gpu_ids=gpu_ids,
                    cpu_ids=cpu_cores
                )
                global_rank += 1
        
        return assignment
    
    def test_mpiexec_cuda_wrapper_subnode(self):
        """Test CUDA wrapper generation for subnode job."""
        assignment = self.create_gpu_assignment(num_nodes=1, gpus_per_node=2)
        job_spec = JobResourceSpec(job_id=54321, ngpus=2, num_nodes=1)
        
        wrapper_path = mpiexec_cuda_gpu_wrapper(assignment, self.polaris_config, job_spec)
        
        with open(wrapper_path, 'r') as f:
            content = f.read()
        
        # Check CUDA-specific features
        assert "CUDA GPU assignment wrapper" in content
        assert "CUDA_VISIBLE_DEVICES=0" in content
        assert "CUDA_VISIBLE_DEVICES=1" in content
        assert "PMI_LOCAL_RANK" in content  # CUDA rank detection
        assert "ZE_ENABLE_PCI_ID_DEVICE_ORDER=1" in content  # Intel compatibility
        assert "polaris" in content.lower()  # System name
        
        os.unlink(wrapper_path)
    
    def test_openmpi_cuda_wrapper_multinode(self):
        """Test CUDA wrapper generation for multinode job."""
        assignment = self.create_gpu_assignment(num_nodes=2, gpus_per_node=2)
        job_spec = JobResourceSpec(job_id=54321, ngpus=4, num_nodes=2)
        
        wrapper_path = openmpi_cuda_gpu_wrapper(assignment, self.polaris_config, job_spec)
        
        with open(wrapper_path, 'r') as f:
            content = f.read()
        
        # Check OpenMPI CUDA wrapper
        assert "CUDA GPU assignment wrapper for OpenMPI" in content
        assert "OMPI_COMM_WORLD_LOCAL_RANK" in content  # OpenMPI rank detection
        assert "GLOBAL_RANK" in content
        assert "exec \"$@\"" in content
        
        # Check all ranks are covered
        for rank in range(4):
            assert f"GLOBAL_RANK\" -eq {rank}" in content
        
        os.unlink(wrapper_path)


class TestIntelGPUWrapperGeneration:
    """Test Intel GPU wrapper generation for Aurora systems."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.aurora_tile_config = AuroraTileConfig()
        self.aurora_gpu_config = AuroraGpuConfig()
    
    def create_aurora_assignment(self, config_type="tile", num_nodes=1):
        """Create Aurora assignment for testing."""
        if config_type == "tile":
            gpus_per_node = 12  # 12 tiles
        else:
            gpus_per_node = 6   # 6 full GPUs
        
        assignment = ResourceAssignment(
            job_id=67890,
            node_ids=[f"node-{i}" for i in range(num_nodes)],
            hostnames=[f"aurora-node-{i:02d}" for i in range(num_nodes)],
            node_occupancy=1.0
        )
        
        global_rank = 0
        for node_idx in range(num_nodes):
            for gpu_idx in range(gpus_per_node):
                cpu_cores = [global_rank * 16 + i for i in range(16)]  # 16 cores per GPU/tile
                gpu_ids = [gpu_idx]  # GPU/tile ID relative to node
                
                assignment.assign_resources_to_rank(
                    rank=global_rank,
                    node_idx=node_idx,
                    gpu_ids=gpu_ids,
                    cpu_ids=cpu_cores
                )
                global_rank += 1
        
        return assignment
    
    def test_mpiexec_intel_wrapper_aurora_tile(self):
        """Test Intel GPU wrapper for Aurora tile mode."""
        assignment = self.create_aurora_assignment("tile", num_nodes=1)
        job_spec = JobResourceSpec(job_id=67890, ngpus=12, num_nodes=1)
        
        wrapper_path = mpiexec_intel_gpu_wrapper(assignment, self.aurora_tile_config, job_spec)
        
        with open(wrapper_path, 'r') as f:
            content = f.read()
        
        # Check Intel GPU features
        assert "Intel GPU assignment wrapper" in content
        assert "Aurora (aurora-tile)" in content
        assert "ZE_ENABLE_PCI_ID_DEVICE_ORDER=1" in content
        assert "MPI_LOCALRANKID" in content  # Aurora rank detection
        assert "ulimit -c 0" in content  # Aurora filesystem workaround
        
        # Check tile affinity format
        assert 'ZE_AFFINITY_MASK="0.0"' in content  # Tile 0 -> GPU 0, tile 0
        assert 'ZE_AFFINITY_MASK="0.1"' in content  # Tile 1 -> GPU 0, tile 1
        assert 'ZE_AFFINITY_MASK="1.0"' in content  # Tile 2 -> GPU 1, tile 0
        assert 'ZE_AFFINITY_MASK="5.1"' in content  # Tile 11 -> GPU 5, tile 1
        
        os.unlink(wrapper_path)
    
    def test_openmpi_intel_wrapper_aurora_gpu(self):
        """Test Intel GPU wrapper for Aurora full GPU mode."""
        assignment = self.create_aurora_assignment("gpu", num_nodes=1)
        job_spec = JobResourceSpec(job_id=67890, ngpus=6, num_nodes=1)
        
        wrapper_path = openmpi_intel_gpu_wrapper(assignment, self.aurora_gpu_config, job_spec)
        
        with open(wrapper_path, 'r') as f:
            content = f.read()
        
        # Check full GPU mode
        assert "Intel GPU assignment wrapper for OpenMPI" in content
        assert "Aurora (aurora-gpu)" in content
        assert "MPI_LOCALRANKID" in content  # Aurora rank detection
        
        # Check full GPU affinity format (no tile specification)
        assert 'ZE_AFFINITY_MASK="0"' in content  # GPU 0
        assert 'ZE_AFFINITY_MASK="1"' in content  # GPU 1
        assert 'ZE_AFFINITY_MASK="5"' in content  # GPU 5
        
        # Should not have tile format
        assert 'ZE_AFFINITY_MASK="0.0"' not in content
        
        os.unlink(wrapper_path)
    
    def test_mpiexec_intel_wrapper_multinode_aurora(self):
        """Test Intel GPU wrapper for multinode Aurora job."""
        assignment = self.create_aurora_assignment("tile", num_nodes=2)
        job_spec = JobResourceSpec(job_id=67890, ngpus=24, num_nodes=2)  # 2 nodes × 12 tiles
        
        wrapper_path = mpiexec_intel_gpu_wrapper(assignment, self.aurora_tile_config, job_spec)
        
        with open(wrapper_path, 'r') as f:
            content = f.read()
        
        # Check multinode coverage
        assert "Intel GPU assignment wrapper" in content
        
        # Check that all 24 ranks are covered
        for rank in range(24):
            assert f"GLOBAL_RANK\" -eq {rank}" in content
        
        # Check tile affinity for different ranks
        assert 'ZE_AFFINITY_MASK="0.0"' in content  # First tile on each node
        
        os.unlink(wrapper_path)


class TestDispatcherFunctions:
    """Test dispatcher functions that route to appropriate implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.polaris_config = PolarisConfig()
        self.aurora_tile_config = AuroraTileConfig()
        self.aurora_gpu_config = AuroraGpuConfig()
    
    def create_test_assignment(self):
        """Create a simple test assignment."""
        assignment = ResourceAssignment(
            job_id=11111,
            node_ids=["node-0"],
            hostnames=["test-node"],
            node_occupancy=1.0
        )
        assignment.assign_resources_to_rank(
            rank=0, node_idx=0, gpu_ids=[0], cpu_ids=[0,1,2,3]
        )
        return assignment
    
    def test_mpiexec_dispatcher_cuda(self):
        """Test MPIEXEC dispatcher routes to CUDA wrapper."""
        assignment = self.create_test_assignment()
        job_spec = JobResourceSpec(job_id=11111, ngpus=1, num_nodes=1)
        
        wrapper_path = generate_mpiexec_gpu_wrapper(assignment, self.polaris_config, job_spec)
        
        with open(wrapper_path, 'r') as f:
            content = f.read()
        
        # Should route to CUDA wrapper
        assert "CUDA GPU assignment wrapper" in content
        assert "CUDA_VISIBLE_DEVICES" in content
        assert "polaris" in content.lower()
        
        os.unlink(wrapper_path)
    
    def test_mpiexec_dispatcher_intel(self):
        """Test MPIEXEC dispatcher routes to Intel wrapper."""
        assignment = self.create_test_assignment()
        job_spec = JobResourceSpec(job_id=11111, ngpus=1, num_nodes=1)
        
        wrapper_path = generate_mpiexec_gpu_wrapper(assignment, self.aurora_tile_config, job_spec)
        
        with open(wrapper_path, 'r') as f:
            content = f.read()
        
        # Should route to Intel wrapper
        assert "Intel GPU assignment wrapper" in content
        assert "ZE_AFFINITY_MASK" in content
        assert "aurora" in content.lower()
        
        os.unlink(wrapper_path)
    
    def test_openmpi_dispatcher_routing(self):
        """Test OpenMPI dispatcher routing for different GPU types."""
        assignment = self.create_test_assignment()
        job_spec = JobResourceSpec(job_id=11111, ngpus=1, num_nodes=1)
        
        # Test CUDA routing
        cuda_wrapper = generate_openmpi_gpu_wrapper(assignment, self.polaris_config, job_spec)
        assert os.path.exists(cuda_wrapper), f"CUDA wrapper not created at {cuda_wrapper}"
        
        with open(cuda_wrapper, 'r') as f:
            cuda_content = f.read()
        
        # Test Intel routing
        intel_wrapper = generate_openmpi_gpu_wrapper(assignment, self.aurora_gpu_config, job_spec)
        assert os.path.exists(intel_wrapper), f"Intel wrapper not created at {intel_wrapper}"
        
        with open(intel_wrapper, 'r') as f:
            intel_content = f.read()
        
        # Verify different wrappers generated
        assert "CUDA GPU assignment wrapper for OpenMPI" in cuda_content
        assert "Intel GPU assignment wrapper for OpenMPI" in intel_content
        assert "CUDA_VISIBLE_DEVICES" in cuda_content
        assert "ZE_AFFINITY_MASK" in intel_content
        
        # Clean up files - check existence first
        if os.path.exists(cuda_wrapper):
            os.unlink(cuda_wrapper)
        if os.path.exists(intel_wrapper):
            os.unlink(intel_wrapper)


class TestJobTypeScenarios:
    """Test comprehensive job type scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.polaris_config = PolarisConfig()
        self.aurora_tile_config = AuroraTileConfig()
    
    def test_subnode_cpu_job_complete_flow(self):
        """Test complete flow for subnode CPU job."""
        # Create subnode CPU assignment
        assignment = ResourceAssignment(
            job_id=22222,
            node_ids=["node-0"],
            hostnames=["test-node"],
            node_occupancy=0.5  # Subnode
        )
        
        # 2 ranks, no GPUs
        assignment.assign_resources_to_rank(rank=0, node_idx=0, gpu_ids=[], cpu_ids=[0,1,2,3])
        assignment.assign_resources_to_rank(rank=1, node_idx=0, gpu_ids=[], cpu_ids=[4,5,6,7])
        
        job_spec = JobResourceSpec(job_id=22222, ngpus=0, num_nodes=1, ranks_per_node=2)
        
        # Test rankfile generation
        rankfile_path = generate_openmpi_rankfile(assignment, self.polaris_config, job_spec)
        with open(rankfile_path, 'r') as f:
            rankfile_content = f.read()
        
        assert "rank 0=test-node slot=0,1,2,3" in rankfile_content
        assert "rank 1=test-node slot=4,5,6,7" in rankfile_content
        
        os.unlink(rankfile_path)
    
    def test_fullnode_gpu_job_complete_flow(self):
        """Test complete flow for fullnode GPU job."""
        # Create fullnode GPU assignment
        assignment = ResourceAssignment(
            job_id=33333,
            node_ids=["node-0"],
            hostnames=["aurora-node"],
            node_occupancy=1.0  # Fullnode
        )
        
        # 12 ranks (tiles), full node
        for rank in range(12):
            cpu_cores = [rank * 16 + i for i in range(16)]
            assignment.assign_resources_to_rank(
                rank=rank, node_idx=0, gpu_ids=[rank], cpu_ids=cpu_cores
            )
        
        job_spec = JobResourceSpec(job_id=33333, ngpus=12, num_nodes=1)
        
        # Test wrapper generation
        wrapper_path = generate_mpiexec_gpu_wrapper(assignment, self.aurora_tile_config, job_spec)
        with open(wrapper_path, 'r') as f:
            wrapper_content = f.read()
        
        # Check all tiles covered
        for rank in range(12):
            assert f"GLOBAL_RANK\" -eq {rank}" in wrapper_content
        
        # Check tile affinity calculations
        assert 'ZE_AFFINITY_MASK="0.0"' in wrapper_content  # Tile 0
        assert 'ZE_AFFINITY_MASK="5.1"' in wrapper_content  # Tile 11
        
        os.unlink(wrapper_path)
    
    def test_multinode_mixed_job_complete_flow(self):
        """Test complete flow for multinode mixed job."""
        # Create multinode assignment
        assignment = ResourceAssignment(
            job_id=44444,
            node_ids=["node-0", "node-1"],
            hostnames=["polaris-node-00", "polaris-node-01"],
            node_occupancy=1.0
        )
        
        # 2 nodes × 4 GPUs = 8 total ranks
        global_rank = 0
        for node_idx in range(2):
            for gpu_idx in range(4):
                cpu_cores = [global_rank * 8 + i for i in range(8)]
                assignment.assign_resources_to_rank(
                    rank=global_rank, node_idx=node_idx, gpu_ids=[gpu_idx], cpu_ids=cpu_cores
                )
                global_rank += 1
        
        job_spec = JobResourceSpec(job_id=44444, ngpus=8, num_nodes=2)
        
        # Test rankfile generation
        rankfile_path = generate_mpiexec_rankfile(assignment, self.polaris_config, job_spec)
        with open(rankfile_path, 'r') as f:
            rankfile_content = f.read()
        
        # Check multinode rankfile
        assert "0 0 0,1,2,3,4,5,6,7" in rankfile_content  # Rank 0 on node 0
        assert "4 1 32,33,34,35,36,37,38,39" in rankfile_content  # Rank 4 on node 1
        
        # Test wrapper generation
        wrapper_path = generate_mpiexec_gpu_wrapper(assignment, self.polaris_config, job_spec)
        with open(wrapper_path, 'r') as f:
            wrapper_content = f.read()
        
        # Check all ranks covered
        for rank in range(8):
            assert f"GLOBAL_RANK\" -eq {rank}" in wrapper_content
        
        os.unlink(rankfile_path)
        os.unlink(wrapper_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
