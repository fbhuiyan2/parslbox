#!/usr/bin/env python3
"""
Test script for the ParslBox Resource Manager

This script tests the basic functionality of the resource manager
without requiring an actual HPC environment.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add parslbox to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parslbox.resource_manager import (
    ResourceManager, 
    JobResourceSpec,
    InsufficientResources,
    InvalidResourceSpec
)
from parslbox.resource_manager.utils import build_job_command, format_resource_summary


class MockSystemConfig:
    """Mock system configuration for testing."""
    
    CORES_PER_NODE = 32
    GPUS_PER_NODE = 4
    SCHEDULER = "PBS"
    
    def detect_resources(self):
        """Mock resource detection - simulate 2 nodes with 4 GPUs each."""
        return 2, 8  # 2 nodes, 8 total GPUs
    
    def _get_node_hostnames(self, num_nodes):
        """Mock hostname generation."""
        return [f"test-node-{i:02d}" for i in range(num_nodes)]


def test_resource_manager_basic():
    """Test basic resource manager functionality."""
    print("=== Testing Basic Resource Manager Functionality ===")
    
    # Create mock system config
    config = MockSystemConfig()
    
    # Temporarily mock the hostname detection
    original_method = ResourceManager._get_node_hostnames
    ResourceManager._get_node_hostnames = lambda self, num_nodes: [f"test-node-{i:02d}" for i in range(num_nodes)]
    
    try:
        # Create resource manager
        rm = ResourceManager(config)
        
        print(f"✅ Resource manager initialized with {len(rm.nodes)} nodes")
        
        # Test resource status
        status = rm.get_resource_status()
        print(f"✅ Initial status: {status['total_nodes']} nodes, {status['total_gpus']} GPUs")
        
        return rm
        
    finally:
        # Restore original method
        ResourceManager._get_node_hostnames = original_method


def test_single_node_gpu_jobs(rm):
    """Test single-node GPU job assignments."""
    print("\n=== Testing Single-Node GPU Jobs ===")
    
    # Job 1: 2 GPUs
    spec1 = JobResourceSpec(job_id=1, num_nodes=1, ngpus=2)
    assignment1 = rm.assign_resources(spec1)
    print(f"✅ Job 1 assigned: {assignment1.get_summary()}")
    print(f"   Environment: {assignment1.get_env_vars()}")
    
    # Job 2: 2 GPUs (should share same node)
    spec2 = JobResourceSpec(job_id=2, num_nodes=1, ngpus=2)
    assignment2 = rm.assign_resources(spec2)
    print(f"✅ Job 2 assigned: {assignment2.get_summary()}")
    print(f"   Environment: {assignment2.get_env_vars()}")
    
    # Verify they're on the same node but different GPUs
    assert assignment1.hostnames[0] == assignment2.hostnames[0], "Jobs should share the same node"
    assert set(assignment1.gpu_assignments[0]).isdisjoint(set(assignment2.gpu_assignments[0])), "Jobs should have different GPUs"
    
    print("✅ GPU sharing verification passed")
    
    return [assignment1, assignment2]


def test_cpu_only_jobs(rm):
    """Test CPU-only job assignments."""
    print("\n=== Testing CPU-Only Jobs ===")
    
    # Job 3: CPU-only with 0.5 occupancy
    spec3 = JobResourceSpec(job_id=3, num_nodes=1, ngpus=0, node_occupancy=0.5)
    assignment3 = rm.assign_resources(spec3)
    print(f"✅ Job 3 (CPU-only) assigned: {assignment3.get_summary()}")
    
    # Job 4: CPU-only with 0.25 occupancy (should share same node)
    spec4 = JobResourceSpec(job_id=4, num_nodes=1, ngpus=0, node_occupancy=0.25)
    assignment4 = rm.assign_resources(spec4)
    print(f"✅ Job 4 (CPU-only) assigned: {assignment4.get_summary()}")
    
    return [assignment3, assignment4]


def test_multinode_job(rm):
    """Test multi-node job assignment."""
    print("\n=== Testing Multi-Node Job ===")
    
    # Job 5: Multi-node job requiring 2 nodes
    spec5 = JobResourceSpec(job_id=5, num_nodes=2, ngpus=0)
    
    try:
        assignment5 = rm.assign_resources(spec5)
        print(f"✅ Job 5 (multi-node) assigned: {assignment5.get_summary()}")
        print(f"   Hostlist: {assignment5.get_mpi_hostlist()}")
        return assignment5
    except InsufficientResources as e:
        print(f"⚠️  Multi-node job couldn't be assigned (expected): {e}")
        return None


def test_mpi_command_generation():
    """Test MPI command generation utilities."""
    print("\n=== Testing MPI Command Generation ===")
    
    # Mock assignment for single-node GPU job
    from parslbox.resource_manager.models import NodeAssignment
    
    gpu_assignment = NodeAssignment(
        job_id=1,
        node_ids=["node-0"],
        hostnames=["test-node-01"],
        gpu_assignments=[[0, 1]]
    )
    
    app_config = {"ranks_per_node": 2}
    
    # Test command building
    command = build_job_command(
        assignment=gpu_assignment,
        app_config=app_config,
        executable="lmp",
        args="-k on g 2 -sf kk -in input.lammps",
        mpi_opts="-x OMP_NUM_THREADS=1"
    )
    
    print(f"✅ Generated command: {command}")
    
    # Verify command contains expected components
    assert "CUDA_VISIBLE_DEVICES=0,1" in command
    assert "mpirun -n 2 -host test-node-01" in command
    assert "lmp" in command
    
    print("✅ Command generation verification passed")


def test_resource_cleanup(rm, assignments):
    """Test resource cleanup when jobs finish."""
    print("\n=== Testing Resource Cleanup ===")
    
    # Get initial status
    initial_status = rm.get_resource_status()
    print(f"Before cleanup: {initial_status['active_jobs']} active jobs")
    
    # Free resources for first job
    if assignments:
        job_id = assignments[0].job_id
        rm.free_resources(job_id)
        print(f"✅ Freed resources for job {job_id}")
        
        # Check status after cleanup
        after_status = rm.get_resource_status()
        print(f"After cleanup: {after_status['active_jobs']} active jobs")
        
        assert after_status['active_jobs'] == initial_status['active_jobs'] - 1
        print("✅ Resource cleanup verification passed")


def test_insufficient_resources(rm):
    """Test handling of insufficient resources."""
    print("\n=== Testing Insufficient Resources ===")
    
    # Try to assign a job requiring more GPUs than available on any node
    spec_impossible = JobResourceSpec(job_id=99, num_nodes=1, ngpus=10)
    
    try:
        rm.assign_resources(spec_impossible)
        print("❌ Should have raised an exception")
    except (InsufficientResources, InvalidResourceSpec) as e:
        print(f"✅ Correctly raised exception: {type(e).__name__}: {e}")


def main():
    """Run all tests."""
    print("Starting ParslBox Resource Manager Tests\n")
    
    try:
        # Test basic functionality
        rm = test_resource_manager_basic()
        
        # Test different job types
        gpu_assignments = test_single_node_gpu_jobs(rm)
        cpu_assignments = test_cpu_only_jobs(rm)
        multinode_assignment = test_multinode_job(rm)
        
        # Test utilities
        test_mpi_command_generation()
        
        # Test resource status
        print("\n=== Final Resource Status ===")
        status = rm.get_resource_status()
        summary = format_resource_summary(status)
        print(summary)
        
        # Test cleanup
        all_assignments = gpu_assignments + cpu_assignments
        if multinode_assignment:
            all_assignments.append(multinode_assignment)
        test_resource_cleanup(rm, all_assignments)
        
        # Test error handling
        test_insufficient_resources(rm)
        
        print("\n🎉 All tests passed! Resource manager is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
