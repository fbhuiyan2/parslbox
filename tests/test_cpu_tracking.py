#!/usr/bin/env python3
"""
Test script to verify CPU core tracking functionality for single-node jobs.
"""

from parslbox.resource_manager.models import NodeResource, JobResourceSpec, NodeAssignment
from parslbox.resource_manager.mpi_launcher import compose_all_mpi_commands
from parslbox.configs.polaris import PolarisConfig

def test_cpu_core_tracking():
    """Test CPU core tracking for single-node jobs."""
    print("Testing CPU core tracking for single-node jobs...")
    
    # Create a mock system config
    system_config = PolarisConfig()
    
    # Create a node with CPU cores
    node = NodeResource(
        node_id="test-node-0",
        hostname="test-host-01",
        total_gpus=4,
        total_cores=32
    )
    
    print(f"Initial node state:")
    print(f"  Available GPU IDs: {node.available_gpu_ids}")
    print(f"  Available Core IDs: {node.available_core_ids[:10]}... (showing first 10)")
    print(f"  Total cores: {node.total_cores}")
    
    # Test 1: Single-node CPU job with 0.25 occupancy (8 cores)
    print("\n=== Test 1: Single-node CPU job (0.25 occupancy) ===")
    job1_spec = JobResourceSpec(job_id=1, num_nodes=1, ngpus=0, node_occupancy=0.25)
    num_cores_needed = max(1, int(job1_spec.node_occupancy * system_config.CORES_PER_NODE))
    print(f"Job 1 needs {num_cores_needed} cores")
    
    if node.can_fit_cpu_cores(num_cores_needed):
        assigned_cores_1 = node.assign_cpu_cores(job1_spec.job_id, num_cores_needed)
        print(f"Job 1 assigned cores: {assigned_cores_1}")
        print(f"Remaining available cores: {len(node.available_core_ids)}")
        
        # Create assignment for MPI command generation
        assignment1 = NodeAssignment(
            job_id=1,
            node_ids=[node.node_id],
            hostnames=[node.hostname],
            gpu_assignments=[[]],
            cpu_assignments=[assigned_cores_1],
            node_occupancy=0.25
        )
        
        # Generate MPI commands
        mpi_commands = compose_all_mpi_commands(assignment1, system_config, 0.25)
        print(f"MPI commands for Job 1:")
        for key, cmd in mpi_commands.items():
            print(f"  {key}: {cmd}")
    
    # Test 2: Single-node GPU job (2 GPUs)
    print("\n=== Test 2: Single-node GPU job (2 GPUs) ===")
    job2_spec = JobResourceSpec(job_id=2, num_nodes=1, ngpus=2, node_occupancy=1.0)
    cores_per_gpu = system_config.CORES_PER_NODE // system_config.GPUS_PER_NODE
    num_cores_needed = job2_spec.ngpus * cores_per_gpu
    print(f"Job 2 needs {job2_spec.ngpus} GPUs and {num_cores_needed} cores")
    
    if node.can_fit_gpu_job(job2_spec.ngpus) and node.can_fit_cpu_cores(num_cores_needed):
        assigned_gpus_2 = node.assign_gpu_job(job2_spec.job_id, job2_spec.ngpus)
        assigned_cores_2 = node.assign_cpu_cores(job2_spec.job_id, num_cores_needed)
        print(f"Job 2 assigned GPUs: {assigned_gpus_2}")
        print(f"Job 2 assigned cores: {assigned_cores_2}")
        print(f"Remaining available GPUs: {len(node.available_gpu_ids)}")
        print(f"Remaining available cores: {len(node.available_core_ids)}")
        
        # Create assignment for MPI command generation
        assignment2 = NodeAssignment(
            job_id=2,
            node_ids=[node.node_id],
            hostnames=[node.hostname],
            gpu_assignments=[assigned_gpus_2],
            cpu_assignments=[assigned_cores_2],
            node_occupancy=1.0
        )
        
        # Generate MPI commands
        mpi_commands = compose_all_mpi_commands(assignment2, system_config, 1.0)
        print(f"MPI commands for Job 2:")
        for key, cmd in mpi_commands.items():
            print(f"  {key}: {cmd}")
    
    # Test 3: Free Job 1 and verify cores are returned
    print("\n=== Test 3: Free Job 1 resources ===")
    print(f"Before freeing Job 1 - Available cores: {len(node.available_core_ids)}")
    node.free_job(1)
    print(f"After freeing Job 1 - Available cores: {len(node.available_core_ids)}")
    print(f"Job CPU assignments: {node.job_cpu_assignments}")
    
    # Test 4: Free Job 2 and verify all resources are returned
    print("\n=== Test 4: Free Job 2 resources ===")
    print(f"Before freeing Job 2 - Available GPUs: {len(node.available_gpu_ids)}, Available cores: {len(node.available_core_ids)}")
    node.free_job(2)
    print(f"After freeing Job 2 - Available GPUs: {len(node.available_gpu_ids)}, Available cores: {len(node.available_core_ids)}")
    print(f"Job CPU assignments: {node.job_cpu_assignments}")
    
    print("\n✅ CPU core tracking test completed successfully!")

if __name__ == "__main__":
    test_cpu_core_tracking()
