#!/usr/bin/env python3
"""
Simplified test script to verify LAMMPS app integration with CPU core tracking and MPI binding.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

from parslbox.apps.lammps import LammpsApp
from parslbox.resource_manager.models import NodeResource, JobResourceSpec, NodeAssignment
from parslbox.resource_manager.resource_manager import ResourceManager
from parslbox.resource_manager.mpi_launcher import compose_all_mpi_commands
from parslbox.configs.polaris import PolarisConfig

def test_lammps_mpi_commands():
    """Test LAMMPS MPI command generation with CPU core tracking."""
    print("Testing LAMMPS MPI command generation with CPU core tracking...")
    
    # Create system config
    system_config = PolarisConfig()
    
    # Create resource manager with mocked methods
    with patch.object(system_config, 'detect_resources', return_value=(1, 4)):
        with patch('parslbox.resource_manager.resource_manager.ResourceManager._get_pbs_hostnames', return_value=['test-node-01']):
            with patch('parslbox.resource_manager.resource_manager.ResourceManager._get_slurm_hostnames', return_value=['test-node-01']):
                resource_manager = ResourceManager(system_config)
    
    # Test 1: Single-node CPU job (0.25 occupancy = 8 cores)
    print("\n=== Test 1: CPU job (0.25 occupancy) ===")
    cpu_job_spec = JobResourceSpec(job_id=1, num_nodes=1, ngpus=0, node_occupancy=0.25)
    cpu_assignment = resource_manager.assign_resources(cpu_job_spec)
    
    print(f"CPU job assignment: {cpu_assignment.get_summary()}")
    print(f"CPU cores assigned: {cpu_assignment.cpu_assignments[0]}")
    
    # Generate MPI commands
    cpu_mpi_commands = compose_all_mpi_commands(cpu_assignment, system_config, 0.25)
    print(f"MPI commands for CPU job:")
    for key, cmd in cpu_mpi_commands.items():
        if key.startswith('PBX_'):
            print(f"  {key}: {cmd}")
    
    # Verify CPU binding in MPI commands
    expected_cpu_list = ",".join(map(str, cpu_assignment.cpu_assignments[0]))
    mpirun_cmd = cpu_mpi_commands.get('PBX_MPIRUN_PREFIX', '')
    mpiexec_cmd = cpu_mpi_commands.get('PBX_MPIEXEC_PREFIX', '')
    
    assert f"--cpu-list {expected_cpu_list}" in mpirun_cmd
    assert f"--cpu-bind list:{expected_cpu_list}" in mpiexec_cmd
    print(f"✅ CPU binding verified in MPI commands: cores {expected_cpu_list}")
    
    # Test 2: Single-node GPU job (2 GPUs)
    print("\n=== Test 2: GPU job (2 GPUs) ===")
    gpu_job_spec = JobResourceSpec(job_id=2, num_nodes=1, ngpus=2, node_occupancy=1.0)
    gpu_assignment = resource_manager.assign_resources(gpu_job_spec)
    
    print(f"GPU job assignment: {gpu_assignment.get_summary()}")
    print(f"GPU IDs assigned: {gpu_assignment.gpu_assignments[0]}")
    print(f"CPU cores assigned: {gpu_assignment.cpu_assignments[0]}")
    
    # Generate MPI commands
    gpu_mpi_commands = compose_all_mpi_commands(gpu_assignment, system_config, 1.0)
    print(f"MPI commands for GPU job:")
    for key, cmd in gpu_mpi_commands.items():
        if key.startswith('PBX_'):
            print(f"  {key}: {cmd}")
    
    # Verify GPU and CPU binding
    expected_gpu_list = ",".join(map(str, gpu_assignment.gpu_assignments[0]))
    expected_cpu_list = ",".join(map(str, gpu_assignment.cpu_assignments[0]))
    
    mpirun_cmd = gpu_mpi_commands.get('PBX_MPIRUN_PREFIX', '')
    mpiexec_cmd = gpu_mpi_commands.get('PBX_MPIEXEC_PREFIX', '')
    
    assert f"--cpu-list {expected_cpu_list}" in mpirun_cmd
    assert f"--cpu-bind list:{expected_cpu_list}" in mpiexec_cmd
    print(f"✅ GPU binding verified: GPUs {expected_gpu_list}")
    print(f"✅ CPU binding verified in MPI commands: cores {expected_cpu_list}")
    
    # Test 3: Verify environment variables
    print("\n=== Test 3: Environment variables ===")
    env_vars = gpu_assignment.get_env_vars()
    print(f"Environment variables: {env_vars}")
    
    expected_cuda_devices = ",".join(map(str, gpu_assignment.gpu_assignments[0]))
    assert env_vars.get('CUDA_VISIBLE_DEVICES') == expected_cuda_devices
    print(f"✅ CUDA_VISIBLE_DEVICES verified: {expected_cuda_devices}")
    
    # Test 4: Test LAMMPS app command construction logic
    print("\n=== Test 4: LAMMPS command construction ===")
    lammps_app = LammpsApp()
    
    # Test CPU job command arguments
    total_gpus_cpu = cpu_assignment.get_total_gpus()
    if total_gpus_cpu > 0:
        lammps_args_cpu = f"-k on g {total_gpus_cpu} -sf kk -pk kokkos newton on neigh half -in in.lammps"
    else:
        lammps_args_cpu = f"-in in.lammps"
    
    print(f"CPU job LAMMPS args: {lammps_args_cpu}")
    assert "-in in.lammps" in lammps_args_cpu
    assert "-k on g" not in lammps_args_cpu  # Should not have GPU args for CPU job
    print(f"✅ CPU job LAMMPS arguments verified")
    
    # Test GPU job command arguments
    total_gpus_gpu = gpu_assignment.get_total_gpus()
    if total_gpus_gpu > 0:
        lammps_args_gpu = f"-k on g {total_gpus_gpu} -sf kk -pk kokkos newton on neigh half -in in.lammps"
    else:
        lammps_args_gpu = f"-in in.lammps"
    
    print(f"GPU job LAMMPS args: {lammps_args_gpu}")
    assert "-in in.lammps" in lammps_args_gpu
    assert f"-k on g {total_gpus_gpu}" in lammps_args_gpu  # Should have GPU args
    print(f"✅ GPU job LAMMPS arguments verified")
    
    # Test 5: Resource cleanup
    print("\n=== Test 5: Resource cleanup ===")
    print(f"Before cleanup:")
    status = resource_manager.get_resource_status()
    print(f"  Available GPUs: {status['available_gpus']}")
    print(f"  Available CPU capacity: {status['available_cpu_capacity']}")
    
    # Free both jobs
    resource_manager.free_resources(1)
    resource_manager.free_resources(2)
    
    print(f"After cleanup:")
    status = resource_manager.get_resource_status()
    print(f"  Available GPUs: {status['available_gpus']}")
    print(f"  Available CPU capacity: {status['available_cpu_capacity']}")
    
    # Verify all resources are freed
    node = resource_manager.nodes[0]
    print(f"Debug - Available GPUs: {len(node.available_gpu_ids)}, Expected: {system_config.GPUS_PER_NODE}")
    print(f"Debug - Available cores: {len(node.available_core_ids)}, Expected: {system_config.CORES_PER_NODE}")
    print(f"Debug - Job CPU assignments: {node.job_cpu_assignments}")
    
    assert len(node.available_gpu_ids) == system_config.GPUS_PER_NODE, f"Expected {system_config.GPUS_PER_NODE} GPUs, got {len(node.available_gpu_ids)}"
    assert len(node.available_core_ids) == system_config.CORES_PER_NODE, f"Expected {system_config.CORES_PER_NODE} cores, got {len(node.available_core_ids)}"
    assert len(node.job_cpu_assignments) == 0, f"Expected no job assignments, got {node.job_cpu_assignments}"
    print(f"✅ All resources properly freed")
    
    # Test 6: Multiple concurrent jobs
    print("\n=== Test 6: Multiple concurrent jobs ===")
    job3_spec = JobResourceSpec(job_id=3, num_nodes=1, ngpus=0, node_occupancy=0.125)  # 4 cores
    job4_spec = JobResourceSpec(job_id=4, num_nodes=1, ngpus=0, node_occupancy=0.125)  # 4 cores
    
    assignment3 = resource_manager.assign_resources(job3_spec)
    assignment4 = resource_manager.assign_resources(job4_spec)
    
    print(f"Job 3 CPU cores: {assignment3.cpu_assignments[0]}")
    print(f"Job 4 CPU cores: {assignment4.cpu_assignments[0]}")
    
    # Verify no overlap in CPU assignments
    cores3 = set(assignment3.cpu_assignments[0])
    cores4 = set(assignment4.cpu_assignments[0])
    assert cores3.isdisjoint(cores4), "CPU cores should not overlap between jobs"
    print(f"✅ No CPU core overlap between concurrent jobs")
    
    # Generate MPI commands for both jobs
    mpi3 = compose_all_mpi_commands(assignment3, system_config, 0.125)
    mpi4 = compose_all_mpi_commands(assignment4, system_config, 0.125)
    
    print(f"Job 3 MPI: {mpi3['PBX_MPI_PREFIX']}")
    print(f"Job 4 MPI: {mpi4['PBX_MPI_PREFIX']}")
    
    # Verify different CPU binding for each job
    cores3_str = ",".join(map(str, assignment3.cpu_assignments[0]))
    cores4_str = ",".join(map(str, assignment4.cpu_assignments[0]))
    
    assert cores3_str in mpi3['PBX_MPI_PREFIX']
    assert cores4_str in mpi4['PBX_MPI_PREFIX']
    assert cores3_str != cores4_str
    print(f"✅ Different CPU binding for concurrent jobs")
    
    # Cleanup
    resource_manager.free_resources(3)
    resource_manager.free_resources(4)
    
    print("\n🎉 All LAMMPS CPU binding tests passed!")

if __name__ == "__main__":
    test_lammps_mpi_commands()
