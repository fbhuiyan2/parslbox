"""
MPI Launcher Helper Functions

This module contains helper functions for MPI command generation including
wrapper script generation and rankfile creation. Supports both CUDA and Intel GPU systems.
"""

import logging
import os
import tempfile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from parslbox.resource_manager.models import ResourceAssignment, JobResourceSpec
    from parslbox.system_configs.base import SystemConfig

logger = logging.getLogger(__name__)


def write_gpu_wrapper(wrapper_content: str, assignment: 'ResourceAssignment', wrapper_type: str, job_path: str = None) -> str:
    """
    Write GPU wrapper script to job directory or temp location.
    
    Args:
        wrapper_content: The bash script content to write
        assignment: Resource assignment for the job
        wrapper_type: Type of wrapper ('openmpi' or 'mpiexec')
        job_path: Optional job directory path
        
    Returns:
        Path to the written wrapper script. When job_path is provided, returns relative path
        (filename only) to avoid OpenMPI buffer limits. When job_path is None, returns full path.
    """
    filename = f"gpu_wrapper_pbx_{wrapper_type}_{assignment.job_id}.sh"
    
    if job_path:
        # Write to job directory
        wrapper_path = os.path.join(job_path, filename)
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)
        logger.debug(f"Generated {wrapper_type} GPU wrapper: {wrapper_path}")
        return "./" + filename  # Return relative path when job_path provided
    else:
        # Fallback to temp location
        wrapper_path = os.path.join(tempfile.gettempdir(), filename)
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)
        logger.debug(f"Generated {wrapper_type} GPU wrapper: {wrapper_path}")
        return wrapper_path  # Return full path for temp location


def write_rankfile(rankfile_content: str, assignment: 'ResourceAssignment', launcher_type: str, job_path: str = None) -> str:
    """
    Write rankfile to job directory or temp location.
    
    Args:
        rankfile_content: The rankfile content to write
        assignment: Resource assignment for the job
        launcher_type: Type of launcher ('openmpi' or 'mpiexec')
        job_path: Optional job directory path
        
    Returns:
        Path to the written rankfile. When job_path is provided, returns relative path
        (filename only) to avoid OpenMPI buffer limits. When job_path is None, returns full path.
    """
    filename = f"rankfile_pbx_{launcher_type}_{assignment.job_id}.txt"
    
    if job_path:
        # Write to job directory
        rankfile_path = os.path.join(job_path, filename)
        with open(rankfile_path, 'w') as f:
            f.write(rankfile_content)
        logger.debug(f"Generated {launcher_type} rankfile: {rankfile_path}")
        return "./" + filename  # Return relative path when job_path provided
    else:
        # Fallback to temp location
        rankfile_path = os.path.join(tempfile.gettempdir(), filename)
        with open(rankfile_path, 'w') as f:
            f.write(rankfile_content)
        logger.debug(f"Generated {launcher_type} rankfile: {rankfile_path}")
        return rankfile_path  # Return full path for temp location


def generate_openmpi_rankfile(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> str:
    """
    Generate OpenMPI rankfile for CPU binding (GPU binding handled by wrapper).
    
    Format: rank <global_rank>=<hostname> slot=<cpu_cores>
    """
    rankfile_content = ""
    global_rank = 0
    
    for node_idx, hostname in enumerate(assignment.hostnames):
        node_ranks = assignment.get_ranks_for_node(node_idx)
        
        for local_rank in node_ranks:
            # Get actual CPU cores assigned to this rank by resource manager
            cpu_cores = assignment.get_cpu_assignments_for_rank(global_rank)
            
            if cpu_cores:
                cpu_cores_str = ",".join(map(str, cpu_cores))
                rankfile_content += f"rank {global_rank}={hostname} slot={cpu_cores_str}\n"
            else:
                logger.warning(f"No CPU assignment found for rank {global_rank} on node {hostname}")
                # Skip binding for this rank rather than using hardcoded fallback
            
            global_rank += 1
    
    return write_rankfile(rankfile_content, assignment, "openmpi", job_path)


def generate_mpiexec_rankfile(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> str:
    """
    Generate MPICH/PALS rankfile for CPU binding only.
    
    Format: <rank> <host_index> <cpu_cores> <optional gpus>
    
    Note: GPU binding via rankfile has issues with mpiexec, so GPUs are assigned 
    via wrapper scripts instead.
    """
    rankfile_content = ""
    global_rank = 0
    
    for node_idx, hostname in enumerate(assignment.hostnames):
        node_ranks = assignment.get_ranks_for_node(node_idx)
        
        for local_rank in node_ranks:
            # Get actual CPU cores assigned to this rank by resource manager
            cpu_cores = assignment.get_cpu_assignments_for_rank(global_rank)
            
            if cpu_cores:
                cpu_cores_str = ",".join(map(str, cpu_cores))
                # Only include CPU binding - GPU assignment handled by wrapper script
                line = f"{global_rank} {node_idx} {cpu_cores_str}"
                rankfile_content += line + "\n"
            else:
                logger.warning(f"No CPU assignment found for rank {global_rank} on node {hostname}")
                # Skip this rank rather than using hardcoded fallback
            
            global_rank += 1
    
    return write_rankfile(rankfile_content, assignment, "mpiexec", job_path)


def mpiexec_cuda_gpu_wrapper(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> str:
    """
    Generate GPU assignment wrapper script for CUDA systems using MPICH/PALS.
    
    Optimized for CUDA GPU systems like Polaris with NVIDIA GPUs.
    """
    # Standard rank detection for CUDA systems
    local_rank_chain = "${PMI_LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}}"
    global_rank_chain = "${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}}"
    
    system_name = getattr(system_config, 'SYSTEM_NAME', 'CUDA System')
    
    wrapper_content = f"""#!/bin/bash
# CUDA GPU assignment wrapper for MPICH/PALS job {assignment.job_id}
# Generated by ParslBox Resource Manager for {system_name}

# Environment variable priority for CUDA systems
LOCAL_RANK={local_rank_chain}
GLOBAL_RANK={global_rank_chain}

# GPU assignments from resource manager (per global rank):
"""
    
    # Build mapping for all ranks across all nodes
    for node_idx in range(len(assignment.hostnames)):
        node_ranks = assignment.get_ranks_for_node(node_idx)
        
        for local_rank_idx, global_rank in enumerate(node_ranks):
            gpu_ids = assignment.get_gpu_assignments_for_rank(global_rank)
            if gpu_ids:
                gpu_id = gpu_ids[0]  # Single GPU per rank
                
                wrapper_content += f"""if [ "$GLOBAL_RANK" -eq {global_rank} ]; then
    export CUDA_VISIBLE_DEVICES={gpu_id}
    # Set Intel GPU variables for potential compatibility
    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    export ZE_AFFINITY_MASK="{gpu_id}.0"
fi
"""
    
    wrapper_content += """
# Execute the application (CPU binding handled by mpiexec --cpu-bind)
exec "$@"
"""
    
    return write_gpu_wrapper(wrapper_content, assignment, "mpiexec", job_path)


def mpiexec_intel_gpu_wrapper(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> str:
    """
    Generate GPU assignment wrapper script for Intel GPU systems using MPICH/PALS.
    
    Supports Intel GPU systems with Aurora-specific optimizations when detected.
    Handles both tile and full GPU modes based on configuration.
    """
    system_name = getattr(system_config, 'SYSTEM_NAME', '').lower()
    
    # Rank detection: Aurora-optimized if Aurora system, otherwise standard
    if 'aurora' in system_name:
        # Aurora systems: prioritize Aurora-specific variables
        local_rank_chain = "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-${PMI_LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}}}}"
        global_rank_chain = "${MPI_RANK:-${PALS_RANKID:-${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}}}}"
        system_display_name = f"Aurora ({system_name})"
    else:
        # Other Intel GPU systems: use standard priority
        local_rank_chain = "${PMI_LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-${SLURM_LOCALID:-0}}}"
        global_rank_chain = "${PMI_RANK:-${OMPI_COMM_WORLD_RANK:-${SLURM_PROCID:-0}}}"
        system_display_name = f"Intel GPU System ({system_name})"
    
    wrapper_content = f"""#!/bin/bash
# Intel GPU assignment wrapper for MPICH/PALS job {assignment.job_id}
# Generated by ParslBox Resource Manager for {system_display_name}

# Environment variable priority optimized for Intel GPU systems
LOCAL_RANK={local_rank_chain}
GLOBAL_RANK={global_rank_chain}

# GPU assignments from resource manager (per global rank):
"""
    
    # Determine if this is tile mode based on config class name
    config_class_name = system_config.__class__.__name__.lower()
    is_tile_mode = 'tile' in config_class_name
    
    # Build mapping for all ranks across all nodes
    for node_idx in range(len(assignment.hostnames)):
        node_ranks = assignment.get_ranks_for_node(node_idx)
        
        for local_rank_idx, global_rank in enumerate(node_ranks):
            gpu_ids = assignment.get_gpu_assignments_for_rank(global_rank)
            if gpu_ids:
                gpu_id = gpu_ids[0]  # Single GPU per rank
                
                wrapper_content += f"""if [ "$GLOBAL_RANK" -eq {global_rank} ]; then
    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
"""
                
                # Set ZE_AFFINITY_MASK based on tile vs full GPU mode
                if is_tile_mode:
                    # Tile mode: Aurora's official script uses gpu_id.tile_id format
                    # For tile mode, each logical GPU ID from resource manager represents a tile
                    # Aurora has 6 physical GPUs with 2 tiles each = 12 tiles total
                    # Our resource manager assigns tile IDs 0-11, which map to:
                    # GPU 0: tiles 0,1 -> 0.0, 0.1
                    # GPU 1: tiles 2,3 -> 1.0, 1.1  
                    # GPU 2: tiles 4,5 -> 2.0, 2.1
                    # etc.
                    physical_gpu = gpu_id // 2  # Which physical GPU (0-5)
                    tile_id = gpu_id % 2        # Which tile on that GPU (0 or 1)
                    wrapper_content += f'    export ZE_AFFINITY_MASK="{physical_gpu}.{tile_id}"\n'
                else:
                    # Full GPU mode: use gpu_id format (no tile specification)
                    wrapper_content += f'    export ZE_AFFINITY_MASK="{gpu_id}"\n'
                
                # Aurora-specific optimizations
                if 'aurora' in system_name:
                    wrapper_content += f'    ulimit -c 0  # Aurora filesystem workaround\n'
                
                wrapper_content += "fi\n"
    
    wrapper_content += """
# Execute the application (CPU binding handled by mpiexec --cpu-bind)
exec "$@"
"""
    
    return write_gpu_wrapper(wrapper_content, assignment, "mpiexec", job_path)


def openmpi_cuda_gpu_wrapper(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> str:
    """
    Generate GPU assignment wrapper script for CUDA systems using OpenMPI.
    
    Optimized for CUDA GPU systems like Polaris with NVIDIA GPUs.
    CPU binding is handled by OpenMPI rankfile.
    """
    # Standard rank detection for CUDA systems
    local_rank_chain = "${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${SLURM_LOCALID:-0}}}"
    global_rank_chain = "${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-0}}}"
    
    system_name = getattr(system_config, 'SYSTEM_NAME', 'CUDA System')
    
    wrapper_content = f"""#!/bin/bash
# CUDA GPU assignment wrapper for OpenMPI job {assignment.job_id}
# Generated by ParslBox Resource Manager for {system_name}

LOCAL_RANK={local_rank_chain}
GLOBAL_RANK={global_rank_chain}

# GPU assignments from resource manager (per global rank):
"""
    
    # Build mapping for all ranks across all nodes
    for node_idx in range(len(assignment.hostnames)):
        node_ranks = assignment.get_ranks_for_node(node_idx)
        
        for local_rank_idx, global_rank in enumerate(node_ranks):
            gpu_ids = assignment.get_gpu_assignments_for_rank(global_rank)
            if gpu_ids:
                gpu_id = gpu_ids[0]  # Single GPU per rank
                
                wrapper_content += f"""if [ "$GLOBAL_RANK" -eq {global_rank} ]; then
    export CUDA_VISIBLE_DEVICES={gpu_id}
    # Set Intel GPU variables for potential compatibility
    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    export ZE_AFFINITY_MASK="{gpu_id}.0"
fi
"""
    
    wrapper_content += """
# Execute the application (CPU binding handled by OpenMPI rankfile)
exec "$@"
"""
    
    return write_gpu_wrapper(wrapper_content, assignment, "openmpi", job_path)


def openmpi_intel_gpu_wrapper(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> str:
    """
    Generate GPU assignment wrapper script for Intel GPU systems using OpenMPI.
    
    Supports Intel GPU systems with Aurora-specific optimizations when detected.
    Handles both tile and full GPU modes based on configuration.
    CPU binding is handled by OpenMPI rankfile.
    """
    system_name = getattr(system_config, 'SYSTEM_NAME', '').lower()
    
    # Rank detection: Aurora-optimized if Aurora system, otherwise standard
    if 'aurora' in system_name:
        # Aurora systems: prioritize Aurora-specific variables
        local_rank_chain = "${MPI_LOCALRANKID:-${PALS_LOCAL_RANKID:-${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${SLURM_LOCALID:-0}}}}}"
        global_rank_chain = "${MPI_RANK:-${PALS_RANKID:-${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-0}}}}}"
        system_display_name = f"Aurora ({system_name})"
    else:
        # Other Intel GPU systems: use standard priority with OpenMPI preference
        local_rank_chain = "${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${SLURM_LOCALID:-0}}}"
        global_rank_chain = "${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-${SLURM_PROCID:-0}}}"
        system_display_name = f"Intel GPU System ({system_name})"
    
    wrapper_content = f"""#!/bin/bash
# Intel GPU assignment wrapper for OpenMPI job {assignment.job_id}
# Generated by ParslBox Resource Manager for {system_display_name}

LOCAL_RANK={local_rank_chain}
GLOBAL_RANK={global_rank_chain}

# GPU assignments from resource manager (per global rank):
"""
    
    # Determine if this is tile mode based on config class name
    config_class_name = system_config.__class__.__name__.lower()
    is_tile_mode = 'tile' in config_class_name
    
    # Build mapping for all ranks across all nodes
    for node_idx in range(len(assignment.hostnames)):
        node_ranks = assignment.get_ranks_for_node(node_idx)
        
        for local_rank_idx, global_rank in enumerate(node_ranks):
            gpu_ids = assignment.get_gpu_assignments_for_rank(global_rank)
            if gpu_ids:
                gpu_id = gpu_ids[0]  # Single GPU per rank
                
                wrapper_content += f"""if [ "$GLOBAL_RANK" -eq {global_rank} ]; then
    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
"""
                
                # Set ZE_AFFINITY_MASK based on tile vs full GPU mode
                if is_tile_mode:
                    # Tile mode: Aurora's official script uses gpu_id.tile_id format
                    physical_gpu = gpu_id // 2  # Which physical GPU (0-5)
                    tile_id = gpu_id % 2        # Which tile on that GPU (0 or 1)
                    wrapper_content += f'    export ZE_AFFINITY_MASK="{physical_gpu}.{tile_id}"\n'
                else:
                    # Full GPU mode: use gpu_id format (no tile specification)
                    wrapper_content += f'    export ZE_AFFINITY_MASK="{gpu_id}"\n'
                
                # Aurora-specific optimizations
                if 'aurora' in system_name:
                    wrapper_content += f'    ulimit -c 0  # Aurora filesystem workaround\n'
                
                wrapper_content += "fi\n"
    
    wrapper_content += """
# Execute the application (CPU binding handled by OpenMPI rankfile)
exec "$@"
"""
    
    return write_gpu_wrapper(wrapper_content, assignment, "openmpi", job_path)


def generate_openmpi_gpu_wrapper(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> str:
    """
    Generate GPU assignment wrapper script for OpenMPI (CPU binding handled by rankfile).
    
    Dispatcher function that routes to appropriate GPU-specific implementation
    based on the system's GPU_TYPE configuration.
    
    Supports both CUDA and Intel GPU systems.
    """
    gpu_type = getattr(system_config, 'GPU_TYPE', 'cuda').lower()
    
    if gpu_type == 'intel':
        return openmpi_intel_gpu_wrapper(assignment, system_config, job_spec, job_path)
    else:
        # Default to CUDA for 'cuda' or any other/unknown GPU types
        return openmpi_cuda_gpu_wrapper(assignment, system_config, job_spec, job_path)


def generate_mpiexec_gpu_wrapper(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> str:
    """
    Generate GPU assignment wrapper script for MPICH/PALS (CPU binding handled by --cpu-bind).
    
    Dispatcher function that routes to appropriate GPU-specific implementation
    based on the system's GPU_TYPE configuration.
    
    Supports both CUDA and Intel GPU systems.
    """
    gpu_type = getattr(system_config, 'GPU_TYPE', 'cuda').lower()
    
    if gpu_type == 'intel':
        return mpiexec_intel_gpu_wrapper(assignment, system_config, job_spec, job_path)
    else:
        # Default to CUDA for 'cuda' or any other/unknown GPU types
        return mpiexec_cuda_gpu_wrapper(assignment, system_config, job_spec, job_path)
