"""
MPI Command Launcher for ParslBox Resource Manager

This module provides MPI command generation functionality similar to Parsl's
launcher system, but adapted for ParslBox's resource management system.
"""

import logging
import os
import tempfile
from typing import Dict, Tuple, TYPE_CHECKING


if TYPE_CHECKING:
    from parslbox.resource_manager.models import ResourceAssignment, JobResourceSpec
    from parslbox.configs.base import SystemConfig

logger = logging.getLogger(__name__)

VALID_LAUNCHERS = ('mpirun', 'mpiexec', 'srun')


def write_gpu_wrapper(wrapper_content: str, assignment: 'ResourceAssignment', wrapper_type: str, job_path: str = None) -> str:
    """
    Write GPU wrapper script to job directory or temp location.
    
    Args:
        wrapper_content: The bash script content to write
        assignment: Resource assignment for the job
        wrapper_type: Type of wrapper ('openmpi' or 'mpiexec')
        job_path: Optional job directory path
        
    Returns:
        Path to the written wrapper script
    """
    if job_path:
        # Write to job directory
        wrapper_path = os.path.join(job_path, f"parslbox_{wrapper_type}_gpu_wrapper_{assignment.job_id}.sh")
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)
    else:
        # Fallback to temp location
        fd, wrapper_path = tempfile.mkstemp(prefix=f"parslbox_{wrapper_type}_gpu_wrapper_{assignment.job_id}_", suffix=".sh")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(wrapper_content)
            os.chmod(wrapper_path, 0o755)
        except:
            os.close(fd)
            raise
    
    logger.debug(f"Generated {wrapper_type} GPU wrapper: {wrapper_path}")
    return wrapper_path


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
    
    # Write rankfile to job directory or temp location
    if job_path:
        rankfile_path = os.path.join(job_path, f"parslbox_openmpi_rankfile_{assignment.job_id}.txt")
        with open(rankfile_path, 'w') as f:
            f.write(rankfile_content)
    else:
        # Fallback to temp location
        fd, rankfile_path = tempfile.mkstemp(prefix=f"parslbox_openmpi_rankfile_{assignment.job_id}_", suffix=".txt")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(rankfile_content)
        except:
            os.close(fd)
            raise
    
    logger.debug(f"Generated OpenMPI rankfile: {rankfile_path}")
    return rankfile_path


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
    
    # Write rankfile to job directory or temp location
    if job_path:
        rankfile_path = os.path.join(job_path, f"parslbox_mpiexec_rankfile_{assignment.job_id}.txt")
        with open(rankfile_path, 'w') as f:
            f.write(rankfile_content)
    else:
        # Fallback to temp location
        fd, rankfile_path = tempfile.mkstemp(prefix=f"parslbox_mpiexec_rankfile_{assignment.job_id}_", suffix=".txt")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(rankfile_content)
        except:
            os.close(fd)
            raise
    
    logger.debug(f"Generated MPICH/PALS rankfile (CPU binding only): {rankfile_path}")
    return rankfile_path


def generate_openmpi_gpu_wrapper(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> str:
    """
    Generate GPU assignment wrapper script for OpenMPI (CPU binding handled by rankfile).
    
    Handles all cases:
    - Single node sub-node jobs (e.g., 2 GPUs out of 4): LOCAL_RANK 0,1 → GPU IDs as assigned
    - Single node full node jobs (e.g., all 4 GPUs): LOCAL_RANK 0,1,2,3 → GPU IDs as assigned  
    - Multi-node jobs: LOCAL_RANK on each node maps to assigned GPU IDs for that node
    """
    # Build comprehensive GPU mapping: global_rank → (local_rank_on_node, gpu_id)
    # We need to map LOCAL_RANK (per-node) to the correct GPU ID
    
    wrapper_content = f"""#!/bin/bash
# GPU assignment wrapper for OpenMPI job {assignment.job_id}
# Generated by ParslBox Resource Manager

LOCAL_RANK=${{OMPI_COMM_WORLD_LOCAL_RANK:-${{PMI_LOCAL_RANK:-${{SLURM_LOCALID:-0}}}}}}
GLOBAL_RANK=${{OMPI_COMM_WORLD_RANK:-${{PMI_RANK:-${{SLURM_PROCID:-0}}}}}}

# GPU assignments from resource manager (per global rank):
"""
    
    # Build mapping for all ranks across all nodes
    for node_idx in range(len(assignment.hostnames)):
        node_ranks = assignment.get_ranks_for_node(node_idx)
        
        # For each global rank on this node, determine its local rank and GPU assignment
        for local_rank_idx, global_rank in enumerate(node_ranks):
            gpu_ids = assignment.get_gpu_assignments_for_rank(global_rank)
            if gpu_ids:
                gpu_id = gpu_ids[0]  # Single GPU per rank
                
                # Add condition for this specific global rank
                wrapper_content += f"""if [ "$GLOBAL_RANK" -eq {global_rank} ]; then
    export CUDA_VISIBLE_DEVICES={gpu_id}
    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    export ZE_AFFINITY_MASK="{gpu_id}.0"
fi
"""
    
    wrapper_content += """
# Execute the application (CPU binding handled by OpenMPI rankfile)
exec "$@"
"""
    
    return write_gpu_wrapper(wrapper_content, assignment, "openmpi", job_path)


def generate_mpiexec_gpu_wrapper(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> str:
    """
    Generate GPU assignment wrapper script for MPICH/PALS (CPU binding handled by --cpu-bind).
    
    Uses MPICH environment variables with fallbacks to OpenMPI and SLURM.
    Priority: PMI (MPICH) → OMPI (OpenMPI) → SLURM
    """
    wrapper_content = f"""#!/bin/bash
# GPU assignment wrapper for MPICH/PALS job {assignment.job_id}
# Generated by ParslBox Resource Manager

# Environment variable priority: MPICH → OpenMPI → SLURM
LOCAL_RANK=${{PMI_LOCAL_RANK:-${{OMPI_COMM_WORLD_LOCAL_RANK:-${{SLURM_LOCALID:-0}}}}}}
GLOBAL_RANK=${{PMI_RANK:-${{OMPI_COMM_WORLD_RANK:-${{SLURM_PROCID:-0}}}}}}

# GPU assignments from resource manager (per global rank):
"""
    
    # Build mapping for all ranks across all nodes
    for node_idx in range(len(assignment.hostnames)):
        node_ranks = assignment.get_ranks_for_node(node_idx)
        
        # For each global rank on this node, determine its local rank and GPU assignment
        for local_rank_idx, global_rank in enumerate(node_ranks):
            gpu_ids = assignment.get_gpu_assignments_for_rank(global_rank)
            if gpu_ids:
                gpu_id = gpu_ids[0]  # Single GPU per rank
                
                # Add condition for this specific global rank
                wrapper_content += f"""if [ "$GLOBAL_RANK" -eq {global_rank} ]; then
    export CUDA_VISIBLE_DEVICES={gpu_id}
    export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1
    export ZE_AFFINITY_MASK="{gpu_id}.0"
fi
"""
    
    wrapper_content += """
# Execute the application (CPU binding handled by mpiexec --cpu-bind)
exec "$@"
"""
    
    return write_gpu_wrapper(wrapper_content, assignment, "mpiexec", job_path)


def compose_mpirun_launch_cmd(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> Tuple[str, str]:
    """
    Compose mpirun launch command prefix using appropriate binding strategy based on job type.
    
    - Sub-node jobs: Use rankfile for precise per-rank binding
    - Full-node CPU jobs: Use --map-by core:PE=N --bind-to core for MPI-managed distribution
    - Full-node GPU jobs: Use rankfile + wrapper for per-rank GPU assignment
    """
    total_ranks = job_spec.get_total_ranks()
    hostlist = ",".join(assignment.hostnames)
    
    # Detect job type to determine binding strategy
    job_type = job_spec.detect_job_type(system_config)
    
    if job_type == "fullnode_cpu":
        # Full-node CPU job: Use MPI's built-in core distribution
        cores_per_rank = system_config.CORES_PER_NODE // job_spec.ranks_per_node
        prefix = f"mpirun -H {hostlist} -np {total_ranks} --map-by core:PE={cores_per_rank} --bind-to core"
        logger.debug(f"Generated OpenMPI full-node CPU command: PE={cores_per_rank} cores per rank")
        
    else:
        # Sub-node jobs and GPU jobs: Use rankfile for precise binding
        rankfile_path = generate_openmpi_rankfile(assignment, system_config, job_spec, job_path)
        base_cmd = f"mpirun -H {hostlist} -np {total_ranks} --map-by rankfile:file={rankfile_path}"
        
        if job_spec.is_gpu_job():
            # GPU job: add wrapper for GPU assignment
            wrapper_path = generate_openmpi_gpu_wrapper(assignment, system_config, job_spec, job_path)
            prefix = f"{base_cmd} {wrapper_path}"
        else:
            # Sub-node CPU job: rankfile only
            prefix = base_cmd
    
    return "PBX_MPIRUN_PREFIX", prefix


def compose_mpiexec_launch_cmd(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> Tuple[str, str]:
    """
    Compose mpiexec launch command prefix using appropriate binding strategy based on job type.
    
    - Sub-node CPU jobs: Use --cpu-bind list for per-rank binding
    - Sub-node GPU jobs: Use --cpu-bind list + wrapper script (--gpu-bind is unreliable)
    - Full-node CPU jobs: Use --ppn and --depth with --cpu-bind depth for MPI-managed distribution
    - Full-node GPU jobs: Use rankfile for per-rank GPU assignment
    """
    total_ranks = job_spec.get_total_ranks()
    
    # Detect job type to determine binding strategy
    job_type = job_spec.detect_job_type(system_config)
    
    if assignment.is_single_node():
        hostname = assignment.hostnames[0]
        
        if job_type == "fullnode_cpu":
            # Full-node CPU job: Use MPICH's built-in core distribution
            ranks_per_node = job_spec.ranks_per_node
            cores_per_rank = system_config.CORES_PER_NODE // ranks_per_node
            prefix = f"mpiexec -n {total_ranks} -host {hostname} --ppn {ranks_per_node} --depth {cores_per_rank} --cpu-bind depth"
            logger.debug(f"Generated MPICH full-node CPU command: ppn={ranks_per_node}, depth={cores_per_rank}")
            
        else:
            # Sub-node jobs: Use --cpu-bind list and wrapper for GPU jobs
            cpu_bind_list = []
            
            node_ranks = assignment.get_ranks_for_node(0)
            for rank in node_ranks:
                # Get actual CPU cores assigned by resource manager
                cpu_cores = assignment.get_cpu_assignments_for_rank(rank)
                if cpu_cores:
                    cpu_cores_str = ",".join(map(str, cpu_cores))
                    cpu_bind_list.append(cpu_cores_str)
                else:
                    logger.warning(f"No CPU assignment for rank {rank}, skipping CPU binding")
                    cpu_bind_list.append("0")  # Minimal fallback
            
            # Build command with CPU binding
            cpu_bind_arg = f"--cpu-bind list:{':'.join(cpu_bind_list)}" if cpu_bind_list else ""
            
            if job_spec.is_gpu_job():
                # Sub-node GPU job: Use wrapper script instead of --gpu-bind
                wrapper_path = generate_mpiexec_gpu_wrapper(assignment, system_config, job_spec, job_path)
                prefix = f"mpiexec -n {total_ranks} -host {hostname} {cpu_bind_arg} {wrapper_path}".strip()
            else:
                # Sub-node CPU job: CPU binding only
                prefix = f"mpiexec -n {total_ranks} -host {hostname} {cpu_bind_arg}".strip()
    else:
        # Multi-node: use rankfile (cleaner than long lists)
        if job_type == "fullnode_cpu":
            # Multi-node full-node CPU job: Use ppn and depth
            ranks_per_node = job_spec.ranks_per_node
            cores_per_rank = system_config.CORES_PER_NODE // ranks_per_node
            hostlist = ",".join(assignment.hostnames)
            prefix = f"mpiexec -n {total_ranks} -ppn {ranks_per_node} -hosts {hostlist} --depth {cores_per_rank} --cpu-bind depth"
            logger.debug(f"Generated MPICH multi-node full-node CPU command: ppn={ranks_per_node}, depth={cores_per_rank}")
            
        else:
            # Multi-node sub-node or GPU jobs: Use rankfile for CPU binding
            rankfile_path = generate_mpiexec_rankfile(assignment, system_config, job_spec, job_path)
            ranks_per_node = len(assignment.get_ranks_for_node(0)) if assignment.hostnames else total_ranks // len(assignment.hostnames)
            hostlist = ",".join(assignment.hostnames)
            
            if job_spec.is_gpu_job():
                # Multi-node GPU job: Use rankfile for CPU binding + wrapper for GPU assignment
                wrapper_path = generate_mpiexec_gpu_wrapper(assignment, system_config, job_spec, job_path)
                prefix = f"mpiexec -n {total_ranks} -ppn {ranks_per_node} -hosts {hostlist} --rankfile {rankfile_path} {wrapper_path}"
                logger.debug(f"Generated MPICH multi-node GPU command: rankfile + wrapper")
            else:
                # Multi-node CPU job: Use rankfile for CPU binding only
                prefix = f"mpiexec -n {total_ranks} -ppn {ranks_per_node} -hosts {hostlist} --rankfile {rankfile_path}"
                logger.debug(f"Generated MPICH multi-node CPU command: rankfile only")
    
    return "PBX_MPIEXEC_PREFIX", prefix


def compose_srun_launch_cmd(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec') -> Tuple[str, str]:
    """
    Compose srun launch command prefix.
    
    Args:
        assignment: Resource assignment for the job
        system_config: System configuration
        job_spec: Job resource specification
        
    Returns:
        Tuple of (env_var_name, command_prefix)
    """
    num_nodes = len(assignment.node_ids)
    total_ranks = job_spec.get_total_ranks()
    
    if assignment.is_single_node():
        ranks_per_node = total_ranks
    else:
        ranks_per_node = job_spec.ranks_per_node if not job_spec.is_gpu_job() else job_spec.ngpus
    
    prefix = (f"srun --ntasks {total_ranks} --ntasks-per-node {ranks_per_node} "
              f"--nodelist {','.join(assignment.hostnames)} --nodes {num_nodes}")
    
    return "PBX_SRUN_PREFIX", prefix


def compose_mpi_command(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None) -> Dict[str, str]:
    """
    Generate the required MPI command prefix based on system configuration.
    
    Args:
        assignment: Resource assignment for the job
        system_config: System configuration
        job_spec: Job resource specification
        job_path: Optional job directory path for wrapper scripts and rankfiles
        
    Returns:
        Dictionary with the generated MPI command prefix
    """
    # Generate only the required command based on system config
    if system_config.MPI_CMD_TO_USE == "mpirun":
        key, prefix = compose_mpirun_launch_cmd(assignment, system_config, job_spec, job_path)
    elif system_config.MPI_CMD_TO_USE == "mpiexec":
        key, prefix = compose_mpiexec_launch_cmd(assignment, system_config, job_spec, job_path)
    elif system_config.MPI_CMD_TO_USE == "srun":
        key, prefix = compose_srun_launch_cmd(assignment, system_config, job_spec)
    else:
        raise ValueError(f"Unknown MPI_CMD_TO_USE: {system_config.MPI_CMD_TO_USE}")
    
    # Return both the specific command and set it as the default
    result = {
        key: prefix,
        "PBX_MPI_PREFIX": prefix
    }
    
    logger.debug(f"Generated MPI command for job {assignment.job_id}: {system_config.MPI_CMD_TO_USE} -> {prefix}")
    return result



