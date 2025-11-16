"""
MPI Command Launcher for ParslBox Resource Manager

This module provides MPI command generation functionality similar to Parsl's
launcher system, but adapted for ParslBox's resource management system.
"""

import logging
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from parslbox.resource_manager.models import NodeAssignment, JobResourceSpec
    from parslbox.configs.base import SystemConfig

logger = logging.getLogger(__name__)

VALID_LAUNCHERS = ('mpirun', 'mpiexec', 'srun')


def compose_mpirun_launch_cmd(assignment: 'NodeAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec') -> Tuple[str, str]:
    """
    Compose mpirun launch command prefix.
    
    Args:
        assignment: Node assignment for the job
        system_config: System configuration
        job_spec: Job resource specification
        
    Returns:
        Tuple of (env_var_name, command_prefix)
    """
    total_ranks = job_spec.get_total_ranks()
    
    if assignment.is_single_node():
        hostname = assignment.hostnames[0]
        
        # Add CPU binding for single-node jobs if CPU assignments are available
        cpu_binding = ""
        if assignment.cpu_assignments and assignment.cpu_assignments[0]:
            cpu_cores = assignment.cpu_assignments[0]
            cpu_list = ",".join(map(str, cpu_cores))
            cpu_binding = f"--cpu-list {cpu_list}"
        
        prefix = f"mpirun -H {hostname} -np {total_ranks} {cpu_binding}"
    else:
        hostlist = ",".join(assignment.hostnames)
        ranks_per_node = job_spec.ranks_per_node if not job_spec.is_gpu_job() else job_spec.ngpus
        prefix = f"mpirun -H {hostlist} --map-by node -np {total_ranks}"    # -npernode {ranks_per_node} --> openmp mpirun manual says this is deprecated
    
    return "PBX_MPIRUN_PREFIX", prefix


def compose_mpiexec_launch_cmd(assignment: 'NodeAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec') -> Tuple[str, str]:
    """
    Compose mpiexec launch command prefix.
    
    Args:
        assignment: Node assignment for the job
        system_config: System configuration
        job_spec: Job resource specification
        
    Returns:
        Tuple of (env_var_name, command_prefix)
    """
    total_ranks = job_spec.get_total_ranks()
    
    if assignment.is_single_node():
        hostname = assignment.hostnames[0]
        
        # Add CPU binding for single-node jobs if CPU assignments are available
        cpu_binding = ""
        if assignment.cpu_assignments and assignment.cpu_assignments[0]:
            cpu_cores = assignment.cpu_assignments[0]
            cpu_list = ",".join(map(str, cpu_cores))
            cpu_binding = f"--cpu-bind list:{cpu_list}"
        
        prefix = f"mpiexec -n {total_ranks} -host {hostname} {cpu_binding}"
    else:
        hostlist = ",".join(assignment.hostnames)
        ranks_per_node = job_spec.ranks_per_node if not job_spec.is_gpu_job() else job_spec.ngpus
        prefix = f"mpiexec -n {total_ranks} -ppn {ranks_per_node} -hosts {hostlist}"
    
    return "PBX_MPIEXEC_PREFIX", prefix


def compose_srun_launch_cmd(assignment: 'NodeAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec') -> Tuple[str, str]:
    """
    Compose srun launch command prefix.
    
    Args:
        assignment: Node assignment for the job
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


def compose_all_mpi_commands(assignment: 'NodeAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec') -> Dict[str, str]:
    """
    Generate all MPI command prefixes and set the default based on system config.
    
    Args:
        assignment: Node assignment for the job
        system_config: System configuration
        job_spec: Job resource specification
        
    Returns:
        Dictionary of environment variable names to command prefixes
    """
    all_prefixes = {}
    
    # Generate all available prefixes
    composers = [
        compose_mpirun_launch_cmd,
        compose_mpiexec_launch_cmd,
        compose_srun_launch_cmd,
    ]
    
    for composer in composers:
        try:
            key, prefix = composer(assignment, system_config, job_spec)
            all_prefixes[key] = prefix
        except Exception:
            logger.exception(f"Failed to compose launch prefix with {composer}")
    
    # Set the default based on system configuration
    if system_config.MPI_CMD_TO_USE == "mpirun":
        all_prefixes["PBX_MPI_PREFIX"] = all_prefixes.get("PBX_MPIRUN_PREFIX", "")
    elif system_config.MPI_CMD_TO_USE == "mpiexec":
        all_prefixes["PBX_MPI_PREFIX"] = all_prefixes.get("PBX_MPIEXEC_PREFIX", "")
    elif system_config.MPI_CMD_TO_USE == "srun":
        all_prefixes["PBX_MPI_PREFIX"] = all_prefixes.get("PBX_SRUN_PREFIX", "")
    else:
        raise RuntimeError(f"Unknown MPI_CMD_TO_USE: {system_config.MPI_CMD_TO_USE}")
    
    logger.debug(f"Generated MPI commands for job {assignment.job_id}: {all_prefixes}")
    return all_prefixes
