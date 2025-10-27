"""
MPI Command Launcher for ParslBox Resource Manager

This module provides MPI command generation functionality similar to Parsl's
launcher system, but adapted for ParslBox's resource management system.
"""

import logging
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from parslbox.resource_manager.models import NodeAssignment
    from parslbox.configs.base import SystemConfig

logger = logging.getLogger(__name__)

VALID_LAUNCHERS = ('mpirun', 'mpiexec', 'srun')


def _calculate_cpu_ranks(assignment: 'NodeAssignment', system_config: 'SystemConfig', node_occupancy: float) -> int:
    """
    Calculate number of CPU ranks for CPU-only jobs based on node occupancy.
    
    Args:
        assignment: Node assignment for the job
        system_config: System configuration
        node_occupancy: Fraction of node to use (0.0 to 1.0)
        
    Returns:
        Number of CPU ranks to use
    """
    if assignment.is_single_node():
        # For single-node CPU jobs, use occupancy * cores_per_node
        return max(1, int(node_occupancy * system_config.CORES_PER_NODE))
    else:
        # For multi-node CPU jobs, use all cores per node
        return system_config.CORES_PER_NODE


def compose_mpirun_launch_cmd(assignment: 'NodeAssignment', system_config: 'SystemConfig', node_occupancy: float = 1.0) -> Tuple[str, str]:
    """
    Compose mpirun launch command prefix.
    
    Args:
        assignment: Node assignment for the job
        system_config: System configuration
        node_occupancy: Node occupancy for CPU-only jobs
        
    Returns:
        Tuple of (env_var_name, command_prefix)
    """
    if assignment.is_single_node():
        hostname = assignment.hostnames[0]
        total_ranks = assignment.get_total_gpus() or _calculate_cpu_ranks(assignment, system_config, node_occupancy)
        
        # Add CPU binding for single-node jobs if CPU assignments are available
        cpu_binding = ""
        if assignment.cpu_assignments and assignment.cpu_assignments[0]:
            cpu_cores = assignment.cpu_assignments[0]
            cpu_list = ",".join(map(str, cpu_cores))
            cpu_binding = f"--cpu-list {cpu_list}"
        
        prefix = f"mpirun -H {hostname} -np {total_ranks} {cpu_binding}"
    else:
        hostlist = ",".join(assignment.hostnames)
        # For multi-node jobs, determine ranks per node
        if assignment.get_total_gpus() > 0:
            # GPU job: use GPUs per node from system config
            ranks_per_node = system_config.GPUS_PER_NODE
        else:
            # CPU job: use all cores per node
            ranks_per_node = system_config.CORES_PER_NODE
        
        total_ranks = len(assignment.node_ids) * ranks_per_node
        prefix = f"mpirun -H {hostlist} --map-by node -np {total_ranks}"    # -npernode {ranks_per_node} --> openmp mpirun manual says this is deprecated
    
    return "PBX_MPIRUN_PREFIX", prefix


def compose_mpiexec_launch_cmd(assignment: 'NodeAssignment', system_config: 'SystemConfig', node_occupancy: float = 1.0) -> Tuple[str, str]:
    """
    Compose mpiexec launch command prefix.
    
    Args:
        assignment: Node assignment for the job
        system_config: System configuration
        node_occupancy: Node occupancy for CPU-only jobs
        
    Returns:
        Tuple of (env_var_name, command_prefix)
    """
    if assignment.is_single_node():
        hostname = assignment.hostnames[0]
        total_ranks = assignment.get_total_gpus() or _calculate_cpu_ranks(assignment, system_config, node_occupancy)
        
        # Add CPU binding for single-node jobs if CPU assignments are available
        cpu_binding = ""
        if assignment.cpu_assignments and assignment.cpu_assignments[0]:
            cpu_cores = assignment.cpu_assignments[0]
            cpu_list = ",".join(map(str, cpu_cores))
            cpu_binding = f"--cpu-bind list:{cpu_list}"
        
        prefix = f"mpiexec -n {total_ranks} -host {hostname} {cpu_binding}"
    else:
        hostlist = ",".join(assignment.hostnames)
        # For multi-node jobs, determine ranks per node
        if assignment.get_total_gpus() > 0:
            # GPU job: use GPUs per node from system config
            ranks_per_node = system_config.GPUS_PER_NODE
        else:
            # CPU job: use all cores per node
            ranks_per_node = system_config.CORES_PER_NODE
        
        total_ranks = len(assignment.node_ids) * ranks_per_node
        prefix = f"mpiexec -n {total_ranks} -ppn {ranks_per_node} -hosts {hostlist}"
    
    return "PBX_MPIEXEC_PREFIX", prefix


def compose_srun_launch_cmd(assignment: 'NodeAssignment', system_config: 'SystemConfig', node_occupancy: float = 1.0) -> Tuple[str, str]:
    """
    Compose srun launch command prefix.
    
    Args:
        assignment: Node assignment for the job
        system_config: System configuration
        node_occupancy: Node occupancy for CPU-only jobs
        
    Returns:
        Tuple of (env_var_name, command_prefix)
    """
    num_nodes = len(assignment.node_ids)
    
    if assignment.is_single_node():
        total_ranks = assignment.get_total_gpus() or _calculate_cpu_ranks(assignment, system_config, node_occupancy)
        ranks_per_node = total_ranks
    else:
        # For multi-node jobs, determine ranks per node
        if assignment.get_total_gpus() > 0:
            # GPU job: use GPUs per node from system config
            ranks_per_node = system_config.GPUS_PER_NODE
        else:
            # CPU job: use all cores per node
            ranks_per_node = system_config.CORES_PER_NODE
        
        total_ranks = num_nodes * ranks_per_node
    
    prefix = (f"srun --ntasks {total_ranks} --ntasks-per-node {ranks_per_node} "
              f"--nodelist {','.join(assignment.hostnames)} --nodes {num_nodes}")
    
    return "PBX_SRUN_PREFIX", prefix


def compose_all_mpi_commands(assignment: 'NodeAssignment', system_config: 'SystemConfig', node_occupancy: float = 1.0) -> Dict[str, str]:
    """
    Generate all MPI command prefixes and set the default based on system config.
    
    Args:
        assignment: Node assignment for the job
        system_config: System configuration
        node_occupancy: Node occupancy for CPU-only jobs (default: 1.0)
        
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
            key, prefix = composer(assignment, system_config, node_occupancy)
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
