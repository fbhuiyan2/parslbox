"""
Resource Manager Utilities

Helper functions and utilities for resource management operations.
"""

import logging
from typing import Dict, List, Optional
from .models import NodeAssignment, JobResourceSpec

logger = logging.getLogger(__name__)


def generate_mpi_command(
    assignment: NodeAssignment,
    app_config: Dict,
    executable: str,
    args: str = "",
    mpi_opts: str = ""
) -> str:
    """
    Generate MPI command based on resource assignment.
    
    Args:
        assignment: Resource assignment for the job
        app_config: Application configuration dictionary
        executable: The executable to run
        args: Arguments to pass to the executable
        mpi_opts: Additional MPI options
        
    Returns:
        Complete MPI command string
    """
    if assignment.is_single_node():
        return _generate_single_node_mpi_command(assignment, app_config, executable, args, mpi_opts)
    else:
        return _generate_multinode_mpi_command(assignment, app_config, executable, args, mpi_opts)


def _generate_single_node_mpi_command(
    assignment: NodeAssignment,
    app_config: Dict,
    executable: str,
    args: str,
    mpi_opts: str
) -> str:
    """Generate MPI command for single-node job."""
    hostname = assignment.hostnames[0]
    
    # Determine number of ranks
    if assignment.gpu_assignments[0]:
        # GPU job: 1 rank per GPU
        num_ranks = len(assignment.gpu_assignments[0])
    else:
        # CPU job: use config or default to 1
        num_ranks = app_config.get('ranks_per_node', 1)
    
    # Build MPI command
    mpi_cmd_parts = [
        "mpirun",
        f"-n {num_ranks}",
        f"-host {hostname}"
    ]
    
    # Add any additional MPI options
    if mpi_opts:
        mpi_cmd_parts.append(mpi_opts)
    
    # Add executable and arguments
    mpi_cmd_parts.extend([executable, args])
    
    return " ".join(filter(None, mpi_cmd_parts))


def _generate_multinode_mpi_command(
    assignment: NodeAssignment,
    app_config: Dict,
    executable: str,
    args: str,
    mpi_opts: str
) -> str:
    """Generate MPI command for multi-node job."""
    hostlist = assignment.get_mpi_hostlist()
    ranks_per_node = app_config.get('ranks_per_node', 4)
    total_ranks = len(assignment.node_ids) * ranks_per_node
    
    # Build MPI command
    mpi_cmd_parts = [
        "mpirun",
        f"-n {total_ranks}",
        f"-host {hostlist}",
        f"-ppn {ranks_per_node}"
    ]
    
    # Add any additional MPI options
    if mpi_opts:
        mpi_cmd_parts.append(mpi_opts)
    
    # Add executable and arguments
    mpi_cmd_parts.extend([executable, args])
    
    return " ".join(filter(None, mpi_cmd_parts))


def generate_environment_setup(assignment: NodeAssignment) -> str:
    """
    Generate environment variable setup commands.
    
    Args:
        assignment: Resource assignment for the job
        
    Returns:
        Environment setup command string
    """
    env_vars = assignment.get_env_vars()
    
    if not env_vars:
        return ""
    
    # Generate export commands
    export_commands = []
    for var_name, var_value in env_vars.items():
        export_commands.append(f"export {var_name}={var_value}")
    
    return " && ".join(export_commands)


def build_job_command(
    assignment: NodeAssignment,
    app_config: Dict,
    executable: str,
    args: str = "",
    mpi_opts: str = ""
) -> str:
    """
    Build complete job command including environment setup and MPI command.
    
    Args:
        assignment: Resource assignment for the job
        app_config: Application configuration dictionary
        executable: The executable to run
        args: Arguments to pass to the executable
        mpi_opts: Additional MPI options
        
    Returns:
        Complete job command string
    """
    command_parts = []
    
    # Add environment setup
    env_setup = generate_environment_setup(assignment)
    if env_setup:
        command_parts.append(env_setup)
    
    # Add MPI command
    mpi_cmd = generate_mpi_command(assignment, app_config, executable, args, mpi_opts)
    command_parts.append(mpi_cmd)
    
    return " && ".join(command_parts)


def validate_job_resource_spec(spec: JobResourceSpec, max_nodes: int, max_gpus_per_node: int) -> List[str]:
    """
    Validate a job resource specification and return any validation errors.
    
    Args:
        spec: Job resource specification to validate
        max_nodes: Maximum nodes available in the system
        max_gpus_per_node: Maximum GPUs per node in the system
        
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check node count
    if spec.num_nodes < 1:
        errors.append("Number of nodes must be at least 1")
    elif spec.num_nodes > max_nodes:
        errors.append(f"Requested {spec.num_nodes} nodes, but only {max_nodes} available")
    
    # Check GPU count
    if spec.gpus_per_node < 0:
        errors.append("GPUs per node cannot be negative")
    elif spec.gpus_per_node > max_gpus_per_node:
        errors.append(f"Requested {spec.gpus_per_node} GPUs per node, but only {max_gpus_per_node} available")
    
    # Check occupancy
    if not 0.0 < spec.node_occupancy <= 1.0:
        errors.append("Node occupancy must be between 0.0 and 1.0")
    
    # Check consistency
    if spec.num_nodes > 1 and spec.node_occupancy != 1.0:
        errors.append("Multi-node jobs must have node_occupancy=1.0")
    
    return errors


def calculate_resource_efficiency(assignment: NodeAssignment, total_system_resources: Dict) -> Dict:
    """
    Calculate resource utilization efficiency for an assignment.
    
    Args:
        assignment: Resource assignment to analyze
        total_system_resources: Total system resources (nodes, gpus, etc.)
        
    Returns:
        Dictionary with efficiency metrics
    """
    total_nodes = total_system_resources.get('total_nodes', 1)
    total_gpus = total_system_resources.get('total_gpus', 0)
    
    assigned_nodes = len(assignment.node_ids)
    assigned_gpus = assignment.get_total_gpus()
    
    node_efficiency = assigned_nodes / total_nodes if total_nodes > 0 else 0.0
    gpu_efficiency = assigned_gpus / total_gpus if total_gpus > 0 else 0.0
    
    return {
        'node_efficiency': node_efficiency,
        'gpu_efficiency': gpu_efficiency,
        'assigned_nodes': assigned_nodes,
        'assigned_gpus': assigned_gpus,
        'total_nodes': total_nodes,
        'total_gpus': total_gpus
    }


def format_resource_summary(status: Dict) -> str:
    """
    Format resource status into a human-readable summary.
    
    Args:
        status: Resource status dictionary from ResourceManager.get_resource_status()
        
    Returns:
        Formatted summary string
    """
    lines = [
        "=== Resource Summary ===",
        f"Nodes: {status['used_nodes']}/{status['total_nodes']} used ({status['free_nodes']} free)",
        f"GPUs: {status['used_gpus']}/{status['total_gpus']} used ({status['available_gpus']} available)",
        f"CPU Capacity: {status['used_cpu_capacity']:.2f}/{status['total_cpu_capacity']:.2f} used",
        f"Active Jobs: {status['active_jobs']}",
        f"Backlogged Jobs: {status['backlogged_jobs']}"
    ]
    
    if status.get('nodes'):
        lines.append("\n=== Node Details ===")
        for node_status in status['nodes']:
            node_line = f"  {node_status['hostname']}: "
            if node_status['is_free']:
                node_line += "FREE"
            else:
                gpu_info = f"{node_status['total_gpus'] - node_status['available_gpus']}/{node_status['total_gpus']} GPUs"
                cpu_info = f"{node_status['cpu_occupancy']:.2f} CPU occupancy"
                job_info = f"Jobs: {node_status['assigned_jobs']}"
                node_line += f"{gpu_info}, {cpu_info}, {job_info}"
            lines.append(node_line)
    
    return "\n".join(lines)


def estimate_job_wait_time(spec: JobResourceSpec, current_status: Dict) -> Optional[str]:
    """
    Estimate wait time for a job based on current resource status.
    
    Args:
        spec: Job resource specification
        current_status: Current resource status
        
    Returns:
        Estimated wait time as string, or None if can be scheduled immediately
    """
    # Simple heuristic-based estimation
    if spec.is_multinode_job():
        free_nodes = current_status['free_nodes']
        if free_nodes >= spec.num_nodes:
            return None  # Can schedule immediately
        else:
            # Estimate based on active jobs (very rough)
            active_jobs = current_status['active_jobs']
            return f"~{active_jobs * 5} minutes (estimated)"
    
    elif spec.is_gpu_job():
        available_gpus = current_status['available_gpus']
        if available_gpus >= spec.gpus_per_node:
            return None  # Can schedule immediately
        else:
            # Estimate based on GPU availability
            return "~10-30 minutes (estimated)"
    
    else:
        # CPU-only job
        available_capacity = current_status['available_cpu_capacity']
        if available_capacity >= spec.node_occupancy:
            return None  # Can schedule immediately
        else:
            return "~5-15 minutes (estimated)"
