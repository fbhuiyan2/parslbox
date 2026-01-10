"""
MPI Command Launcher for ParslBox Resource Manager

This module provides MPI command generation functionality similar to Parsl's
launcher system, but adapted for ParslBox's resource management system.
"""

import logging
from typing import Dict, TYPE_CHECKING

# Import helper functions
from parslbox.resource_manager.helpers.mpi_launcher_helpers import (
    generate_openmpi_rankfile,
    generate_mpiexec_rankfile,
    generate_openmpi_gpu_wrapper,
    generate_mpiexec_gpu_wrapper
)

if TYPE_CHECKING:
    from parslbox.resource_manager.models import ResourceAssignment, JobResourceSpec
    from parslbox.system_configs.base_sysconf import SystemConfig

logger = logging.getLogger(__name__)

VALID_LAUNCHERS = ('mpirun', 'mpiexec', 'srun')


class MPICommandBuilder:
    """
    Modular MPI command builder that supports app-specific overrides.
    
    This class breaks down MPI command generation into discrete components
    and allows apps to disable or modify specific parts of the command.
    """
    
    def __init__(self, launcher_type: str, overrides: dict = None):
        """
        Initialize the MPI command builder.
        
        Args:
            launcher_type: Type of MPI launcher ('mpirun', 'mpiexec', 'srun')
            overrides: Dictionary with 'disable' and 'add' lists for command modification
        """
        if launcher_type not in VALID_LAUNCHERS:
            raise ValueError(f"Invalid launcher type: {launcher_type}. Must be one of {VALID_LAUNCHERS}")
        
        self.launcher_type = launcher_type
        self.overrides = overrides or {}
        
    def build_command(self, assignment: 'ResourceAssignment', system_config: 'SystemConfig', 
                     job_spec: 'JobResourceSpec', job_path: str = None) -> str:
        """
        Build complete MPI command by calling individual flag builders and applying overrides.
        
        Args:
            assignment: Resource assignment for the job
            system_config: System configuration
            job_spec: Job resource specification
            job_path: Optional job directory path for wrapper scripts and rankfiles
            
        Returns:
            Complete MPI command string
        """
        flags = []
        
        # Build each type of flag based on launcher type
        if self.launcher_type == "mpirun":
            flags.extend(self._build_mpirun_flags(assignment, system_config, job_spec, job_path))
        elif self.launcher_type == "mpiexec":
            flags.extend(self._build_mpiexec_flags(assignment, system_config, job_spec, job_path))
        elif self.launcher_type == "srun":
            flags.extend(self._build_srun_flags(assignment, system_config, job_spec))
        
        # Apply overrides
        flags = self._apply_overrides(flags)
        
        # Build final command
        command = f"{self.launcher_type} {' '.join(flags)}"
        logger.debug(f"Built MPI command with overrides: {command}")
        return command
    
    def _build_mpirun_flags(self, assignment: 'ResourceAssignment', system_config: 'SystemConfig', 
                           job_spec: 'JobResourceSpec', job_path: str = None) -> list[str]:
        """Build mpirun-specific flags."""
        flags = []
        total_ranks = job_spec.get_total_ranks()
        hostlist = ",".join(assignment.hostnames)
        job_type = job_spec.detect_job_type(system_config)
        
        # Process count flags
        flags.extend(["-np", str(total_ranks)])
        
        # Hostlist flags
        flags.extend(["-H", hostlist])
        
        if job_type == "fullnode_cpu":
            # Full-node CPU job: Use MPI's built-in core distribution
            cores_per_rank = system_config.CORES_PER_NODE // job_spec.ranks_per_node
            flags.extend([f"--map-by", f"core:PE={cores_per_rank}"])
            flags.extend(["--bind-to", "core"])
        else:
            # Sub-node jobs and GPU jobs: Use rankfile for precise binding
            rankfile_path = generate_openmpi_rankfile(assignment, system_config, job_spec, job_path)
            flags.extend(["--map-by", f"rankfile:file={rankfile_path}"])
            
            if job_spec.is_gpu_job():
                # GPU job: add wrapper for GPU assignment
                wrapper_path = generate_openmpi_gpu_wrapper(assignment, system_config, job_spec, job_path)
                flags.append(wrapper_path)
        
        return flags
    
    def _build_mpiexec_flags(self, assignment: 'ResourceAssignment', system_config: 'SystemConfig', 
                            job_spec: 'JobResourceSpec', job_path: str = None) -> list[str]:
        """Build mpiexec-specific flags."""
        flags = []
        total_ranks = job_spec.get_total_ranks()
        job_type = job_spec.detect_job_type(system_config)
        
        # Process count flags
        flags.extend(["-n", str(total_ranks)])
        
        if assignment.is_single_node():
            hostname = assignment.hostnames[0]
            flags.extend(["-host", hostname])
            
            if job_type == "fullnode_cpu":
                # Full-node CPU job: Use MPICH's built-in core distribution
                ranks_per_node = job_spec.ranks_per_node
                cores_per_rank = system_config.CORES_PER_NODE // ranks_per_node
                flags.extend(["--ppn", str(ranks_per_node)])
                flags.extend(["--depth", str(cores_per_rank)])
                flags.extend(["--cpu-bind", "depth"])
            else:
                # Sub-node jobs: Use --cpu-bind list and wrapper for GPU jobs
                cpu_bind_list = []
                node_ranks = assignment.get_ranks_for_node(0)
                
                for rank in node_ranks:
                    cpu_cores = assignment.get_cpu_assignments_for_rank(rank)
                    if cpu_cores:
                        cpu_cores_str = ",".join(map(str, cpu_cores))
                        cpu_bind_list.append(cpu_cores_str)
                    else:
                        logger.warning(f"No CPU assignment for rank {rank}, using fallback")
                        cpu_bind_list.append("0")
                
                if cpu_bind_list:
                    flags.extend(["--cpu-bind", f"list:{':'.join(cpu_bind_list)}"])
                
                if job_spec.is_gpu_job():
                    wrapper_path = generate_mpiexec_gpu_wrapper(assignment, system_config, job_spec, job_path)
                    flags.append(wrapper_path)
        else:
            # Multi-node
            hostlist = ",".join(assignment.hostnames)
            flags.extend(["-hosts", hostlist])
            
            if job_type == "fullnode_cpu":
                # Multi-node full-node CPU job
                ranks_per_node = job_spec.ranks_per_node
                cores_per_rank = system_config.CORES_PER_NODE // ranks_per_node
                flags.extend(["-ppn", str(ranks_per_node)])
                flags.extend(["--depth", str(cores_per_rank)])
                flags.extend(["--cpu-bind", "depth"])
            else:
                # Multi-node sub-node or GPU jobs: Use rankfile
                rankfile_path = generate_mpiexec_rankfile(assignment, system_config, job_spec, job_path)
                ranks_per_node = len(assignment.get_ranks_for_node(0)) if assignment.hostnames else total_ranks // len(assignment.hostnames)
                flags.extend(["-ppn", str(ranks_per_node)])
                flags.extend(["--rankfile", rankfile_path])
                
                if job_spec.is_gpu_job():
                    wrapper_path = generate_mpiexec_gpu_wrapper(assignment, system_config, job_spec, job_path)
                    flags.append(wrapper_path)
        
        return flags
    
    def _build_srun_flags(self, assignment: 'ResourceAssignment', system_config: 'SystemConfig', 
                         job_spec: 'JobResourceSpec') -> list[str]:
        """Build srun-specific flags."""
        flags = []
        num_nodes = len(assignment.node_ids)
        total_ranks = job_spec.get_total_ranks()
        
        if assignment.is_single_node():
            ranks_per_node = total_ranks
        else:
            ranks_per_node = job_spec.ranks_per_node if not job_spec.is_gpu_job() else job_spec.ngpus
        
        flags.extend(["--ntasks", str(total_ranks)])
        flags.extend(["--ntasks-per-node", str(ranks_per_node)])
        flags.extend(["--nodelist", ",".join(assignment.hostnames)])
        flags.extend(["--nodes", str(num_nodes)])
        
        return flags
    
    def _apply_overrides(self, flags: list[str]) -> list[str]:
        """
        Apply disable/add overrides to the flag list.
        
        Args:
            flags: List of command flags to modify
            
        Returns:
            Modified list of flags
        """
        if not self.overrides:
            return flags
        
        # Apply disable rules
        disable_list = self.overrides.get('disable', [])
        filtered_flags = self._filter_disabled_flags(flags, disable_list)
        
        # Apply add rules
        add_list = self.overrides.get('add', [])
        if add_list:
            # Parse add_list to handle multi-word flags properly
            for add_item in add_list:
                # Split on spaces to handle flags with arguments
                add_flags = add_item.split()
                filtered_flags.extend(add_flags)
        
        return filtered_flags
    
    def _filter_disabled_flags(self, flags: list[str], disable_list: list[str]) -> list[str]:
        """
        Remove flags based on disable rules.
        
        Args:
            flags: List of command flags
            disable_list: List of disable patterns/flags
            
        Returns:
            Filtered list of flags
        """
        if not disable_list:
            return flags
        
        result = []
        i = 0
        
        while i < len(flags):
            flag = flags[i]
            should_skip = False
            skip_next = False  # Whether to skip the next argument
            
            for disable_rule in disable_list:
                if disable_rule.startswith('--') or disable_rule.startswith('-'):
                    # Exact flag match - remove flag and its argument
                    if flag == disable_rule:
                        should_skip = True
                        # Skip the argument too if it exists and doesn't start with -
                        # This is necessary because flags contain ['--flag', 'arguments', '--next_flag', 'arguments']
                        if i + 1 < len(flags) and not flags[i + 1].startswith('-'):
                            skip_next = True
                        break
                else:
                    # Substring match - check both flag and its argument
                    if disable_rule in flag:
                        should_skip = True
                        # Skip the argument too if it exists and doesn't start with -
                        if i + 1 < len(flags) and not flags[i + 1].startswith('-'):
                            skip_next = True
                        break
                    # Also check if the substring is in the argument
                    elif (i + 1 < len(flags) and 
                          not flags[i + 1].startswith('-') and 
                          disable_rule in flags[i + 1]):
                        should_skip = True
                        skip_next = True
                        break
            
            if not should_skip:
                result.append(flag)
            
            i += 1
            
            # Skip the next flag if it was an argument to a disabled flag
            if skip_next and i < len(flags):
                i += 1
        
        return result


def compose_mpi_command(assignment: 'ResourceAssignment', system_config: 'SystemConfig', job_spec: 'JobResourceSpec', job_path: str = None, mpi_overrides: dict = None) -> Dict[str, str]:
    """
    Generate the required MPI command prefix based on system configuration.
    
    Args:
        assignment: Resource assignment for the job
        system_config: System configuration
        job_spec: Job resource specification
        job_path: Optional job directory path for wrapper scripts and rankfiles
        mpi_overrides: Optional dictionary with 'disable' and 'add' lists for command modification
        
    Returns:
        Dictionary with the generated MPI command prefix
    """
    # Use new modular builder for all MPI command generation
    builder = MPICommandBuilder(system_config.MPI_CMD_TO_USE, mpi_overrides)
    prefix = builder.build_command(assignment, system_config, job_spec, job_path)
    key = f"PBX_{system_config.MPI_CMD_TO_USE.upper()}_PREFIX"
    
    # Return both the specific command and set it as the default
    result = {
        key: prefix,
        "PBX_MPI_PREFIX": prefix
    }
    
    logger.debug(f"Generated MPI command for job {assignment.job_id}: {system_config.MPI_CMD_TO_USE} -> {prefix}")
    return result
