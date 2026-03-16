"""
Simplified MPI Command Builder for ParslBox

This module generates MPI commands using the new "start simple, opt-in complexity" approach.
Commands start with minimal flags and users opt-in to advanced features.

Command Structure:
    {mpi_cmd} {mpi_args} {mpi_extra} {wrapper} {exe} {exe_args}

Where:
    - mpi_cmd: The launcher command (mpirun, mpiexec, srun)
    - mpi_args: Core MPI flags (ranks, ranks per node)
    - mpi_extra: Optional advanced flags (CPU binding, hostlist, etc.)
    - wrapper: Optional GPU wrapper script
    - exe: The executable
    - exe_args: Application arguments
"""

import logging
import os
from typing import Dict, Optional, TYPE_CHECKING

from parslbox.resource_manager.mpi_config import MPIConfig, MPIBackend
from parslbox.resource_manager.helpers.mpi_launcher_helpers import (
    generate_openmpi_rankfile,
    generate_mpich_rankfile,
    generate_openmpi_gpu_wrapper,
    generate_mpich_gpu_wrapper,
)

if TYPE_CHECKING:
    from parslbox.resource_manager.models import ResourceAssignment, JobResourceSpec
    from parslbox.system_configs.base_sysconf import SystemConfig

logger = logging.getLogger(__name__)


class MPICommandBuilder:
    """
    Simplified MPI command builder that starts with minimal commands
    and adds complexity only when explicitly configured.
    """
    
    def __init__(self, mpi_config: MPIConfig, system_config: 'SystemConfig'):
        """
        Initialize the builder.
        
        Args:
            mpi_config: MPI configuration object
            system_config: System configuration for resource info
        """
        self.config = mpi_config
        self.system_config = system_config
    
    def build_command(
        self,
        assignment: 'ResourceAssignment',
        job_spec: 'JobResourceSpec',
        job_path: Optional[str] = None
    ) -> str:
        """
        Build the MPI command.
        
        Args:
            assignment: Resource assignment for the job
            job_spec: Job resource specification
            job_path: Optional job directory for generated files
            
        Returns:
            Complete MPI command string (without executable)
        """
        # Get basic info
        total_ranks = job_spec.get_total_ranks()
        ranks_per_node = self._calculate_ranks_per_node(job_spec, assignment)
        hostlist = self._get_hostlist(assignment)
        
        # Build context for template substitution
        context = self._build_context(assignment, job_spec, job_path, hostlist, ranks_per_node)
        
        # Build command parts
        mpi_cmd = self.config.get_mpi_command()
        mpi_args = self._build_mpi_args(total_ranks, ranks_per_node)
        mpi_extra = self._build_mpi_extra(assignment, job_spec, job_path, context)
        
        # Apply disable rules first
        mpi_extra = self._apply_disable(mpi_extra)
        
        # Apply add rules with template substitution
        mpi_extra = self._apply_add(mpi_extra, context)
        
        # Add GPU wrapper if enabled
        wrapper = ""
        if self.config.use_gpu_wrapper and job_spec.is_gpu_job():
            wrapper = self._generate_gpu_wrapper(assignment, job_spec, job_path)
            if "gpu-wrapper" in " ".join(self.config.disable):
                wrapper = ""  # Disabled via config
        
        # Build final command
        parts = [mpi_cmd] + mpi_args + mpi_extra
        if wrapper:
            parts.append(wrapper)
        
        command = " ".join(parts)
        logger.debug(f"Built MPI command: {command}")
        return command
    
    def _calculate_ranks_per_node(
        self,
        job_spec: 'JobResourceSpec',
        assignment: 'ResourceAssignment'
    ) -> int:
        """Calculate ranks per node."""
        if job_spec.is_gpu_job():
            # For GPU jobs, 1 rank per GPU
            if job_spec.num_nodes > 1:
                return job_spec.ngpus // job_spec.num_nodes
            return job_spec.ngpus
        else:
            return job_spec.ranks_per_node
    
    def _get_hostlist(self, assignment: 'ResourceAssignment') -> str:
        """Get hostlist, optionally using short hostnames."""
        hostnames = assignment.hostnames
        if self.config.use_short_hostnames:
            hostnames = [h.split('.')[0] for h in hostnames]
        return ",".join(hostnames)
    
    def _build_context(
        self,
        assignment: 'ResourceAssignment',
        job_spec: 'JobResourceSpec',
        job_path: Optional[str],
        hostlist: str,
        ranks_per_node: int
    ) -> Dict[str, str]:
        """Build template context for substitution."""
        total_ranks = job_spec.get_total_ranks()
        
        # Calculate cores per rank
        excluded_cores = getattr(self.system_config, 'EXCLUDE_CORES', None) or []
        effective_cores = self.system_config.CORES_PER_NODE - len(excluded_cores)
        cores_per_rank = effective_cores // ranks_per_node if ranks_per_node > 0 else effective_cores
        
        context = {
            "total_ranks": str(total_ranks),
            "ranks_per_node": str(ranks_per_node),
            "cores_per_rank": str(cores_per_rank),
            "hostlist": hostlist,
            "rankfile_path": None,
            "wrapper_path": None,
        }
        
        # Generate rankfile on-demand (will be populated if needed)
        # We defer actual generation until template is used
        self._deferred_rankfile_path = None
        self._deferred_wrapper_path = None
        self._deferred_context = {
            "assignment": assignment,
            "job_spec": job_spec,
            "job_path": job_path,
        }
        
        return context
    
    def _generate_rankfile_if_needed(self, context: Dict[str, str]) -> str:
        """Generate rankfile and return path."""
        if self._deferred_rankfile_path:
            return self._deferred_rankfile_path
        
        deferred = self._deferred_context
        assignment = deferred["assignment"]
        job_spec = deferred["job_spec"]
        job_path = deferred["job_path"]
        
        if self.config.backend == MPIBackend.OPENMPI:
            path = generate_openmpi_rankfile(assignment, self.system_config, job_spec, job_path)
        else:
            path = generate_mpich_rankfile(assignment, self.system_config, job_spec, job_path)
        
        self._deferred_rankfile_path = path
        return path
    
    def _generate_gpu_wrapper(
        self,
        assignment: 'ResourceAssignment',
        job_spec: 'JobResourceSpec',
        job_path: Optional[str]
    ) -> str:
        """Generate GPU wrapper script."""
        if self.config.backend == MPIBackend.OPENMPI:
            return generate_openmpi_gpu_wrapper(assignment, self.system_config, job_spec, job_path)
        else:
            return generate_mpich_gpu_wrapper(assignment, self.system_config, job_spec, job_path)
    
    def _build_mpi_args(self, total_ranks: int, ranks_per_node: int) -> list:
        """
        Build minimal MPI arguments based on backend.
        
        These are the essential flags that every job needs.
        
        Note: For OpenMPI, we skip --map-by if cpu_bind_method will add its own
        mapping (rankfile, list, or depth), to avoid duplicate --map-by flags.
        """
        backend = self.config.backend
        cpu_bind = self.config.cpu_bind_method
        
        if backend == MPIBackend.OPENMPI:
            # OpenMPI minimal: -np N
            args = ["-np", str(total_ranks)]
            
            # Only add default --map-by if cpu_bind_method won't add its own
            # rankfile, list, and depth all add their own --map-by flags
            if cpu_bind == "none":
                args.extend(["--map-by", f"ppr:{ranks_per_node}:node"])
            
            return args
        
        elif backend == MPIBackend.MPICH:
            # MPICH minimal: -n N --ppn M
            return [
                "-n", str(total_ranks),
                "--ppn", str(ranks_per_node)
            ]
        
        elif backend == MPIBackend.SRUN:
            # SRUN minimal: -n N --ntasks-per-node M
            return [
                "-n", str(total_ranks),
                "--ntasks-per-node", str(ranks_per_node)
            ]
        
        return []
    
    def _build_mpi_extra(
        self,
        assignment: 'ResourceAssignment',
        job_spec: 'JobResourceSpec',
        job_path: Optional[str],
        context: Dict[str, str]
    ) -> list:
        """
        Build optional MPI flags based on configuration.
        
        These are only added when explicitly enabled.
        """
        extra = []
        backend = self.config.backend
        
        # Hostlist (if enabled)
        if self.config.use_hostlist:
            hostlist = context["hostlist"]
            if backend == MPIBackend.OPENMPI:
                extra.extend(["-H", hostlist])
            elif backend == MPIBackend.MPICH:
                extra.extend(["-hosts", hostlist])
            elif backend == MPIBackend.SRUN:
                extra.extend(["--nodelist", hostlist])
        
        # CPU binding
        extra.extend(self._build_cpu_bind_flags(assignment, job_spec, job_path, context))
        
        return extra
    
    def _build_cpu_bind_flags(
        self,
        assignment: 'ResourceAssignment',
        job_spec: 'JobResourceSpec',
        job_path: Optional[str],
        context: Dict[str, str]
    ) -> list:
        """Build CPU binding flags based on cpu_bind_method."""
        method = self.config.cpu_bind_method
        backend = self.config.backend
        
        if method == "none":
            return []
        
        if method == "rankfile":
            return self._build_rankfile_binding(assignment, job_spec, job_path)
        
        if method == "list":
            return self._build_list_binding(assignment, job_spec, job_path)
        
        if method.startswith("depth"):
            return self._build_depth_binding(context)
        
        logger.warning(f"Unknown cpu_bind_method: {method}, ignoring")
        return []
    
    def _build_rankfile_binding(
        self,
        assignment: 'ResourceAssignment',
        job_spec: 'JobResourceSpec',
        job_path: Optional[str]
    ) -> list:
        """Build rankfile-based CPU binding flags."""
        backend = self.config.backend
        
        if backend == MPIBackend.OPENMPI:
            rankfile_path = self._generate_rankfile_if_needed({})
            return ["--map-by", f"rankfile:file={rankfile_path}"]
        
        elif backend == MPIBackend.MPICH:
            rankfile_path = self._generate_rankfile_if_needed({})
            return ["--rankfile", rankfile_path]
        
        elif backend == MPIBackend.SRUN:
            # SRUN doesn't have native rankfile, fall back to cpu-bind map
            logger.info("SRUN doesn't support rankfile, using cpu-bind map instead")
            return self._build_list_binding(assignment, job_spec, job_path)
        
        return []
    
    def _build_list_binding(
        self,
        assignment: 'ResourceAssignment',
        job_spec: 'JobResourceSpec',
        job_path: Optional[str]
    ) -> list:
        """Build list-based CPU binding flags."""
        backend = self.config.backend
        
        if backend == MPIBackend.OPENMPI:
            # OpenMPI uses rankfile for list binding too
            rankfile_path = self._generate_rankfile_if_needed({})
            return ["--map-by", f"rankfile:file={rankfile_path}"]
        
        elif backend == MPIBackend.MPICH:
            # Build cpu-bind list from assignment
            cpu_bind_list = []
            for node_idx in range(len(assignment.node_ids)):
                node_ranks = assignment.get_ranks_for_node(node_idx)
                for rank in node_ranks:
                    cpu_cores = assignment.get_cpu_assignments_for_rank(rank)
                    if cpu_cores:
                        cpu_bind_list.append(",".join(map(str, cpu_cores)))
                    else:
                        cpu_bind_list.append("0")
            
            if cpu_bind_list:
                return ["--cpu-bind", f"list:{':'.join(cpu_bind_list)}"]
            return []
        
        elif backend == MPIBackend.SRUN:
            # Build cpu-bind map for SRUN
            cpu_map = []
            for node_idx in range(len(assignment.node_ids)):
                node_ranks = assignment.get_ranks_for_node(node_idx)
                for rank in node_ranks:
                    cpu_cores = assignment.get_cpu_assignments_for_rank(rank)
                    if cpu_cores:
                        cpu_map.append(",".join(map(str, cpu_cores)))
            
            if cpu_map:
                return [f"--cpu-bind=map_cpu:{','.join(cpu_map)}"]
            return []
        
        return []
    
    def _build_depth_binding(self, context: Dict[str, str]) -> list:
        """Build depth-based CPU binding flags."""
        backend = self.config.backend
        
        # Get depth value (user-specified or auto-calculated)
        depth = self.config.get_cpu_bind_depth()
        if depth is None:
            depth = int(context["cores_per_rank"])
        
        if backend == MPIBackend.OPENMPI:
            # OpenMPI: use --map-by core:PE=N
            return ["--map-by", f"core:PE={depth}", "--bind-to", "core"]
        
        elif backend == MPIBackend.MPICH:
            # MPICH: use --cpu-bind depth --depth N
            return ["--cpu-bind", "depth", "--depth", str(depth)]
        
        elif backend == MPIBackend.SRUN:
            return ["--cpus-per-task", str(depth), "--cpu-bind=cores"]
        
        return []
    
    def _apply_disable(self, flags: list) -> list:
        """Apply disable rules to remove flags."""
        if not self.config.disable:
            return flags
        
        result = []
        i = 0
        
        while i < len(flags):
            flag = flags[i]
            should_skip = False
            skip_next = False
            
            for disable_rule in self.config.disable:
                # Skip special "gpu-wrapper" rule (handled separately)
                if disable_rule == "gpu-wrapper":
                    continue
                
                if disable_rule.startswith('--') or disable_rule.startswith('-'):
                    # Exact flag match
                    if flag == disable_rule:
                        should_skip = True
                        if i + 1 < len(flags) and not flags[i + 1].startswith('-'):
                            skip_next = True
                        break
                else:
                    # Substring match
                    if disable_rule in flag:
                        should_skip = True
                        if i + 1 < len(flags) and not flags[i + 1].startswith('-'):
                            skip_next = True
                        break
                    elif i + 1 < len(flags) and disable_rule in flags[i + 1]:
                        should_skip = True
                        skip_next = True
                        break
            
            if not should_skip:
                result.append(flag)
            
            i += 1
            if skip_next and i < len(flags):
                i += 1
        
        return result
    
    def _apply_add(self, flags: list, context: Dict[str, str]) -> list:
        """Apply add rules with template substitution."""
        if not self.config.add:
            return flags
        
        result = list(flags)
        
        for add_item in self.config.add:
            # Substitute templates
            substituted = add_item
            
            for key, value in context.items():
                placeholder = "{" + key + "}"
                if placeholder in substituted:
                    # Generate rankfile/wrapper on-demand
                    if key == "rankfile_path" and value is None:
                        value = self._generate_rankfile_if_needed(context)
                    elif key == "wrapper_path" and value is None:
                        deferred = self._deferred_context
                        value = self._generate_gpu_wrapper(
                            deferred["assignment"],
                            deferred["job_spec"],
                            deferred["job_path"]
                        )
                    
                    if value is not None:
                        substituted = substituted.replace(placeholder, str(value))
            
            # Split on spaces and add to result
            result.extend(substituted.split())
        
        return result


def build_mpi_command(
    mpi_config: MPIConfig,
    system_config: 'SystemConfig',
    assignment: 'ResourceAssignment',
    job_spec: 'JobResourceSpec',
    job_path: Optional[str] = None
) -> Dict[str, str]:
    """
    Convenience function to build MPI command.

    Args:
        mpi_config: MPI configuration
        system_config: System configuration
        assignment: Resource assignment
        job_spec: Job resource specification
        job_path: Optional job directory

    Returns:
        Dictionary with MPI command prefix
    """
    builder = MPICommandBuilder(mpi_config, system_config)
    prefix = builder.build_command(assignment, job_spec, job_path)

    return {
        "PBX_MPI_PREFIX": prefix,
        f"PBX_{mpi_config.get_mpi_command().upper()}_PREFIX": prefix,
    }


def build_resource_launcher(
    mpi_config: MPIConfig,
    system_config: 'SystemConfig',
    assignment: 'ResourceAssignment',
    job_spec: 'JobResourceSpec',
    job_path: Optional[str] = None
) -> str:
    """
    Build a single-process MPI launcher for non-MPI apps (USES_MPI=False).

    Non-MPI apps don't use MPI for parallelization. The MPI launcher's only
    role is to place a single process on the assigned node with the assigned
    CPU/GPU resources. The app handles any internal parallelism itself.

    This function reuses MPICommandBuilder.build_command() with modified inputs:
    - A synthetic JobResourceSpec that forces -n 1 --ppn 1 output, regardless
      of the actual job's resource spec (which can be multi-node, multi-GPU, etc.)
    - A modified ResourceAssignment preserving ALL node IDs/hostnames (for
      hostlist) but with CPU cores consolidated into rank 0 on node 0
    - No GPU wrapper — GPU binding is via CUDA_VISIBLE_DEVICES env var
      (already exported in the bash engine via _format_env_vars())

    All user MPI config settings are respected (use_hostlist, cpu_bind_method,
    use_short_hostnames, disable/add overrides). Nothing is forced or overridden.

    Args:
        mpi_config: MPI configuration (used for backend type and options)
        system_config: System configuration
        assignment: Resource assignment from the resource manager
        job_spec: Original job resource specification
        job_path: Optional job directory for generated files

    Returns:
        Resource launcher command string
    """
    from parslbox.resource_manager.models import JobResourceSpec, ResourceAssignment

    # Synthetic spec to make MPICommandBuilder produce -n 1 --ppn 1.
    # The actual job can be any size — PBX always launches 1 process
    # since the app handles its own parallelism (USES_MPI=False).
    single_spec = JobResourceSpec(
        job_id=job_spec.job_id,
        num_nodes=1,
        ngpus=0,  # No GPU handling via MPI — use env vars instead
        node_occupancy=job_spec.node_occupancy,
        ranks_per_node=1,
    )

    # Consolidate all CPU cores from all ranks on node 0 → rank 0
    all_cores = []
    if assignment.cpu_assignments and assignment.cpu_assignments[0]:
        for rank, cores in assignment.cpu_assignments[0].items():
            all_cores.extend(cores)
    all_cores = sorted(set(all_cores))

    # Preserve ALL nodes/hostnames for the hostlist,
    # but only rank 0 on node 0 has CPU assignments
    n_nodes = len(assignment.node_ids)
    launcher_assignment = ResourceAssignment(
        job_id=assignment.job_id,
        node_ids=assignment.node_ids,
        hostnames=assignment.hostnames,
        gpu_assignments=[{} for _ in range(n_nodes)],
        cpu_assignments=(
            [{0: all_cores}] + [{} for _ in range(n_nodes - 1)]
            if all_cores
            else [{} for _ in range(n_nodes)]
        ),
        node_occupancy=assignment.node_occupancy,
    )

    # Reuse the existing command builder
    builder = MPICommandBuilder(mpi_config, system_config)
    prefix = builder.build_command(launcher_assignment, single_spec, job_path)

    logger.info(f"Job {assignment.job_id}: Resource launcher: {prefix}")
    return prefix
