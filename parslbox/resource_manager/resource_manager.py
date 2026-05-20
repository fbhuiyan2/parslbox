"""
ParslBox Resource Manager

Main resource manager class that handles resource allocation, tracking,
and scheduling for jobs across HPC systems.
"""

import logging
import time
from typing import List, Dict, Optional, TYPE_CHECKING

from .models import NodeResource, JobResourceSpec, ResourceAssignment, create_job_resource_spec
from .exceptions import InsufficientResources, JobNotFound, InvalidResourceSpec
from .cpu_affinity import CPUAffinityManager
from .node_failure_tracker import NodeFailureTracker
from .job_tracker import JobTracker

if TYPE_CHECKING:
    from parslbox.system_configs.base_sysconf import SystemConfig

logger = logging.getLogger(__name__)


class ResourceManager:
    """
    Main resource manager for ParslBox.
    
    Key design philosophy: 
        - 1 node jobs with sub-node resource requirements (e.g. 2 gpus out of 4) can share the node with other 1 node jobs
        - multi-node jobs do not share nodes and take up the ful nodes
        - 1 node cpu jobs (e.g., ngpus=0) use the node_occupancy which determines what portion of the node the job will use
    Handles allocation and tracking of nodes, CPUs, and GPUs across
    different HPC systems. Supports both single-node and multi-node jobs
    with intelligent resource sharing.
    """
    
    def __init__(self, system_config: 'SystemConfig', job_tracker: JobTracker):
        """
        Initialize the resource manager.
        
        Args:
            system_config: System configuration object (Polaris, Sophia, etc.)
            job_tracker: JobTracker for efficient dependency checking
        """
        self.system_config = system_config
        self.job_tracker = job_tracker
        self.nodes: List[NodeResource] = []
        self.job_assignments: Dict[int, ResourceAssignment] = {}
        self._backlogged_jobs_set: set = set()  # Track job IDs in backlog
        
        # Initialize CPU affinity manager
        self.cpu_affinity_manager = CPUAffinityManager(
            system_config.WORKER_CPU_AFFINITY,
            system_config.CORES_PER_NODE
        )
        
        # Initialize node failure tracker for fault tolerance
        max_failures = getattr(system_config, 'MAX_CONSECUTIVE_FAILURES', 4)
        quarantine_duration = getattr(system_config, 'QUARANTINE_DURATION', 600)  # 10 minutes
        self.failure_tracker = NodeFailureTracker(max_failures, quarantine_duration)
        
        # Initialize nodes from system configuration
        self._initialize_nodes()
        
        logger.info(f"Initialized resource manager with {len(self.nodes)} nodes")
        logger.info(f"Fault tolerance: max_failures={max_failures}, quarantine_duration={quarantine_duration}s")
    
    def _initialize_nodes(self) -> None:
        """Initialize node resources from system configuration."""
        try:
            total_nodes, total_gpus = self.system_config.detect_resources()
            gpus_per_node = total_gpus//total_nodes #self.system_config.GPUS_PER_NODE
                                                    # gpus_per_node is initialized this way to account for sub-node batch jobs like in ALCF Sophia
            # Ideally total_cores should also be calculated here like gpus_per_node above
            # But I decided to intiate total_cores using CORES_PER_NODE so that affinity cores can be assigned even to subnode gpu batchjobs 
            # Get node list from scheduler
            node_hostnames = self._get_node_hostnames(total_nodes)
            
            # Create NodeResource objects
            for i, hostname in enumerate(node_hostnames):
                node_id = f"node-{i}"
                node = NodeResource(
                    node_id=node_id,
                    hostname=hostname,
                    total_gpus=gpus_per_node,
                    total_cores=self.system_config.CORES_PER_NODE,
                    excluded_cores=getattr(self.system_config, 'EXCLUDE_CORES', None)
                )
                self.nodes.append(node)
                
            logger.info(f"Initialized {len(self.nodes)} nodes with {gpus_per_node} GPUs each")
            
        except Exception as e:
            logger.error(f"Failed to initialize nodes: {e}")
            raise
    
    def _get_node_hostnames(self, num_nodes: int) -> List[str]:
        """
        Get actual hostnames from the scheduler.
        
        Args:
            num_nodes: Expected number of nodes
            
        Returns:
            List of hostnames
        """
        if hasattr(self.system_config, '_get_node_hostnames'):
            return self.system_config._get_node_hostnames(num_nodes)
        
        scheduler = getattr(self.system_config, 'SCHEDULER', 'UNKNOWN')
        
        if scheduler == "PBS":
            return self._get_pbs_hostnames()
        elif scheduler == "SLURM":
            return self._get_slurm_hostnames()
        else:
            # Fallback to generic names
            logger.warning(f"Unknown scheduler {scheduler}, using generic hostnames")
            return [f"compute-node-{i:02d}" for i in range(num_nodes)]
    
    def _get_pbs_hostnames(self) -> List[str]:
        """Get hostnames from PBS_NODEFILE."""
        import os
        
        node_file = os.environ.get("PBS_NODEFILE")
        if not node_file or not os.path.exists(node_file):
            raise RuntimeError("PBS_NODEFILE not found or not accessible")
        
        with open(node_file, 'r') as f:
            # Get unique hostnames (PBS_NODEFILE may have duplicates)
            hostnames = list(set(line.strip() for line in f.readlines() if line.strip()))
        
        return sorted(hostnames)
    
    def _get_slurm_hostnames(self) -> List[str]:
        """Get hostnames from SLURM environment."""
        import subprocess
        
        try:
            cmd = "scontrol show hostname $SLURM_NODELIST"
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, check=True
            )
            hostnames = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            return hostnames
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to get SLURM hostnames: {e}")
    
    def assign_resources(self, job: dict) -> ResourceAssignment:
        """
        Assign resources to a job based on its metadata.
        
        Sub-node jobs:
        - Single-node with partial occupancy (< 1.0)
        - Single-node GPU job with fewer GPUs than available per node
        
        Full-node jobs:
        - Single-node with full occupancy (= 1.0)
        - Single-node GPU job using all GPUs per node
        - All multi-node jobs

        Args:
            job: Job dictionary containing metadata and resource requirements
            
        Returns:
            ResourceAssignment with allocated resources
            
        Raises:
            InsufficientResources: If resources cannot be allocated
            InvalidResourceSpec: If resource specification is invalid
        """
        try:
            # Create resource specification from job metadata
            resource_spec = create_job_resource_spec(job)
            
            # Validate resource spec
            self._validate_resource_spec(resource_spec)
            
            # Check if job already has resources assigned
            if resource_spec.job_id in self.job_assignments:
                raise ValueError(f"Job {resource_spec.job_id} already has resources assigned")
            
            # Classify job type and assign resources accordingly
            if self._is_subnode_job(resource_spec):
                if resource_spec.is_gpu_job():
                    assignment = self._assign_subnode_gpu_job(resource_spec)
                else:
                    assignment = self._assign_subnode_cpu_job(resource_spec)
            else:
                # Full-node jobs
                if resource_spec.is_gpu_job():
                    assignment = self._assign_fullnode_gpu_job(resource_spec)
                else:
                    assignment = self._assign_fullnode_cpu_job(resource_spec)
            
            # Store the assignment
            self.job_assignments[resource_spec.job_id] = assignment
            
            logger.info(f"Assigned resources to job {resource_spec.job_id}: {assignment.get_summary()}")
            return assignment
            
        except InsufficientResources:
            # Add to backlog using centralized method
            self.add_to_backlog(resource_spec.job_id)
            raise
    
    def _validate_resource_spec(self, spec: JobResourceSpec) -> None:
        """Validate resource specification against system capabilities."""
        # For single-node jobs, check GPU requirements
        if spec.num_nodes == 1 and spec.ngpus > self.system_config.GPUS_PER_NODE:
            raise InvalidResourceSpec(
                f"Requested {spec.ngpus} GPUs per node, "
                f"but system only has {self.system_config.GPUS_PER_NODE}"
            )
        
        if spec.num_nodes > len(self.nodes):
            raise InvalidResourceSpec(
                f"Requested {spec.num_nodes} nodes, "
                f"but only {len(self.nodes)} nodes available"
            )
    
    def _is_subnode_job(self, spec: JobResourceSpec) -> bool:
        """
        Determine if this is a sub-node job requiring per-rank resource assignment.
        
        Sub-node jobs:
        - Single-node with partial occupancy (< 1.0)
        - Single-node GPU job with fewer GPUs than available per node
        
        Args:
            spec: Job resource specification
            
        Returns:
            True if this is a sub-node job, False if full-node job
        """
        if spec.num_nodes > 1:
            return False  # Multi-node = always full-node
        
        # Single-node job classification
        if spec.is_gpu_job():
            # node occupancy is designed to be ignored for gpu jobs, so just check ngpus
            return spec.ngpus < self.system_config.GPUS_PER_NODE
        else:
            return spec.node_occupancy < 1.0
    

    def _assign_subnode_cpu_job(self, spec: JobResourceSpec) -> ResourceAssignment:
        """Assign resources for a sub-node CPU-only job with per-rank allocation."""
        # Calculate effective cores per node accounting for excluded cores
        excluded_cores = getattr(self.system_config, 'EXCLUDE_CORES', None) or []
        effective_cores_per_node = self.system_config.CORES_PER_NODE - len(excluded_cores)
        
        # Calculate number of CPU cores needed based on occupancy
        num_cores_needed = max(1, int(spec.node_occupancy * effective_cores_per_node))
        
        # Find a healthy node with enough CPU cores
        for node in self.nodes:
            if node.can_fit_cpu_cores(num_cores_needed) and node.health_tracker.can_accept_jobs():
                # Use simple core assignment (no GPU affinity for CPU-only jobs)
                assigned_cores = node.assign_cpu_job(spec.job_id, num_cores_needed)
                
                # Create ResourceAssignment with per-rank allocation
                assignment = ResourceAssignment(
                    job_id=spec.job_id,
                    node_ids=[node.node_id],
                    hostnames=[node.hostname],
                    node_occupancy=spec.node_occupancy
                )
                
                # Distribute CPU cores evenly across ranks
                cores_per_rank = max(1, num_cores_needed // spec.ranks_per_node)
                remaining_cores = num_cores_needed % spec.ranks_per_node
                
                core_idx = 0
                for rank in range(spec.ranks_per_node):
                    # Give some ranks one extra core if cores don't divide evenly
                    rank_core_count = cores_per_rank + (1 if rank < remaining_cores else 0)
                    rank_cores = assigned_cores[core_idx:core_idx + rank_core_count]
                    core_idx += rank_core_count
                    
                    assignment.assign_resources_to_rank(
                        rank=rank,
                        node_idx=0,
                        gpu_ids=[],  # No GPUs for CPU-only jobs
                        cpu_ids=rank_cores
                    )
                
                logger.info(f"Assigned CPU-only job {spec.job_id}: {num_cores_needed} cores across {spec.ranks_per_node} ranks")
                return assignment
        
        # No suitable node found
        max_available_cores = max(len(node.available_core_ids) for node in self.nodes)
        raise InsufficientResources(
            f"Not enough CPU cores available for sub-node CPU job",
            requested={'cores': num_cores_needed},
            available={'cores': max_available_cores}
        )
    
    def _assign_subnode_gpu_job(self, spec: JobResourceSpec) -> ResourceAssignment:
        """
        Assign resources for a sub-node GPU job with per-rank allocation and GPU-CPU affinity awareness.
        Affinity is a priority but not a strict requirement. If affinity CPUs are available for a GPU, then those CPUs will be assigned.
        If affinity CPUs are not available but other CPUs are, then GPUs will get those CPUs

        """
        # Calculate cores per GPU for balanced allocation
        # Account for excluded cores when calculating cores per GPU
        excluded_cores = getattr(self.system_config, 'EXCLUDE_CORES', None) or []
        effective_cores_per_node = self.system_config.CORES_PER_NODE - len(excluded_cores)
        cores_per_gpu = effective_cores_per_node // self.system_config.GPUS_PER_NODE
        
        # Find a healthy node with enough GPUs and CPU cores
        for node in self.nodes:
            if node.can_fit_gpu_job(spec.ngpus, cores_per_gpu) and node.health_tracker.can_accept_jobs():
                # Use the new combined assignment method
                assigned_gpus, cpu_assignments = node.assign_gpu_job(
                    spec.job_id, spec.ngpus, cores_per_gpu, self.cpu_affinity_manager
                )
                
                # Create ResourceAssignment with per-rank allocation
                assignment = ResourceAssignment(
                    job_id=spec.job_id,
                    node_ids=[node.node_id],
                    hostnames=[node.hostname],
                    node_occupancy=spec.node_occupancy
                )
                
                # Assign resources per rank
                gpus_per_rank = max(1, spec.ngpus // spec.ranks_per_node)
                cores_per_rank = len(cpu_assignments) // spec.ranks_per_node if cpu_assignments else 0
                for rank in range(spec.ranks_per_node):
                    rank_gpus = assigned_gpus[rank * gpus_per_rank : (rank + 1) * gpus_per_rank]
                    rank_cores = []
                    for g in range(rank * gpus_per_rank, min((rank + 1) * gpus_per_rank, len(cpu_assignments))):
                        rank_cores.extend(cpu_assignments[g])

                    assignment.assign_resources_to_rank(
                        rank=rank,
                        node_idx=0,
                        gpu_ids=rank_gpus,
                        cpu_ids=rank_cores
                    )
                
                total_cores = sum(len(cores) for cores in cpu_assignments)
                logger.info(f"Assigned GPU job {spec.job_id}: GPUs {assigned_gpus}, {total_cores} total CPU cores ({cores_per_gpu} per GPU)")
                return assignment
        
        # No suitable node found
        max_available_gpus = max(len(node.available_gpu_ids) for node in self.nodes)
        max_available_cores = max(len(node.available_core_ids) for node in self.nodes)
        raise InsufficientResources(
            f"Not enough resources for sub-node GPU job",
            requested={'gpus': spec.ngpus, 'cores': spec.ngpus * cores_per_gpu},
            available={'gpus': max_available_gpus, 'cores': max_available_cores}
        )
    
    
    def _assign_fullnode_cpu_job(self, spec: JobResourceSpec) -> ResourceAssignment:
        """
        Assign resources for full-node CPU jobs (single-node full or multi-node).
        
        Key assumptions:
        - Jobs get exclusive access to entire nodes
        - No per-rank resource assignment for CPU-only jobs (MPI handles core distribution)
        - Full-node CPU-only jobs blocks all CPUs as well as GPUs
        
        """
        # Find enough completely free and healthy nodes
        free_nodes = [node for node in self.nodes if node.is_completely_free() and node.health_tracker.can_accept_jobs()]
        
        if len(free_nodes) < spec.num_nodes:
            available = len(free_nodes)
            requested = spec.num_nodes
            raise InsufficientResources(
                f"Not enough free nodes for full-node CPU job",
                requested={'nodes': requested},
                available={'nodes': available}
            )
        
        # Assign the first N free nodes
        # Capture available core IDs BEFORE assign_fullnode_cpu_job() clears them
        assigned_nodes = free_nodes[:spec.num_nodes]
        node_ids = []
        hostnames = []
        node_cores = []  # Per-node list of available core IDs

        for node in assigned_nodes:
            node_cores.append(sorted(list(node.available_core_ids)))
            node.assign_fullnode_cpu_job(spec.job_id)
            node_ids.append(node.node_id)
            hostnames.append(node.hostname)

        assignment = ResourceAssignment(
            job_id=spec.job_id,
            node_ids=node_ids,
            hostnames=hostnames,
            node_occupancy=spec.node_occupancy
        )

        # Distribute all available cores evenly across ranks per node
        current_rank = 0
        for node_idx in range(spec.num_nodes):
            all_cores = node_cores[node_idx]
            cores_per_rank = max(1, len(all_cores) // spec.ranks_per_node)
            remaining = len(all_cores) % spec.ranks_per_node

            core_idx = 0
            for local_rank in range(spec.ranks_per_node):
                rank_core_count = cores_per_rank + (1 if local_rank < remaining else 0)
                rank_cores = all_cores[core_idx:core_idx + rank_core_count]
                core_idx += rank_core_count

                assignment.assign_resources_to_rank(
                    rank=current_rank,
                    node_idx=node_idx,
                    gpu_ids=[],
                    cpu_ids=rank_cores,
                )
                current_rank += 1
        
        logger.info(f"Assigned full-node CPU job {spec.job_id}: {spec.num_nodes} nodes, {spec.get_total_ranks()} total ranks")
        return assignment
    
    def _assign_fullnode_gpu_job(self, spec: JobResourceSpec) -> ResourceAssignment:
        """
        Assign resources for full-node GPU jobs (single-node full or multi-node).
        
        Key assumptions:
        - Jobs get exclusive access to entire nodes
        - GPU jobs still use per-rank assignment (1 GPU + affinity cores per rank)
        - For multi-node GPU jobs: each node gets all its GPUs with affinity
        - For single-node full GPU jobs: use all GPUs on the node
        - Since all CPUs and GPUs are being used here, affinity CPU assignment is (at least should be) guarnteed
        """
        # Find enough completely free and healthy nodes
        free_nodes = [node for node in self.nodes if node.is_completely_free() and node.health_tracker.can_accept_jobs()]
        
        if len(free_nodes) < spec.num_nodes:
            available = len(free_nodes)
            requested = spec.num_nodes
            raise InsufficientResources(
                f"Not enough free nodes for full-node GPU job",
                requested={'nodes': requested},
                available={'nodes': available}
            )
        
        # Assign the first N free nodes
        assigned_nodes = free_nodes[:spec.num_nodes]
        node_ids = []
        hostnames = []
        
        # Calculate cores per GPU for balanced allocation
        # Account for excluded cores when calculating cores per GPU
        excluded_cores = getattr(self.system_config, 'EXCLUDE_CORES', None) or []
        effective_cores_per_node = self.system_config.CORES_PER_NODE - len(excluded_cores)
        cores_per_gpu = effective_cores_per_node // self.system_config.GPUS_PER_NODE
        
        # For each node, assign all GPUs with affinity
        for node in assigned_nodes:
            if spec.is_multinode_job():
                # Multi-node GPU job: each node gets all its GPUs
                gpus_to_assign = self.system_config.GPUS_PER_NODE
            else:
                # Single-node full GPU job: use the requested number of GPUs
                gpus_to_assign = spec.ngpus
            
            node.assign_gpu_job(spec.job_id, gpus_to_assign, cores_per_gpu, self.cpu_affinity_manager)
            node_ids.append(node.node_id)
            hostnames.append(node.hostname)
        
        # Create ResourceAssignment with per-rank allocation for GPU jobs
        assignment = ResourceAssignment(
            job_id=spec.job_id,
            node_ids=node_ids,
            hostnames=hostnames,
            node_occupancy=spec.node_occupancy
        )
        
        # Assign resources per rank
        current_rank = 0
        for node_idx, node in enumerate(assigned_nodes):
            if spec.is_multinode_job():
                gpus_on_node = self.system_config.GPUS_PER_NODE
            else:
                gpus_on_node = spec.ngpus

            assigned_gpus = node.job_gpu_assignments[spec.job_id]
            assigned_cores = node.job_cpu_assignments[spec.job_id]

            if len(assigned_gpus) != gpus_on_node:
                raise RuntimeError(f"GPU assignment mismatch: expected {gpus_on_node}, got {len(assigned_gpus)}")

            gpus_per_rank = max(1, gpus_on_node // spec.ranks_per_node)
            cores_per_gpu_actual = len(assigned_cores) // len(assigned_gpus) if assigned_gpus else 0

            for local_rank in range(spec.ranks_per_node):
                rank_gpus = assigned_gpus[local_rank * gpus_per_rank : (local_rank + 1) * gpus_per_rank]

                start_core_idx = local_rank * gpus_per_rank * cores_per_gpu_actual
                end_core_idx = min(start_core_idx + gpus_per_rank * cores_per_gpu_actual, len(assigned_cores))
                rank_cores = assigned_cores[start_core_idx:end_core_idx]

                assignment.assign_resources_to_rank(
                    rank=current_rank,
                    node_idx=node_idx,
                    gpu_ids=rank_gpus,
                    cpu_ids=rank_cores
                )
                current_rank += 1
        
        total_gpus = sum(len(node.job_gpu_assignments[spec.job_id]) for node in assigned_nodes)
        total_cores = sum(len(node.job_cpu_assignments[spec.job_id]) for node in assigned_nodes)
        logger.info(f"Assigned full-node GPU job {spec.job_id}: {spec.num_nodes} nodes, {total_gpus} GPUs, {total_cores} CPU cores")
        return assignment
    
    
    def add_to_backlog(self, job_id: int) -> None:
        """
        Add a job to the backlog queue.
        
        Args:
            job_id: The job ID to add to backlog
        """
        if job_id not in self._backlogged_jobs_set:
            self._backlogged_jobs_set.add(job_id)
            logger.info(f"Job {job_id} added to backlog")

    
    def free_resources(self, job_id: int) -> None:
        """
        Free up resources assigned to a job.
        
        .. deprecated:: 
            Use :func:`free_resources_with_health_check` instead for proper fault tolerance.
            This method does not update node health tracking and should only be used
            for internal operations where health tracking is handled separately.
        
        Args:
            job_id: The job ID to free resources for
            
        Raises:
            JobNotFound: If job has no assigned resources
        """
        if job_id not in self.job_assignments:
            raise JobNotFound(f"Job {job_id} has no assigned resources")
        
        assignment = self.job_assignments[job_id]
        
        # Free resources on all assigned nodes
        for node_id in assignment.node_ids:
            node = self._get_node_by_id(node_id)
            if node:
                node.free_job(job_id)
        
        # Remove from assignments
        del self.job_assignments[job_id]
        
        logger.info(f"Freed resources for job {job_id}")
    
    def _get_node_by_id(self, node_id: str) -> Optional[NodeResource]:
        """Get node by ID."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None
    
    @property
    def backlog(self) -> List[dict]:
        """Get list of jobs currently in backlog using JobTracker."""
        if not self._backlogged_jobs_set:
            return []
        
        job_ids = list(self._backlogged_jobs_set)
        return self.job_tracker.get_jobs_by_ids(job_ids)
    
    def get_dependency_ready_jobs_from_backlog(self) -> List[dict]:
        """
        Get backlogged jobs whose dependencies are satisfied using JobTracker.
        
        Returns:
            List of job dictionaries for jobs whose dependencies are satisfied
        """
        if not self._backlogged_jobs_set:
            return []
        
        backlog_job_ids = list(self._backlogged_jobs_set)
        return self.job_tracker.get_dependency_ready_jobs(backlog_job_ids)
    
    def schedule_backlog(self, candidate_jobs: List[dict]) -> List[dict]:
        """Schedule jobs from candidate list that are in backlog."""
        
        if not candidate_jobs:
            return []
        
        # Filter candidates that are actually in backlog
        candidates_in_backlog = [job for job in candidate_jobs if job['job_id'] in self._backlogged_jobs_set]
        
        if not candidates_in_backlog:
            return []
        
        # Sort by priority (LOWER num_nodes first - easier to schedule)
        candidates_in_backlog.sort(key=lambda job: job.get('num_nodes', 1))
        
        scheduled_jobs = []
        for job in candidates_in_backlog:
            job_id = job['job_id']
            
            try:
                assignment = self.assign_resources(job)
                scheduled_jobs.append(job)
                
                # Remove from backlog tracking
                self._backlogged_jobs_set.discard(job_id)
                logger.info(f"Successfully rescheduled job {job_id}: {assignment.get_summary()}")
                
            except InsufficientResources as e:
                logger.debug(f"Job {job_id} still cannot be scheduled: {e}")
                continue  # Try next job
            except Exception as e:
                logger.error(f"Error scheduling job {job_id}: {e}")
                continue
        
        return scheduled_jobs
    
    def get_resource_status(self) -> Dict:
        """
        Get current resource status.
        
        Returns:
            Dictionary with resource status information
        """
        total_nodes = len(self.nodes)
        free_nodes = len([node for node in self.nodes if len(node.assigned_jobs) == 0])
        total_gpus = sum(node.total_gpus for node in self.nodes)
        available_gpus = sum(len(node.available_gpu_ids) for node in self.nodes)
        available_nodes = sum(1 for node in self.nodes if node.cpu_occupancy < 1.0)
        total_cpu_capacity = float(total_nodes)
        used_cpu_capacity = sum(node.cpu_occupancy for node in self.nodes)
        
        return {
            'total_nodes': total_nodes,
            'free_nodes': free_nodes,
            'used_nodes': total_nodes - free_nodes,
            'total_gpus': total_gpus,
            'available_gpus': available_gpus,
            'available_nodes': available_nodes,
            'used_gpus': total_gpus - available_gpus,
            'total_cpu_capacity': total_cpu_capacity,
            'used_cpu_capacity': used_cpu_capacity,
            'available_cpu_capacity': total_cpu_capacity - used_cpu_capacity,
            'active_jobs': len(self.job_assignments),
            'backlogged_jobs': len(self._backlogged_jobs_set),
            'nodes': [node.get_status() for node in self.nodes]
        }
    
    def get_job_assignment(self, job_id: int) -> Optional[ResourceAssignment]:
        """
        Get resource assignment for a specific job.
        
        Args:
            job_id: The job ID to get assignment for
            
        Returns:
            ResourceAssignment if job has resources assigned, None otherwise
        """
        return self.job_assignments.get(job_id)
    
    def list_active_jobs(self) -> List[int]:
        """Get list of job IDs with active resource assignments."""
        return list(self.job_assignments.keys())
    
    def get_backlog_size(self) -> int:
        """Get number of jobs in the backlog."""
        return len(self._backlogged_jobs_set)
    
    def get_backlog_jobids(self) -> List[int]:
        """Get job ids of jobs in the backlog."""
        return list(self._backlogged_jobs_set)
    
    # Fault tolerance methods
    
    def record_job_failure(self, job_id: int, error_message: str = None) -> None:
        """
        Record a job failure and handle node health tracking.
        
        Args:
            job_id: The job ID that failed
            error_message: Error message from the failure
        """
        if job_id not in self.job_assignments:
            logger.warning(f"Cannot record failure for job {job_id}: no resource assignment found")
            return
        
        assignment = self.job_assignments[job_id]
        
        # Record failure for each node used by the job
        for node_id in assignment.node_ids:
            node = self._get_node_by_id(node_id)
            if node:
                # Record failure in node's health tracker
                should_quarantine = node.health_tracker.record_failure(error_message)
                
                if should_quarantine:
                    logger.error(f"Node {node_id} ({node.hostname}) quarantined after job {job_id} failure")
                    
                    # Also record in the centralized failure tracker
                    self.failure_tracker.record_job_failure(node_id, job_id, error_message)
    
    def record_job_success(self, job_id: int) -> None:
        """
        Record a successful job completion and update node health.
        
        Args:
            job_id: The job ID that succeeded
        """
        if job_id not in self.job_assignments:
            logger.warning(f"Cannot record success for job {job_id}: no resource assignment found")
            return
        
        assignment = self.job_assignments[job_id]
        
        # Record success for each node used by the job
        for node_id in assignment.node_ids:
            node = self._get_node_by_id(node_id)
            if node:
                # Record success in node's health tracker
                old_status = node.health_tracker.health_status
                node.health_tracker.record_success()
                
                if old_status != node.health_tracker.health_status:
                    logger.info(f"Node {node_id} ({node.hostname}) health improved from {old_status.value} "
                               f"to {node.health_tracker.health_status.value} after job {job_id} success")
                
                # Also record in the centralized failure tracker
                self.failure_tracker.record_job_success(node_id, job_id)
    
    def free_resources_with_health_check(self, job_id: int, job_succeeded: bool = True, error_message: str = None) -> None:
        """
        Free resources and update node health based on job outcome.
        
        This is the recommended method to use instead of free_resources() directly
        when you know the job outcome.
        
        Args:
            job_id: The job ID to free resources for
            job_succeeded: Whether the job succeeded or failed
            error_message: Error message if job failed
        """
        # Record job outcome for health tracking
        if job_succeeded:
            self.record_job_success(job_id)
        else:
            self.record_job_failure(job_id, error_message)
        
        # Free the resources
        self.free_resources(job_id)
    
    def get_node_health_summary(self) -> Dict:
        """
        Get comprehensive node health information.
        
        Returns:
            Dictionary with node health summary
        """
        health_summary = {
            'system_health': self.failure_tracker.get_system_health_summary(),
            'node_details': {},
            'quarantined_nodes': []
        }
        
        # Get detailed health for each node
        for node in self.nodes:
            node_health = node.health_tracker.get_status_summary()
            health_summary['node_details'][node.node_id] = {
                'hostname': node.hostname,
                'health': node_health,
                'resource_status': {
                    'available_gpus': len(node.available_gpu_ids),
                    'total_gpus': node.total_gpus,
                    'cpu_occupancy': node.cpu_occupancy,
                    'assigned_jobs': node.assigned_jobs.copy()
                }
            }
            
            # Track quarantined nodes
            if not node_health['can_accept_jobs']:
                health_summary['quarantined_nodes'].append({
                    'node_id': node.node_id,
                    'hostname': node.hostname,
                    'health_status': node_health['health_status'],
                    'consecutive_failures': node_health['consecutive_failures'],
                    'last_failure_error': node_health['last_failure_error']
                })
        
        return health_summary
    
    def force_quarantine_node(self, node_id: str, reason: str = "Manual quarantine") -> bool:
        """
        Manually quarantine a node.
        
        Args:
            node_id: ID of the node to quarantine
            reason: Reason for quarantine
            
        Returns:
            True if node was quarantined, False if node not found
        """
        node = self._get_node_by_id(node_id)
        if not node:
            logger.error(f"Cannot quarantine node {node_id}: node not found")
            return False
        
        node.health_tracker.health_status = node.health_tracker.health_status.QUARANTINED
        node.health_tracker.quarantine_start_time = time.time()
        node.health_tracker.last_failure_error = reason
        
        # Also record in centralized tracker
        self.failure_tracker.force_quarantine_node(node_id, reason)
        
        logger.warning(f"Node {node_id} ({node.hostname}) manually quarantined: {reason}")
        return True
    
    def force_recover_node(self, node_id: str) -> bool:
        """
        Manually recover a quarantined node.
        
        Args:
            node_id: ID of the node to recover
            
        Returns:
            True if node was recovered, False if node not found or not quarantined
        """
        node = self._get_node_by_id(node_id)
        if not node:
            logger.error(f"Cannot recover node {node_id}: node not found")
            return False
        
        if node.health_tracker.health_status != node.health_tracker.health_status.QUARANTINED:
            logger.warning(f"Node {node_id} is not quarantined (status: {node.health_tracker.health_status.value})")
            return False
        
        node.health_tracker.health_status = node.health_tracker.health_status.HEALTHY
        node.health_tracker.consecutive_failures = 0
        node.health_tracker.quarantine_start_time = None
        
        # Also record in centralized tracker
        self.failure_tracker.force_recover_node(node_id)
        
        logger.info(f"Node {node_id} ({node.hostname}) manually recovered from quarantine")
        return True
    
    def attempt_node_recovery(self) -> List[str]:
        """
        Attempt to recover nodes whose quarantine period has expired.
        
        Returns:
            List of node IDs that were recovered
        """
        recovered_nodes = []
        
        for node in self.nodes:
            if node.health_tracker.attempt_recovery():
                recovered_nodes.append(node.node_id)
                logger.info(f"Node {node.node_id} ({node.hostname}) automatically recovered from quarantine")
        
        return recovered_nodes
