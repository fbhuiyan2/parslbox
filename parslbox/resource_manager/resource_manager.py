"""
ParslBox Resource Manager

Main resource manager class that handles resource allocation, tracking,
and scheduling for jobs across HPC systems.
"""

import logging
from typing import List, Dict, Optional, TYPE_CHECKING

from .models import NodeResource, JobResourceSpec, ResourceAssignment, create_job_resource_spec
from .exceptions import InsufficientResources, JobNotFound, InvalidResourceSpec
from .cpu_affinity import CPUAffinityManager

if TYPE_CHECKING:
    from parslbox.configs.base import SystemConfig

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
    
    def __init__(self, system_config: 'SystemConfig'):
        """
        Initialize the resource manager.
        
        Args:
            system_config: System configuration object (Polaris, Sophia, etc.)
        """
        self.system_config = system_config
        self.nodes: List[NodeResource] = []
        self.job_assignments: Dict[int, ResourceAssignment] = {}
        self._backlogged_jobs_set: set = set()  # Track job IDs in backlog
        
        # Initialize CPU affinity manager
        self.cpu_affinity_manager = CPUAffinityManager(
            system_config.WORKER_CPU_AFFINITY,
            system_config.CORES_PER_NODE
        )
        
        # Initialize nodes from system configuration
        self._initialize_nodes()
        
        logger.info(f"Initialized resource manager with {len(self.nodes)} nodes")
    
    def _initialize_nodes(self) -> None:
        """Initialize node resources from system configuration."""
        try:
            total_nodes, total_gpus = self.system_config.detect_resources()
            gpus_per_node = total_gpus//total_nodes #self.system_config.GPUS_PER_NODE
            
            # Get node list from scheduler
            node_hostnames = self._get_node_hostnames(total_nodes)
            
            # Create NodeResource objects
            for i, hostname in enumerate(node_hostnames):
                node_id = f"node-{i}"
                node = NodeResource(
                    node_id=node_id,
                    hostname=hostname,
                    total_gpus=gpus_per_node,
                    total_cores=self.system_config.CORES_PER_NODE
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
                if resource_spec.is_gpu_job() or resource_spec.is_multinode_job():
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
        
        Full-node jobs:
        - Single-node with full occupancy (= 1.0)
        - Single-node GPU job using all GPUs per node
        - All multi-node jobs
        
        Args:
            spec: Job resource specification
            
        Returns:
            True if this is a sub-node job, False if full-node job
        """
        if spec.num_nodes > 1:
            return False  # Multi-node = always full-node
        
        # Single-node job classification
        if spec.is_gpu_job():
            return spec.ngpus < self.system_config.GPUS_PER_NODE
        else:
            return spec.node_occupancy < 1.0
    

    def _assign_subnode_cpu_job(self, spec: JobResourceSpec) -> ResourceAssignment:
        """Assign resources for a sub-node CPU-only job with per-rank allocation."""
        # Calculate number of CPU cores needed based on occupancy
        num_cores_needed = max(1, int(spec.node_occupancy * self.system_config.CORES_PER_NODE))
        
        # Find a node with enough CPU cores
        for node in self.nodes:
            if node.can_fit_cpu_cores(num_cores_needed):
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
        """Assign resources for a sub-node GPU job with per-rank allocation and GPU-CPU affinity awareness."""
        # Calculate cores per GPU for balanced allocation
        cores_per_gpu = self.system_config.CORES_PER_NODE // self.system_config.GPUS_PER_NODE
        
        # Find a node with enough GPUs and CPU cores
        for node in self.nodes:
            if node.can_fit_gpu_job(spec.ngpus, cores_per_gpu):
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
                
                # Assign resources per rank (1 rank per GPU for GPU jobs)
                for rank in range(spec.ngpus):
                    gpu_id = assigned_gpus[rank]
                    rank_cores = cpu_assignments[rank]
                    
                    assignment.assign_resources_to_rank(
                        rank=rank,
                        node_idx=0,
                        gpu_ids=[gpu_id],
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
        - No per-rank resource assignment (MPI handles core distribution)
        - Each node gets all cores, MPI distributes among ranks
        """
        # Find enough completely free nodes
        free_nodes = [node for node in self.nodes if node.is_completely_free()]
        
        if len(free_nodes) < spec.num_nodes:
            available = len(free_nodes)
            requested = spec.num_nodes
            raise InsufficientResources(
                f"Not enough free nodes for full-node CPU job",
                requested={'nodes': requested},
                available={'nodes': available}
            )
        
        # Assign the first N free nodes
        assigned_nodes = free_nodes[:spec.num_nodes]
        node_ids = []
        hostnames = []
        
        for node in assigned_nodes:
            node.assign_fullnode_job(spec.job_id)
            node_ids.append(node.node_id)
            hostnames.append(node.hostname)
        
        # Create ResourceAssignment with empty per-rank allocations
        # (MPI will handle core distribution)
        assignment = ResourceAssignment(
            job_id=spec.job_id,
            node_ids=node_ids,
            hostnames=hostnames,
            node_occupancy=spec.node_occupancy
        )
        
        # For full-node CPU jobs, assign ranks across nodes with empty resource lists
        # MPI will handle the actual core binding
        current_rank = 0
        for node_idx in range(spec.num_nodes):
            for local_rank in range(spec.ranks_per_node):
                assignment.assign_resources_to_rank(
                    rank=current_rank,
                    node_idx=node_idx,
                    gpu_ids=[],  # No per-rank GPU assignment
                    cpu_ids=[]   # No per-rank CPU assignment (MPI handles distribution)
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
        """
        # Find enough completely free nodes
        free_nodes = [node for node in self.nodes if node.is_completely_free()]
        
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
        cores_per_gpu = self.system_config.CORES_PER_NODE // self.system_config.GPUS_PER_NODE
        
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
        
        # Assign resources per rank (1 rank per GPU for GPU jobs)
        current_rank = 0
        for node_idx, node in enumerate(assigned_nodes):
            if spec.is_multinode_job():
                gpus_on_node = self.system_config.GPUS_PER_NODE
            else:
                gpus_on_node = spec.ngpus
            
            # Get the GPU and CPU assignments from the node
            assigned_gpus = node.job_gpu_assignments[spec.job_id]
            assigned_cores = node.job_cpu_assignments[spec.job_id]
            
            # Validate that we have the expected number of GPUs
            if len(assigned_gpus) != gpus_on_node:
                raise RuntimeError(f"GPU assignment mismatch: expected {gpus_on_node}, got {len(assigned_gpus)}")
            
            # Distribute cores among GPUs (assuming equal distribution)
            cores_per_gpu_actual = len(assigned_cores) // len(assigned_gpus)
            
            for local_rank in range(gpus_on_node):                 
                gpu_id = assigned_gpus[local_rank]
                
                # Assign cores for this GPU rank
                # assigned_cores is a flat list. but the method below still maintains affinity because 
                # each gpu gets the same number of cores (equal distribution assumption) and are being looped through linearly
                # in the same order as in assign_gpu_job()
                start_core_idx = local_rank * cores_per_gpu_actual
                end_core_idx = min(start_core_idx + cores_per_gpu_actual, len(assigned_cores))
                rank_cores = assigned_cores[start_core_idx:end_core_idx]
                
                
                assignment.assign_resources_to_rank(
                    rank=current_rank,
                    node_idx=node_idx,
                    gpu_ids=[gpu_id],
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
        """Get list of jobs currently in backlog by querying database."""
        if not self._backlogged_jobs_set:
            return []
        
        # Import here to avoid circular imports
        from parslbox.helpers import database, path_utils
        
        # Get job details from database
        job_ids = list(self._backlogged_jobs_set)
        return database.get_jobs_by_ids(path_utils.DB_FILE, job_ids)
    
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
