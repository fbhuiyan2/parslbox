"""
Resource Manager Data Models

Data classes and models for representing resources, job specifications,
and resource assignments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .cpu_affinity import CPUAffinityManager

from .node_failure_tracker import NodeHealthTracker

logger = logging.getLogger(__name__)


def create_job_resource_spec(job_data: dict) -> 'JobResourceSpec':
    """
    Create JobResourceSpec directly from job database data.
    
    Args:
        job_data: Dictionary containing job data from database
        
    Returns:
        JobResourceSpec object
    """
    return JobResourceSpec(
        job_id=job_data['job_id'],
        num_nodes=job_data.get('num_nodes', 1),
        ngpus=job_data.get('ngpus', 0),
        node_occupancy=job_data.get('node_occupancy', 1.0),
        ranks_per_node=job_data.get('ranks_per_node', 1)
    )


@dataclass
class NodeResource:
    """
    Represents a compute node and its available resources.
    
    Tracks GPU assignments individually and CPU usage via occupancy.
    For single-node jobs, also tracks individual CPU core assignments.
    Includes health tracking for fault tolerance.
    """
    node_id: str
    hostname: str
    total_gpus: int
    total_cores: int = 0
    available_gpu_ids: List[int] = field(default_factory=list)
    available_core_ids: List[int] = field(default_factory=list)
    excluded_cores: List[int] = None
    cpu_occupancy: float = 0.0  # 0.0 = free, 1.0 = fully occupied
    assigned_jobs: List[int] = field(default_factory=list)
    job_cpu_assignments: Dict[int, List[int]] = field(default_factory=dict)  # job_id -> assigned core IDs
    job_gpu_assignments: Dict[int, List[int]] = field(default_factory=dict)  # job_id -> assigned GPU IDs
    health_tracker: NodeHealthTracker = field(default_factory=NodeHealthTracker)  # Health tracking for fault tolerance
    
    def _get_available_cores(self) -> List[int]:
        """Get list of available cores excluding any excluded cores."""
        if self.total_cores <= 0:
            return []
        
        all_cores = list(range(self.total_cores))
        
        if not self.excluded_cores:
            return all_cores
        
        # Filter out excluded cores, with validation
        valid_excluded = [core for core in self.excluded_cores 
                         if 0 <= core < self.total_cores]
        
        if len(valid_excluded) != len(self.excluded_cores):
            invalid_cores = [core for core in self.excluded_cores 
                            if core < 0 or core >= self.total_cores]
            logger.warning(f"Invalid excluded cores {invalid_cores} for node {self.node_id} "
                          f"(valid range: 0-{self.total_cores-1})")
        
        return [core for core in all_cores if core not in valid_excluded]

    def __post_init__(self):
        """Initialize available GPU and CPU core IDs if not provided."""
        if not self.available_gpu_ids and self.total_gpus > 0:
            self.available_gpu_ids = list(range(self.total_gpus))
        
        if not self.available_core_ids and self.total_cores > 0:
            self.available_core_ids = self._get_available_cores()
    
    def can_fit_gpu_job(self, num_gpus: int, cores_per_gpu: int = None) -> bool:
        """Check if this node can accommodate a GPU job with optional CPU requirements."""
        gpu_available = len(self.available_gpu_ids) >= num_gpus
        if cores_per_gpu is not None:
            total_cores_needed = num_gpus * cores_per_gpu
            cpu_available = len(self.available_core_ids) >= total_cores_needed
            return gpu_available and cpu_available
        return gpu_available
    
    def can_fit_cpu_cores(self, num_cores: int) -> bool:
        """Check if this node can accommodate a sub-node job requiring specific CPU cores."""
        return len(self.available_core_ids) >= num_cores and self.cpu_occupancy < 1.0
    
    def is_completely_free(self) -> bool:
        """Check if this node is completely free for full-node job."""
        return len(self.assigned_jobs) == 0 and self.cpu_occupancy == 0.0
    
    def _update_cpu_occupancy(self) -> None:
        """
        Update CPU occupancy based on currently available cores.
        
        Note: This is primarily for tracking/monitoring purposes. 
        Single-node jobs use individual core assignments, but we maintain
        occupancy for consistency with multi-node jobs and status reporting.
        """
        if self.total_cores == 0:
            # No cores available = fully occupied (can't run anything)
            self.cpu_occupancy = 1.0
        else:
            # Calculate effective cores (total - excluded)
            excluded_count = len(self.excluded_cores) if self.excluded_cores else 0
            effective_cores = self.total_cores - excluded_count
            
            if effective_cores == 0:
                self.cpu_occupancy = 1.0
            else:
                used_cores = effective_cores - len(self.available_core_ids)
                self.cpu_occupancy = used_cores / effective_cores
    

    def assign_cpu_job(self, job_id: int, num_cores: int) -> List[int]:
        """
        Assign CPU cores for a CPU-only job and return the assigned core IDs.
        
        Note: This method is specifically for CPU-only jobs. GPU jobs should use assign_gpu_job().
        
        Args:
            job_id: The job ID to assign resources to
            num_cores: Number of CPU cores to assign
            
        Returns:
            List of assigned CPU core IDs
            
        Raises:
            ValueError: If not enough CPU cores are available
        """
        if not self.can_fit_cpu_cores(num_cores):
            raise ValueError(f"Cannot assign {num_cores} CPU cores to node {self.node_id}")
        
        # Assign the first N available cores
        assigned_cores = self.available_core_ids[:num_cores]
        self.available_core_ids = self.available_core_ids[num_cores:]
        
        # Track the assignment for this job
        self.job_cpu_assignments[job_id] = assigned_cores
        
        if job_id not in self.assigned_jobs:
            self.assigned_jobs.append(job_id)
        
        # Update CPU occupancy based on actual core usage (for tracking purposes)
        self._update_cpu_occupancy()
        
        logger.debug(f"Assigned CPU cores {assigned_cores} to job {job_id} on node {self.node_id}")
        return assigned_cores
    

    def assign_gpu_job(self, job_id: int, num_gpus: int, cores_per_gpu: int = None, affinity_manager: 'CPUAffinityManager' = None):
        """
        Assign GPUs and CPU cores to a job with affinity preference.
        
        Key assumptions:
        - Each GPU rank gets exactly cores_per_gpu CPU cores
        - Affinity is preferred but not required (falls back to any available cores)
        - GPU-CPU affinity groups are non-overlapping in well-designed HPC systems
        - cores_per_gpu = CORES_PER_NODE // GPUS_PER_NODE for balanced allocation
        - Sequential assignment: GPU 0 gets affinity cores first, then GPU 1, etc.
        
        Args:
            job_id: The job ID to assign resources to
            num_gpus: Number of GPUs to assign
            cores_per_gpu: Number of CPU cores per GPU (if None, only assigns GPUs)
            affinity_manager: CPU affinity manager for intelligent core selection
            
        Returns:
            If cores_per_gpu is None: List of assigned GPU IDs
            If cores_per_gpu is provided: Tuple of (assigned_gpu_ids, assigned_cpu_cores_per_rank)
            
        Raises:
            ValueError: If not enough GPUs or CPU cores are available
        """
        # Check if we can fit the job (including CPU requirements if specified)
        if not self.can_fit_gpu_job(num_gpus, cores_per_gpu):
            gpu_msg = f"Cannot assign {num_gpus} GPUs"
            if cores_per_gpu is not None:
                cpu_msg = f" and {num_gpus * cores_per_gpu} CPU cores"
                raise ValueError(f"{gpu_msg}{cpu_msg} to node {self.node_id}")
            else:
                raise ValueError(f"{gpu_msg} to node {self.node_id}")
        
        # Assign the first N available GPUs
        assigned_gpus = self.available_gpu_ids[:num_gpus]
        self.available_gpu_ids = self.available_gpu_ids[num_gpus:]
        
        # Track GPU assignment for this job
        self.job_gpu_assignments[job_id] = assigned_gpus
        
        if job_id not in self.assigned_jobs:
            self.assigned_jobs.append(job_id)
        
        logger.debug(f"Assigned GPUs {assigned_gpus} to job {job_id} on node {self.node_id}")
        
        # If no CPU assignment needed, return just GPU IDs
        if cores_per_gpu is None:
            return assigned_gpus
        
        # Assign CPU cores for each GPU with affinity preference
        cpu_assignments = []
        for gpu_id in assigned_gpus:
            # Try affinity cores first if affinity manager is provided
            if affinity_manager and affinity_manager.has_affinity:
                affinity_cores = affinity_manager.get_cores_for_gpu(gpu_id, self.available_core_ids)
                
                if len(affinity_cores) >= cores_per_gpu:
                    # Use affinity cores
                    rank_cores = affinity_cores[:cores_per_gpu]
                    logger.debug(f"Assigned GPU {gpu_id} affinity cores {rank_cores} to job {job_id}")
                else:
                    # Fall back to any available cores
                    rank_cores = self.available_core_ids[:cores_per_gpu]
                    logger.debug(f"No sufficient affinity cores for GPU {gpu_id}, using fallback cores {rank_cores}")
            else:
                # No affinity manager or no affinity configured - use any available cores
                rank_cores = self.available_core_ids[:cores_per_gpu]
                logger.debug(f"Assigned cores {rank_cores} to GPU {gpu_id} (no affinity)")
            
            # Remove assigned cores from available pool
            for core in rank_cores:
                if core in self.available_core_ids:
                    self.available_core_ids.remove(core)
            
            cpu_assignments.append(rank_cores)
        
        # Update CPU tracking for this job
        all_assigned_cores = [core for rank_cores in cpu_assignments for core in rank_cores]
        self.job_cpu_assignments[job_id] = all_assigned_cores
        self._update_cpu_occupancy()
        
        logger.debug(f"Assigned {len(all_assigned_cores)} total CPU cores to job {job_id} on node {self.node_id}")
        
        return assigned_gpus, cpu_assignments
    
    
    
    def assign_fullnode_cpu_job(self, job_id: int) -> None:
        """
        Assign entire node to a full-node CPU-only job.
        
        For CPU-only jobs that take the entire node, mark node resources as fully used.
        
        Args:
            job_id: The job ID to assign the entire node to
            
        Raises:
            ValueError: If node is not completely free
        """
        if not self.is_completely_free():
            raise ValueError(f"Node {self.node_id} is not free for full-node CPU job")
        
        # For full-node CPU jobs, mark node resources as fully used
        self.cpu_occupancy = 1.0
        self.available_core_ids = []  # No cores available for other jobs
        self.available_gpu_ids = []   # No GPUs available for other jobs
        self.assigned_jobs.append(job_id)
        
        logger.debug(f"Assigned entire node {self.node_id} to full-node CPU job {job_id}")
    
    def free_job(self, job_id: int) -> None:
        """
        Free all resources assigned to a job.
        
        Args:
            job_id: The job ID to free resources for
        """
        if job_id not in self.assigned_jobs:
            logger.warning(f"Job {job_id} not found on node {self.node_id}")
            return
        
        # Check if this was a multi-node job (occupancy = 1.0 and only one job)
        is_multinode_job = self.cpu_occupancy == 1.0 and len(self.assigned_jobs) == 1
        
        # Free CPU cores if this job had specific core assignments
        if job_id in self.job_cpu_assignments:
            freed_cores = self.job_cpu_assignments[job_id]
            self.available_core_ids.extend(freed_cores)
            self.available_core_ids.sort()  # Keep cores sorted for consistent assignment
            del self.job_cpu_assignments[job_id]
            logger.debug(f"Freed CPU cores {freed_cores} for job {job_id} on node {self.node_id}")
        
        # Free GPUs if this job had specific GPU assignments
        if job_id in self.job_gpu_assignments:
            freed_gpus = self.job_gpu_assignments[job_id]
            self.available_gpu_ids.extend(freed_gpus)
            self.available_gpu_ids.sort()  # Keep GPUs sorted for consistent assignment
            del self.job_gpu_assignments[job_id]
            logger.debug(f"Freed GPUs {freed_gpus} for job {job_id} on node {self.node_id}")
        
        
        if is_multinode_job:
            # For multi-node jobs, reset entire node
            self.available_gpu_ids = list(range(self.total_gpus))
            self.available_core_ids = self._get_available_cores()
            self.job_cpu_assignments.clear()
            self.job_gpu_assignments.clear()
            self.cpu_occupancy = 0.0
        else:
            # For single-node jobs, update occupancy based on actual core usage
            self._update_cpu_occupancy()
        
        # Remove job from assigned jobs list
        self.assigned_jobs.remove(job_id)
        logger.debug(f"Freed resources for job {job_id} on node {self.node_id}")
    
    def get_status(self) -> Dict:
        """Get current status of the node including health information."""
        status = {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'total_gpus': self.total_gpus,
            'available_gpus': len(self.available_gpu_ids),
            'cpu_occupancy': self.cpu_occupancy,
            'assigned_jobs': self.assigned_jobs.copy(),
            'is_free': len(self.assigned_jobs) == 0,
            'can_accept_jobs': self.health_tracker.can_accept_jobs()
        }
        
        # Add health information
        status.update(self.health_tracker.get_status_summary())
        
        return status


@dataclass
class JobResourceSpec:
    """
    Specification for job resource requirements.
    """
    job_id: int
    num_nodes: int = 1
    ngpus: int = 0  # Total GPUs needed (only relevant for single-node jobs)
    node_occupancy: float = 1.0  # For CPU-only jobs, fraction of node to use
    ranks_per_node: int = 1  # MPI ranks per node (for CPU jobs)
    
    def __post_init__(self):
        """Validate the resource specification."""
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be at least 1")
        
        if self.ngpus < 0:
            raise ValueError("ngpus cannot be negative")
        
        if not 0.0 < self.node_occupancy <= 1.0:
            raise ValueError("node_occupancy must be between 0.0 and 1.0")
        
        # For multi-node jobs, occupancy should be 1.0
        if self.num_nodes > 1 and self.node_occupancy != 1.0:
            logger.warning(f"Multi-node job {self.job_id} should have node_occupancy=1.0, got {self.node_occupancy}")
            self.node_occupancy = 1.0
    
    def is_gpu_job(self) -> bool:
        """Check if this job requires GPUs"""
        return self.ngpus > 0
    
    def is_multinode_job(self) -> bool:
        """Check if this is a multi-node job."""
        return self.num_nodes > 1
    
    def get_total_ranks(self) -> int:
        """Calculate total number of MPI ranks for this job."""
        if self.is_gpu_job():
            # GPU jobs: 1 rank per GPU
            return self.ngpus
        else:
            # CPU jobs: use ranks_per_node
            return self.num_nodes * self.ranks_per_node
    
    def detect_job_type(self, system_config) -> str:
        """
        Detect job type for appropriate MPI command generation.
        
        Args:
            system_config: System configuration object
            
        Returns:
            Job type string:
            - "subnode_cpu": Sub-node CPU job (per-rank core assignment)
            - "subnode_gpu": Sub-node GPU job (per-rank GPU + core assignment)  
            - "fullnode_cpu": Full-node CPU job (MPI handles core distribution)
            - "fullnode_gpu": Full-node GPU job (per-rank GPU + core assignment)
        """
        # Check if this is a GPU job
        if self.is_gpu_job():
            # GPU jobs: check if sub-node or full-node
            if self.num_nodes == 1 and self.ngpus < system_config.GPUS_PER_NODE:
                return "subnode_gpu"
            else:
                return "fullnode_gpu"
        else:
            # CPU-only jobs: check if sub-node or full-node
            if self.num_nodes == 1 and self.node_occupancy < 1.0:
                return "subnode_cpu"
            else:
                return "fullnode_cpu"
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the resource spec."""
        if self.is_multinode_job():
            return f"Multi-node job: {self.num_nodes} nodes and {self.ngpus} GPUs"
        elif self.num_nodes == 1 and self.is_gpu_job():
            return f"Single-node GPU job: {self.ngpus} GPUs"
        else:
            return f"Single-node CPU job: {self.node_occupancy:.2f} occupancy"


@dataclass
class ResourceAssignment:
    """
    Result of resource assignment for a job.
    
    Contains the specific nodes, hostnames, and per-rank GPU/CPU assignments for a job.
    The new structure supports per-rank allocation:
    - gpu_assignments[node_idx][rank] = [gpu_ids] 
    - cpu_assignments[node_idx][rank] = [cpu_core_ids]
    """
    job_id: int
    node_ids: List[str]
    hostnames: List[str]
    gpu_assignments: List[Dict[int, List[int]]] = field(default_factory=list)  # [node][rank] -> GPU IDs
    cpu_assignments: List[Dict[int, List[int]]] = field(default_factory=list)  # [node][rank] -> CPU core IDs
    node_occupancy: float = 1.0  # Node occupancy for CPU-only jobs
    
    def __post_init__(self):
        """Validate the assignment."""
        if len(self.node_ids) != len(self.hostnames):
            raise ValueError("node_ids and hostnames must have the same length")
        
        # Initialize empty GPU assignments if not provided
        if not self.gpu_assignments:
            self.gpu_assignments = [{} for _ in self.node_ids]
        
        if len(self.gpu_assignments) != len(self.node_ids):
            raise ValueError("gpu_assignments must have same length as node_ids")
        
        # Initialize empty CPU assignments if not provided
        if not self.cpu_assignments:
            self.cpu_assignments = [{} for _ in self.node_ids]
        
        if len(self.cpu_assignments) != len(self.node_ids):
            raise ValueError("cpu_assignments must have same length as node_ids")
    
    def get_total_gpus(self) -> int:
        """Get total number of GPUs assigned."""
        total = 0
        for node_assignments in self.gpu_assignments:
            for rank, gpu_list in node_assignments.items():
                total += len(gpu_list)
        return total
    
    def is_single_node(self) -> bool:
        """Check if this is a single-node assignment."""
        return len(self.node_ids) == 1
    
    def get_env_vars(self, mpi_backend: Optional[str] = None) -> Dict[str, str]:
        """
        Generate environment variables for this assignment.

        Args:
            mpi_backend: MPI backend name (e.g., "srun"). When "srun",
                GPU env vars are skipped because SLURM handles GPU
                binding natively via --gpus-per-node/--gpus-per-task.

        Returns:
            Dictionary of environment variable name -> value
        """
        env_vars = {}

        # Skip GPU env vars for srun — SLURM handles GPU binding natively
        if mpi_backend == "srun":
            return env_vars

        # For single-node jobs with GPUs, set CUDA_VISIBLE_DEVICES
        if self.is_single_node() and self.gpu_assignments[0]:
            all_gpu_ids = []
            for rank, gpu_list in self.gpu_assignments[0].items():
                all_gpu_ids.extend(gpu_list)

            if all_gpu_ids:
                gpu_ids_str = ",".join(map(str, sorted(set(all_gpu_ids))))
                env_vars["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
                env_vars["ZE_AFFINITY_MASK"] = gpu_ids_str

        return env_vars
    
    def get_mpi_hostlist(self) -> str:
        """Get comma-separated hostlist for MPI commands."""
        return ",".join(self.hostnames)
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the assignment."""
        if self.is_single_node():
            if self.gpu_assignments[0]:
                # Collect all GPU IDs from all ranks on the first node
                all_gpu_ids = []
                for rank, gpu_list in self.gpu_assignments[0].items():
                    all_gpu_ids.extend(gpu_list)
                gpu_str = f" (GPUs: {sorted(set(all_gpu_ids))})"
            else:
                gpu_str = " (CPU-only)"
            return f"Node: {self.hostnames[0]}{gpu_str}"
        else:
            total_gpus = self.get_total_gpus()
            gpu_str = f" ({total_gpus} total GPUs)" if total_gpus > 0 else ""
            return f"Nodes: {len(self.node_ids)} nodes{gpu_str}"
    
    def get_gpu_assignments_for_rank(self, rank: int) -> List[int]:
        """
        Get GPU assignments for a specific rank across all nodes.
        
        Args:
            rank: The MPI rank to get GPU assignments for
            
        Returns:
            List of GPU IDs assigned to the rank
        """
        gpu_ids = []
        for node_assignments in self.gpu_assignments:
            if rank in node_assignments:
                gpu_ids.extend(node_assignments[rank])
        return gpu_ids
    
    def get_cpu_assignments_for_rank(self, rank: int) -> List[int]:
        """
        Get CPU core assignments for a specific rank across all nodes.
        
        Args:
            rank: The MPI rank to get CPU assignments for
            
        Returns:
            List of CPU core IDs assigned to the rank
        """
        cpu_ids = []
        for node_assignments in self.cpu_assignments:
            if rank in node_assignments:
                cpu_ids.extend(node_assignments[rank])
        return cpu_ids
    
    def get_all_ranks(self) -> List[int]:
        """
        Get all MPI ranks that have resource assignments.
        
        Returns:
            Sorted list of all rank numbers
        """
        all_ranks = set()
        for node_assignments in self.gpu_assignments:
            all_ranks.update(node_assignments.keys())
        for node_assignments in self.cpu_assignments:
            all_ranks.update(node_assignments.keys())
        return sorted(all_ranks)
    
    def get_ranks_for_node(self, node_idx: int) -> List[int]:
        """
        Get all ranks assigned to a specific node.
        
        Args:
            node_idx: Index of the node (0-based)
            
        Returns:
            Sorted list of rank numbers for the node
        """
        if node_idx >= len(self.node_ids):
            return []
        
        ranks = set()
        if node_idx < len(self.gpu_assignments):
            ranks.update(self.gpu_assignments[node_idx].keys())
        if node_idx < len(self.cpu_assignments):
            ranks.update(self.cpu_assignments[node_idx].keys())
        return sorted(ranks)
    
    def assign_resources_to_rank(self, rank: int, node_idx: int, gpu_ids: List[int] = None, cpu_ids: List[int] = None) -> None:
        """
        Assign resources to a specific rank on a specific node.
        
        Args:
            rank: The MPI rank to assign resources to
            node_idx: Index of the node (0-based)
            gpu_ids: List of GPU IDs to assign (optional)
            cpu_ids: List of CPU core IDs to assign (optional)
        """
        if node_idx >= len(self.node_ids):
            raise ValueError(f"Node index {node_idx} out of range")
        
        # Ensure we have enough assignment dictionaries
        while len(self.gpu_assignments) <= node_idx:
            self.gpu_assignments.append({})
        while len(self.cpu_assignments) <= node_idx:
            self.cpu_assignments.append({})
        
        # Assign GPU resources
        if gpu_ids is not None:
            self.gpu_assignments[node_idx][rank] = gpu_ids
        
        # Assign CPU resources
        if cpu_ids is not None:
            self.cpu_assignments[node_idx][rank] = cpu_ids


# Backward compatibility alias
NodeAssignment = ResourceAssignment
