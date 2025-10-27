"""
Resource Manager Data Models

Data classes and models for representing resources, job specifications,
and resource assignments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

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
        node_occupancy=job_data.get('node_occupancy', 1.0)
    )


@dataclass
class NodeResource:
    """
    Represents a compute node and its available resources.
    
    Tracks GPU assignments individually and CPU usage via occupancy.
    For single-node jobs, also tracks individual CPU core assignments.
    """
    node_id: str
    hostname: str
    total_gpus: int
    total_cores: int = 0
    available_gpu_ids: List[int] = field(default_factory=list)
    available_core_ids: List[int] = field(default_factory=list)
    cpu_occupancy: float = 0.0  # 0.0 = free, 1.0 = fully occupied
    assigned_jobs: List[int] = field(default_factory=list)
    job_cpu_assignments: Dict[int, List[int]] = field(default_factory=dict)  # job_id -> assigned core IDs
    job_gpu_assignments: Dict[int, List[int]] = field(default_factory=dict)  # job_id -> assigned GPU IDs
    
    def __post_init__(self):
        """Initialize available GPU and CPU core IDs if not provided."""
        if not self.available_gpu_ids and self.total_gpus > 0:
            self.available_gpu_ids = list(range(self.total_gpus))
        
        if not self.available_core_ids and self.total_cores > 0:
            self.available_core_ids = list(range(self.total_cores))
    
    def can_fit_gpu_job(self, num_gpus: int) -> bool:
        """Check if this node can accommodate a GPU job."""
        return len(self.available_gpu_ids) >= num_gpus
    
    def can_fit_cpu_job(self, occupancy: float) -> bool:
        """Check if this node can accommodate a CPU-only job with given occupancy."""
        return (self.cpu_occupancy + occupancy) <= 1.0
    
    def can_fit_cpu_cores(self, num_cores: int) -> bool:
        """Check if this node can accommodate a job requiring specific CPU cores."""
        return len(self.available_core_ids) >= num_cores
    
    def can_fit_multinode_job(self) -> bool:
        """Check if this node is completely free for multi-node job."""
        return len(self.assigned_jobs) == 0 and self.cpu_occupancy == 0.0
    
    def assign_gpu_job(self, job_id: int, num_gpus: int) -> List[int]:
        """
        Assign GPUs to a job and return the assigned GPU IDs.
        
        Args:
            job_id: The job ID to assign resources to
            num_gpus: Number of GPUs to assign
            
        Returns:
            List of assigned GPU IDs
            
        Raises:
            ValueError: If not enough GPUs are available
        """
        if not self.can_fit_gpu_job(num_gpus):
            raise ValueError(f"Cannot assign {num_gpus} GPUs to node {self.node_id}")
        
        # Assign the first N available GPUs
        assigned_gpus = self.available_gpu_ids[:num_gpus]
        self.available_gpu_ids = self.available_gpu_ids[num_gpus:]
        
        # Track the assignment for this job
        self.job_gpu_assignments[job_id] = assigned_gpus
        
        if job_id not in self.assigned_jobs:
            self.assigned_jobs.append(job_id)
        
        logger.debug(f"Assigned GPUs {assigned_gpus} to job {job_id} on node {self.node_id}")
        return assigned_gpus
    
    def assign_cpu_job(self, job_id: int, occupancy: float) -> None:
        """
        Assign CPU resources to a job via occupancy.
        
        Args:
            job_id: The job ID to assign resources to
            occupancy: Fraction of node to occupy (0.0 to 1.0)
            
        Raises:
            ValueError: If occupancy would exceed 1.0
        """
        if not self.can_fit_cpu_job(occupancy):
            raise ValueError(f"Cannot assign occupancy {occupancy} to node {self.node_id}")
        
        self.cpu_occupancy += occupancy
        self.assigned_jobs.append(job_id)
        
        logger.debug(f"Assigned CPU occupancy {occupancy} to job {job_id} on node {self.node_id}")
    
    def assign_cpu_cores(self, job_id: int, num_cores: int) -> List[int]:
        """
        Assign specific CPU cores to a job and return the assigned core IDs.
        
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
        
        logger.debug(f"Assigned CPU cores {assigned_cores} to job {job_id} on node {self.node_id}")
        return assigned_cores
    
    def assign_multinode_job(self, job_id: int) -> None:
        """
        Assign entire node to a multi-node job.
        
        Args:
            job_id: The job ID to assign the entire node to
            
        Raises:
            ValueError: If node is not completely free
        """
        if not self.can_fit_multinode_job():
            raise ValueError(f"Node {self.node_id} is not free for multi-node job")
        
        self.cpu_occupancy = 1.0
        self.assigned_jobs.append(job_id)
        
        logger.debug(f"Assigned entire node {self.node_id} to multi-node job {job_id}")
    
    def free_job(self, job_id: int) -> None:
        """
        Free all resources assigned to a job.
        
        Args:
            job_id: The job ID to free resources for
        """
        if job_id not in self.assigned_jobs:
            logger.warning(f"Job {job_id} not found on node {self.node_id}")
            return
        
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
        
        # If this was a multi-node job (occupancy = 1.0 and only one job)
        if self.cpu_occupancy == 1.0 and len(self.assigned_jobs) == 1:
            # Free entire node
            self.cpu_occupancy = 0.0
            self.available_gpu_ids = list(range(self.total_gpus))
            self.available_core_ids = list(range(self.total_cores))
            self.job_cpu_assignments.clear()
            self.job_gpu_assignments.clear()
        
        self.assigned_jobs.remove(job_id)
        logger.debug(f"Freed resources for job {job_id} on node {self.node_id}")
    
    def get_status(self) -> Dict:
        """Get current status of the node."""
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'total_gpus': self.total_gpus,
            'available_gpus': len(self.available_gpu_ids),
            'cpu_occupancy': self.cpu_occupancy,
            'assigned_jobs': self.assigned_jobs.copy(),
            'is_free': len(self.assigned_jobs) == 0
        }


@dataclass
class JobResourceSpec:
    """
    Specification for job resource requirements.
    """
    job_id: int
    num_nodes: int = 1
    ngpus: int = 0  # Total GPUs needed (only relevant for single-node jobs)
    node_occupancy: float = 1.0  # For CPU-only jobs, fraction of node to use
    
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
        """Check if this job requires GPUs (only meaningful for single-node jobs)."""
        return self.num_nodes == 1 and self.ngpus > 0
    
    def is_multinode_job(self) -> bool:
        """Check if this is a multi-node job."""
        return self.num_nodes > 1
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the resource spec."""
        if self.is_multinode_job():
            return f"Multi-node job: {self.num_nodes} nodes"
        elif self.is_gpu_job():
            return f"Single-node GPU job: {self.ngpus} GPUs"
        else:
            return f"Single-node CPU job: {self.node_occupancy:.2f} occupancy"


@dataclass
class NodeAssignment:
    """
    Result of resource assignment for a job.
    
    Contains the specific nodes, hostnames, GPU assignments, and CPU core assignments for a job.
    """
    job_id: int
    node_ids: List[str]
    hostnames: List[str]
    gpu_assignments: List[List[int]] = field(default_factory=list)  # GPU IDs per node
    cpu_assignments: List[List[int]] = field(default_factory=list)  # CPU core IDs per node
    node_occupancy: float = 1.0  # Node occupancy for CPU-only jobs
    
    def __post_init__(self):
        """Validate the assignment."""
        if len(self.node_ids) != len(self.hostnames):
            raise ValueError("node_ids and hostnames must have the same length")
        
        # Initialize empty GPU assignments if not provided
        if not self.gpu_assignments:
            self.gpu_assignments = [[] for _ in self.node_ids]
        
        if len(self.gpu_assignments) != len(self.node_ids):
            raise ValueError("gpu_assignments must have same length as node_ids")
        
        # Initialize empty CPU assignments if not provided
        if not self.cpu_assignments:
            self.cpu_assignments = [[] for _ in self.node_ids]
        
        if len(self.cpu_assignments) != len(self.node_ids):
            raise ValueError("cpu_assignments must have same length as node_ids")
    
    def get_total_gpus(self) -> int:
        """Get total number of GPUs assigned."""
        return sum(len(gpu_list) for gpu_list in self.gpu_assignments)
    
    def is_single_node(self) -> bool:
        """Check if this is a single-node assignment."""
        return len(self.node_ids) == 1
    
    def get_env_vars(self) -> Dict[str, str]:
        """
        Generate environment variables for this assignment.
        
        Returns:
            Dictionary of environment variable name -> value
        """
        env_vars = {}
        
        # For single-node jobs with GPUs, set CUDA_VISIBLE_DEVICES
        if self.is_single_node() and self.gpu_assignments[0]:
            gpu_ids = ",".join(map(str, self.gpu_assignments[0]))
            env_vars["CUDA_VISIBLE_DEVICES"] = gpu_ids
            # For Intel GPUs (future support)
            env_vars["ZE_AFFINITY_MASK"] = gpu_ids
        
        return env_vars
    
    def get_mpi_hostlist(self) -> str:
        """Get comma-separated hostlist for MPI commands."""
        return ",".join(self.hostnames)
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the assignment."""
        if self.is_single_node():
            if self.gpu_assignments[0]:
                gpu_str = f" (GPUs: {self.gpu_assignments[0]})"
            else:
                gpu_str = " (CPU-only)"
            return f"Node: {self.hostnames[0]}{gpu_str}"
        else:
            total_gpus = self.get_total_gpus()
            gpu_str = f" ({total_gpus} total GPUs)" if total_gpus > 0 else ""
            return f"Nodes: {len(self.node_ids)} nodes{gpu_str}"
