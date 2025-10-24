"""
Resource Manager Data Models

Data classes and models for representing resources, job specifications,
and resource assignments.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class NodeResource:
    """
    Represents a compute node and its available resources.
    
    Tracks GPU assignments individually and CPU usage via occupancy.
    """
    node_id: str
    hostname: str
    total_gpus: int
    available_gpu_ids: List[int] = field(default_factory=list)
    cpu_occupancy: float = 0.0  # 0.0 = free, 1.0 = fully occupied
    assigned_jobs: List[int] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize available GPU IDs if not provided."""
        if not self.available_gpu_ids and self.total_gpus > 0:
            self.available_gpu_ids = list(range(self.total_gpus))
    
    def can_fit_gpu_job(self, num_gpus: int) -> bool:
        """Check if this node can accommodate a GPU job."""
        return len(self.available_gpu_ids) >= num_gpus
    
    def can_fit_cpu_job(self, occupancy: float) -> bool:
        """Check if this node can accommodate a CPU-only job with given occupancy."""
        return (self.cpu_occupancy + occupancy) <= 1.0
    
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
        
        # If this was a multi-node job (occupancy = 1.0 and only one job)
        if self.cpu_occupancy == 1.0 and len(self.assigned_jobs) == 1:
            # Free entire node
            self.cpu_occupancy = 0.0
            self.available_gpu_ids = list(range(self.total_gpus))
        else:
            # This is more complex - we need to track what this specific job was using
            # For now, we'll implement a simple approach
            # TODO: Implement more sophisticated resource tracking per job
            logger.warning(f"Simplified resource freeing for job {job_id} on node {self.node_id}")
        
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
    gpus_per_node: int = 0  # 0 = no GPUs needed
    node_occupancy: float = 1.0  # For CPU-only jobs, fraction of node to use
    
    def __post_init__(self):
        """Validate the resource specification."""
        if self.num_nodes < 1:
            raise ValueError("num_nodes must be at least 1")
        
        if self.gpus_per_node < 0:
            raise ValueError("gpus_per_node cannot be negative")
        
        if not 0.0 < self.node_occupancy <= 1.0:
            raise ValueError("node_occupancy must be between 0.0 and 1.0")
        
        # For multi-node jobs, occupancy should be 1.0
        if self.num_nodes > 1 and self.node_occupancy != 1.0:
            logger.warning(f"Multi-node job {self.job_id} should have node_occupancy=1.0, got {self.node_occupancy}")
            self.node_occupancy = 1.0
    
    def is_gpu_job(self) -> bool:
        """Check if this job requires GPUs."""
        return self.gpus_per_node > 0
    
    def is_multinode_job(self) -> bool:
        """Check if this is a multi-node job."""
        return self.num_nodes > 1
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the resource spec."""
        if self.is_multinode_job():
            return f"Multi-node job: {self.num_nodes} nodes"
        elif self.is_gpu_job():
            return f"Single-node GPU job: {self.gpus_per_node} GPUs"
        else:
            return f"Single-node CPU job: {self.node_occupancy:.2f} occupancy"


@dataclass
class NodeAssignment:
    """
    Result of resource assignment for a job.
    
    Contains the specific nodes, hostnames, and GPU assignments for a job.
    """
    job_id: int
    node_ids: List[str]
    hostnames: List[str]
    gpu_assignments: List[List[int]] = field(default_factory=list)  # GPU IDs per node
    
    def __post_init__(self):
        """Validate the assignment."""
        if len(self.node_ids) != len(self.hostnames):
            raise ValueError("node_ids and hostnames must have the same length")
        
        # Initialize empty GPU assignments if not provided
        if not self.gpu_assignments:
            self.gpu_assignments = [[] for _ in self.node_ids]
        
        if len(self.gpu_assignments) != len(self.node_ids):
            raise ValueError("gpu_assignments must have same length as node_ids")
    
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
