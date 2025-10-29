"""
ParslBox Resource Manager

Main resource manager class that handles resource allocation, tracking,
and scheduling for jobs across HPC systems.
"""

import logging
import queue
from typing import List, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from .models import NodeResource, JobResourceSpec, NodeAssignment, create_job_resource_spec
from .exceptions import InsufficientResources, JobNotFound, InvalidResourceSpec

if TYPE_CHECKING:
    from parslbox.configs.base import SystemConfig

logger = logging.getLogger(__name__)


@dataclass(order=True)
class PrioritizedJob:
    """Job with priority for backlog queue."""
    priority: int
    job: dict = field(compare=False)


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
        self.job_assignments: Dict[int, NodeAssignment] = {}
        self._backlog_queue: queue.PriorityQueue[PrioritizedJob] = queue.PriorityQueue()
        self._queued_jobs: set = set()  # Track job IDs in backlog to prevent duplicates
        
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
    
    def assign_resources(self, job: dict) -> NodeAssignment:
        """
        Assign resources to a job based on its metadata.
        
        Args:
            job: Job dictionary containing metadata and resource requirements
            
        Returns:
            NodeAssignment with allocated resources
            
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
            
            # Try to assign resources
            if resource_spec.is_multinode_job():
                assignment = self._assign_multinode_job(resource_spec)
            elif resource_spec.is_gpu_job():
                assignment = self._assign_single_node_gpu_job(resource_spec)
            else:
                assignment = self._assign_single_node_cpu_job(resource_spec)
            
            # Store the assignment
            self.job_assignments[resource_spec.job_id] = assignment
            
            logger.info(f"Assigned resources to job {resource_spec.job_id}: {assignment.get_summary()}")
            return assignment
            
        except InsufficientResources:
            # Add to backlog if resources not available and not already queued
            if resource_spec.job_id not in self._queued_jobs:
                self._queued_jobs.add(resource_spec.job_id)
                priority = resource_spec.num_nodes  # Higher node count = higher priority
                self._backlog_queue.put(PrioritizedJob(priority, job))
                logger.info(f"Job {resource_spec.job_id} added to backlog")
            else:
                logger.info(f"Job {resource_spec.job_id} already in backlog, skipping duplicate")
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
    
    def _assign_multinode_job(self, spec: JobResourceSpec) -> NodeAssignment:
        """Assign resources for a multi-node job."""
        # Find enough free nodes
        free_nodes = [node for node in self.nodes if node.can_fit_multinode_job()]
        
        if len(free_nodes) < spec.num_nodes:
            available = len(free_nodes)
            requested = spec.num_nodes
            raise InsufficientResources(
                f"Not enough free nodes for multi-node job",
                requested={'nodes': requested},
                available={'nodes': available}
            )
        
        # Assign the first N free nodes
        assigned_nodes = free_nodes[:spec.num_nodes]
        node_ids = []
        hostnames = []
        
        for node in assigned_nodes:
            node.assign_multinode_job(spec.job_id)
            node_ids.append(node.node_id)
            hostnames.append(node.hostname)
        
        return NodeAssignment(
            job_id=spec.job_id,
            node_ids=node_ids,
            hostnames=hostnames,
            gpu_assignments=[[] for _ in range(spec.num_nodes)],  # No specific GPU assignments for multi-node
            node_occupancy=spec.node_occupancy
        )
    
    def _assign_single_node_gpu_job(self, spec: JobResourceSpec) -> NodeAssignment:
        """Assign resources for a single-node GPU job."""
        # For GPU jobs, we typically assign cores equal to the number of GPUs
        # or based on the system's cores per GPU ratio
        cores_per_gpu = self.system_config.CORES_PER_NODE // self.system_config.GPUS_PER_NODE
        num_cores_needed = spec.ngpus * cores_per_gpu
        
        # Find a node with enough GPUs and CPU cores
        for node in self.nodes:
            if node.can_fit_gpu_job(spec.ngpus) and node.can_fit_cpu_cores(num_cores_needed):
                assigned_gpus = node.assign_gpu_job(spec.job_id, spec.ngpus)
                assigned_cores = node.assign_cpu_cores(spec.job_id, num_cores_needed)
                
                return NodeAssignment(
                    job_id=spec.job_id,
                    node_ids=[node.node_id],
                    hostnames=[node.hostname],
                    gpu_assignments=[assigned_gpus],
                    cpu_assignments=[assigned_cores],
                    node_occupancy=spec.node_occupancy
                )
        
        # No suitable node found
        available_gpus = max(len(node.available_gpu_ids) for node in self.nodes)
        available_cores = max(len(node.available_core_ids) for node in self.nodes)
        raise InsufficientResources(
            f"Not enough GPUs or CPU cores available for single-node job",
            requested={'gpus': spec.ngpus, 'cores': num_cores_needed},
            available={'gpus': available_gpus, 'cores': available_cores}
        )
    
    def _assign_single_node_cpu_job(self, spec: JobResourceSpec) -> NodeAssignment:
        """Assign resources for a single-node CPU-only job."""
        # Calculate number of CPU cores needed based on occupancy
        num_cores_needed = max(1, int(spec.node_occupancy * self.system_config.CORES_PER_NODE))
        
        # Find a node with enough CPU cores
        for node in self.nodes:
            if node.can_fit_cpu_cores(num_cores_needed):
                assigned_cores = node.assign_cpu_cores(spec.job_id, num_cores_needed)
                
                return NodeAssignment(
                    job_id=spec.job_id,
                    node_ids=[node.node_id],
                    hostnames=[node.hostname],
                    gpu_assignments=[[]],  # No GPUs
                    cpu_assignments=[assigned_cores],  # Assigned CPU cores
                    node_occupancy=spec.node_occupancy
                )
        
        # No suitable node found
        max_available_cores = max(len(node.available_core_ids) for node in self.nodes)
        raise InsufficientResources(
            f"Not enough CPU cores available for single-node job",
            requested={'cores': num_cores_needed},
            available={'cores': max_available_cores}
        )
    
    def free_resources(self, job_id: int) -> None:
        """
        Free resources assigned to a job.
        
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
    
    def schedule_backlog(self) -> List[dict]:
        """
        Attempt to schedule jobs from the backlog.
        
        Returns:
            List of job metadata for successfully scheduled jobs
        """
        scheduled_jobs = []
        
        # Try to schedule jobs from backlog
        while not self._backlog_queue.empty():
            try:
                prioritized_job = self._backlog_queue.get(block=False)
                job_id = prioritized_job.job['job_id']
                
                # Remove from queued set since we're processing it
                self._queued_jobs.discard(job_id)
                
                # Skip if already has assignment (handles existing duplicates)
                if job_id in self.job_assignments:
                    logger.info(f"Job {job_id} already has resources, skipping duplicate")
                    continue
                
                assignment = self.assign_resources(prioritized_job.job)
                scheduled_jobs.append(prioritized_job.job)  # Return full job metadata
                logger.info(f"Scheduled backlogged job {job_id}")
                
            except InsufficientResources:
                # Put the job back if it still can't be scheduled
                self._queued_jobs.add(prioritized_job.job['job_id'])  # Re-add to tracking set
                self._backlog_queue.put(prioritized_job)
                break
            except queue.Empty:
                break
        
        if scheduled_jobs:
            logger.info(f"Scheduled {len(scheduled_jobs)} jobs from backlog")
        
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
            'backlogged_jobs': self._backlog_queue.qsize(),
            'nodes': [node.get_status() for node in self.nodes]
        }
    
    def get_job_assignment(self, job_id: int) -> Optional[NodeAssignment]:
        """
        Get resource assignment for a specific job.
        
        Args:
            job_id: The job ID to get assignment for
            
        Returns:
            NodeAssignment if job has resources assigned, None otherwise
        """
        return self.job_assignments.get(job_id)
    
    def list_active_jobs(self) -> List[int]:
        """Get list of job IDs with active resource assignments."""
        return list(self.job_assignments.keys())
    
    def get_backlog_size(self) -> int:
        """Get number of jobs in the backlog queue."""
        return self._backlog_queue.qsize()
