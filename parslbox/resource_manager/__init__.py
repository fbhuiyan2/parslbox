"""
ParslBox Resource Manager

This module provides resource management capabilities for tracking and allocating
nodes, CPUs, and GPUs across different HPC systems.
"""

from .models import NodeResource, JobResourceSpec, ResourceAssignment, NodeAssignment, create_job_resource_spec
from .resource_manager import ResourceManager
from .exceptions import InsufficientResources, JobNotFound, InvalidResourceSpec

__all__ = [
    'ResourceManager',
    'NodeResource',
    'JobResourceSpec',
    'ResourceAssignment',
    'NodeAssignment',  # Backward compatibility alias
    'create_job_resource_spec',
    'InsufficientResources',
    'JobNotFound',
    'InvalidResourceSpec'
]
