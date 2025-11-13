"""
Base System Configuration for ParslBox

This module defines the abstract base class for system configurations,
providing a consistent interface for accessing system specifications
and generating Parsl configurations.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from parsl.config import Config

if TYPE_CHECKING:
    from parslbox.resource_manager import ResourceManager


class SystemConfig(ABC):
    """
    Abstract base class for system configurations.
    
    Each system (e.g., Polaris, Sophia) should inherit from this class
    and implement the required methods and class attributes.
    """
    
    # System specifications (must be defined in subclasses)
    CORES_PER_NODE: int
    GPUS_PER_NODE: int
    SCHEDULER: str
    MPI_CMD_TO_USE: str
    MAX_WORKERS_PER_NODE: int
    WORKER_CPU_AFFINITY: Optional[str] = None
    
    @abstractmethod
    def detect_resources(self) -> tuple[int, int]:
        """
        Detect available nodes and GPUs for this system.
        
        Returns:
            tuple[int, int]: A tuple of (nodes, total_gpus)
        """
        pass
    
    @abstractmethod
    def get_config(self, run_dir: Path, retries: int = 0) -> Config:
        """
        Generate Parsl configuration for this system.
        
        Args:
            run_dir (Path): The path for Parsl's run directory
            retries (int): The number of retries for failed Parsl apps
            
        Returns:
            Config: A fully instantiated Parsl configuration object
        """
        pass
    
    def create_resource_manager(self) -> 'ResourceManager':
        """
        Create and initialize a resource manager for this system.
        
        Returns:
            ResourceManager: Initialized resource manager
        """
        from parslbox.resource_manager import ResourceManager
        return ResourceManager(self)
