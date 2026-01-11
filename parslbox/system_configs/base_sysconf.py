"""
Base System Configuration for ParslBox

This module defines the abstract base class for system configurations,
providing a consistent interface for accessing system specifications
and generating Parsl configurations.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Optional, List
from parsl.config import Config

if TYPE_CHECKING:
    from parslbox.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class SystemConfig(ABC):
    """
    Abstract base class for system configurations.
    
    Each system (e.g., Polaris, Sophia) should inherit from this class
    and implement the required methods and class attributes.
    
    Key assumptions for CPU affinity validation:
    - WORKER_CPU_AFFINITY format: "list:group1:group2:..." where each group is comma-separated ranges
    - For GPU systems: number of affinity groups should match GPUS_PER_NODE
    - Core IDs must be within valid range (0 to CORES_PER_NODE-1)
    - Overlapping cores generate warnings but don't prevent operation
    - Unbalanced groups (different core counts) generate warnings
    """
    
    # System specifications (must be defined in subclasses)
    SYSTEM_NAME: str  # Human-readable system name
    CORES_PER_NODE: int
    GPUS_PER_NODE: int
    CORES_PER_GPU: Optional[int] = None     # Some systems can have a weird design where number of affinity core does not match CORES_PER_NODE//GPUS_PER_NODE
    SCHEDULER: str
    MPI_CMD_TO_USE: str
    MAX_WORKERS_PER_NODE: int
    WORKER_CPU_AFFINITY: Optional[str] = None
    EXCLUDE_CORES: Optional[List[int]] = None
    GPU_TYPE: str     # GPU type: 'cuda', 'intel', None, etc.
    
    def __init__(self):
        """
        Initialize and validate system configuration.
        
        Validates WORKER_CPU_AFFINITY string if provided to catch configuration
        errors early in the workflow, before jobs are created or executed.
        """
        self._validate_configuration()
    
    def _validate_configuration(self):
        """
        Validate system configuration parameters.
        
        Raises:
            ValueError: If WORKER_CPU_AFFINITY or EXCLUDE_CORES is invalid
        """
        if self.WORKER_CPU_AFFINITY:
            is_valid, error_msg, warnings = self._validate_affinity_string(
                self.WORKER_CPU_AFFINITY,
                self.CORES_PER_NODE,
                self.GPUS_PER_NODE
            )
            
            if not is_valid:
                raise ValueError(
                    f"Invalid WORKER_CPU_AFFINITY in {self.__class__.__name__}: {error_msg}\n"
                    f"Expected format: 'list:group1:group2:...' where each group is comma-separated core ranges.\n"
                    f"Example: 'list:0-7,32-39:8-15,40-47:16-23,48-55:24-31,56-63'"
                )
            
            # Log warnings for non-fatal issues
            for warning in warnings:
                logger.warning(f"WORKER_CPU_AFFINITY in {self.__class__.__name__}: {warning}")
        
        # Validate EXCLUDE_CORES
        if self.EXCLUDE_CORES:
            invalid_cores = [core for core in self.EXCLUDE_CORES if core < 0 or core >= self.CORES_PER_NODE]
            if invalid_cores:
                raise ValueError(
                    f"Invalid EXCLUDE_CORES in {self.__class__.__name__}: {invalid_cores}\n"
                    f"Core IDs must be within valid range (0 to {self.CORES_PER_NODE-1})"
                )
            
            # Check if too many cores are excluded
            if len(self.EXCLUDE_CORES) >= self.CORES_PER_NODE:
                raise ValueError(
                    f"Invalid EXCLUDE_CORES in {self.__class__.__name__}: Cannot exclude all cores\n"
                    f"Excluding {len(self.EXCLUDE_CORES)} cores out of {self.CORES_PER_NODE} total cores"
                )
            
            # Log info about excluded cores
            logger.info(f"EXCLUDE_CORES in {self.__class__.__name__}: Excluding cores {sorted(self.EXCLUDE_CORES)}")
    
    def _validate_affinity_string(self, affinity_str: str, total_cores: int, gpus_per_node: int = 0) -> tuple[bool, str, list[str]]:
        """
        Validate an affinity string format and return errors and warnings.
        
        Args:
            affinity_str: Affinity string to validate
            total_cores: Total number of cores available on the node
            gpus_per_node: Number of GPUs per node (for group count validation)
            
        Returns:
            Tuple of (is_valid, error_message, warnings_list)
        """
        warnings = []
        
        try:
            # Parse the affinity string using a temporary manager instance
            from parslbox.resource_manager.cpu_affinity import CPUAffinityManager
            temp_manager = CPUAffinityManager(affinity_str, total_cores)
            groups = temp_manager.affinity_groups
            
            if not groups:
                return False, "No valid affinity groups found", warnings
            
            # ERROR: Check for cores outside valid range
            all_cores = set()
            for group in groups:
                for core in group:
                    if core < 0 or core >= total_cores:
                        return False, f"Invalid core ID {core} (outside 0-{total_cores-1})", warnings
                    all_cores.add(core)
            
            # ERROR: For GPU systems, check group count matches GPU count
            if gpus_per_node > 0 and len(groups) != gpus_per_node:
                return False, f"Expected {gpus_per_node} affinity groups for {gpus_per_node} GPUs, found {len(groups)}", warnings
            
            # WARNING: Check for overlapping cores
            seen_cores = set()
            overlapping_cores = set()
            for group in groups:
                group_set = set(group)
                overlap = seen_cores.intersection(group_set)
                if overlap:
                    overlapping_cores.update(overlap)
                seen_cores.update(group_set)
            
            if overlapping_cores:
                warnings.append(f"Overlapping cores detected: {sorted(overlapping_cores)}. This may cause resource conflicts.")
            
            # WARNING: Check for unbalanced groups
            group_sizes = [len(group) for group in groups]
            if len(set(group_sizes)) > 1:  # Not all groups have same size
                min_size, max_size = min(group_sizes), max(group_sizes)
                if max_size > min_size * 1.5:  # More than 50% difference
                    warnings.append(f"Unbalanced affinity groups: {group_sizes}. Consider balancing core distribution.")
            
            return True, "Valid affinity string", warnings
            
        except Exception as e:
            return False, f"Parse error: {e}", warnings
    
    @abstractmethod
    def detect_resources(self) -> tuple[int, int]:
        """
        Detect available nodes and GPUs for this system.
        
        Returns:
            tuple[int, int]: A tuple of (nodes, total_gpus)
        """
        pass
    
    @abstractmethod
    def get_config(self, run_dir: Path, retries: int = 0, max_workers: Optional[int] = None) -> Config:
        """
        Generate Parsl configuration for this system.
        
        Args:
            run_dir (Path): The path for Parsl's run directory
            retries (int): The number of retries for failed Parsl apps
            max_workers (Optional[int]): Optional override for total workers across all nodes.
                                        If None, uses MAX_WORKERS_PER_NODE * nodes (default behavior).
                                        If provided, will be capped at MAX_WORKERS_PER_NODE * nodes.
            
        Returns:
            Config: A fully instantiated Parsl configuration object
        """
        pass
    
    def create_resource_manager(self, job_tracker=None) -> 'ResourceManager':
        """
        Create and initialize a resource manager for this system.
        
        Args:
            job_tracker: Optional JobTracker for efficient dependency checking
        
        Returns:
            ResourceManager: Initialized resource manager
        """
        from parslbox.resource_manager import ResourceManager
        return ResourceManager(self, job_tracker)
