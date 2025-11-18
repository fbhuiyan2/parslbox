"""
CPU Affinity Manager for ParslBox Resource Manager

This module provides CPU affinity parsing and management functionality
for intelligent CPU core assignment based on system-specific affinity groups.
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)



class CPUAffinityManager:

    
    """
    Manages CPU affinity groups and provides intelligent core assignment.
    
    Parses affinity strings in the format:
    "list:0-15,128-143:16-31,144-159:32-47,160-175:..."
    
    Each group (separated by :) represents cores that should be assigned together
    for optimal performance (e.g., NUMA domains, CPU-GPU locality).
    """
    
    def __init__(self, affinity_string: Optional[str], total_cores: int):
        """
        Initialize the CPU affinity manager.
        
        Args:
            affinity_string: Affinity string in "list:group1:group2:..." format
            total_cores: Total number of cores available on the node
        """
        self.total_cores = total_cores
        self.affinity_groups = self._parse_affinity_string(affinity_string)
        self.has_affinity = affinity_string is not None and len(self.affinity_groups) > 0
        
        if self.has_affinity:
            logger.debug(f"Initialized CPU affinity with {len(self.affinity_groups)} groups")
        else:
            logger.debug("No CPU affinity configured - using linear assignment")
    
    def _parse_affinity_string(self, affinity_str: Optional[str]) -> List[List[int]]:
        """
        Parse affinity string into groups of core IDs.
        
        Args:
            affinity_str: String like "list:0-15,128-143:16-31,144-159:..."
            
        Returns:
            List of core ID groups: [[0,1,...,15,128,...,143], [16,17,...,31,144,...,159], ...]
        """
        if not affinity_str:
            return []
        
        try:
            # Remove "list:" prefix if present
            if affinity_str.startswith("list:"):
                affinity_str = affinity_str[5:]
            
            # Split into groups by ":"
            group_strings = affinity_str.split(":")
            groups = []
            
            for group_str in group_strings:
                if not group_str.strip():
                    continue
                
                cores = self._parse_core_range(group_str)
                if cores:
                    groups.append(cores)
            
            return groups
            
        except Exception as e:
            logger.warning(f"Failed to parse affinity string '{affinity_str}': {e}")
            return []
    
    def _parse_core_range(self, range_str: str) -> List[int]:
        """
        Parse a core range string like "0-15,128-143" into list of core IDs.
        
        Args:
            range_str: String like "0-15,128-143" or "24-31,56-63"
            
        Returns:
            List of core IDs: [0,1,2,...,15,128,129,...,143]
        """
        cores = []
        
        # Split by comma to handle multiple ranges
        for part in range_str.split(","):
            part = part.strip()
            if not part:
                continue
            
            if "-" in part:
                # Range like "0-15"
                start, end = part.split("-", 1)
                start_core = int(start.strip())
                end_core = int(end.strip())
                cores.extend(range(start_core, end_core + 1))
            else:
                # Single core
                cores.append(int(part.strip()))
        
        return sorted(cores)
    
    
    def get_cores_for_gpu(self, gpu_id: int, available_cores: List[int]) -> List[int]:
        """
        Get CPU cores that have affinity with a specific GPU.
        
        Args:
            gpu_id: GPU ID to get affinity cores for
            available_cores: List of currently available core IDs
            
        Returns:
            List of core IDs that have affinity with the GPU (may be empty if none available)
        """
        if not self.has_affinity or gpu_id >= len(self.affinity_groups):
            # No affinity configured or GPU ID out of range
            return []
        
        # Get the affinity group for this GPU
        gpu_affinity_cores = self.affinity_groups[gpu_id]
        available_set = set(available_cores)
        
        # Find cores from this GPU's affinity group that are available
        affinity_available = [core for core in gpu_affinity_cores if core in available_set]
        
        return affinity_available
