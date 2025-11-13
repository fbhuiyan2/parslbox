"""
CPU Affinity Manager for ParslBox Resource Manager

This module provides CPU affinity parsing and management functionality
for intelligent CPU core assignment based on system-specific affinity groups.
"""

import logging
from typing import List, Optional, Tuple
import re

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
    
    def get_preferred_cores(self, num_cores: int, available_cores: List[int]) -> List[int]:
        """
        Get preferred core assignment based on affinity groups.
        
        Tries to assign cores from complete affinity groups first, then falls back
        to partial groups if needed.
        
        Args:
            num_cores: Number of cores needed
            available_cores: List of currently available core IDs
            
        Returns:
            List of preferred core IDs (may be fewer than requested if not available)
        """
        if not self.has_affinity or not available_cores:
            return available_cores[:num_cores]
        
        available_set = set(available_cores)
        preferred_cores = []
        
        # First pass: try to assign from complete affinity groups
        for group in self.affinity_groups:
            if len(preferred_cores) >= num_cores:
                break
            
            # Find cores from this group that are available
            group_available = [core for core in group if core in available_set]
            
            if not group_available:
                continue
            
            # Take cores from this group
            cores_needed = min(len(group_available), num_cores - len(preferred_cores))
            selected_cores = group_available[:cores_needed]
            preferred_cores.extend(selected_cores)
            
            # Remove selected cores from available set
            for core in selected_cores:
                available_set.discard(core)
        
        # Second pass: if we still need more cores, take any remaining available cores
        if len(preferred_cores) < num_cores:
            remaining_available = [core for core in available_cores if core in available_set]
            cores_needed = num_cores - len(preferred_cores)
            # Take as many as available, even if fewer than needed
            additional_cores = remaining_available[:cores_needed]
            preferred_cores.extend(additional_cores)
        
        # Return what we could assign (may be fewer than requested if not enough available)
        return preferred_cores
    
    def get_cores_for_occupancy(self, occupancy: float, available_cores: List[int]) -> List[int]:
        """
        Get core assignment for a given node occupancy.
        
        Tries to assign complete affinity groups when possible.
        
        Args:
            occupancy: Node occupancy (0.0 to 1.0)
            available_cores: List of currently available core IDs
            
        Returns:
            List of assigned core IDs
        """
        num_cores = max(1, int(occupancy * self.total_cores))
        return self.get_preferred_cores(num_cores, available_cores)
    
    def get_affinity_group_for_cores(self, core_ids: List[int]) -> Optional[int]:
        """
        Determine which affinity group the given cores belong to.
        
        Args:
            core_ids: List of core IDs to check
            
        Returns:
            Affinity group index, or None if cores span multiple groups
        """
        if not self.has_affinity or not core_ids:
            return None
        
        core_set = set(core_ids)
        
        for i, group in enumerate(self.affinity_groups):
            group_set = set(group)
            if core_set.issubset(group_set):
                return i
        
        return None
    
    def get_group_info(self) -> List[Tuple[int, List[int]]]:
        """
        Get information about all affinity groups.
        
        Returns:
            List of (group_index, core_list) tuples
        """
        return [(i, group.copy()) for i, group in enumerate(self.affinity_groups)]
    
    def validate_affinity_string(self, affinity_str: str) -> Tuple[bool, str]:
        """
        Validate an affinity string format.
        
        Args:
            affinity_str: Affinity string to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            groups = self._parse_affinity_string(affinity_str)
            
            if not groups:
                return False, "No valid affinity groups found"
            
            # Check for overlapping cores
            all_cores = set()
            for group in groups:
                group_set = set(group)
                overlap = all_cores.intersection(group_set)
                if overlap:
                    return False, f"Overlapping cores found: {sorted(overlap)}"
                all_cores.update(group_set)
            
            # Check for cores outside valid range
            invalid_cores = [core for core in all_cores if core < 0 or core >= self.total_cores]
            if invalid_cores:
                return False, f"Invalid core IDs (outside 0-{self.total_cores-1}): {sorted(invalid_cores)}"
            
            return True, "Valid affinity string"
            
        except Exception as e:
            return False, f"Parse error: {e}"
