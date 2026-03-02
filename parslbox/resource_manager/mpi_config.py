"""
MPI Configuration Schema and Loading Logic for ParslBox

This module defines the MPI configuration schema and provides functions to
load and merge MPI configurations from system-level and app-level settings.

Configuration Hierarchy (highest to lowest priority):
1. App-level (lammps-kk.polaris.mpi)
2. System-level (polaris.mpi)
3. Backend defaults (hardcoded minimal commands)

Scalar values are replaced by higher priority; disable/add lists from
highest priority completely replace lower priority lists.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class MPIBackend(Enum):
    """Supported MPI backends."""
    OPENMPI = "openmpi"
    MPICH = "mpich"
    SRUN = "srun"


@dataclass
class MPIConfig:
    """
    MPI configuration for a job.
    
    This class holds all MPI-related settings that control how the MPI
    command is generated. Settings can come from system-level defaults,
    app-level overrides, or job-level overrides.
    
    Attributes:
        backend: MPI backend type (openmpi, mpich, srun)
        mpi_cmd: Custom MPI command path (overrides default for backend)
        use_gpu_wrapper: Whether to generate GPU assignment wrapper script
        use_hostlist: Whether to explicitly pass hostlist to MPI command
        use_short_hostnames: Whether to strip domain from hostnames
        cpu_bind_method: CPU binding method (none, rankfile, list, depth, depth <N>)
        disable: List of flags/substrings to remove from command
        add: List of flags to append to command (supports templates)
    """
    backend: MPIBackend = MPIBackend.OPENMPI
    mpi_cmd: Optional[str] = None
    use_gpu_wrapper: bool = False
    use_hostlist: bool = False
    use_short_hostnames: bool = False
    cpu_bind_method: str = "none"
    disable: List[str] = field(default_factory=list)
    add: List[str] = field(default_factory=list)
    
    def get_mpi_command(self) -> str:
        """Get the MPI command to use."""
        if self.mpi_cmd:
            return self.mpi_cmd
        
        # Default commands per backend
        defaults = {
            MPIBackend.OPENMPI: "mpirun",
            MPIBackend.MPICH: "mpiexec",
            MPIBackend.SRUN: "srun",
        }
        return defaults.get(self.backend, "mpirun")
    
    def get_cpu_bind_depth(self) -> Optional[int]:
        """
        Parse depth value from cpu_bind_method if specified.
        
        Returns:
            int if depth value is specified (e.g., "depth 8" -> 8)
            None if auto-calculate (e.g., "depth" -> None)
            None if not a depth method
        """
        if not self.cpu_bind_method.startswith("depth"):
            return None
        
        parts = self.cpu_bind_method.split()
        if len(parts) == 2:
            try:
                return int(parts[1])
            except ValueError:
                logger.warning(f"Invalid depth value in cpu_bind_method: {self.cpu_bind_method}")
                return None
        return None  # Auto-calculate
    
    def is_depth_binding(self) -> bool:
        """Check if using depth-based CPU binding."""
        return self.cpu_bind_method.startswith("depth")
    
    def is_rankfile_binding(self) -> bool:
        """Check if using rankfile-based CPU binding."""
        return self.cpu_bind_method == "rankfile"
    
    def is_list_binding(self) -> bool:
        """Check if using list-based CPU binding."""
        return self.cpu_bind_method == "list"


# Default MPI configurations per backend
DEFAULT_MPI_CONFIGS = {
    MPIBackend.OPENMPI: MPIConfig(backend=MPIBackend.OPENMPI),
    MPIBackend.MPICH: MPIConfig(backend=MPIBackend.MPICH),
    MPIBackend.SRUN: MPIConfig(backend=MPIBackend.SRUN),
}


def parse_backend(backend_str: str) -> MPIBackend:
    """
    Parse backend string to MPIBackend enum.
    
    Args:
        backend_str: Backend name (openmpi, mpich, srun)
        
    Returns:
        MPIBackend enum value
        
    Raises:
        ValueError: If backend string is not recognized
    """
    backend_str = backend_str.lower().strip()
    try:
        return MPIBackend(backend_str)
    except ValueError:
        valid = [b.value for b in MPIBackend]
        raise ValueError(f"Unknown MPI backend: '{backend_str}'. Valid options: {valid}")


def mpi_config_from_dict(config_dict: Dict[str, Any]) -> MPIConfig:
    """
    Create MPIConfig from a dictionary (parsed from YAML).
    
    Args:
        config_dict: Dictionary with MPI configuration options
        
    Returns:
        MPIConfig instance
    """
    if not config_dict:
        return MPIConfig()
    
    # Parse backend
    backend = MPIBackend.OPENMPI
    if "backend" in config_dict:
        backend = parse_backend(config_dict["backend"])
    
    return MPIConfig(
        backend=backend,
        mpi_cmd=config_dict.get("mpi_cmd"),
        use_gpu_wrapper=config_dict.get("use_gpu_wrapper", False),
        use_hostlist=config_dict.get("use_hostlist", False),
        use_short_hostnames=config_dict.get("use_short_hostnames", False),
        cpu_bind_method=config_dict.get("cpu_bind_method", "none"),
        disable=config_dict.get("disable", []),
        add=config_dict.get("add", []),
    )


def merge_mpi_config_dicts(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two MPI config dictionaries, with override taking precedence.
    
    This operates on raw dictionaries (before parsing to MPIConfig) so that
    explicit values like `use_gpu_wrapper: false` properly override `true`.
    
    Args:
        base_dict: Base configuration dict (lower priority)
        override_dict: Override configuration dict (higher priority)
        
    Returns:
        Merged dictionary
    """
    if not base_dict:
        return override_dict.copy() if override_dict else {}
    if not override_dict:
        return base_dict.copy()
    
    # Start with base, then override with explicit values from override_dict
    merged = base_dict.copy()
    
    for key, value in override_dict.items():
        # Override dict values always win (including explicit false/empty)
        merged[key] = value
    
    return merged


def load_mpi_config(
    system_name: str,
    app_name: str,
    system_config_class: Any,
    yaml_config: Dict[str, Any]
) -> MPIConfig:
    """
    Load MPI configuration with proper hierarchy.
    
    Priority (highest to lowest):
    1. App-level config from YAML (app_name.system_name.mpi)
    2. System-level config from YAML (system_name.mpi)
    3. System config class default (MPI_BACKEND attribute)
    4. Backend defaults
    
    Merging is done at the dictionary level so that explicit values
    (including `use_gpu_wrapper: false`) properly override parent configs.
    
    Args:
        system_name: Name of the system (e.g., "polaris")
        app_name: Name of the application (e.g., "lammps-kk")
        system_config_class: System configuration class instance
        yaml_config: Full parsed YAML config dictionary
        
    Returns:
        Final merged MPIConfig
    """
    # Start with backend default based on system config class
    backend_str = "openmpi"
    if hasattr(system_config_class, 'MPI_BACKEND'):
        backend_str = system_config_class.MPI_BACKEND
    elif hasattr(system_config_class, 'MPI_CMD_TO_USE'):
        # Legacy support: infer backend from MPI_CMD_TO_USE
        cmd = system_config_class.MPI_CMD_TO_USE.lower()
        if cmd == "mpiexec":
            backend_str = "mpich"
        elif cmd == "srun":
            backend_str = "srun"
        else:
            backend_str = "openmpi"
    
    # Start with base dict containing backend from system config class
    merged_dict = {"backend": backend_str}
    
    # Layer 1: Merge with system-level config from YAML
    system_yaml = yaml_config.get(system_name, {})
    if isinstance(system_yaml, dict) and "mpi" in system_yaml:
        system_mpi_dict = system_yaml["mpi"]
        if isinstance(system_mpi_dict, dict):
            merged_dict = merge_mpi_config_dicts(merged_dict, system_mpi_dict)
    
    # Layer 2: Merge with app-level config from YAML (highest priority)
    app_yaml = yaml_config.get(app_name, {})
    if isinstance(app_yaml, dict):
        app_system_yaml = app_yaml.get(system_name, {})
        if isinstance(app_system_yaml, dict) and "mpi" in app_system_yaml:
            app_mpi_dict = app_system_yaml["mpi"]
            if isinstance(app_mpi_dict, dict):
                merged_dict = merge_mpi_config_dicts(merged_dict, app_mpi_dict)
    
    # Parse the merged dict to MPIConfig
    config = mpi_config_from_dict(merged_dict)
    
    logger.debug(f"Loaded MPI config for {app_name} on {system_name}: backend={config.backend.value}, "
                 f"gpu_wrapper={config.use_gpu_wrapper}, cpu_bind={config.cpu_bind_method}")
    
    return config
