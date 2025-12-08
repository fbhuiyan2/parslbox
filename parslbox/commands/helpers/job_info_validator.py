"""
Job Information Validator Helper Module

This module contains validation functions used by both add.py and update.py
to ensure consistent validation logic and reduce code duplication.
"""

import sqlite3
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from parslbox.apps.app_registry import get_app_config, is_app_registered, get_registered_apps
from parslbox.system_configs.loader import get_system_config
from parslbox.database import database


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass

class ResourceConflictError(ValidationError):
    """Exception raised for resource parameter conflicts."""
    pass


def validate_app_configuration(app: str) -> dict:
    """
    Validates that the application exists in the registry and returns its configuration.
    
    Args:
        app: Application name to validate
        
    Returns:
        dict: Application configuration containing INPUT_REQUIRED and DFLT_INPUT
        
    Raises:
        ValidationError: If the application is not registered
    """
    if not is_app_registered(app):
        available_apps = ", ".join(get_registered_apps())
        raise ValidationError(f"Unknown application '{app}'. Available applications: {available_apps}")
    
    try:
        return get_app_config(app)
    except ValueError as e:
        raise ValidationError(str(e))


def validate_system_configuration(config_name: str):
    """
    Validates and loads the system configuration.
    
    Args:
        config_name: Name of the system configuration
        
    Returns:
        System configuration instance
        
    Raises:
        ValidationError: If the configuration cannot be loaded
    """
    try:
        return get_system_config(config_name)
    except Exception as e:
        raise ValidationError(f"Could not load system configuration: {e}")


def validate_input_file(app: str, input_file: Optional[str] = None) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Validates and determines the final input file based on app requirements.
    
    This implements the 4-scenario input file logic:
    1. Input required, no default -> must provide input_file (raises error if missing)
    2. Input required, has default -> use input_file or default
    3. Input not required -> use default (usually None)
    4. Special handling for user-provided input when not required
    
    Args:
        app: Application name
        input_file: User-provided input file (optional)
        
    Returns:
        Tuple of (final_input_file, info_messages, warning_messages)
        
    Raises:
        ValidationError: If app is invalid or required input is missing
    """
    info_messages = []
    warning_messages = []
    app_config = validate_app_configuration(app)
    input_required = app_config["INPUT_REQUIRED"]
    default_input = app_config["DFLT_INPUT"]
    
    if input_required and default_input is None:
        # Must have input file, no default available
        if input_file is None:
            # Core function should always validate - CLI handles this with interactive prompt
            # API calls this and gets the error, which is the correct behavior
            raise ValidationError(f"Input file is required for {app} application but none was provided")
        else:
            return input_file, info_messages, warning_messages
    elif input_required and default_input is not None:
        # Input required but has default
        final_input = input_file if input_file is not None else default_input
        if input_file is None:
            info_messages.append(f"Using default input file '{default_input}' for {app}")
        return final_input, info_messages, warning_messages
    else:
        # Input not required (input_required = False)
        if input_file is not None:
            warning_messages.append(f"Input file ignored for {app} (not required), using default behavior")
        return default_input, info_messages, warning_messages  # Will be None for apps that don't need input


def validate_resource_parameters(
    ngpus: int = 0,
    nnodes: int = 1,
    node_occupancy: Optional[float] = None,
    ranks_per_node: Optional[int] = None,
    system_config = None
) -> Tuple[Dict[str, Any], List[str], List[str]]:
    """
    Validates resource parameters and calculates final values.
    
    Args:
        ngpus: Number of GPUs requested
        nnodes: Number of nodes requested
        node_occupancy: Node occupancy fraction for CPU jobs
        ranks_per_node: Number of MPI ranks per node
        system_config: System configuration instance
        
    Returns:
        Tuple of (resource_params_dict, info_messages, warning_messages) where resource_params_dict contains:
            - final_num_nodes: int
            - final_ngpus: int
            - final_node_occupancy: float
            - final_ranks_per_node: int
            
    Raises:
        ValidationError: If resource parameters are invalid
        ResourceConflictError: If GPU and CPU parameters conflict
    """
    info_messages = []
    warning_messages = []
    
    # Validate parameter ranges
    if node_occupancy is not None and not (0.0 < node_occupancy <= 1.0):
        raise ValidationError("--nocc must be between 0.0 and 1.0")
    
    if ranks_per_node is not None and ranks_per_node < 1:
        raise ValidationError(f"--ranks-per-node must be a positive integer, got {ranks_per_node}")
    
    if nnodes < 1:
        raise ValidationError(f"--nnodes must be at least 1, got {nnodes}")
    
    if system_config is None:
        raise ValidationError("System configuration is required for resource validation")
    
    gpus_per_node = system_config.GPUS_PER_NODE
    
    # Determine final resource parameters
    if nnodes > 1:
        # Multi-node job: ignore ngpus and node occupancy
        final_num_nodes = nnodes
        final_ngpus = nnodes * gpus_per_node  # For display purposes
        final_node_occupancy = 1.0
        
        # Add info for multi-node GPU jobs
        if final_ngpus > 0:
            info_messages.append(f"Multi-node job will use {nnodes * gpus_per_node} total GPUs ({gpus_per_node} per node)")
    else:
        # Single-node job: check for conflicting parameters
        handle_gpu_cpu_conflict(ngpus, node_occupancy)  # raises ResourceConflictError if both > 0
        
        if ngpus > 0:
            # Single-node GPU job
            if ngpus > gpus_per_node:
                raise ValidationError(f"Requested {ngpus} GPUs but only {gpus_per_node} available per node")
            
            final_num_nodes = 1
            final_ngpus = ngpus
            final_node_occupancy = 1.0
        else:
            # Single-node CPU-only job
            final_num_nodes = 1
            final_ngpus = 0
            final_node_occupancy = node_occupancy if node_occupancy is not None else 1.0
    
    # Calculate smart default for ranks_per_node if not specified
    if ranks_per_node is None:
        if final_ngpus > 0:
            # GPU jobs: 1 rank per GPU
            final_ranks_per_node = 1
        else:
            # CPU jobs: calculate based on cores_per_node * node_occupancy
            calculated_ranks = int(system_config.CORES_PER_NODE * final_node_occupancy)
            final_ranks_per_node = max(1, calculated_ranks)  # Ensure at least 1
            info_messages.append(f"Using smart default: ranks_per_node = {final_ranks_per_node} (cores_per_node={system_config.CORES_PER_NODE} * node_occupancy={final_node_occupancy})")
    else:
        # User specified ranks_per_node
        if final_ngpus > 0:
            # GPU jobs: force to 1 regardless of user input
            final_ranks_per_node = 1
            if ranks_per_node != 1:
                warning_messages.append("ranks-per-node is ignored for GPU jobs (1 rank per GPU)")
        else:
            # CPU jobs: use user-specified value
            final_ranks_per_node = ranks_per_node
    
    resource_params = {
        'final_num_nodes': final_num_nodes,
        'final_ngpus': final_ngpus,
        'final_node_occupancy': final_node_occupancy,
        'final_ranks_per_node': final_ranks_per_node
    }
    
    return resource_params, info_messages, warning_messages


def handle_gpu_cpu_conflict(ngpus: int, node_occupancy: Optional[float]) -> None:
    """
    Validates that GPU and CPU parameters don't conflict.
    
    Args:
        ngpus: Number of GPUs requested
        node_occupancy: Node occupancy for CPU jobs
        
    Raises:
        ResourceConflictError: If both ngpus > 0 and node_occupancy > 0
    """
    if ngpus > 0 and node_occupancy is not None and node_occupancy > 0:
        # CLI handles this with interactive prompt
        # API calls this and gets the error, which is the correct behavior
        raise ResourceConflictError("Cannot specify both ngpus and node_occupancy to be > 0.")
    

def validate_environment_file(env_file: Optional[str]) -> Tuple[Optional[str], List[str], List[str]]:
    """
    Validates environment file path and converts to absolute path.
    
    Args:
        env_file: Environment file path (relative or absolute)
        
    Returns:
        Tuple of (absolute_path_or_none, info_messages, warning_messages)
        
    Raises:
        ValidationError: If file doesn't exist or isn't a file
    """
    info_messages = []
    warning_messages = []
    
    if not env_file:
        return None, info_messages, warning_messages
    
    # Convert relative path to absolute path
    env_file_path = Path(env_file)
    if not env_file_path.is_absolute():
        env_file_path = Path.cwd() / env_file_path
    
    # Validate that the environment file exists
    if not env_file_path.exists():
        raise ValidationError(f"Environment file '{env_file}' does not exist")
    
    if not env_file_path.is_file():
        raise ValidationError(f"Environment file '{env_file}' is not a file")
    
    resolved_path = str(env_file_path.resolve())
    info_messages.append(f"Using environment file: {resolved_path}")
    
    return resolved_path, info_messages, warning_messages


def validate_parent_dependencies(
    parents: Optional[List[int]] = None,
    parent_tag: Optional[str] = None,
    db_path: Optional[Path] = None
) -> Tuple[List[int], List[str], List[str]]:
    """
    Validates parent dependencies and converts parent_tag to IDs.
    
    Args:
        parents: List of parent job IDs
        parent_tag: Tag to convert to parent IDs (all Done jobs with this tag)
        db_path: Database path
        
    Returns:
        Tuple of (final_parent_ids, info_messages, warning_messages)
        
    Raises:
        ValidationError: If parent IDs don't exist in database
    """
    info_messages = []
    warning_messages = []
    
    if db_path is None:
        from parslbox.utils import path_utils
        db_path = path_utils.DB_FILE
    
    final_parents = []
    
    # Parse parents list if provided
    if parents:
        final_parents = list(parents)  # Create a copy
    
    # Handle parent_tag conversion to parent IDs
    if parent_tag:
        # Get all jobs with the specified tag that are Done
        tag_jobs = database.get_jobs(db_path, tag=parent_tag, status='Done')
        tag_parent_ids = [job['job_id'] for job in tag_jobs]
        
        if tag_parent_ids:
            info_messages.append(f"Added {len(tag_parent_ids)} parent jobs from tag '{parent_tag}': {tag_parent_ids}")
            final_parents.extend(tag_parent_ids)
        else:
            warning_messages.append(f"No completed jobs found with tag '{parent_tag}'")
    
    # Validate parent job IDs exist
    if final_parents:
        existing_jobs = database.get_jobs_by_ids(db_path, final_parents)
        existing_ids = {job['job_id'] for job in existing_jobs}
        missing_ids = set(final_parents) - existing_ids
        
        if missing_ids:
            raise ValidationError(f"Parent job IDs do not exist: {sorted(missing_ids)}")
    
    return final_parents, info_messages, warning_messages


def validate_paths(paths: List[str]) -> Tuple[List[Path], List[Tuple[str, str]]]:
    """
    Validates paths and handles 'all' subdirectory expansion.
    
    Args:
        paths: List of path strings, or ['all'] for all subdirectories
        
    Returns:
        Tuple of (valid_paths, failed_paths_with_errors)
        
    Raises:
        ValidationError: If 'all' is specified but no subdirectories found
    """
    paths_to_add: List[Path] = []
    failed_jobs: List[Tuple[str, str]] = []
    
    if len(paths) == 1 and paths[0].lower() == 'all':
        current_dir = Path.cwd()
        subdirectories = [p for p in current_dir.iterdir() if p.is_dir()]
        
        if not subdirectories:
            raise ValidationError("No subdirectories found in the current directory")
            
        paths_to_add = [p.resolve() for p in subdirectories]
    else:
        # Validate user-provided paths - collect failures instead of raising immediately
        for path_str in paths:
            path_obj = Path(path_str)
            if not path_obj.exists():
                failed_jobs.append((path_str, f"Path '{path_str}' does not exist"))
                continue
            if not path_obj.is_dir():
                failed_jobs.append((path_str, f"Path '{path_str}' is not a directory"))
                continue
            paths_to_add.append(path_obj.resolve())
    
    return paths_to_add, failed_jobs


def validate_python_app_env_file(app: str, env_file: Optional[str]) -> Optional[str]:
    """
    Special validation for Python app environment file requirements.
    
    Args:
        app: Application name
        env_file: Environment file path or special 'pass' value
        
    Returns:
        Environment file path or None
        
    Raises:
        ValidationError: If Python app requires env file but none provided
    """
    if app == "python":
        if env_file == 'pass':
            return None
        elif env_file is None:
            raise ValidationError("Python app requires an environment file. " \
            "Please specify env_file parameter. " \
            "If you wish to proceed without an environment file, set env_file='pass'.")
    
    return env_file


def calculate_resource_display_info(resource_params: Dict[str, Any], system_config) -> Dict[str, Any]:
    """
    Calculates display parameters for CLI output.
    
    Args:
        resource_params: Dictionary from validate_resource_parameters()
        system_config: System configuration instance
        
    Returns:
        Dictionary with display information including resource specification string
    """
    final_num_nodes = resource_params['final_num_nodes']
    final_ngpus = resource_params['final_ngpus']
    final_node_occupancy = resource_params['final_node_occupancy']
    final_ranks_per_node = resource_params['final_ranks_per_node']
    
    # Calculate total ranks for display
    if final_ngpus > 0:
        total_ranks = final_num_nodes * final_ngpus
    else:
        total_ranks = final_num_nodes * final_ranks_per_node
    
    # Generate resource specification string
    if final_num_nodes > 1:
        if final_ngpus > 0:
            display_str = f"n:{final_num_nodes}-r:{final_ngpus}-g:{final_ngpus}-nocc:NA"
        else:
            display_str = f"n:{final_num_nodes}-r:{total_ranks}-g:0-nocc:NA"
    elif final_ngpus > 0:
        display_str = f"n:1-r:{total_ranks}-g:{final_ngpus}-nocc:NA"
    else:
        display_str = f"n:1-r:{total_ranks}-g:0-nocc:{final_node_occupancy}"
    
    return {
        'total_ranks': total_ranks,
        'resource_spec_string': display_str,
        'is_multi_node': final_num_nodes > 1,
        'is_gpu_job': final_ngpus > 0
    }


def validate_update_app_compatibility(
    job_ids: List[int],
    new_app: str,
    input_file: Optional[str],
    db_path: Optional[Path] = None
) -> Tuple[List[Tuple[int, str]], List[str], List[str], Optional[str]]:
    """
    Validates app update compatibility and handles input file logic for update operations.
    
    Args:
        job_ids: List of job IDs being updated
        new_app: New application to update to
        input_file: New input file (if provided)
        db_path: Database path
        
    Returns:
        Tuple of (failed_jobs, info_messages, warning_messages, final_input_file)
    """
    if db_path is None:
        from parslbox.utils import path_utils
        db_path = path_utils.DB_FILE
    
    failed_jobs = []
    info_messages = []
    warning_messages = []
    final_input_file = input_file
    
    # Get current jobs to check their existing apps
    current_jobs = database.get_jobs_by_ids(db_path, job_ids)
    
    for job in current_jobs:
        job_id = job['job_id']
        original_app = job['app']
        
        try:
            # Get app configurations
            original_config = get_app_config(original_app)
            new_config = get_app_config(new_app)
            
            original_input_required = original_config['INPUT_REQUIRED']
            original_default_input = original_config['DFLT_INPUT']
            new_input_required = new_config['INPUT_REQUIRED']
            new_default_input = new_config['DFLT_INPUT']
            
            # Apply the four scenarios for input file handling
            if original_input_required and not new_input_required:
                # Scenario 1: Original requires input, new doesn't -> set in_file to None
                if input_file is None:  # Only auto-set if user didn't explicitly provide input_file
                    final_input_file = None
                    info_messages.append(f"Job {job_id}: App updated from '{original_app}' to '{new_app}'. Input file set to None since new app doesn't require input.")
                else:
                    warning_messages.append(f"Job {job_id}: App updated from '{original_app}' to '{new_app}'. User-specified input file '{input_file}' will be kept despite new app not requiring input.")
            
            elif not original_input_required and new_input_required:
                # Scenario 2: Original doesn't require input, new does -> throw error unless user provided input
                if input_file is None:
                    failed_jobs.append((job_id, f"Cannot update job {job_id} from '{original_app}' to '{new_app}': new app requires input file but none provided. Please specify --input filename."))
                    continue
                else:
                    info_messages.append(f"Job {job_id}: App updated from '{original_app}' to '{new_app}'. User-specified input file '{input_file}' will be used.")
            
            elif original_input_required and new_input_required:
                # Scenario 3: Both require input -> check default inputs
                if input_file is None:  # User didn't specify input file
                    if original_default_input == new_default_input:
                        # Same default, update quietly
                        info_messages.append(f"Job {job_id}: App updated from '{original_app}' to '{new_app}'. Both apps have same default input file.")
                    else:
                        # Different defaults, require user to specify
                        failed_jobs.append((job_id, f"Cannot update job {job_id} from '{original_app}' to '{new_app}': apps have different default input files ('{original_default_input}' vs '{new_default_input}'). Please specify --input filename."))
                        continue
                else:
                    info_messages.append(f"Job {job_id}: App updated from '{original_app}' to '{new_app}'. User-specified input file '{input_file}' will be used.")
            
            else:
                # Scenario 4: Both don't require input -> update with warning
                info_messages.append(f"Job {job_id}: App updated from '{original_app}' to '{new_app}'. Both apps don't require input files, but user should verify input file compatibility.")
            
        except ValueError as e:
            # App doesn't exist
            failed_jobs.append((job_id, f"Cannot update job {job_id}: {str(e)}"))
            continue
    
    return failed_jobs, info_messages, warning_messages, final_input_file
