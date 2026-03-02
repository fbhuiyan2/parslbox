import typer
import sqlite3
from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated
from parslbox.database import database
from parslbox.utils import path_utils
from parslbox.apps.app_registry import get_app_config, is_app_registered, get_registered_apps
from parslbox.system_configs.loader import get_system_config
from parslbox.commands.helpers.job_info_validator import (
    ValidationError,
    ResourceConflictError,
    validate_app_configuration,
    validate_system_configuration,
    validate_input_file,
    validate_resource_parameters,
    validate_environment_file,
    validate_parent_dependencies,
    validate_paths,
    validate_python_app_env_file,
    calculate_resource_display_info
)

app = typer.Typer()


def add_jobs(
    paths: List[str],
    app: str,
    config_name: str,
    tag: Optional[str] = None,
    input_file: Optional[str] = None,
    ngpus: int = 0,
    nnodes: int = 1,
    node_occupancy: Optional[float] = None,
    ranks_per_node: Optional[int] = None,
    mpi_opts: Optional[str] = None,
    env_file: Optional[str] = None,
    parents: Optional[List[int]] = None,
    parent_tag: Optional[str] = None,
    status: str = "Ready",
    app_args: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> tuple[List[int], List[tuple[str, str]], dict]:
    """
    Core job addition logic - used by both CLI and API.

    Args:
        paths: List of paths to job directories, or ['all'] for all subdirectories
        app: Application type (e.g., 'lammps-kk', 'vasp')
        config_name: System configuration name (e.g., 'polaris')
        tag: Optional tag to categorize the job(s)
        input_file: Input filename for the job(s)
        ngpus: Number of GPUs required (default: 0)
        nnodes: Number of nodes required (default: 1)
        node_occupancy: Node occupancy fraction for CPU-only jobs (0.0-1.0)
        ranks_per_node: Number of MPI ranks per node
        mpi_opts: Additional MPI options
        env_file: Path to environment setup file
        parents: List of parent job IDs
        parent_tag: Tag to wait for (all jobs with this tag must be Done)
        status: Initial job status (default: 'Ready')
        app_args: Additional arguments to append to the application command
        db_path: Database path (uses default if None)
    
    Returns:
        Tuple of (successful_job_ids, failed_jobs, msg_log) where:
        - successful_job_ids: List of created job IDs
        - failed_jobs: List of tuples (path, error_message) for failed jobs
        - msg_log: Dictionary with 'warnings' and 'info' lists
        
    Raises:
        ValidationError: If global validation fails (e.g., invalid app, invalid parameters)
    """
    if db_path is None:
        db_path = path_utils.DB_FILE
    
    # --- Use helper functions for validation and collect messages ---
    info_messages = []
    warning_messages = []
    
    # Validate application exists in registry
    validate_app_configuration(app)
    
    # API-specific validation for Python apps
    env_file = validate_python_app_env_file(app, env_file)
    
    # Handle input file logic based on app configuration
    final_input_file, input_info, input_warnings = validate_input_file(app, input_file)
    info_messages.extend(input_info)
    warning_messages.extend(input_warnings)
    
    # Get system configuration and validate resource parameters
    system_config = validate_system_configuration(config_name)
    resource_params, resource_info, resource_warnings = validate_resource_parameters(ngpus, nnodes, node_occupancy, ranks_per_node, system_config)
    info_messages.extend(resource_info)
    warning_messages.extend(resource_warnings)
    
    # Extract final resource values
    final_num_nodes = resource_params['final_num_nodes']
    final_ngpus = resource_params['final_ngpus']
    final_node_occupancy = resource_params['final_node_occupancy']
    final_ranks_per_node = resource_params['final_ranks_per_node']
    
    # Handle environment file validation and processing
    final_env_file, env_info, env_warnings = validate_environment_file(env_file)
    info_messages.extend(env_info)
    warning_messages.extend(env_warnings)
    
    # Handle parent dependencies
    final_parents, parent_info, parent_warnings = validate_parent_dependencies(parents, parent_tag, db_path)
    info_messages.extend(parent_info)
    warning_messages.extend(parent_warnings)
    
    # Append app_args to input file if provided
    if app_args and final_input_file:
        final_input_file = f"{final_input_file} {app_args}"

    # Determine the list of paths to process
    paths_to_add, failed_jobs = validate_paths(paths)

    # --- Add the determined paths to the database ---
    created_job_ids = []
    
    for path in paths_to_add:
        try:
            new_id = database.add_job(
                db_path=db_path,
                path=str(path),
                app=app,
                num_nodes=final_num_nodes,
                ngpus=final_ngpus,
                node_occupancy=final_node_occupancy,
                ranks_per_node=final_ranks_per_node,
                tag=tag,
                in_file=final_input_file,
                mpi_opts=mpi_opts,
                env_file=final_env_file,
                parents=final_parents,
                status=status
            )
            created_job_ids.append(new_id)
        except sqlite3.IntegrityError:
            input_display = f"input file '{final_input_file}'" if final_input_file else "no input file"
            failed_jobs.append((str(path), f"A job in the same path and {input_display} already exists in the database"))
    
    # Create message log dictionary
    msg_log = {
        "warnings": warning_messages,
        "info": info_messages
    }
    
    return created_job_ids, failed_jobs, msg_log


# Main CLI command

@app.command()
def add(
    paths: Annotated[
        List[str],
        typer.Argument(
            help="One or more paths to job directories, or 'all' to add all subdirectories in the current location."
        ),
    ],
    app: Annotated[
        str,
        typer.Option("--app", "-a", help="The application type (e.g., 'lammps-kk', 'vasp')."),
    ],
    config_name: Annotated[
        str,
        typer.Option("--config", "-c", help="The name of the configuration to use (e.g., 'polaris').")
    ],
    tag: Annotated[
        Optional[str],
        typer.Option("--tag", "-t", help="An optional tag to categorize the job(s)."),
    ] = None,
    input_file: Annotated[
        Optional[str],
        typer.Option("--input", "-i", help="Input filename for the job(s)."),
    ] = None,
    ngpus: Annotated[
        int,
        typer.Option("--ngpus", "-g", help="Number of GPUs required for the job(s)."),
    ] = 0,
    nnodes: Annotated[
        int,
        typer.Option("--nnodes", "-n", help="Number of nodes required for the job(s)."),
    ] = 1,
    node_occupancy: Annotated[
        Optional[float],
        typer.Option("--nocc", "-o", help="Node occupancy fraction for CPU-only jobs (0.0-1.0)."),
    ] = None,
    ranks_per_node: Annotated[
        Optional[int],
        typer.Option("--ranks-per-node", "-rpn", help="Number of MPI ranks per node. For CPU jobs only; ignored for GPU jobs. If not specified, defaults to cores_per_node * node_occupancy."),
    ] = None,
    mpi_opts: Annotated[
        Optional[str],
        typer.Option("--mpiopts", help="Additional MPI options to append to the MPI command."),
    ] = None,
    env_file: Annotated[
        Optional[str],
        typer.Option("--envfile", "-e", help="Path to environment setup file (relative or absolute)."),
    ] = None,
    parents: Annotated[
        Optional[str],
        typer.Option("--parents", "-P", help="Space-separated job IDs in quotes (e.g., '1 2 3')"),
    ] = None,
    parent_tag: Annotated[
        Optional[str],
        typer.Option("--parent-tag", help="Wait for all jobs with this tag to complete"),
    ] = None,
    app_args: Annotated[
        Optional[str],
        typer.Option("--args", help="Additional arguments to append to the application command (e.g., '--file afile -o 8 bfile')."),
    ] = None,
    status: Annotated[
        str,
        typer.Option("--status", "-s", help="Initial status for the job(s)."),
    ] = "Ready",
):
    """
    Adds one or more new jobs to the database.
    Can add specific directories by path, or all subdirectories with 'all'.
    """
    # Handle CLI-specific interactive prompts and validations
    try:
        # Handle input file prompting for CLI
        final_input_file = input_file
        if input_file is None:
            try:
                app_config = get_app_config(app)
                input_required = app_config["INPUT_REQUIRED"]
                default_input = app_config["DFLT_INPUT"]
                
                if input_required and default_input is None:
                    # Must have input file, no default available - prompt user
                    final_input_file = typer.prompt(f"Input filename is required for {app}")
                elif input_required and default_input is not None:
                    # Input required but has default - inform user
                    final_input_file = default_input
                    typer.secho(f"ℹ️  Using default input file '{default_input}' for {app}", fg=typer.colors.BLUE)
                else:
                    # Input not required - inform user if they provided one anyway
                    if input_file is not None:
                        typer.secho(f"ℹ️  Input file ignored for {app} (not required), using default behavior", fg=typer.colors.YELLOW)
            except ValueError as e:
                typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
                raise typer.Exit(code=1)
        
        # Handle GPU/CPU conflict resolution for CLI
        if ngpus > 0 and node_occupancy is not None:
            typer.secho("⚠️  Warning: Both --ngpus and --nocc specified.", fg=typer.colors.YELLOW)
            typer.secho(f"--ngpus {ngpus} suggests GPU job", fg=typer.colors.BLUE)
            typer.secho(f"--nocc {node_occupancy} suggests CPU job with {node_occupancy} node occupancy", fg=typer.colors.BLUE)
            
            use_gpus = typer.confirm("Do you want to use GPUs for this job?")
            if not use_gpus:
                ngpus = 0
                typer.secho("Setting ngpus=0 for CPU-only job", fg=typer.colors.GREEN)
        
        # Handle Python app environment file confirmation for CLI
        if app == "python" and env_file is None:
            typer.secho("⚠️  Warning: No environment file specified for Python app.", fg=typer.colors.YELLOW)
            typer.secho("Python jobs typically need environment setup (conda activate, module load, etc.)", fg=typer.colors.YELLOW)
            
            proceed_without_env = typer.confirm("Do you want to proceed without an environment file?")
            if not proceed_without_env:
                typer.secho("❌ Job creation cancelled. Please specify an environment file with --envfile/-e", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            env_file = 'pass'  # Special value to indicate no env file for core function
        
        # Parse parents string for CLI
        final_parents = None
        if parents:
            try:
                final_parents = [int(x) for x in parents.split()]
            except ValueError:
                typer.secho("❌ Error: Invalid parent job IDs. Use space-separated integers in quotes.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
        
        # Call the core function
        job_ids, failed_jobs, msg_log = add_jobs(
            paths=paths,
            app=app,
            config_name=config_name,
            tag=tag,
            input_file=final_input_file,
            ngpus=ngpus,
            nnodes=nnodes,
            node_occupancy=node_occupancy,
            ranks_per_node=ranks_per_node,
            mpi_opts=mpi_opts,
            env_file=env_file,
            parents=final_parents,
            parent_tag=parent_tag,
            status=status,
            app_args=app_args,
        )
        
        # Display messages from core function
        for warning in msg_log["warnings"]:
            typer.secho(f"⚠️  Warning: {warning}", fg=typer.colors.YELLOW)
        for info in msg_log["info"]:
            typer.secho(f"ℹ️  {info}", fg=typer.colors.BLUE)
        
        # Display failed jobs with better formatting
        if failed_jobs:
            typer.secho(f"❌ Failed to add {len(failed_jobs)} job(s):", fg=typer.colors.RED)
            for path, error_msg in failed_jobs:
                typer.secho(f"  - {path}: {error_msg}", fg=typer.colors.RED)
        
        # CLI-specific success output
        success_count = len(job_ids)
        
        # Display resource information for CLI users
        if success_count > 0:
            # Get system config for display
            try:
                system_config = get_system_config(config_name)
                
                # Calculate display parameters using helper function
                resource_params, _, _ = validate_resource_parameters(ngpus, nnodes, node_occupancy, ranks_per_node, system_config)
                display_info = calculate_resource_display_info(resource_params, system_config)
                
                # Display resource specification
                typer.secho(f"ℹ️  Resource specification: {display_info['resource_spec_string']}", fg=typer.colors.BLUE)
                
                # Display parent info
                if final_parents:
                    typer.secho(f"ℹ️  Job will depend on parent jobs: {final_parents}", fg=typer.colors.BLUE)
                
            except Exception:
                pass  # Don't fail on display issues
        
        # Display job creation results
        if job_ids:
            input_info = f" (input: {final_input_file})" if final_input_file else " (no input file)"
            typer.secho(f"✅ Added {len(job_ids)} job(s) with IDs: {', '.join(map(str, job_ids))}{input_info}", fg=typer.colors.GREEN)
                
        typer.echo("---")  # Separator
        
        # Summary with both successes and failures
        if success_count > 0 and failed_jobs:
            typer.secho(f"Summary: Successfully added {success_count} job(s), {len(failed_jobs)} failed.", fg=typer.colors.BLUE)
        elif success_count > 0:
            typer.secho(f"Summary: Successfully added {success_count} job(s).", fg=typer.colors.GREEN)
        elif failed_jobs:
            typer.secho(f"Summary: Failed to add {len(failed_jobs)} job(s).", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        else:
            typer.secho("Summary: No jobs were processed.", fg=typer.colors.YELLOW)
        
    except ValidationError as e:
        typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except sqlite3.IntegrityError as e:
        typer.secho(f"⚠️  Skipped: {e}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
