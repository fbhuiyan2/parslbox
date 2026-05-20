import typer
from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated
from parslbox.database import database
from parslbox.utils import path_utils
from parslbox.commands.helpers.job_info_validator import (
    ValidationError,
    validate_environment_file,
    validate_input_file,
)

app = typer.Typer()


def update_jobs(
    job_ids: List[int],
    status: Optional[str] = None,
    tag: Optional[str] = None,
    input_file: Optional[str] = None,
    ngpus: Optional[int] = None,
    env_file: Optional[str] = None,
    nnodes: Optional[int] = None,
    node_occupancy: Optional[float] = None,
    ranks_per_node: Optional[int] = None,
    add_deps: Optional[List[int]] = None,
    rm_deps: Optional[List[int]] = None,
    app_args: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> tuple[List[int], List[tuple[int, str]], dict]:
    """
    Core job update logic - used by both CLI and API.

    Args:
        job_ids: List of job IDs to update
        status: New status
        tag: New tag
        input_file: New input file
        ngpus: New number of GPUs
        env_file: New environment file path
        nnodes: New number of nodes
        node_occupancy: New node occupancy
        ranks_per_node: New ranks per node
        add_deps: Parent job IDs to add
        rm_deps: Parent job IDs to remove
        app_args: Additional arguments to append to the application command
        db_path: Database path (uses default if None)
    
    Returns:
        Tuple of (successful_job_ids, failed_jobs, msg_log) where:
        - successful_job_ids: List of job IDs that were successfully updated
        - failed_jobs: List of tuples (job_id, error_message) for failed jobs
        - msg_log: Dictionary with 'warnings' and 'info' lists
        
    Raises:
        ValidationError: If validation fails
    """
    if db_path is None:
        db_path = path_utils.DB_FILE
    
    # --- Use helper functions for validation and collect messages ---
    info_messages = []
    warning_messages = []
    
    # Validate that at least one update option was provided
    if all(opt is None for opt in [status, tag, input_file, ngpus, env_file, nnodes, node_occupancy, ranks_per_node, add_deps, rm_deps, app_args]):
        raise ValidationError("You must provide at least one field to update")

    # Handle app_args: reconstruct in_file by appending args
    if app_args is not None:
        if input_file is not None:
            # User provided both -i and --args: use the new input_file as base
            input_file = f"{input_file} {app_args}".strip()
        else:
            # Only --args provided: extract base script name from current in_file per job
            jobs = database.get_jobs_by_ids(db_path, job_ids)
            for job in jobs:
                base = job['in_file'].split()[0] if job.get('in_file') else None
                if base:
                    new_in_file = f"{base} {app_args}".strip()
                    database.update_jobs(
                        db_path=db_path,
                        job_ids=[job['job_id']],
                        in_file=new_in_file
                    )
                    info_messages.append(f"Job {job['job_id']}: updated args → {new_in_file}")

    # Basic parameter validation
    if nnodes is not None and nnodes < 1:
        raise ValidationError(f"--nnodes must be at least 1, got {nnodes}")

    if ngpus is not None and ngpus < 0:
        raise ValidationError(f"--ngpus must be non-negative, got {ngpus}")

    if node_occupancy is not None and not (0.0 < node_occupancy <= 1.0):
        raise ValidationError("--nocc must be between 0.0 and 1.0")

    if ranks_per_node is not None and ranks_per_node < 1:
        raise ValidationError(f"--ranks-per-node must be a positive integer, got {ranks_per_node}")

    # Handle environment file validation and processing
    final_env_file, env_info, env_warnings = validate_environment_file(env_file)
    info_messages.extend(env_info)
    warning_messages.extend(env_warnings)

    # Handle input file validation for existing jobs
    failed_jobs = []
    final_input_file = input_file
    
    if input_file is not None:
        # Validate input file against each job's existing app
        current_jobs = database.get_jobs_by_ids(db_path, job_ids)
        for job in current_jobs:
            job_id = job['job_id']
            job_app = job['app']
            
            try:
                # Validate input file against the job's existing app
                validated_input, input_info, input_warnings = validate_input_file(job_app, input_file)
                
                # Fail the job if there are any warnings (e.g., input file ignored)
                if input_warnings:
                    failed_jobs.append((job_id, f"Cannot update input file to '{input_file}' for job {job_id} app '{job_app}': {'; '.join(input_warnings)}"))

                
            except ValidationError as e:
                # Input file validation failed for this job's app
                failed_jobs.append((job_id, f"Input file '{input_file}' is not compatible with job {job_id} app '{job_app}': {str(e)}"))
    
    # Check for direct conflict: both ngpus and node_occupancy specified
    if ngpus is not None and node_occupancy is not None and ngpus > 0 and node_occupancy > 0:
        raise ValidationError("Cannot specify both ngpus and node_occupancy to be > 0.")
    
    # Validate GPU/CPU conflicts for each job
    if node_occupancy is not None and ngpus is None:
        # Setting node_occupancy but not ngpus - check if jobs currently have GPUs
        jobs = database.get_jobs_by_ids(db_path, job_ids)
        for job in jobs:
            job_id = job['job_id']
            current_ngpus = job.get('ngpus', 0)
            if current_ngpus > 0:
                failed_jobs.append((job_id, f"Cannot set node_occupancy for job {job_id} which currently has {current_ngpus} GPUs. Set ngpus=0 first."))
    
    # Handle GPU scaling when nnodes is updated without explicit ngpus
    # This needs to be done per-job since each job may have different GPU configurations
    per_job_gpu_updates = {}  # Maps job_id -> new_ngpus value
    
    if nnodes is not None and ngpus is None:
        # Check if any jobs are GPU jobs that need GPU count adjustment
        jobs = database.get_jobs_by_ids(db_path, job_ids)
        for job in jobs:
            job_id = job['job_id']
            current_ngpus = job.get('ngpus', 0)
            current_nnodes = job.get('num_nodes', 1)
            
            if current_ngpus > 0:  # This is a GPU job
                if current_nnodes == 1 and nnodes > 1:
                    # Case 2: Single-node → Multi-node
                    failed_jobs.append((job_id, 
                        f"Cannot automatically scale GPU job {job_id} from 1 to {nnodes} nodes. "
                        f"Please use -g flag to specify total GPUs (total_gpus = nnodes * gpus_per_node)."))
                elif current_nnodes > 1:
                    # Case 1: Multi-node → different node count
                    gpus_per_node = current_ngpus / current_nnodes
                    if not gpus_per_node.is_integer():
                        failed_jobs.append((job_id,
                            f"Job {job_id} has {current_ngpus} GPUs across {current_nnodes} nodes "
                            f"({gpus_per_node:.2f} GPUs per node - not an integer). "
                            f"Cannot auto-scale. Please use -g flag to specify total GPUs."))
                    else:
                        # Auto-calculate new GPU count for this job
                        new_ngpus = int(gpus_per_node * nnodes)
                        per_job_gpu_updates[job_id] = new_ngpus
                        info_messages.append(
                            f"Auto-scaling job {job_id}: {current_ngpus} GPUs on {current_nnodes} nodes "
                            f"→ {new_ngpus} GPUs on {nnodes} nodes ({int(gpus_per_node)} GPUs/node)")
        
    # Remove failed job IDs from the list to process
    if failed_jobs:
        failed_job_ids = {job_id for job_id, _ in failed_jobs}
        job_ids = [job_id for job_id in job_ids if job_id not in failed_job_ids]
    
    # Handle node occupancy vs ngpus conflict
    final_ngpus = ngpus
    final_node_occupancy = node_occupancy
    

    # Handle dependency management
    dependency_updated_job_ids = []
    if add_deps is not None or rm_deps is not None:
        # Get current jobs to work with their existing dependencies
        current_jobs = database.get_jobs_by_ids(db_path, job_ids)
        if not current_jobs:
            raise ValidationError("No jobs found with the specified IDs")
        
        # Validate parent job IDs exist in database
        all_parent_ids = []
        if add_deps:
            all_parent_ids.extend(add_deps)
        if rm_deps:
            all_parent_ids.extend(rm_deps)
        
        if all_parent_ids:
            # Remove duplicates and validate
            unique_parent_ids = list(set(all_parent_ids))
            invalid_parent_ids = database.validate_parent_job_ids(db_path, unique_parent_ids)
            
            if invalid_parent_ids:
                raise ValidationError(f"The following parent job IDs do not exist: {', '.join(map(str, invalid_parent_ids))}")
        
        # Prevent circular dependencies (job can't be parent of itself)
        if add_deps:
            circular_deps = [dep for dep in add_deps if dep in job_ids]
            if circular_deps:
                raise ValidationError(f"Jobs cannot be parents of themselves: {', '.join(map(str, circular_deps))}")
        
        # Process dependency changes for each job
        for job in current_jobs:
            job_id = job['job_id']
            existing_parents = database.parse_existing_parents(job.get('parents'))
            updated_parents = existing_parents.copy()
            
            # Process removals first
            if rm_deps:
                updated_parents, not_found = database.remove_dependencies(updated_parents, rm_deps)
            
            # Process additions
            if add_deps:
                updated_parents = database.add_dependencies(updated_parents, add_deps)
            
            # Update if changed
            if existing_parents != updated_parents:
                database.update_jobs(
                    db_path=db_path,
                    job_ids=[job_id],
                    parents=updated_parents
                )
                dependency_updated_job_ids.append(job_id)

    # Update other fields (non-dependency fields)
    non_dependency_updates = any(opt is not None for opt in [status, tag, input_file, final_ngpus, final_env_file, nnodes, final_node_occupancy, ranks_per_node])
    
    non_dependency_updated_job_ids = []
    if non_dependency_updates:
        # Get jobs that actually exist before updating
        existing_jobs = database.get_jobs_by_ids(db_path, job_ids)
        existing_job_ids = [job['job_id'] for job in existing_jobs]
        
        if existing_job_ids:
            # Check if we have per-job GPU updates (from auto-scaling)
            if per_job_gpu_updates:
                # Update jobs individually when they have different GPU counts
                for job_id in existing_job_ids:
                    # Use per-job GPU value if available, otherwise use the common value
                    job_ngpus = per_job_gpu_updates.get(job_id, final_ngpus)
                    
                    updated_count = database.update_jobs(
                        db_path=db_path,
                        job_ids=[job_id],
                        status=status,
                        tag=tag,
                        in_file=final_input_file,
                        ngpus=job_ngpus,
                        env_file=final_env_file,
                        num_nodes=nnodes,
                        node_occupancy=final_node_occupancy,
                        ranks_per_node=ranks_per_node
                    )
                    if updated_count > 0:
                        non_dependency_updated_job_ids.append(job_id)
            else:
                # No per-job GPU updates - update all jobs with same values
                updated_count = database.update_jobs(
                    db_path=db_path,
                    job_ids=existing_job_ids,
                    status=status,
                    tag=tag,
                    in_file=final_input_file,
                    ngpus=final_ngpus,
                    env_file=final_env_file,
                    num_nodes=nnodes,
                    node_occupancy=final_node_occupancy,
                    ranks_per_node=ranks_per_node
                )
                # If update was successful, all existing jobs were updated
                if updated_count > 0:
                    non_dependency_updated_job_ids = existing_job_ids
    
    # Combine all updated job IDs and remove duplicates
    all_updated_job_ids = list(set(dependency_updated_job_ids + non_dependency_updated_job_ids))
    
    # Create message log dictionary
    msg_log = {
        "warnings": warning_messages,
        "info": info_messages
    }
    
    return sorted(all_updated_job_ids), failed_jobs, msg_log

@app.command()
def update(
    job_ids: Annotated[
        List[str],
        typer.Argument(help="ID(s) of the job(s) to update. Supports ranges (e.g., 1-5 8 14-20).")
    ],
    status: Annotated[
        Optional[str],
        typer.Option("--status", "-s", help="Update the job status.")
    ] = None,
    tag: Annotated[
        Optional[str],
        typer.Option("--tag", "-t", help="Update the job tag.")
    ] = None,
    input_file: Annotated[
        Optional[str],
        typer.Option("--input", "-i", help="Update the input filename.")
    ] = None,
    ngpus: Annotated[
        Optional[int],
        typer.Option("--ngpus", "-g", help="Update the number of GPUs.")
    ] = None,
    env_file: Annotated[
        Optional[str],
        typer.Option("--envfile", "-e", help="Update the environment setup file path.")
    ] = None,
    nnodes: Annotated[
        Optional[int],
        typer.Option("--nnodes", "-n", help="Update the number of nodes required for the job(s).")
    ] = None,
    node_occupancy: Annotated[
        Optional[float],
        typer.Option("--nocc", "-o", help="Update the node occupancy fraction for CPU-only jobs (0.0-1.0).")
    ] = None,
    ranks_per_node: Annotated[
        Optional[int],
        typer.Option("--ranks-per-node", "-rpn", help="Update the number of MPI ranks per node.")
    ] = None,
    add_deps: Annotated[
        Optional[str],
        typer.Option("--add_deps", "--padd", help="Space-separated job IDs to add as parents (e.g., '1 2 3')")
    ] = None,
    rm_deps: Annotated[
        Optional[str],
        typer.Option("--rm_deps", "--parm", help="Space-separated job IDs to remove from parents (e.g., '1 2 3')")
    ] = None,
    app_args: Annotated[
        Optional[str],
        typer.Option("--args", help="Update the arguments appended to the application command (e.g., '--file afile -o 8 bfile')."),
    ] = None,
):
    """
    Updates one or more fields for a given set of jobs.
    """
    from parslbox.commands.helpers.job_id_parser import parse_job_ids
    try:
        job_ids = parse_job_ids(job_ids)
    except ValueError as e:
        typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        # Handle CLI-specific interactive prompts and validations
        
        # Handle node occupancy vs ngpus conflict for CLI
        if node_occupancy is not None:
            # Check if any of the jobs currently have ngpus > 0
            jobs = database.get_jobs_by_ids(path_utils.DB_FILE, job_ids)
            jobs_with_gpus = [job for job in jobs if job.get('ngpus', 0) > 0]
            
            if jobs_with_gpus:
                job_ids_with_gpus = [str(job['job_id']) for job in jobs_with_gpus]
                typer.secho(f"⚠️  Warning: Setting node occupancy will set ngpus to 0 for jobs: {', '.join(job_ids_with_gpus)}", fg=typer.colors.YELLOW)
                typer.secho("Node occupancy is for CPU-only jobs.", fg=typer.colors.YELLOW)
                
                proceed = typer.confirm("Do you want to proceed and set ngpus=0 for these jobs?")
                if not proceed:
                    typer.secho("❌ Update cancelled.", fg=typer.colors.RED)
                    raise typer.Exit(code=1)
                
                # Set ngpus to 0 when node occupancy is specified
                ngpus = 0
                typer.secho("Setting ngpus=0 for CPU-only jobs with node occupancy", fg=typer.colors.GREEN)

        # Parse dependency strings for CLI
        parsed_add_deps = None
        parsed_rm_deps = None
        
        if add_deps:
            try:
                parsed_add_deps = [int(x) for x in add_deps.split()]
            except ValueError:
                typer.secho("❌ Error: Invalid add_deps job IDs. Use space-separated integers in quotes.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
        
        if rm_deps:
            try:
                parsed_rm_deps = [int(x) for x in rm_deps.split()]
            except ValueError:
                typer.secho("❌ Error: Invalid rm_deps job IDs. Use space-separated integers in quotes.", fg=typer.colors.RED)
                raise typer.Exit(code=1)

        # Environment file info will be displayed by the validator helper function

        # Call the core function
        updated_job_ids, failed_jobs, msg_log = update_jobs(
            job_ids=job_ids,
            status=status,
            tag=tag,
            input_file=input_file,
            ngpus=ngpus,
            env_file=env_file,
            nnodes=nnodes,
            node_occupancy=node_occupancy,
            ranks_per_node=ranks_per_node,
            add_deps=parsed_add_deps,
            rm_deps=parsed_rm_deps,
            app_args=app_args,
        )
        
        # Display messages from core function
        for warning in msg_log["warnings"]:
            typer.secho(f"⚠️  {warning}", fg=typer.colors.YELLOW)
        for info in msg_log["info"]:
            typer.secho(f"ℹ️  {info}", fg=typer.colors.BLUE)
        
        # Display failed jobs
        for job_id, error_msg in failed_jobs:
            typer.secho(f"❌ {error_msg}", fg=typer.colors.RED)
        
        # CLI-specific success output
        if updated_job_ids:
            typer.secho(f"🔄 Successfully updated {len(updated_job_ids)} job(s): {', '.join(map(str, updated_job_ids))}", fg=typer.colors.BLUE)
        else:
            typer.secho("⚠️ No jobs were updated.", fg=typer.colors.YELLOW)
        
        # Provide summary if there were both successes and failures
        if updated_job_ids and failed_jobs:
            typer.secho(f"ℹ️  Summary: {len(updated_job_ids)} succeeded, {len(failed_jobs)} failed", fg=typer.colors.BLUE)
        
    except ValidationError as e:
        typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
