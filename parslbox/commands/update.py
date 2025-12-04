import typer
from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated
from parslbox.helpers import database, path_utils

app = typer.Typer()


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def update_jobs(
    job_ids: List[int],
    status: Optional[str] = None,
    app: Optional[str] = None,
    tag: Optional[str] = None,
    input_file: Optional[str] = None,
    ngpus: Optional[int] = None,
    env_file: Optional[str] = None,
    nnodes: Optional[int] = None,
    node_occupancy: Optional[float] = None,
    ranks_per_node: Optional[int] = None,
    add_deps: Optional[List[int]] = None,
    rm_deps: Optional[List[int]] = None,
    interactive_prompts: bool = True,
    db_path: Optional[Path] = None,
) -> List[int]:
    """
    Core job update logic - used by both CLI and API.
    
    Args:
        job_ids: List of job IDs to update
        status: New status
        app: New application
        tag: New tag
        input_file: New input file
        ngpus: New number of GPUs
        env_file: New environment file path
        nnodes: New number of nodes
        node_occupancy: New node occupancy
        ranks_per_node: New ranks per node
        add_deps: Parent job IDs to add
        rm_deps: Parent job IDs to remove
        interactive_prompts: Whether to allow interactive prompts (CLI only)
        db_path: Database path (uses default if None)
    
    Returns:
        List of job IDs that were successfully updated
        
    Raises:
        ValidationError: If validation fails
    """
    if db_path is None:
        db_path = path_utils.DB_FILE
    
    # Validate that at least one update option was provided
    if all(opt is None for opt in [status, app, tag, input_file, ngpus, env_file, nnodes, node_occupancy, ranks_per_node, add_deps, rm_deps]):
        raise ValidationError("You must provide at least one field to update")

    # Validate parameters
    if nnodes is not None and nnodes < 1:
        raise ValidationError(f"--nnodes must be at least 1, got {nnodes}")

    if node_occupancy is not None and not (0.0 < node_occupancy <= 1.0):
        raise ValidationError("--nocc must be between 0.0 and 1.0")

    if ranks_per_node is not None and ranks_per_node < 1:
        raise ValidationError(f"--ranks-per-node must be a positive integer, got {ranks_per_node}")

    # Handle environment file validation and processing
    final_env_file = None
    if env_file:
        # Convert relative path to absolute path
        env_file_path = Path(env_file)
        if not env_file_path.is_absolute():
            env_file_path = Path.cwd() / env_file_path
        
        # Validate that the environment file exists
        if not env_file_path.exists():
            raise ValidationError(f"Environment file '{env_file}' does not exist")
        
        if not env_file_path.is_file():
            raise ValidationError(f"Environment file '{env_file}' is not a file")
        
        final_env_file = str(env_file_path.resolve())

    # Handle node occupancy vs ngpus conflict
    final_ngpus = ngpus
    final_node_occupancy = node_occupancy
    
    if node_occupancy is not None:
        # Check if any of the jobs currently have ngpus > 0
        jobs = database.get_jobs_by_ids(db_path, job_ids)
        jobs_with_gpus = [job for job in jobs if job.get('ngpus', 0) > 0]
        
        if jobs_with_gpus and interactive_prompts:
            # This will be handled by CLI wrapper
            raise ValidationError("Setting node occupancy for GPU jobs requires interactive confirmation")
        elif jobs_with_gpus:
            # API usage - automatically set ngpus to 0
            final_ngpus = 0

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
    non_dependency_updates = any(opt is not None for opt in [status, app, tag, input_file, final_ngpus, final_env_file, nnodes, final_node_occupancy, ranks_per_node])
    
    non_dependency_updated_job_ids = []
    if non_dependency_updates:
        # Get jobs that actually exist before updating
        existing_jobs = database.get_jobs_by_ids(db_path, job_ids)
        existing_job_ids = [job['job_id'] for job in existing_jobs]
        
        if existing_job_ids:
            updated_count = database.update_jobs(
                db_path=db_path,
                job_ids=existing_job_ids,
                status=status,
                app=app,
                tag=tag,
                in_file=input_file,
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
    return sorted(all_updated_job_ids)

@app.command()
def update(
    job_ids: Annotated[
        List[int],
        typer.Argument(help="ID(s) of the job(s) to update.")
    ],
    status: Annotated[
        Optional[str],
        typer.Option("--status", "-s", help="Update the job status.")
    ] = None,
    app: Annotated[
        Optional[str],
        typer.Option("--app", "-a", help="Update the job application.")
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
        typer.Option("--ranks-per-node", "-rpn", help="Update the number of MPI ranks per node. For CPU jobs only; ignored for GPU jobs.")
    ] = None,
    add_deps: Annotated[
        Optional[str],
        typer.Option("--add_deps", "--padd", help="Space-separated job IDs to add as parents (e.g., '1 2 3')")
    ] = None,
    rm_deps: Annotated[
        Optional[str],
        typer.Option("--rm_deps", "--parm", help="Space-separated job IDs to remove from parents (e.g., '1 2 3')")
    ] = None,
):
    """
    Updates one or more fields for a given set of jobs.
    """
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

        # Display environment file info for CLI
        if env_file:
            env_file_path = Path(env_file)
            if not env_file_path.is_absolute():
                env_file_path = Path.cwd() / env_file_path
            typer.secho(f"ℹ️  Using environment file: {env_file_path.resolve()}", fg=typer.colors.BLUE)

        # Call the core function
        updated_job_ids = update_jobs(
            job_ids=job_ids,
            status=status,
            app=app,
            tag=tag,
            input_file=input_file,
            ngpus=ngpus,
            env_file=env_file,
            nnodes=nnodes,
            node_occupancy=node_occupancy,
            ranks_per_node=ranks_per_node,
            add_deps=parsed_add_deps,
            rm_deps=parsed_rm_deps,
            interactive_prompts=True,
        )
        
        # CLI-specific success output
        if updated_job_ids:
            typer.secho(f"🔄 Successfully updated {len(updated_job_ids)} job(s): {', '.join(map(str, updated_job_ids))}", fg=typer.colors.BLUE)
        else:
            typer.secho("⚠️ No jobs were updated.", fg=typer.colors.YELLOW)
        
    except ValidationError as e:
        typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
