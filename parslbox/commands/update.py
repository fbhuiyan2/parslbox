import typer
from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated
from parslbox.helpers import database, path_utils

app = typer.Typer()

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
    nodealloc: Annotated[
        Optional[float],
        typer.Option("--nodealloc", "-na", help="Update the node allocation fraction for CPU-only jobs (0.0-1.0).")
    ] = None,
    add_deps: Annotated[
        Optional[List[int]],
        typer.Option("--add_deps", "--padd", help="Job IDs to add as new parent dependencies.")
    ] = None,
    rm_deps: Annotated[
        Optional[List[int]],
        typer.Option("--rm_deps", "--parm", help="Job IDs to remove from parent dependencies.")
    ] = None,
):
    """
    Updates one or more fields for a given set of jobs.
    """
    # Validate that at least one update option was provided
    if all(opt is None for opt in [status, app, tag, input_file, ngpus, env_file, nnodes, nodealloc, add_deps, rm_deps]):
        typer.secho("❌ Error: You must provide at least one field to update.", fg=typer.colors.RED)
        typer.echo("Example: pbx update 1 --status Submitted")
        typer.echo("         pbx update 1 --add_deps 2 3 --rm_deps 4")
        raise typer.Exit(code=1)

    # Handle environment file validation and processing
    final_env_file = None
    if env_file:
        # Convert relative path to absolute path
        env_file_path = Path(env_file)
        if not env_file_path.is_absolute():
            env_file_path = Path.cwd() / env_file_path
        
        # Validate that the environment file exists
        if not env_file_path.exists():
            typer.secho(f"❌ Error: Environment file '{env_file}' does not exist.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        if not env_file_path.is_file():
            typer.secho(f"❌ Error: Environment file '{env_file}' is not a file.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        final_env_file = str(env_file_path.resolve())
        typer.secho(f"ℹ️  Using environment file: {final_env_file}", fg=typer.colors.BLUE)

    # Validate nnodes parameter
    if nnodes is not None and nnodes < 1:
        typer.secho(f"❌ Error: --nnodes must be at least 1, got {nnodes}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Validate nodealloc parameter
    if nodealloc is not None and not (0.0 < nodealloc <= 1.0):
        typer.secho(f"❌ Error: --nodealloc must be between 0.0 and 1.0", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Handle nodealloc vs ngpus conflict
    final_ngpus = ngpus
    final_node_occupancy = nodealloc
    
    if nodealloc is not None:
        # Check if any of the jobs currently have ngpus > 0
        jobs = database.get_jobs_by_ids(path_utils.DB_FILE, job_ids)
        jobs_with_gpus = [job for job in jobs if job.get('ngpus', 0) > 0]
        
        if jobs_with_gpus:
            job_ids_with_gpus = [str(job['job_id']) for job in jobs_with_gpus]
            typer.secho(f"⚠️  Warning: Setting node allocation will set ngpus to 0 for jobs: {', '.join(job_ids_with_gpus)}", fg=typer.colors.YELLOW)
            typer.secho("Node allocation is for CPU-only jobs.", fg=typer.colors.YELLOW)
            
            proceed = typer.confirm("Do you want to proceed and set ngpus=0 for these jobs?")
            if not proceed:
                typer.secho("❌ Update cancelled.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            
            # Set ngpus to 0 when nodealloc is specified
            final_ngpus = 0
            typer.secho("Setting ngpus=0 for CPU-only jobs with node allocation", fg=typer.colors.GREEN)

    # Handle dependency management first before other flags
    final_parents = None
    if add_deps is not None or rm_deps is not None:
        # Get current jobs to work with their existing dependencies
        current_jobs = database.get_jobs_by_ids(path_utils.DB_FILE, job_ids)
        if not current_jobs:
            typer.secho("❌ Error: No jobs found with the specified IDs.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        # Validate parent job IDs exist in database
        all_parent_ids = []
        if add_deps:
            all_parent_ids.extend(add_deps)
        if rm_deps:
            all_parent_ids.extend(rm_deps)
        
        if all_parent_ids:
            # Remove duplicates and validate
            unique_parent_ids = list(set(all_parent_ids))
            invalid_parent_ids = database.validate_parent_job_ids(path_utils.DB_FILE, unique_parent_ids)
            
            if invalid_parent_ids:
                typer.secho(f"❌ Error: The following parent job IDs do not exist: {', '.join(map(str, invalid_parent_ids))}", fg=typer.colors.RED)
                raise typer.Exit(code=1)
        
        # Prevent circular dependencies (job can't be parent of itself)
        if add_deps:
            circular_deps = [dep for dep in add_deps if dep in job_ids]
            if circular_deps:
                typer.secho(f"❌ Error: Jobs cannot be parents of themselves: {', '.join(map(str, circular_deps))}", fg=typer.colors.RED)
                raise typer.Exit(code=1)
        
        # Process dependency changes for each job
        dependency_updates = {}
        all_warnings = []
        
        for job in current_jobs:
            job_id = job['job_id']
            existing_parents = database.parse_existing_parents(job.get('parents'))
            updated_parents = existing_parents.copy()
            
            # Process removals first
            if rm_deps:
                updated_parents, not_found = database.remove_dependencies(updated_parents, rm_deps)
                if not_found:
                    all_warnings.append(f"Job {job_id}: Parent IDs {', '.join(map(str, not_found))} were not found in existing dependencies")
            
            # Process additions
            if add_deps:
                updated_parents = database.add_dependencies(updated_parents, add_deps)
            
            dependency_updates[job_id] = {
                'old_parents': existing_parents,
                'new_parents': updated_parents
            }
        
        # Show warnings for non-existent parent removals
        for warning in all_warnings:
            typer.secho(f"⚠️  Warning: {warning}", fg=typer.colors.YELLOW)
        
        # Show summary of dependency changes
        changes_made = False
        for job_id, update_info in dependency_updates.items():
            old_parents = update_info['old_parents']
            new_parents = update_info['new_parents']
            
            if old_parents != new_parents:
                changes_made = True
                #old_str = ', '.join(map(str, old_parents)) if old_parents else 'None'
                #new_str = ', '.join(map(str, new_parents)) if new_parents else 'None'
                typer.secho(f"Job {job_id}: Dependencies updated", fg=typer.colors.BLUE)
        
        if not changes_made:
            typer.secho("ℹ️  No dependency changes were made.", fg=typer.colors.BLUE)
        
        # For database update, we need to use the same parent list for all jobs
        # Since we're updating multiple jobs at once, we'll need to update them individually for dependencies
        if changes_made:
            # Update dependencies for each job individually
            for job_id, update_info in dependency_updates.items():
                new_parents = update_info['new_parents']
                if update_info['old_parents'] != new_parents:
                    database.update_jobs(
                        db_path=path_utils.DB_FILE,
                        job_ids=[job_id],
                        parents=new_parents
                    )
            
            typer.secho("✅ Dependencies updated successfully.", fg=typer.colors.GREEN)

    # Update other fields (non-dependency fields)
    non_dependency_updates = any(opt is not None for opt in [status, app, tag, input_file, final_ngpus, final_env_file, nnodes, final_node_occupancy])
    
    if non_dependency_updates:
        count = database.update_jobs(
            db_path=path_utils.DB_FILE,
            job_ids=job_ids,
            status=status,
            app=app,
            tag=tag,
            in_file=input_file,
            ngpus=final_ngpus,
            env_file=final_env_file,
            num_nodes=nnodes,
            node_occupancy=final_node_occupancy
        )
        
        if count > 0:
            typer.secho(f"🔄 Successfully updated {count} job(s).", fg=typer.colors.BLUE)
        else:
            typer.secho("⚠️ No jobs found with the specified IDs to update.", fg=typer.colors.YELLOW)
    elif add_deps is None and rm_deps is None:
        # meaning no flags were used in the update command
        typer.secho("⚠️ Please specify what to update.", fg=typer.colors.YELLOW)
