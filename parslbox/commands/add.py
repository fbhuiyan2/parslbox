import typer
import sqlite3
from pathlib import Path
from typing import List, Optional
from typing_extensions import Annotated
from parslbox.helpers import database, path_utils
from parslbox.apps.app_registry import get_app_config, is_app_registered
from parslbox.configs.loader import get_system_config

app = typer.Typer()

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
        typer.Option("--app", "-a", help="The application type (e.g., 'lammps', 'vasp')."),
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
    nodealloc: Annotated[
        Optional[float],
        typer.Option("--nodealloc", "-na", help="Node allocation fraction for CPU-only jobs (0.0-1.0)."),
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
    status: Annotated[
        str,
        typer.Option("--status", "-s", help="Initial status for the job(s)."),
    ] = "Ready",
):
    """
    Adds one or more new jobs to the database.
    Can add specific directories by path, or all subdirectories with 'all'.
    """
    # --- Validate application exists in registry ---
    if not is_app_registered(app):
        typer.secho(f"❌ Error: Unknown application '{app}'.", fg=typer.colors.RED)
        from parslbox.apps.app_registry import get_registered_apps
        available_apps = ", ".join(get_registered_apps())
        typer.secho(f"Available applications: {available_apps}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)

    # --- Handle input file logic based on app configuration ---
    try:
        app_config = get_app_config(app)
        input_required = app_config["INPUT_REQUIRED"]
        default_input = app_config["DFLT_INPUT"]
        
        final_input_file = None
        
        if input_required and default_input is None:
            # Must have input file, no default available
            if input_file is None:
                final_input_file = typer.prompt(f"Input filename is required for {app}")
            else:
                final_input_file = input_file
        elif input_required and default_input is not None:
            # Input required but has default
            if input_file is None:
                final_input_file = default_input
                typer.secho(f"ℹ️  Using default input file '{default_input}' for {app}", fg=typer.colors.BLUE)
            else:
                final_input_file = input_file
        else:
            # Input not required (input_required = False)
            if input_file is not None:
                typer.secho(f"ℹ️  Input file ignored for {app} (not required), using default behavior", fg=typer.colors.YELLOW)
            final_input_file = default_input  # Will be None for apps that don't need input
            
    except ValueError as e:
        typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # --- Validate resource parameters and generate resource string ---
    # Validate nodealloc range
    if nodealloc is not None and not (0.0 < nodealloc <= 1.0):
        typer.secho(f"❌ Error: --nodealloc must be between 0.0 and 1.0", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    # Validate nnodes
    if nnodes < 1:
        typer.secho(f"❌ Error: --nnodes must be at least 1, got {nnodes}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    # Get system configuration to determine GPUs per node
    try:
        system_config = get_system_config(config_name)
        gpus_per_node = system_config.GPUS_PER_NODE
    except Exception as e:
        typer.secho(f"❌ Error: Could not load system configuration: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    # Determine final resource parameters
    if nnodes > 1:
        # Multi-node job: ignore ngpus and nodealloc
        if ngpus > 0 or nodealloc is not None:
            ignored_flags = []
            if ngpus > 0:
                ignored_flags.append(f"--ngpus {ngpus}")
            if nodealloc is not None:
                ignored_flags.append(f"--nodealloc {nodealloc}")
            typer.secho(f"⚠️  Warning: Ignoring {' and '.join(ignored_flags)} for multi-node job", fg=typer.colors.YELLOW)
        
        # Multi-node jobs: set parameters
        final_num_nodes = nnodes
        final_ngpus = 0  # Not used for multi-node jobs
        final_node_occupancy = 1.0
        typer.secho(f"ℹ️  Multi-node job will use {nnodes * gpus_per_node} total GPUs ({gpus_per_node} per node)", fg=typer.colors.BLUE)
    else:
        # Single-node job: check for conflicting parameters
        if ngpus > 0 and nodealloc is not None:
            typer.secho("⚠️  Warning: Both --ngpus and --nodealloc specified.", fg=typer.colors.YELLOW)
            typer.secho(f"--ngpus {ngpus} suggests GPU job", fg=typer.colors.BLUE)
            typer.secho(f"--nodealloc {nodealloc} suggests CPU job with {nodealloc} node occupancy", fg=typer.colors.BLUE)
            
            use_gpus = typer.confirm("Do you want to use GPUs for this job?")
            if not use_gpus:
                ngpus = 0
                typer.secho("Setting ngpus=0 for CPU-only job", fg=typer.colors.GREEN)
        
        if ngpus > 0:
            # Single-node GPU job
            if ngpus > gpus_per_node:
                typer.secho(f"❌ Error: Requested {ngpus} GPUs but only {gpus_per_node} available per node", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            
            final_num_nodes = 1
            final_ngpus = ngpus
            final_node_occupancy = 1.0
        else:
            # Single-node CPU-only job
            final_num_nodes = 1
            final_ngpus = 0
            final_node_occupancy = nodealloc if nodealloc is not None else 1.0
    
    # Display resource specification for user
    if final_num_nodes > 1:
        display_str = f"n:{final_num_nodes}-g:auto-nocc:NA"
    elif final_ngpus > 0:
        display_str = f"n:1-g:{final_ngpus}-nocc:NA"
    else:
        display_str = f"n:1-g:0-nocc:{final_node_occupancy}"
    
    typer.secho(f"ℹ️  Resource specification: {display_str}", fg=typer.colors.BLUE)

    # --- Handle environment file validation and processing ---
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
    
    # --- Special handling for Python app without environment file ---
    elif app == "python":
        typer.secho("⚠️  Warning: No environment file specified for Python app.", fg=typer.colors.YELLOW)
        typer.secho("Python jobs typically need environment setup (conda activate, module load, etc.)", fg=typer.colors.YELLOW)
        
        proceed_without_env = typer.confirm("Do you want to proceed without an environment file?")
        if not proceed_without_env:
            typer.secho("❌ Job creation cancelled. Please specify an environment file with --envfile/-e", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    # --- Handle parent dependencies ---
    final_parents = []
    
    # Parse parents string if provided
    if parents:
        try:
            final_parents = [int(x) for x in parents.split()]
        except ValueError:
            typer.secho("❌ Error: Invalid parent job IDs. Use space-separated integers in quotes.", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    
    # Handle parent_tag conversion to parent IDs
    if parent_tag:
        # Get all jobs with the specified tag that are Done
        tag_jobs = database.get_jobs(path_utils.DB_FILE, tag=parent_tag, status='Done')
        tag_parent_ids = [job['job_id'] for job in tag_jobs]
        final_parents.extend(tag_parent_ids)
        
        if tag_parent_ids:
            typer.secho(f"ℹ️  Added {len(tag_parent_ids)} parent jobs from tag '{parent_tag}': {tag_parent_ids}", 
                       fg=typer.colors.BLUE)
        else:
            typer.secho(f"⚠️  Warning: No completed jobs found with tag '{parent_tag}'", 
                       fg=typer.colors.YELLOW)
    
    # Validate parent job IDs exist
    if final_parents:
        existing_jobs = database.get_jobs_by_ids(path_utils.DB_FILE, final_parents)
        existing_ids = {job['job_id'] for job in existing_jobs}
        missing_ids = set(final_parents) - existing_ids
        
        if missing_ids:
            typer.secho(f"❌ Error: Parent job IDs do not exist: {sorted(missing_ids)}", 
                       fg=typer.colors.RED)
            raise typer.Exit(code=1)
        
        # Check for circular dependencies (basic check - job can't depend on itself) 
        # ^ This cannot happen since the job id for this job does not exist yet and non-existing job ids lead to missing_ids error
        # More sophisticated cycle detection could be added later
        typer.secho(f"ℹ️  Job will depend on parent jobs: {final_parents}", fg=typer.colors.BLUE)

    paths_to_add: List[Path] = []

    # --- Determine the list of paths to process ---
    if len(paths) == 1 and paths[0].lower() == 'all':
        typer.secho("Scanning current directory for subdirectories...", fg=typer.colors.BLUE)
        current_dir = Path.cwd()
        subdirectories = [p for p in current_dir.iterdir() if p.is_dir()]
        
        if not subdirectories:
            typer.secho("No subdirectories found in the current directory.", fg=typer.colors.YELLOW)
            raise typer.Exit()
            
        paths_to_add = [p.resolve() for p in subdirectories]
        typer.secho(f"Found {len(paths_to_add)} directories to add.", fg=typer.colors.BLUE)
    else:
        # Validate user-provided paths
        for path_str in paths:
            path_obj = Path(path_str)
            if not path_obj.exists():
                typer.secho(f"❌ Error: Path '{path_str}' does not exist.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            if not path_obj.is_dir():
                typer.secho(f"❌ Error: Path '{path_str}' is not a directory.", fg=typer.colors.RED)
                raise typer.Exit(code=1)
            paths_to_add.append(path_obj.resolve())

    # --- Add the determined paths to the database ---
    success_count = 0
    fail_count = 0

    for path in paths_to_add:
        try:
            new_id = database.add_job(
                db_path=path_utils.DB_FILE,
                path=str(path),
                app=app,
                num_nodes=final_num_nodes,
                ngpus=final_ngpus,
                node_occupancy=final_node_occupancy,
                tag=tag,
                in_file=final_input_file,
                mpi_opts=mpi_opts,
                env_file=final_env_file,
                parents=final_parents,
                status=status
            )
            input_info = f" (input: {final_input_file})" if final_input_file else " (no input file)"
            parent_info = f" (parents: {final_parents})" if final_parents else ""
            typer.secho(f"✅ Added job '{path}' with ID {new_id}{input_info}{parent_info}", fg=typer.colors.GREEN)
            success_count += 1
        except sqlite3.IntegrityError:
            typer.secho(f"⚠️  Skipped: Job path '{path}' already exists in the database.", fg=typer.colors.YELLOW)
            fail_count += 1

    typer.echo("---") # Separator
    if success_count > 0:
        typer.secho(f"Summary: Successfully added {success_count} job(s).", fg=typer.colors.GREEN)
    if fail_count > 0:
        typer.secho(f"Summary: Skipped {fail_count} job(s) that already existed.", fg=typer.colors.YELLOW)
