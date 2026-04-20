import math
import typer
from typing import List, Optional
from rich.console import Console
from rich.table import Table

from parslbox.database import database
from parslbox.utils import path_utils

console = Console()

app = typer.Typer()


def parse_parents(parents_str: str) -> List[int]:
    """Parse JSON parent string to list of integers"""
    if not parents_str:
        return []
    import json
    return [int(x) for x in json.loads(parents_str)]


def format_job_id_with_parents(job_id: int, parents: List[int], show_all: bool = False) -> str:
    """Format job ID with parent dependencies"""
    if not parents:
        return str(job_id)
    
    if show_all or len(parents) <= 4:
        parents_str = ",".join(str(p) for p in parents)
        return f"{job_id} ({parents_str})"
    else:
        # Truncate after 4 parents: "100 (1,2,...,5)"
        first_parents = ",".join(str(p) for p in parents[:2])
        last_parent = parents[-1]
        return f"{job_id} ({first_parents},...,{last_parent})"


# Main CLI command

@app.command()
def info(
    job_ids: List[str] = typer.Argument(..., help="ID(s) of the job(s) to get information about. Supports ranges (e.g., 1-5 8 14-20)."),
    path: bool = typer.Option(False, "--path", "-p", help="Show only the path field."),
    ngpus: bool = typer.Option(False, "--ngpus", "-n", help="Show only the number of GPUs field."),
    app_name: bool = typer.Option(False, "--app", "-a", help="Show only the application field."),
    status: bool = typer.Option(False, "--status", "-s", help="Show only the status field."),
    tag: bool = typer.Option(False, "--tag", "-t", help="Show only the tag field."),
    input_file: bool = typer.Option(False, "--input", "-i", help="Show only the input file field."),
    sched_job_id: bool = typer.Option(False, "--sched-job-id", "-j", help="Show only the scheduler job ID field."),
    timestamp: bool = typer.Option(False, "--timestamp", "-ts", help="Show only the timestamp field."),
    env_file: bool = typer.Option(False, "--envfile", "-e", help="Show only the environment file field."),
    parents: bool = typer.Option(False, "--parents", "-P", help="Show all parent dependencies without truncation."),
    req: Optional[str] = typer.Option(None, "--req", "-r", help="Calculate resource requirements for a target system (e.g., polaris, crux, sophia)."),
    cmdline: Optional[str] = typer.Option(None, "--cmdline", "-c", help="Show MPI/srun command line for a target system (e.g., polaris, crux, sophia)."),
):
    """
    Shows detailed information about specific jobs.
    """
    from parslbox.commands.helpers.job_id_parser import parse_job_ids
    try:
        job_ids = parse_job_ids(job_ids)
    except ValueError as e:
        typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # Get jobs from database
    jobs = database.get_jobs_by_ids(path_utils.DB_FILE, job_ids)
    
    # Check if any jobs were found
    found_job_ids = {job['job_id'] for job in jobs}
    missing_job_ids = set(job_ids) - found_job_ids
    
    if missing_job_ids:
        missing_ids_str = ", ".join(map(str, sorted(missing_job_ids)))
        console.print(f"[yellow]⚠️ Warning: Job ID(s) {missing_ids_str} not found in database.[/yellow]")
    
    if not jobs:
        console.print("[red]❌ No jobs found with the specified IDs.[/red]")
        raise typer.Exit(code=1)
    
    # Determine which fields to show
    selected_fields = []
    if path:
        selected_fields.append(('path', 'Path'))
    if ngpus:
        selected_fields.append(('ngpus', 'NGPUs'))
    if app_name:
        selected_fields.append(('app', 'App'))
    if status:
        selected_fields.append(('status', 'Status'))
    if tag:
        selected_fields.append(('tag', 'Tag'))
    if input_file:
        selected_fields.append(('in_file', 'Input'))
    if sched_job_id:
        selected_fields.append(('sched_job_id', 'Sched Job ID'))
    if timestamp:
        selected_fields.append(('timestamp', 'Timestamp'))
    if env_file:
        selected_fields.append(('env_file', 'Env File'))
    if parents:
        selected_fields.append(('parents', 'Parents'))
    
    # If no specific fields selected, show all fields
    if not selected_fields:
        selected_fields = [
            ('job_id', 'ID'),
            ('app', 'App'),
            ('status', 'Status'),
            ('ngpus', 'NGPUs'),
            ('sched_job_id', 'Sched Job ID'),
            ('tag', 'Tag'),
            ('in_file', 'Input'),
            ('env_file', 'Env File'),
            ('timestamp', 'Timestamp'),
            ('path', 'Path')
        ]
    else:
        # Always include job_id as the first column when specific fields are selected
        selected_fields.insert(0, ('job_id', 'ID'))
    
    # Create and populate table
    headers = [field[1] for field in selected_fields]
    
    # Check if path is being displayed to configure wrapping
    path_in_fields = any(field[0] == 'path' for field in selected_fields)
    
    if path_in_fields:
        # Configure table to allow wrapping for long paths
        table = Table(*headers, expand=True)
        # Find the path column index and configure it for wrapping
        for i, (field_key, _) in enumerate(selected_fields):
            if field_key == 'path':
                table.columns[i].no_wrap = False
                table.columns[i].overflow = "fold"
    else:
        table = Table(*headers)
    
    for job in jobs:
        row_data = []
        for field_key, _ in selected_fields:
            value = job[field_key]
            if value is None:
                row_data.append("None")
            elif field_key == 'ngpus':
                row_data.append(str(value))
            elif field_key == 'job_id':
                # When parents flag is used, show plain job ID (parents will be in separate column)
                if parents:
                    row_data.append(str(job['job_id']))
                else:
                    # Format job ID with parent dependencies
                    job_parents = parse_parents(job.get('parents'))
                    formatted_id = format_job_id_with_parents(job['job_id'], job_parents, show_all=parents)
                    row_data.append(formatted_id)
            elif field_key == 'parents':
                # Handle the separate parents column
                job_parents = parse_parents(job.get('parents'))
                if job_parents:
                    parents_str = ",".join(str(p) for p in job_parents)
                    row_data.append(parents_str)
                else:
                    row_data.append("None")
            else:
                row_data.append(str(value))
        table.add_row(*row_data)
    
    # Print table unless --req or --cmdline is used (those have their own output)
    if not req and not cmdline:
        console.print(table)

        # Show summary
        if len(jobs) == 1:
            console.print(f"[green]Showing information for 1 job.[/green]")
        else:
            console.print(f"[green]Showing information for {len(jobs)} jobs.[/green]")

    # Resource requirement analysis
    if req:
        _print_resource_requirements(jobs, req)

    # Command line preview
    if cmdline:
        _print_cmdline(jobs, cmdline)


def _print_resource_requirements(jobs: list, system_name: str):
    """Print resource requirement analysis for jobs against a target system."""
    from parslbox.system_configs.loader import get_system_config, get_available_systems

    # Validate system name
    available = get_available_systems()
    if system_name not in available:
        console.print(f"\n[red]Unknown system '{system_name}'. Available: {', '.join(available)}[/red]")
        return

    sys_config = get_system_config(system_name)
    gpus_per_node = sys_config.GPUS_PER_NODE
    cores_per_node = sys_config.CORES_PER_NODE

    # Filter schedulable jobs (Ready/Restart), warn about others
    schedulable = [j for j in jobs if j['status'] in ('Ready', 'Restart')]
    non_schedulable = [j for j in jobs if j['status'] not in ('Ready', 'Restart')]

    console.print()
    console.print(f"[bold]Resource Requirements for {system_name}[/bold] "
                  f"({gpus_per_node} GPUs/node, {cores_per_node} cores/node):")

    if non_schedulable:
        statuses = {}
        for j in non_schedulable:
            statuses[j['status']] = statuses.get(j['status'], 0) + 1
        status_str = ", ".join(f"{count} {s}" for s, count in statuses.items())
        console.print(f"  [yellow]Skipping {len(non_schedulable)} non-schedulable jobs ({status_str})[/yellow]")

    if not schedulable:
        console.print(f"  [yellow]No schedulable (Ready/Restart) jobs to analyze.[/yellow]")
        return

    # Classify jobs
    gpu_jobs = [j for j in schedulable if j['ngpus'] > 0]
    cpu_jobs = [j for j in schedulable if j['ngpus'] == 0]

    console.print(f"  Schedulable jobs: {len(schedulable)}")
    if gpu_jobs and cpu_jobs:
        console.print(f"    GPU jobs: {len(gpu_jobs)}")
        console.print(f"    CPU jobs: {len(cpu_jobs)}")

    # --- GPU analysis ---
    if gpu_jobs:
        _print_gpu_analysis(gpu_jobs, gpus_per_node, system_name)

    # --- CPU analysis ---
    if cpu_jobs:
        _print_cpu_analysis(cpu_jobs, gpus_per_node)

    # --- Combined total ---
    if gpu_jobs and cpu_jobs:
        gpu_sim, gpu_opt = _calc_gpu_nodes(gpu_jobs, gpus_per_node)
        cpu_sim, cpu_opt = _calc_cpu_nodes(cpu_jobs)
        console.print()
        console.print(f"  [bold]Total node estimate: {gpu_sim + cpu_sim} (simultaneous) / "
                      f"{gpu_opt + cpu_opt} (optimal)[/bold]")


def _print_gpu_analysis(gpu_jobs: list, gpus_per_node: int, system_name: str):
    """Print GPU job resource analysis."""
    if gpus_per_node == 0:
        total_gpus = sum(j['ngpus'] for j in gpu_jobs)
        console.print(f"\n  Total GPUs required: {total_gpus}")
        console.print(f"  [red]WARNING: {system_name} has no GPUs. "
                      f"{len(gpu_jobs)} GPU job(s) cannot run on this system.[/red]")
        return

    # Check for oversized single-node jobs
    oversized = [j for j in gpu_jobs if j['num_nodes'] == 1 and j['ngpus'] > gpus_per_node]
    if oversized:
        for j in oversized:
            console.print(f"\n  [red]WARNING: Job {j['job_id']} requires {j['ngpus']} GPUs "
                          f"but {system_name} only has {gpus_per_node} GPUs/node. "
                          f"Cannot run as single-node job.[/red]")

    total_gpus = sum(j['ngpus'] for j in gpu_jobs)
    multinode_jobs = [j for j in gpu_jobs if j['num_nodes'] > 1]
    singlenode_jobs = [j for j in gpu_jobs if j['num_nodes'] == 1 and j not in oversized]

    console.print(f"\n  Total GPUs required: {total_gpus}")
    if multinode_jobs:
        mn_nodes = sum(j['num_nodes'] for j in multinode_jobs)
        console.print(f"    Multi-node jobs: {len(multinode_jobs)} ({mn_nodes} dedicated nodes)")

    sim, opt = _calc_gpu_nodes(gpu_jobs, gpus_per_node)
    min_nodes = max((j['num_nodes'] for j in gpu_jobs), default=1)

    idle = sim * gpus_per_node - total_gpus
    console.print(f"\n  Nodes for simultaneous execution: {sim}"
                  + (f" ({sim * gpus_per_node} GPU slots, {idle} idle)" if idle > 0 else ""))

    if opt < sim:
        queued = total_gpus - opt * gpus_per_node
        console.print(f"  Nodes for optimal packing:        {opt}"
                      f" ({opt * gpus_per_node} GPU slots, {queued} GPU(s) queued by pbx)")
    else:
        console.print(f"  Nodes for optimal packing:        {opt}")

    if min_nodes > 1 and min_nodes < sim:
        console.print(f"  Minimum viable nodes:             {min_nodes} (largest job's requirement)")


def _print_cpu_analysis(cpu_jobs: list, gpus_per_node: int):
    """Print CPU job resource analysis."""
    singlenode = [j for j in cpu_jobs if j['num_nodes'] == 1]
    multinode = [j for j in cpu_jobs if j['num_nodes'] > 1]

    total_occupancy = sum(j['node_occupancy'] for j in singlenode)
    mn_nodes = sum(j['num_nodes'] for j in multinode)

    console.print(f"\n  Total CPU node occupancy: {total_occupancy:.1f}"
                  + (f" + {mn_nodes} multi-node dedicated" if multinode else ""))

    sim, opt = _calc_cpu_nodes(cpu_jobs)
    full = int(total_occupancy)
    frac = total_occupancy - full

    console.print(f"\n  Nodes for simultaneous execution: {sim}")
    if frac > 0 and full > 0:
        console.print(f"  Nodes for optimal packing:        {opt} "
                      f"({full} full + 1 at {frac:.0%})"
                      + (f" + {mn_nodes} multi-node" if multinode else ""))
    else:
        console.print(f"  Nodes for optimal packing:        {opt}")

    if gpus_per_node > 0:
        console.print(f"\n  [dim]Note: CPU jobs will run on GPU-equipped nodes (GPUs will be idle).[/dim]")


def _calc_gpu_nodes(gpu_jobs: list, gpus_per_node: int) -> tuple:
    """Calculate (simultaneous, optimal) node counts for GPU jobs."""
    if gpus_per_node == 0:
        return (0, 0)

    multinode_nodes = sum(j['num_nodes'] for j in gpu_jobs if j['num_nodes'] > 1)
    singlenode_gpus = sum(j['ngpus'] for j in gpu_jobs
                         if j['num_nodes'] == 1 and j['ngpus'] <= gpus_per_node)

    sim = multinode_nodes + math.ceil(singlenode_gpus / gpus_per_node) if singlenode_gpus else multinode_nodes
    opt = multinode_nodes + (singlenode_gpus // gpus_per_node) if singlenode_gpus else multinode_nodes

    # Ensure optimal has at least 1 node if there are remaining GPUs
    if singlenode_gpus % gpus_per_node > 0 and opt == multinode_nodes:
        opt += 1

    return (sim, opt)


def _calc_cpu_nodes(cpu_jobs: list) -> tuple:
    """Calculate (simultaneous, optimal) node counts for CPU jobs."""
    total_occupancy = sum(j['node_occupancy'] for j in cpu_jobs if j['num_nodes'] == 1)
    multinode_nodes = sum(j['num_nodes'] for j in cpu_jobs if j['num_nodes'] > 1)

    sim = multinode_nodes + math.ceil(total_occupancy)
    opt = multinode_nodes + math.ceil(total_occupancy)  # CPU can't pack tighter than occupancy

    return (sim, opt)


def _print_cmdline(jobs: list, system_name: str):
    """Print MPI/srun command line preview for jobs."""
    from parslbox.system_configs.loader import get_system_config, get_available_systems
    from parslbox.resource_manager.mpi_config import load_mpi_config
    from parslbox.resource_manager.mpi_command_builder import MPICommandBuilder
    from parslbox.resource_manager.models import create_job_resource_spec
    from parslbox.utils.pbx_config_utils import load_full_config
    import tempfile
    import shutil

    # Validate system name
    available = get_available_systems()
    if system_name not in available:
        console.print(f"\n[red]Unknown system '{system_name}'. Available: {', '.join(available)}[/red]")
        return

    sys_config = get_system_config(system_name)
    gpus_per_node = sys_config.GPUS_PER_NODE
    cores_per_node = sys_config.CORES_PER_NODE
    excluded_cores = getattr(sys_config, 'EXCLUDE_CORES', None) or []
    effective_cores = cores_per_node - len(excluded_cores)

    # Load MPI config from config.yaml (per-app, since MPI settings can differ per app)
    full_config = load_full_config()
    mpi_configs = {}  # cache per app_name

    # Use a temp directory for any generated files (rankfiles, gpu wrappers)
    tmp_dir = tempfile.mkdtemp(prefix="pbx_cmdline_preview_")

    try:
        # Build table
        table = Table("ID", "Resources", "Command", expand=True)
        table.columns[2].no_wrap = False
        table.columns[2].overflow = "fold"

        for job in jobs:
            job_spec = create_job_resource_spec(job)
            app_name = job['app']

            # Load or reuse MPI config for this app
            if app_name not in mpi_configs:
                try:
                    mpi_cfg = load_mpi_config(
                        system_name=system_name,
                        app_name=app_name,
                        system_config_class=sys_config,
                        yaml_config=full_config
                    )
                    mpi_configs[app_name] = mpi_cfg
                except Exception as e:
                    mpi_configs[app_name] = None
                    table.add_row(str(job['job_id']), "?", f"[red]Error loading MPI config for app '{app_name}': {e}[/red]")
                    continue

            mpi_cfg = mpi_configs[app_name]
            if mpi_cfg is None:
                table.add_row(str(job['job_id']), "?", f"[red]MPI config unavailable for app '{app_name}'[/red]")
                continue

            # Build resource string
            if job_spec.ngpus > 0:
                res_str = f"{job_spec.num_nodes}N/{job_spec.ngpus}G"
            elif job_spec.node_occupancy < 1.0:
                res_str = f"{job_spec.num_nodes}N/{job_spec.node_occupancy}occ"
            else:
                res_str = f"{job_spec.num_nodes}N/CPU"

            # Create synthetic assignment
            assignment = _create_synthetic_assignment(job_spec, gpus_per_node, effective_cores)

            # Build command (use tmp_dir for any generated files)
            builder = MPICommandBuilder(mpi_cfg, sys_config)
            try:
                cmd = builder.build_command(assignment, job_spec, job_path=tmp_dir)
            except Exception as e:
                cmd = f"[error: {e}]"

            table.add_row(str(job['job_id']), res_str, cmd)

        console.print()
        console.print(f"[bold]Command line preview for {system_name}[/bold] "
                      f"(hostnames are placeholders):")
        console.print(table)

    finally:
        # Clean up temp directory with generated files
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _create_synthetic_assignment(job_spec, gpus_per_node, effective_cores):
    """Create a synthetic ResourceAssignment with placeholder hostnames for command preview."""
    from parslbox.resource_manager.models import ResourceAssignment

    num_nodes = job_spec.num_nodes
    node_ids = [f"node-{i}" for i in range(num_nodes)]
    hostnames = [f"host-{i}" for i in range(num_nodes)]

    # Calculate ranks per node
    if job_spec.is_gpu_job():
        if num_nodes > 1:
            ranks_per_node = job_spec.ngpus // num_nodes
        else:
            ranks_per_node = job_spec.ngpus
    else:
        ranks_per_node = job_spec.ranks_per_node

    cores_per_rank = effective_cores // ranks_per_node if ranks_per_node > 0 else effective_cores

    # Build per-node GPU and CPU assignments
    gpu_assignments = []
    cpu_assignments = []
    for node_idx in range(num_nodes):
        node_gpus = {}
        node_cpus = {}
        for rank in range(ranks_per_node):
            if job_spec.is_gpu_job():
                node_gpus[rank] = [rank]  # GPU ID = rank index within node
            start_core = rank * cores_per_rank
            node_cpus[rank] = list(range(start_core, start_core + cores_per_rank))
        gpu_assignments.append(node_gpus)
        cpu_assignments.append(node_cpus)

    return ResourceAssignment(
        job_id=job_spec.job_id,
        node_ids=node_ids,
        hostnames=hostnames,
        gpu_assignments=gpu_assignments,
        cpu_assignments=cpu_assignments,
        node_occupancy=job_spec.node_occupancy,
    )
