import typer
from typing import List, Optional
from rich.console import Console
from rich.table import Table

from parslbox.database import database
from parslbox.utils import path_utils
from parslbox.commands.helpers.ls_cmd_helpers import truncate_path

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
    path_full: bool = typer.Option(False, "--path", help="Show the path field with full (untruncated) display."),
    path_auto: bool = typer.Option(False, "-p", help="Show the path field; auto-truncates when more than 3 fields are selected."),
    ngpus: bool = typer.Option(False, "--ngpus", "-g", help="Show only the number of GPUs field."),
    nodes: bool = typer.Option(False, "--nodes", "-n", help="Show only the number of nodes field."),
    ranks: bool = typer.Option(False, "--ranks", help="Show only the ranks-per-node field."),
    nocc: bool = typer.Option(False, "--nocc", "-o", help="Show only the node occupancy field."),
    resrc: bool = typer.Option(False, "--resrc", "-r", help="Show the full resource string (n/r/g/nocc)."),
    app_name: bool = typer.Option(False, "--app", "-a", help="Show only the application field."),
    status: bool = typer.Option(False, "--status", "-s", help="Show only the status field."),
    tag: bool = typer.Option(False, "--tag", "-t", help="Show only the tag field."),
    input_file: bool = typer.Option(False, "--input", "-i", help="Show only the input file field."),
    sched_job_id: bool = typer.Option(False, "--sched-job-id", "-j", help="Show only the scheduler job ID field."),
    timestamp: bool = typer.Option(False, "--timestamp", "-ts", help="Show only the timestamp field."),
    env_file: bool = typer.Option(False, "--envfile", "-e", help="Show only the environment file field."),
    parents: bool = typer.Option(False, "--parents", "-P", help="Show all parent dependencies without truncation."),
    req: Optional[str] = typer.Option(None, "--req", help="Calculate resource requirements for a target system (e.g., polaris, crux, sophia)."),
    cmdline: Optional[str] = typer.Option(None, "--cmdline", help="Show MPI/srun command line for a target system (e.g., polaris, crux, sophia)."),
):
    """
    Shows detailed information about specific jobs.
    """
    from parslbox.commands.helpers.job_id_parser import resolve_job_ids
    try:
        job_ids = resolve_job_ids(job_ids, path_utils.DB_FILE)
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
    
    # Path selection: --path forces full display, -p uses auto-truncation rule
    path_selected = path_full or path_auto

    # Determine which fields to show
    selected_fields = []
    if path_selected:
        selected_fields.append(('path', 'Path'))
    if ngpus:
        selected_fields.append(('ngpus', 'NGPUs'))
    if nodes:
        selected_fields.append(('num_nodes', 'NNodes'))
    if ranks:
        selected_fields.append(('ranks_per_node', 'Ranks/Node'))
    if nocc:
        selected_fields.append(('node_occupancy', 'NOcc'))
    if resrc:
        selected_fields.append(('_resrc', 'Resources'))
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

    is_default_view = not selected_fields

    # If no specific fields selected, show all fields
    if is_default_view:
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

    # Truncate path when -p (auto) is used and more than 3 fields are shown.
    # --path forces full display; default view keeps existing wrap behavior.
    should_truncate_path = path_auto and not path_full and len(selected_fields) > 3
    
    # Create and populate table
    headers = [field[1] for field in selected_fields]

    # Check if path is being displayed to configure wrapping
    path_in_fields = any(field[0] == 'path' for field in selected_fields)

    if path_in_fields and not should_truncate_path:
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
            if field_key == '_resrc':
                row_data.append(_format_resource_string(job))
                continue
            value = job[field_key]
            if value is None:
                row_data.append("None")
            elif field_key == 'path' and should_truncate_path:
                row_data.append(truncate_path(str(value)))
            elif field_key in ('ngpus', 'num_nodes', 'ranks_per_node', 'node_occupancy'):
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


def _format_resource_string(job: dict) -> str:
    """Build resource summary string matching the ls command format."""
    num_nodes = job.get('num_nodes', 1)
    ngpus = job.get('ngpus', 0)
    node_occupancy = job.get('node_occupancy', 1.0)
    ranks_per_node = job.get('ranks_per_node', 1)
    total_ranks = num_nodes * ranks_per_node
    if num_nodes > 1:
        return f"n:{num_nodes}-r:{total_ranks}-g:{ngpus}-nocc:NA"
    elif ngpus > 0:
        return f"n:1-r:{total_ranks}-g:{ngpus}-nocc:NA"
    else:
        return f"n:1-r:{total_ranks}-g:0-nocc:{node_occupancy}"


def _print_resource_requirements(jobs: list, system_name: str):
    """Print resource requirement analysis for jobs against a target system."""
    from parslbox.commands.helpers.resource_req import (
        UnknownSystemError, compute_requirements,
    )

    try:
        req = compute_requirements(jobs, system_name)
    except UnknownSystemError as e:
        console.print(f"\n[red]{e}[/red]")
        return

    console.print()
    console.print(f"[bold]Resource Requirements for {system_name}[/bold] "
                  f"({req['gpus_per_node']} GPUs/node, {req['cores_per_node']} cores/node):")

    if req["skipped"]:
        n = sum(req["skipped"].values())
        status_str = ", ".join(f"{c} {s}" for s, c in req["skipped"].items())
        console.print(f"  [yellow]Skipping {n} non-schedulable jobs ({status_str})[/yellow]")

    if not req["schedulable"]:
        console.print(f"  [yellow]No schedulable (Ready/Restart) jobs to analyze.[/yellow]")
        return

    gpu, cpu = req["gpu"], req["cpu"]
    console.print(f"  Schedulable jobs: {req['schedulable']}")
    if gpu and cpu:
        console.print(f"    GPU jobs: {gpu['jobs']}")
        console.print(f"    CPU jobs: {cpu['jobs']}")

    if gpu:
        _print_gpu_analysis(gpu, system_name)
    if cpu:
        _print_cpu_analysis(cpu)

    if req["total"]:
        console.print()
        console.print(f"  [bold]Total node estimate: {req['total']['simultaneous']} (simultaneous) / "
                      f"{req['total']['optimal']} (optimal)[/bold]")


def _print_gpu_analysis(gpu: dict, system_name: str):
    """Print GPU job resource analysis from a computed block."""
    if not gpu["system_has_gpus"]:
        console.print(f"\n  Total GPUs required: {gpu['total_gpus']}")
        console.print(f"  [red]WARNING: {system_name} has no GPUs. "
                      f"{gpu['jobs']} GPU job(s) cannot run on this system.[/red]")
        return

    for j in gpu["oversized"]:
        console.print(f"\n  [red]WARNING: Job {j['job_id']} requires {j['ngpus']} GPUs "
                      f"but {system_name} only has {gpu['gpus_per_node']} GPUs/node. "
                      f"Cannot run as single-node job.[/red]")

    console.print(f"\n  Total GPUs required: {gpu['total_gpus']}")
    if gpu["multinode_jobs"]:
        console.print(f"    Multi-node jobs: {gpu['multinode_jobs']} "
                      f"({gpu['multinode_nodes']} dedicated nodes)")

    sim, opt = gpu["simultaneous"], gpu["optimal"]
    slots = sim * gpu["gpus_per_node"]
    console.print(f"\n  Nodes for simultaneous execution: {sim}"
                  + (f" ({slots} GPU slots, {gpu['idle_slots']} idle)"
                     if gpu["idle_slots"] > 0 else ""))

    if opt < sim:
        console.print(f"  Nodes for optimal packing:        {opt}"
                      f" ({opt * gpu['gpus_per_node']} GPU slots, "
                      f"{gpu['queued_gpus']} GPU(s) queued by pbx)")
    else:
        console.print(f"  Nodes for optimal packing:        {opt}")

    min_nodes = gpu["min_viable_nodes"]
    if min_nodes > 1 and min_nodes < sim:
        console.print(f"  Minimum viable nodes:             {min_nodes} (largest job's requirement)")


def _print_cpu_analysis(cpu: dict):
    """Print CPU job resource analysis from a computed block."""
    mn_nodes = cpu["multinode_nodes"]
    console.print(f"\n  Total CPU node occupancy: {cpu['total_occupancy']:.1f}"
                  + (f" + {mn_nodes} multi-node dedicated" if mn_nodes else ""))

    full, frac = cpu["full_nodes"], cpu["fraction"]
    console.print(f"\n  Nodes for simultaneous execution: {cpu['simultaneous']}")
    if frac > 0 and full > 0:
        console.print(f"  Nodes for optimal packing:        {cpu['optimal']} "
                      f"({full} full + 1 at {frac:.0%})"
                      + (f" + {mn_nodes} multi-node" if mn_nodes else ""))
    else:
        console.print(f"  Nodes for optimal packing:        {cpu['optimal']}")

    if cpu["gpus_would_idle"]:
        console.print(f"\n  [dim]Note: CPU jobs will run on GPU-equipped nodes (GPUs will be idle).[/dim]")


def _print_cmdline(jobs: list, system_name: str):
    """Print MPI/srun command line preview for jobs."""
    from parslbox.commands.helpers.resource_req import (
        UnknownSystemError, build_command_lines,
    )

    try:
        preview = build_command_lines(jobs, system_name)
    except UnknownSystemError as e:
        console.print(f"\n[red]{e}[/red]")
        return

    table = Table("ID", "Resources", "Command", expand=True)
    table.columns[2].no_wrap = False
    table.columns[2].overflow = "fold"
    for row in preview["jobs"]:
        if row["error"]:
            table.add_row(str(row["job_id"]), row["resources"] or "?",
                          f"[red]{row['error']}[/red]")
        else:
            table.add_row(str(row["job_id"]), row["resources"], row["command"])

    console.print()
    console.print(f"[bold]Command line preview for {system_name}[/bold] "
                  f"(hostnames are placeholders):")
    console.print(table)
