import typer
import subprocess
import yaml
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated

from parslbox.helpers import path_utils


app = typer.Typer()


def minutes_to_hms(minutes: int) -> str:
    """Convert minutes to HH:MM:SS format for PBS/SLURM."""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}:00"


def get_default_run_dir() -> Path:
    """Generate default run directory with current time and date in hhmmss_ddmmyy format."""
    now = datetime.now()
    time_str = now.strftime("%H%M%S")  # hhmmss format (hours + minutes + seconds)
    date_str = now.strftime("%d%m%y")  # ddmmyy format
    dir_name = f"{time_str}_{date_str}"
    return Path.home() / ".parslbox" / "runs" / dir_name


def load_config() -> dict:
    """Load the parslbox configuration file."""
    config_path = path_utils.FLOW_CONFIG_FILE
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@app.command()
def qsub(
    config_name: Annotated[
        str,
        typer.Option("--config", "-c", help="The name of the configuration to use (e.g., 'sophia').")
    ],
    job_name: Annotated[
        str,
        typer.Option("--job-name", "-N", help="PBS job name.")
    ],
    queue: Annotated[
        str,
        typer.Option("--queue", "-q", help="PBS queue name.")
    ],
    select: Annotated[
        int,
        typer.Option("--select", help="Number of nodes to request.")
    ],
    walltime: Annotated[
        int,
        typer.Option("--walltime", "-T", help="Wall time in minutes (e.g., 90 for 1.5 hours).")
    ],
    project: Annotated[
        str,
        typer.Option("--project", "-A", help="Project/account name.")
    ],
    filesystems: Annotated[
        Optional[str],
        typer.Option("--filesystems", help="Comma-separated list of filesystems (e.g., 'home:eagle').")
    ] = None,
    run_dir: Annotated[
        Optional[Path],
        typer.Option("--run-dir", help="Custom run directory (default: timestamped directory).")
    ] = None,
    apps: Annotated[
        Optional[str],
        typer.Option("--apps", "-a", help="Comma-separated list of apps to run (e.g., 'lammps,vasp').")
    ] = None,
    tags: Annotated[
        Optional[str],
        typer.Option("--tags", "-t", help="Comma-separated list of tags to run (e.g., 'run1,run2').")
    ] = None,
    retries: Annotated[
        int,
        typer.Option("--retries", help="Number of retries for failed tasks.")
    ] = 0,
):
    """
    Generate and submit a PBS job script for running parslbox workflows.
    """
    # Load configuration
    try:
        config = load_config()
    except FileNotFoundError as e:
        typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    # Validate scheduler template exists
    if 'schedulers' not in config or 'pbs' not in config['schedulers']:
        typer.secho("❌ Error: PBS scheduler template not found in configuration.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    # Validate system configuration exists
    if config_name not in config:
        typer.secho(f"❌ Error: System '{config_name}' not found in configuration.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    
    # Get system-specific python environment setup
    system_config = config[config_name]
    python_env_setup = system_config.get('python_env_setup', '')
    
    # Determine run directory
    if run_dir is None:
        run_dir = get_default_run_dir()
    
    # Create run directory
    run_dir.mkdir(parents=True, exist_ok=True)
    typer.secho(f"📁 Created run directory: {run_dir}", fg=typer.colors.BLUE)
    
    # Convert walltime to HH:MM:SS format
    walltime_formatted = minutes_to_hms(walltime)
    
    # Build run options string
    run_options = []
    if apps:
        run_options.append(f"--apps {apps}")
    if tags:
        run_options.append(f"--tags {tags}")
    if retries > 0:
        run_options.append(f"--retries {retries}")
    
    run_options_str = " ".join(run_options)
    
    # Prepare template variables
    template_vars = {
        'job_name': job_name,
        'queue': queue,
        'select': select,
        'walltime': walltime_formatted,
        'filesystems': filesystems or '',
        'project': project,
        'python_env_setup': python_env_setup,
        'config': config_name,
        'run_dir': './', #str(run_dir.resolve()),  # Use absolute path for the run directory
        'run_options': run_options_str
    }
    
    # Get PBS template and format it
    pbs_template = config['schedulers']['pbs']['template']
    
    # Handle optional filesystems directive
    if not filesystems:
        # Remove the filesystems line if not provided
        pbs_template = '\n'.join(line for line in pbs_template.split('\n') 
                                if '#PBS -l filesystems=' not in line)
    
    submit_script = pbs_template.format(**template_vars)
    
    # Write submit script
    submit_file = run_dir / "submit.sh"
    with open(submit_file, 'w') as f:
        f.write(submit_script)
    
    typer.secho(f"📝 Generated submit script: {submit_file}", fg=typer.colors.GREEN)
    
    # Submit the job
    try:
        # Change to run directory and submit
        result = subprocess.run(
            ['qsub', 'submit.sh'],
            cwd=run_dir,
            capture_output=True,
            text=True,
            check=True
        )
        
        job_id = result.stdout.strip()
        typer.secho(f"🚀 Job submitted successfully! Job ID: {job_id}", fg=typer.colors.GREEN)
        typer.secho(f"📊 Monitor with: qstat {job_id}", fg=typer.colors.BLUE)
        typer.secho(f"📁 Run directory: {run_dir}", fg=typer.colors.BLUE)
        
    except subprocess.CalledProcessError as e:
        typer.secho(f"❌ Error submitting job: {e.stderr}", fg=typer.colors.RED)
        typer.secho(f"📄 Submit script saved at: {submit_file}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    except FileNotFoundError:
        typer.secho("❌ Error: 'qsub' command not found. Make sure PBS is available.", fg=typer.colors.RED)
        typer.secho(f"📄 Submit script saved at: {submit_file}", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
