import typer
import subprocess
import yaml
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from typing_extensions import Annotated

# Import helper functions
from parslbox.commands.helpers.qsub_cmd_helpers import (
    minutes_to_hms,
    get_default_run_dir,
    load_config,
)

app = typer.Typer()


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def submit_to_scheduler(
    config_name: str,
    job_name: str,
    queue: str,
    select: int,
    walltime: int,
    project: str,
    filesystems: Optional[str] = None,
    run_dir: Optional[Path] = None,
    apps: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    retries: int = 0,
    config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Core scheduler submission logic - used by both CLI and API.
    
    Args:
        config_name: System configuration name
        job_name: PBS job name
        queue: PBS queue name
        select: Number of nodes to request
        walltime: Wall time in minutes
        project: Project/account name
        filesystems: Comma-separated list of filesystems
        run_dir: Custom run directory (default: timestamped)
        apps: List of apps to run
        tags: List of tags to run
        retries: Number of retries for failed tasks
        config_path: Path to configuration file
    
    Returns:
        Dictionary with submission details including job_id and run_dir
    
    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If qsub command is not found
    """
    # Load configuration
    try:
        config = load_config(config_path)
    except FileNotFoundError as e:
        raise ValidationError(str(e))
    except yaml.YAMLError as e:
        raise ValidationError(f"Invalid YAML in configuration file: {e}")
    
    # Validate that config is not None (empty file or None YAML)
    if config is None:
        raise ValidationError("Configuration file is empty or contains no data")
    
    # Validate scheduler template exists
    if 'schedulers' not in config or 'pbs' not in config['schedulers']:
        raise ValidationError("PBS scheduler template not found in configuration")
    
    # Validate system configuration exists
    if config_name not in config:
        raise ValidationError(f"System '{config_name}' not found in configuration")
    
    # Get system-specific python environment setup
    system_config = config[config_name]
    pbx_python_env_setup = system_config.get('pbx_python_env_setup', '')
    
    # Determine run directory
    if run_dir is None:
        run_dir = get_default_run_dir()
    
    # Create run directory
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert walltime to HH:MM:SS format
    walltime_formatted = minutes_to_hms(walltime)
    
    # Build run options string
    run_options = []
    if apps:
        if isinstance(apps, list):
            run_options.append(f"--apps {','.join(apps)}")
        else:
            run_options.append(f"--apps {apps}")
    if tags:
        if isinstance(tags, list):
            run_options.append(f"--tags {','.join(tags)}")
        else:
            run_options.append(f"--tags {tags}")
    if retries > 0:
        run_options.append(f"--retries {retries}")
    
    run_options_str = " ".join(run_options)
    
    # Capture environment variables for parslbox paths
    pbx_env_vars = ""
    if os.getenv("PBX_DB_PATH"):
        pbx_env_vars += f'export PBX_DB_PATH="{os.getenv("PBX_DB_PATH")}"\n'
    if os.getenv("PBX_CONFIG_PATH"):
        pbx_env_vars += f'export PBX_CONFIG_PATH="{os.getenv("PBX_CONFIG_PATH")}"\n'
    
    # Prepare template variables
    template_vars = {
        'job_name': job_name,
        'queue': queue,
        'select': select,
        'walltime': walltime_formatted,
        'filesystems': filesystems or '',
        'project': project,
        'pbx_python_env_setup': pbx_python_env_setup,
        'pbx_env_vars': pbx_env_vars,
        'config': config_name,
        'run_dir': './',
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
        
        return {
            "success": True,
            "pbs_job_id": job_id,
            "run_dir": str(run_dir),
            "submit_file": str(submit_file),
        }
        
    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr,
            "run_dir": str(run_dir),
            "submit_file": str(submit_file),
        }
    except FileNotFoundError:
        raise ValidationError("qsub command not found. Make sure PBS is available.")


# Main CLI command

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
    try:
        # Convert CLI string arguments to lists for core function
        apps_list = apps.split(',') if apps else None
        tags_list = tags.split(',') if tags else None
        
        # Call core function
        result = submit_to_scheduler(
            config_name=config_name,
            job_name=job_name,
            queue=queue,
            select=select,
            walltime=walltime,
            project=project,
            filesystems=filesystems,
            run_dir=run_dir,
            apps=apps_list,
            tags=tags_list,
            retries=retries,
        )
        
        # CLI-specific output formatting
        typer.secho(f"📁 Created run directory: {result['run_dir']}", fg=typer.colors.BLUE)
        typer.secho(f"📝 Generated submit script: {result['submit_file']}", fg=typer.colors.GREEN)
        
        if result["success"]:
            typer.secho(f"🚀 Job submitted successfully! Job ID: {result['pbs_job_id']}", fg=typer.colors.GREEN)
            typer.secho(f"📊 Monitor with: qstat {result['pbs_job_id']}", fg=typer.colors.BLUE)
            typer.secho(f"📁 Run directory: {result['run_dir']}", fg=typer.colors.BLUE)
        else:
            typer.secho(f"❌ Error submitting job: {result['error']}", fg=typer.colors.RED)
            typer.secho(f"� Submit script saved at: {result['submit_file']}", fg=typer.colors.YELLOW)
            raise typer.Exit(code=1)
            
    except ValidationError as e:
        typer.secho(f"❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.secho(f"❌ Unexpected error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
