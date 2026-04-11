"""
Shared Submission Logic for ParslBox

This module provides the core submit_job() function used by both
qsub (PBS) and sbatch (SLURM) commands.
"""

import subprocess
import yaml
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from parslbox.commands.helpers.qsub_cmd_helpers import (
    minutes_to_hms,
    get_default_run_dir,
    load_config,
)
from parslbox.commands.helpers.sched_opts_helpers import merge_sched_opts


class ValidationError(Exception):
    """Exception raised for validation errors."""
    pass


def submit_job(
    config_name: str,
    job_name: str,
    queue: str,
    select: str,
    walltime: int,
    project: str,
    run_dir: Optional[Path] = None,
    apps: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    retries: int = 0,
    loglevel: str = "info",
    config_path: Optional[Path] = None,
    sched_opts: Optional[List[str]] = None,
    scheduler_type: str = "pbs",
    submit_command: str = "qsub",
    dynamic: bool = True,
) -> Dict[str, Any]:
    """
    Core scheduler submission logic - used by both PBS (qsub) and SLURM (sbatch).

    Args:
        config_name: System configuration name
        job_name: Job name
        queue: Queue/partition name
        select: Resource selection (nodes for SLURM, select spec for PBS)
        walltime: Wall time in minutes
        project: Project/account name
        run_dir: Custom run directory (default: timestamped)
        apps: List of apps to run
        tags: List of tags to run
        retries: Number of retries for failed tasks
        loglevel: Logging level
        config_path: Path to configuration file
        sched_opts: List of extra scheduler directive strings from CLI
        scheduler_type: "pbs" or "slurm"
        submit_command: "qsub" or "sbatch"

    Returns:
        Dictionary with submission details including job_id and run_dir

    Raises:
        ValidationError: If configuration is invalid
        FileNotFoundError: If submit command is not found
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
    if 'schedulers' not in config or scheduler_type not in config['schedulers']:
        raise ValidationError(
            f"{scheduler_type.upper()} scheduler template not found in configuration"
        )

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
    if loglevel != "info":  # Only add if not default
        run_options.append(f"--loglevel {loglevel}")
    if not dynamic:  # Only add when disabling (default is dynamic)
        run_options.append("--static")

    run_options_str = " ".join(run_options)

    # Capture environment variables for parslbox paths
    pbx_env_vars = ""
    if os.getenv("PBX_DB_PATH"):
        pbx_env_vars += f'export PBX_DB_PATH="{os.getenv("PBX_DB_PATH")}"\n'
    if os.getenv("PBX_CONFIG_PATH"):
        pbx_env_vars += f'export PBX_CONFIG_PATH="{os.getenv("PBX_CONFIG_PATH")}"\n'

    # Prepare template variables (no filesystems, sched_opts as placeholder)
    template_vars = {
        'job_name': job_name,
        'queue': queue,
        'select': select,
        'walltime': walltime_formatted,
        'project': project,
        'pbx_python_env_setup': pbx_python_env_setup,
        'pbx_env_vars': pbx_env_vars,
        'config': config_name,
        'run_dir': './',
        'run_options': run_options_str,
        'sched_opts': '{sched_opts}',  # Keep placeholder for merge step
    }

    # Get scheduler template and format it
    sched_template = config['schedulers'][scheduler_type]['template']
    rendered = sched_template.format(**template_vars)

    # Get config-level sched_opts
    config_sched_opts = system_config.get('sched_opts', None)

    # Get system-class default sched_opts (lowest priority)
    try:
        from parslbox.system_configs.loader import get_system_config
        sys_config_obj = get_system_config(config_name)
        system_default_sched_opts = sys_config_obj.get_default_sched_opts()
    except (ValueError, Exception):
        system_default_sched_opts = None

    # Combine: system defaults (lowest) + config.yaml (higher)
    # Within a single override layer, later lines with same key replace earlier
    if system_default_sched_opts and config_sched_opts:
        combined_config_opts = system_default_sched_opts + "\n" + config_sched_opts
    elif system_default_sched_opts:
        combined_config_opts = system_default_sched_opts
    else:
        combined_config_opts = config_sched_opts

    # Merge directives: template → system+config → CLI
    submit_script = merge_sched_opts(rendered, combined_config_opts, sched_opts)

    # Write submit script
    submit_file = run_dir / "submit.sh"
    with open(submit_file, 'w') as f:
        f.write(submit_script)

    # Submit the job
    try:
        result = subprocess.run(
            [submit_command, 'submit.sh'],
            cwd=run_dir,
            capture_output=True,
            text=True,
            check=True,
        )

        job_id = result.stdout.strip()

        # Build result with generic job_id key + backward-compat key
        result_dict = {
            "success": True,
            "job_id": job_id,
            "run_dir": str(run_dir),
            "submit_file": str(submit_file),
        }
        if scheduler_type == "pbs":
            result_dict["pbs_job_id"] = job_id
        elif scheduler_type == "slurm":
            result_dict["slurm_job_id"] = job_id

        return result_dict

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "error": e.stderr,
            "run_dir": str(run_dir),
            "submit_file": str(submit_file),
        }
    except FileNotFoundError:
        raise ValidationError(
            f"{submit_command} command not found. "
            f"Make sure {'PBS' if scheduler_type == 'pbs' else 'SLURM'} is available."
        )
