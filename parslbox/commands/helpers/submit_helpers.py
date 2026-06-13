"""
Shared Submission Logic for ParslBox

This module provides the core submit_job() function used by both
qsub (PBS) and sbatch (SLURM) commands.
"""

import re
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


def count_matching_runnable_jobs(
    db_path: Path,
    apps_list: Optional[List[str]],
    tags_list: Optional[List[str]],
) -> int:
    """Count Ready/Restart jobs matching the given apps/tags filters.

    Apps and tags are matched exactly. `None` for either means "no filter".
    Used by qsub/sbatch to verify there's actually work to do before submitting.
    """
    from parslbox.database import database
    jobs = database.get_jobs(db_path, status='Ready')
    jobs += database.get_jobs(db_path, status='Restart')

    app_set = set(apps_list) if apps_list else None
    tag_set = set(tags_list) if tags_list else None

    count = 0
    for j in jobs:
        if app_set and j['app'] not in app_set:
            continue
        if tag_set and j['tag'] not in tag_set:
            continue
        count += 1
    return count


def render_submit_script_panel(submit_file_path, title: str = "Submit Script"):
    """Build a Rich Panel containing the submit script with the `pbx run` line
    highlighted in red. Returns the Panel ready to be `console.print`-ed.
    """
    from rich.panel import Panel
    from rich.text import Text

    script = Path(submit_file_path).read_text()
    text = Text()
    for line in script.splitlines():
        if 'pbx run' in line:
            text.append(line, style="bold red")
        else:
            text.append(line)
        text.append("\n")
    return Panel(
        text,
        title=f"[bold cyan]{title}[/bold cyan] [dim]({submit_file_path})[/dim]",
        border_style="cyan",
        expand=True,
    )


RESPAWN_TEMPLATE_HEADER = """#######################################################################
# PBX RESPAWN TEMPLATE
#
# Generated once by `pbx qsub --respawn N`. NEVER overwritten by pbx.
#
# RESOURCE PLACEHOLDERS - safe to edit:
#   <<PBX_AUTO_SELECT>>   <<PBX_AUTO_NODES>>   <<PBX_AUTO_NGPUS>>
# Replace with concrete numbers to fix size for all subsequent
# respawn links; otherwise pbx auto-computes from remaining work,
# capped at the original allocation size.
#
# DO NOT EDIT the `pbx run ...` line below. The --respawn value
# is pbx-managed; any edit will be overwritten each cycle. If any
# required arg is missing from that line, pbx will error out and
# stop the chain.
#
# To STOP the chain: `pbx qdel <jobid>` or `pbx scancel <jobid>`.
# Do not try to stop it by editing or deleting this file.
#######################################################################"""


def _prepend_respawn_header(script: str) -> str:
    """Insert the respawn-template header comment after the shebang."""
    lines = script.split('\n', 1)
    if lines and lines[0].startswith('#!'):
        return f"{lines[0]}\n{RESPAWN_TEMPLATE_HEADER}\n{lines[1] if len(lines) > 1 else ''}"
    return f"{RESPAWN_TEMPLATE_HEADER}\n{script}"


def submit_job(
    config_name: str,
    job_name: str,
    queue: str,
    select: str,
    walltime: float,
    project: Optional[str] = None,
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
    respawn: Optional[int] = None,
    validate_runnable: bool = True,
) -> Dict[str, Any]:
    """
    Core scheduler submission logic - used by both PBS (qsub) and SLURM (sbatch).

    Args:
        config_name: System configuration name
        job_name: Job name
        queue: Queue/partition name
        select: Resource selection (nodes for SLURM, select spec for PBS)
        walltime: Wall time in minutes (supports float for fractional minutes)
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
        respawn: When set, generate a respawn_template.sh alongside submit.sh and
            embed --respawn N in both scripts' pbx run line so the chain
            self-perpetuates at every walltime boundary. The integer is the
            number of remaining auto-resubmissions (decremented per link;
            0 = no resubmit, chain ends after this run). Pass None to disable.

    Returns:
        Dictionary with submission details including job_id and run_dir.
        When respawn is set, also includes 'respawn_template_file' pointing
        to the generated template path.

    Raises:
        ValidationError: If configuration is invalid, or if respawn is negative.
        FileNotFoundError: If submit command is not found
    """
    if respawn is not None and respawn < 0:
        raise ValidationError(
            f"--respawn must be >= 0, got {respawn}."
        )

    # Tag glob expansion + runnable-jobs guard (skipped when caller has already
    # validated, e.g. unit tests that bypass the DB).
    matched_count = None
    if validate_runnable:
        from parslbox.utils import path_utils
        if tags:
            from parslbox.utils.tag_match import expand_tag_patterns
            try:
                tags = expand_tag_patterns(path_utils.DB_FILE, list(tags))
            except ValueError as e:
                raise ValidationError(str(e))
        matched_count = count_matching_runnable_jobs(path_utils.DB_FILE, list(apps) if apps else None, list(tags) if tags else None)
        if matched_count == 0:
            raise ValidationError(
                "No Ready/Restart jobs match the given --apps/--tags filters. "
                "Refusing to submit (would waste the allocation)."
            )

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
    if respawn is not None:
        run_options.append(f"--respawn {respawn}")

    # Always pass walltime to `pbx run` so it can trigger graceful shutdown
    # 30s before the batch job's walltime expires.
    run_options.append(f"--walltime-seconds {int(walltime * 60)}")

    run_options_str = " ".join(run_options)

    # Capture environment variables for parslbox paths
    pbx_env_vars = ""
    if os.getenv("PBX_DB_PATH"):
        pbx_env_vars += f'export PBX_DB_PATH="{os.getenv("PBX_DB_PATH")}"\n'
    if os.getenv("PBX_CONFIG_PATH"):
        pbx_env_vars += f'export PBX_CONFIG_PATH="{os.getenv("PBX_CONFIG_PATH")}"\n'
    if os.getenv("PBX_RUN_DELAY"):
        pbx_env_vars += f'export PBX_RUN_DELAY="{os.getenv("PBX_RUN_DELAY")}"\n'

    # Prepare template variables (no filesystems, sched_opts as placeholder)
    template_vars = {
        'job_name': job_name,
        'queue': queue,
        'select': select,
        'walltime': walltime_formatted,
        'project': project or '',
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

    # If no project was provided, remove the account/project directive line
    if not project:
        rendered = '\n'.join(
            line for line in rendered.split('\n')
            if not re.match(r'\s*#(PBS\s+-A|SBATCH\s+--account=)\s*$', line)
        )

    # Get config-level sched_opts
    config_sched_opts = system_config.get('sched_opts', None)

    # System defaults from get_default_sched_opts() are already written
    # into config.yaml by `pbx config`. Only inject them if config.yaml
    # has no sched_opts (e.g., hand-edited config without system defaults).
    if config_sched_opts:
        combined_config_opts = config_sched_opts
    else:
        try:
            from parslbox.system_configs.loader import get_system_config
            sys_config_obj = get_system_config(config_name)
            combined_config_opts = sys_config_obj.get_default_sched_opts()
        except (ValueError, Exception):
            combined_config_opts = None

    # Merge directives: template → system+config → CLI
    submit_script = merge_sched_opts(rendered, combined_config_opts, sched_opts)

    # When --respawn is set, the scheduler log rolls per chain link instead of
    # being overwritten each cycle. Link 0 uses pbx_scheduler_link-0.out;
    # build_respawn_link_script bumps the digit for subsequent links.
    if respawn is not None:
        submit_script = submit_script.replace(
            "pbx_scheduler.out", "pbx_scheduler_link-0.out"
        )

    # Write submit script
    submit_file = run_dir / "submit.sh"
    with open(submit_file, 'w') as f:
        f.write(submit_script)

    # When --respawn is set, also generate respawn_template.sh alongside submit.sh.
    # Same content except the resource line has a placeholder for auto-fill at
    # respawn time, and the file starts with an educational header comment.
    respawn_template_file: Optional[Path] = None
    if respawn is not None:
        respawn_template_vars = dict(template_vars)
        respawn_template_vars['select'] = (
            "<<PBX_AUTO_SELECT>>" if scheduler_type == "pbs" else "<<PBX_AUTO_NODES>>"
        )
        respawn_rendered = sched_template.format(**respawn_template_vars)
        if not project:
            respawn_rendered = '\n'.join(
                line for line in respawn_rendered.split('\n')
                if not re.match(r'\s*#(PBS\s+-A|SBATCH\s+--account=)\s*$', line)
            )
        respawn_script = merge_sched_opts(respawn_rendered, combined_config_opts, sched_opts)
        respawn_script = _prepend_respawn_header(respawn_script)
        respawn_script = respawn_script.replace(
            "pbx_scheduler.out", "pbx_scheduler_link-0.out"
        )
        respawn_template_file = run_dir / "respawn_template.sh"
        with open(respawn_template_file, 'w') as f:
            f.write(respawn_script)

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
            "respawn_template_file": str(respawn_template_file) if respawn_template_file else None,
            "matched_jobs": matched_count,
            "resolved_tags": list(tags) if tags else None,
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
            "respawn_template_file": str(respawn_template_file) if respawn_template_file else None,
            "matched_jobs": matched_count,
            "resolved_tags": list(tags) if tags else None,
        }
    except FileNotFoundError:
        raise ValidationError(
            f"{submit_command} command not found. "
            f"Make sure {'PBS' if scheduler_type == 'pbs' else 'SLURM'} is available."
        )
