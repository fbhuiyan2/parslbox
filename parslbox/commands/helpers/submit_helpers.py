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
from typing import Optional, List, Dict, Any, Tuple

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
) -> Tuple[int, List[str]]:
    """Count Ready/Restart jobs matching the given apps/tags filters.

    Apps and tags are matched exactly. `None` for either means "no filter".
    Used by qsub/sbatch to verify there's actually work to do before submitting.

    Returns (count, sorted distinct app names among the matched jobs).
    """
    from parslbox.database import database
    jobs = database.get_jobs(db_path, status='Ready')
    jobs += database.get_jobs(db_path, status='Restart')

    app_set = set(apps_list) if apps_list else None
    tag_set = set(tags_list) if tags_list else None

    count = 0
    matched_apps = set()
    for j in jobs:
        if app_set and j['app'] not in app_set:
            continue
        if tag_set and j['tag'] not in tag_set:
            continue
        count += 1
        matched_apps.add(j['app'])
    return count, sorted(matched_apps)


def summarize_app_configs(
    app_names: List[str],
    config_name: str,
    full_config: Optional[Dict[str, Any]],
    config_path: Path,
) -> List[Dict[str, Any]]:
    """Resolve what each app will actually run with on `config_name`.

    An app whose per-system block is missing still submits and still runs --
    `load_app_config` returns {} and the executable ends up empty. Reporting
    the three cases apart is what separates "edited the right block" from
    "edited a block with the same shape and a different name".

    `env_state` is one of: 'configured', 'no_system_entry', 'no_app_block'.
    """
    config = full_config or {}
    summary = []
    for app in app_names:
        app_block = config.get(app)
        if not isinstance(app_block, dict):
            summary.append({
                "app": app,
                "executable": None,
                "env_state": "no_app_block",
                "detail": f"no `{app}` block in {config_path}",
            })
            continue
        system_block = app_block.get(config_name)
        if not isinstance(system_block, dict):
            summary.append({
                "app": app,
                "executable": None,
                "env_state": "no_system_entry",
                "detail": f"no `{config_name}` entry under `{app}`",
            })
            continue
        summary.append({
            "app": app,
            "executable": system_block.get("executable_path") or None,
            "env_state": "configured",
            "detail": f"configured for {config_name}",
        })
    return summary


def render_app_summary(result: Dict[str, Any], config_name: str, cancel_cmd: str) -> None:
    """Print the per-app resolution block plus the cancel-now hint.

    `cancel_cmd` is "qdel" or "scancel". Nothing here blocks or validates --
    the job is already in the queue by the time this runs.
    """
    import typer

    summary = result.get("app_summary")
    job_id = result.get("job_id")

    if summary:
        typer.secho(f"\U0001f4e6 Apps in this submission ({config_name}):",
                    fg=typer.colors.CYAN)
        for entry in summary:
            typer.secho(f"   {entry['app']}", fg=typer.colors.WHITE, bold=True)
            if entry["executable"]:
                typer.echo(f"     executable: {entry['executable']}")
            else:
                typer.secho(
                    "     executable: (not set)  ← make sure this is intended",
                    fg=typer.colors.YELLOW,
                )
            if entry["env_state"] == "configured":
                typer.echo(f"     env setup:  {entry['detail']}")
            else:
                typer.secho(f"     env setup:  {entry['detail']}",
                            fg=typer.colors.YELLOW)

    if not job_id:
        return

    typer.secho(
        "⚠  If anything above looks wrong, cancel now and fix it before "
        "the job starts:",
        fg=typer.colors.YELLOW,
    )
    typer.echo(f"     {cancel_cmd} {job_id}")
    typer.echo("   or, if the job already started:")
    typer.echo(f"     pbx {cancel_cmd} {job_id}")


def render_submit_script_panel(submit_file_path, title: str = "Submit Script"):
    """Build a Rich Panel containing the submit script with the `pbx run` line
    highlighted in red. Returns the Panel ready to be `console.print`-ed.
    """
    from rich.panel import Panel
    from rich.text import Text

    abs_path = Path(submit_file_path).resolve()
    script = abs_path.read_text()
    text = Text()
    for line in script.splitlines():
        if 'pbx run' in line:
            text.append(line, style="bold red")
        else:
            text.append(line)
        text.append("\n")
    return Panel(
        text,
        title=f"[bold cyan]{title}[/bold cyan] [dim]({abs_path})[/dim]",
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
    db_path: Optional[Path] = None,
    no_local: bool = False,
    push_dirs: bool = False,
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
        db_path: Database to validate against and to bake into the generated
            script's PBX_DB_PATH export. Defaults to path_utils.DB_FILE.
        no_local: Submit here even if the database belongs to a local project.
        push_dirs: In a local project, also send the matching jobs' directories
            over Globus Transfer before submitting.
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

    from parslbox.utils import path_utils
    if db_path is None:
        db_path = path_utils.DB_FILE
    if config_path is None:
        config_path = path_utils.PBX_CONFIG_FILE

    # Both are exported into submit.sh, where they are resolved against the
    # allocation's working directory rather than this one.
    try:
        db_path = path_utils.require_absolute("db_path", db_path)
        config_path = path_utils.require_absolute("config_path", config_path)
    except path_utils.PbxPathError as e:
        raise ValidationError(str(e))

    # Tag glob expansion + runnable-jobs guard (skipped when caller has already
    # validated, e.g. unit tests that bypass the DB).
    matched_count = None
    matched_apps: List[str] = []
    if validate_runnable:
        if tags:
            from parslbox.utils.tag_match import expand_tag_patterns
            try:
                tags = expand_tag_patterns(db_path, list(tags))
            except ValueError as e:
                raise ValidationError(str(e))
        matched_count, matched_apps = count_matching_runnable_jobs(
            db_path, list(apps) if apps else None, list(tags) if tags else None
        )
        if matched_count == 0:
            raise ValidationError(
                "No Ready/Restart jobs match the given --apps/--tags filters. "
                "Refusing to submit (would waste the allocation)."
            )

    # A local project's database holds remote paths, and the scheduler that
    # can act on them is on the other machine. Hand the whole submission over.
    if not no_local:
        from parslbox.local.project import find_project, LocalProjectError
        try:
            local_proj = find_project(db_path)
        except LocalProjectError as e:
            raise ValidationError(str(e))
        if local_proj is not None:
            from parslbox.local import sync as local_sync
            remote_kwargs = {
                "config_name": config_name,
                "job_name": job_name,
                "queue": queue,
                "select": select,
                "walltime": walltime,
                "project": project,
                "apps": list(apps) if apps else None,
                "tags": list(tags) if tags else None,
                "retries": retries,
                "loglevel": loglevel,
                "sched_opts": list(sched_opts) if sched_opts else None,
                "scheduler_type": scheduler_type,
                "submit_command": submit_command,
                "dynamic": dynamic,
                "respawn": respawn,
                "validate_runnable": validate_runnable,
            }
            from parslbox.local.guard import SyncConflict
            try:
                return local_sync.submit_remote(
                    db_path, remote_kwargs, run_dir=run_dir,
                    push_dirs=push_dirs,
                )
            except (LocalProjectError, SyncConflict) as e:
                raise ValidationError(str(e))

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

    app_summary = summarize_app_configs(matched_apps, config_name, config, config_path)

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

    # Pin the resolved db/config into the script so `pbx run` inside the
    # allocation opens exactly what was validated here, rather than
    # re-resolving against whatever environment the batch job inherits.
    pbx_env_vars = f'export PBX_DB_PATH="{db_path}"\n'
    pbx_env_vars += f'export PBX_CONFIG_PATH="{config_path}"\n'
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
            "app_summary": app_summary,
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
            "app_summary": app_summary,
        }
    except FileNotFoundError:
        raise ValidationError(
            f"{submit_command} command not found. "
            f"Make sure {'PBS' if scheduler_type == 'pbs' else 'SLURM'} is available."
        )


def render_remote_result(
    result: Dict[str, Any],
    monitor_cmd: str = "qstat",
    config_name: Optional[str] = None,
    cancel_cmd: str = "qdel",
) -> bool:
    """Print the outcome of a submission that ran on the remote machine.

    The submit script and run directory named in `result` live over there, so
    there is nothing local to open -- only paths to report. The config.yaml the
    run will read lives over there too, so no app summary is printed here: the
    local one may not even hold those blocks.

    Returns True when the remote scheduler accepted the job.
    """
    import typer

    push = result.get("push") or {}
    if push.get("bytes"):
        typer.secho(
            f"⬆️  Pushed the database to {push.get('remote_db_path')} "
            f"({push.get('fingerprint', {}).get('count')} jobs)",
            fg=typer.colors.BLUE,
        )
    dirs = result.get("dirs")
    if dirs:
        from parslbox.commands.local import render_dir_report
        render_dir_report(dirs)

    typer.secho(f"\U0001f30e Submitted on endpoint {str(result.get('endpoint'))[:8]}…",
                fg=typer.colors.CYAN)
    typer.secho(f"\U0001f4c1 Remote run directory: {result.get('remote_run_dir')}",
                fg=typer.colors.BLUE)
    typer.secho(f"\U0001f4dd Remote submit script: {result.get('submit_file')}",
                fg=typer.colors.BLUE)

    if result.get("pull_error"):
        typer.secho(f"⚠  Could not pull afterwards: {result['pull_error']}",
                    fg=typer.colors.YELLOW)

    if result.get("success"):
        job_id = result.get("job_id", "UNKNOWN")
        typer.secho(f"\U0001f680 Job submitted successfully! Job ID: {job_id}",
                    fg=typer.colors.GREEN)
        typer.secho(f"\U0001f4ca Monitor on the remote with: {monitor_cmd} {job_id}",
                    fg=typer.colors.BLUE)
        typer.secho("\U0001f4a1 Run 'pbx local pull' to bring progress back here.",
                    fg=typer.colors.CYAN)
        for_system = f" for {config_name}" if config_name else ""
        typer.secho(
            "\U0001f30e This is a local project; the submission uses the config "
            "file on the remote host.",
            fg=typer.colors.CYAN,
        )
        typer.secho(
            f"   Check the app executables and env blocks in the remote "
            f"config.yaml to make sure nothing is missing or wrong{for_system}.",
            fg=typer.colors.CYAN,
        )
        typer.secho(
            "⚠  If anything is wrong, cancel on the remote host:",
            fg=typer.colors.YELLOW,
        )
        typer.echo(f"     {cancel_cmd} {job_id}")
        typer.echo("   or, if the job already started:")
        typer.echo(f"     pbx {cancel_cmd} {job_id}")
        return True

    typer.secho(f"❌ The remote scheduler refused the job: {result.get('error')}",
                fg=typer.colors.RED)
    return False
