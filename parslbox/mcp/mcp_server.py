import functools
import sys

from mcp.server.fastmcp import FastMCP

from parslbox.api import ParslBox
from parslbox.local.project import (
    LocalProjectError,
    count_rows,
    find_project,
    resolve_db_path,
)
from parslbox.utils import path_utils
from parslbox.utils.path_utils import PbxPathError
from parslbox.commands.helpers.job_id_parser import format_job_ids, group_failures
from parslbox.mcp.schemas import (
    CommandLinesSchema,
    ResourceRequirementsSchema,
    schema_text_block,
    AddJobSchema,
    LocalInitSchema,
    LocalPullSchema,
    LocalPushSchema,
    LocalStatusSchema,
    CancelJobSchema,
    FilterJobsSchema,
    GetJobSchema,
    GetJobsByIdsSchema,
    ListJobsSchema,
    QSubSchema,
    SBatchSchema,
    RemoveJobsSchema,
    UpdateJobSchema,
    UseDatabaseSchema,
)

# Start ParslBox and MCP.
def _start() -> ParslBox:
    """ParslBox, or a readable reason why the server cannot start.

    A wrong PBX_DB_PATH or a missing config is the caller's environment, not
    a bug here. An MCP client shows the server's stderr and nothing else, so
    a traceback would bury the one line that says what to fix.
    """
    try:
        return ParslBox()
    except (LocalProjectError, PbxPathError, FileNotFoundError) as e:
        sys.stderr.write(f"\nParslBox MCP server did not start.\n\n{e}\n\n")
        raise SystemExit(1)


pbx = _start()


def _db_line() -> str:
    """One-line statement of which database the answer came from.

    The binding is fixed until use_database changes it and does not follow the
    caller's working directory, so every job-facing answer says where it came
    from. The directory name leads because a wrong project is far easier to
    notice as a name than as a path.
    """
    return f"db: {pbx.db_path.parent.name} ({pbx.db_path})"


def _with_db_line(fn):
    """Prefix a tool's answer with the database it came from.

    Applied below @mcp.tool so FastMCP still sees the real signature. Covers
    every return path, error ones included: an exception raised against the
    wrong project is just as misleading as a wrong job list.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return f"{_db_line()}\n{fn(*args, **kwargs)}"
    return wrapper


def _resolved_params(params) -> dict:
    """Dump a schema, expanding its job-id tokens against the active database.

    The schema validator deliberately leaves ranges unexpanded, because it
    runs before this function has a database to bound them with. Here `pbx`
    is bound, so '1-100000000' is refused on its endpoint rather than
    becoming a hundred million integers.
    """
    from parslbox.commands.helpers.job_id_parser import resolve_job_ids

    out = params.model_dump()
    if out.get("job_ids") is not None:
        out["job_ids"] = resolve_job_ids(out["job_ids"], pbx.db_path)
    return out


def _fmt_resources(job: dict) -> str:
    """Compact resource summary: 'nodes:5, ranks:60, gpus:60, nocc:1.0'."""
    nn = job.get('num_nodes', 1)
    g = job.get('ngpus', 0)
    rpn = job.get('ranks_per_node', 1)
    occ = job.get('node_occupancy', 1.0)
    return f"nodes:{nn}, ranks:{nn * rpn}, gpus:{g}, nocc:{occ}"


def _job_one_line(job: dict) -> str:
    """Single-line job summary for list_jobs."""
    return " | ".join([
        f"#{job['job_id']}",
        f"{job.get('app', '?')}",
        f"{job.get('status', '?')}",
        _fmt_resources(job),
        f"tag:{job.get('tag')}",
        f"sched:{job.get('sched_job_id')}",
        f"in:{job.get('in_file')}",
        f"env:{job.get('env_file')}",
        f"parents:{job.get('parents')}",
        f"{job.get('timestamp', '')}",
        f"{job.get('path', '')}",
    ])

mcp = FastMCP(
    name="ParslBox Tool MCP",
    instructions=(
        "You expose tools for managing and running ParslBox jobs.\n\n"
        f"ACTIVE DATABASE: {pbx.db_path}\n"
        f"ACTIVE CONFIG:   {pbx.config_path}\n\n"
        "That binding is fixed for this server and does NOT follow the caller's "
        "working directory. It was read from PBX_DB_PATH when the server started; "
        "a later `cd`, or an `export PBX_DB_PATH=...` in a shell, runs in a "
        "different process and cannot change it. Only use_database changes it.\n\n"
        "So: before using any job tool against a project directory you have not "
        "already bound to, call use_database with that directory. Every job tool "
        "prefixes its answer with the database it used; if that line names the "
        "wrong project, you are on the wrong database -- call use_database rather "
        "than acting on the result. The pbx CLI is unaffected by all of this: it "
        "reads PBX_DB_PATH fresh on every invocation.\n\n"
        "Available capabilities:\n"
        "- use_database: rebind this server to another project's database (and optionally its config).\n"
        "- add_jobs: create new jobs and add them to the ParslBox database.\n"
        "- submit_pbs_job: submit PBS jobs to the queue using a qsub-style configuration.\n"
        "- submit_slurm_job: submit SLURM jobs to the queue using an sbatch-style configuration.\n"
        "- cancel_pbs_job: gracefully cancel a running PBS batch job (preferred over raw qdel — keeps DB accurate).\n"
        "- cancel_slurm_job: gracefully cancel a running SLURM batch job (preferred over raw scancel — keeps DB accurate).\n"
        "- remove_jobs: remove one or more jobs from the database.\n"
        "- update_jobs: modify fields of one or more existing jobs (status, tag, input file, resources, dependencies, etc.). Note: app cannot be changed after job creation.\n"
        "- list_systems_and_apps: the HPC systems and app identifiers this server accepts. Read-only.\n"
        "- get_resource_requirements: how many nodes of a target system the jobs need. Read-only.\n"
        "- get_command_lines: the MPI/srun line each job would be launched with on a target system. A preview; nothing runs.\n"
        "- filter_jobs: filter jobs by status, app, tag, path, or input file and return their IDs.\n"
        "- list_jobs: list jobs with full details, with optional filtering by status, app, tag, path, or input file.\n"
        "- get_job: get a single job's full details by its ID.\n"
        "- get_jobs: get multiple jobs' full details by their IDs.\n"
        "- local_init: create a local project in a directory, so jobs authored there run on a remote machine.\n"
        "- local_status: for a local project (jobs authored here, run on a remote machine), "
        "report which side is ahead and whether the Compute endpoint answers.\n"
        "- local_push: send the local project's database to the remote machine.\n"
        "- local_pull: bring the remote database back down.\n\n"
    ),
)


@mcp.tool(
    name="list_systems_and_apps",
    description=(
        "The HPC systems and app identifiers every other tool accepts.\n\n"
        "Systems come with the hardware that decides how a job is sized: scheduler, "
        "cores and GPUs per node, and MPI backend. Apps are the built-ins plus "
        "whatever the bound config.yaml declares under custom_apps, so the answer "
        "follows use_database.\n\n"
        "Call this before asking the user to choose a system -- the names alone do "
        "not say which are PBS and which are SLURM, or which have GPUs."
    ),
)
def list_systems_and_apps() -> str:
    lines = []
    for s in pbx.available_systems():
        gpus = (f"{s['gpus_per_node']}x {s['gpu_type']}" if s["gpus_per_node"]
                else "no gpu")
        lines.append(
            f"{s['name']}: {s['scheduler']}, {gpus}, {s['cores_per_node']} cores, "
            f"{s['mpi_backend']}"
        )
    apps = pbx.available_apps()
    lines.append(f"apps builtin: {', '.join(apps['builtin'])}")
    lines.append(f"apps custom: {', '.join(apps['custom']) or 'none'}")
    return "\n".join(lines)


@mcp.tool(
    name="use_database",
    description=(
        "Point this server at a different ParslBox database, for the rest of the session.\n\n"
        "THE BINDING IS FIXED UNTIL THIS TOOL CHANGES IT. It is set from PBX_DB_PATH when "
        "the server starts and does NOT follow your working directory, so a `cd` or an "
        "`export PBX_DB_PATH=...` in a shell has no effect on it -- those run in a "
        "different process. Before using any job tool against a project directory you "
        "have not already bound to, call this first. Every job tool reports the database "
        "it used on a `db:` line; if that line names the wrong project, call this.\n\n"
        "Takes the project directory (or a .db file directly), resolving a local "
        "project to its -local database. Refuses a database that does not exist "
        "unless create is set. Reports the resolved paths and the job count."
    ),
)
def use_database(params: UseDatabaseSchema) -> str:
    global pbx

    try:
        db_path = resolve_db_path(params.path)
    except (PbxPathError, LocalProjectError) as e:
        return f"Cannot use that path. {e}"

    if not db_path.exists() and not params.create:
        return (
            f"No database at {db_path}. Nothing was changed.\n"
            f"Still using {pbx.db_path}.\n"
            "If this path is a typo, fix it. To start a new project here, either call "
            "local_init (for a laptop-to-remote project) or run `pbx add` there, or "
            "call this again with create=true."
        )

    config_path = None
    if params.config_path:
        try:
            config_path = path_utils.require_absolute("config_path", params.config_path)
        except PbxPathError as e:
            return f"Cannot use that config path. {e}"
        if config_path.is_dir():
            config_path = config_path / "config.yaml"
        if not config_path.is_file():
            return f"No config file at {config_path}. Nothing was changed."

    previous = pbx.db_path
    try:
        pbx = ParslBox(db_path=db_path, config_path=config_path or pbx.config_path)
    except Exception as e:
        return f"Could not switch to {db_path}: {e}\nStill using {previous}."

    lines = [f"Now using {pbx.db_path}"]
    if previous != pbx.db_path:
        lines.append(f"(was {previous})")
    lines.append(f"config: {pbx.config_path}")
    try:
        lines.append(f"jobs: {count_rows(pbx.db_path)}")
    except Exception:
        pass
    try:
        project = find_project(pbx.db_path)
    except LocalProjectError:
        project = None
    if project is not None:
        lines.append(f"local project -> {project.remote_root}")
    return "\n".join(lines)


@mcp.tool(
    name="add_jobs",
    description="Add jobs to ParslBox",
)
@_with_db_line
def add_jobs(params: AddJobSchema) -> str:
    input_dict = params.model_dump()
    
    try:
        successful_job_ids, failed_jobs, msg_log = pbx.add_jobs(**input_dict)
    except Exception as e:
        return f"Exception occurred when adding jobs. Exception: {e}"
    
    # Build response message
    response_parts = []
    
    if successful_job_ids:
        response_parts.append(f"Successfully added {len(successful_job_ids)} job(s), IDs: {format_job_ids(successful_job_ids)}")

    if failed_jobs:
        response_parts.append(f"Failed to add {len(failed_jobs)} job(s):")
        for error, paths in group_failures(failed_jobs).items():
            response_parts.append(f"[{len(paths)} job(s)] {error}:")
            response_parts.extend(f"  {p}" for p in sorted(paths))
    
    # Add warnings and info messages
    for warning in msg_log.get('warnings', []):
        response_parts.append(f"Warning: {warning}")
    
    for info in msg_log.get('info', []):
        response_parts.append(f"Info: {info}")
    
    if not successful_job_ids and not failed_jobs:
        response_parts.append("No jobs were processed.")
    
    return "\n".join(response_parts)


def _app_summary_lines(status: dict, cancel_cmd: str) -> list:
    """Compact per-app resolution + the cancel hint.

    Remote submits carry no summary: the config.yaml that matters is on the
    other machine, so say that instead of resolving against the local one.
    """
    job_id = status.get("job_id") or "UNKNOWN"
    if status.get("remote"):
        return [
            "This is a local project; the submission uses the config file on the "
            "remote host. Check the app executables and env blocks in the remote "
            "config.yaml to make sure nothing is missing or wrong.",
            f"If anything is wrong, cancel on the remote host: {cancel_cmd} {job_id} "
            f"(or `pbx {cancel_cmd} {job_id}` if it already started).",
        ]

    summary = status.get("app_summary")
    if not summary:
        return []

    parts = []
    for e in summary:
        exe = e.get("executable") or "NOT SET"
        state = {
            "configured": "env:configured",
            "no_system_entry": "env:NO SYSTEM ENTRY",
            "no_app_block": "env:NO APP BLOCK",
        }.get(e.get("env_state"), "env:?")
        parts.append(f"{e['app']} [exe:{exe}, {state}]")
    return [
        "apps: " + " | ".join(parts),
        f"If anything above looks wrong, cancel now: {cancel_cmd} {job_id} "
        f"(or `pbx {cancel_cmd} {job_id}` if it already started).",
    ]


def _remote_lines(status: dict) -> list:
    """Extra lines when the submission went to another machine.

    Without these an agent cannot tell a remote submission from a local one:
    the run directory it is shown does not exist here, and a failed post-submit
    pull -- which means the local database is now behind -- is invisible.
    """
    if not status.get("remote"):
        return []
    lines = [f"Submitted on remote Compute endpoint {status.get('endpoint')}.",
             f"Remote run directory: {status.get('remote_rundir')}"]
    push = status.get("push") or {}
    if push.get("bytes") is not None:
        lines.append(f"Pushed {push['bytes']} bytes of database first.")
    transfer = (push.get("transfer") or {}).get("outcome")
    if transfer:
        lines.append(f"Directory transfer {transfer['status']}, "
                     f"{transfer.get('files_transferred')} files.")
    if status.get("pull_error"):
        lines.append(f"WARNING: the post-submit pull failed ({status['pull_error']}), "
                     f"so the local database may be behind the remote.")
    return lines


@mcp.tool(
    name="submit_pbs_job",
    description=(
        "Submit a PBS job via ParslBox.qsub.\n\n"
        "Takes configuration (system config name, job name, queue, select, walltime, project) "
        "plus optional run directory, apps, tags, retries, and sched_opts for extra PBS directives. "
        "Walltime defaults to minutes; supports h/d suffixes (e.g., 90, 4.25h, 3.5d). "
        "Set respawn=N to enable a self-respawn chain: at walltime, in-flight jobs are "
        "marked Restart and the next link is auto-submitted, until all jobs finish or "
        "respawn reaches 0. Returns a short message describing whether submission succeeded, "
        "including an `apps:` line resolving each app in the submission to its executable "
        "and env block on the target system (env:configured / NO SYSTEM ENTRY / NO APP BLOCK). "
        "Check that line: a missing block still submits and still runs, with an empty "
        "executable. Cancel with the command given there if it looks wrong."
    ),
)
@_with_db_line
def submit_pbs_job(params: QSubSchema) -> str:
    input_dict = params.model_dump()

    try:
        status = pbx.qsub(**input_dict)
    except Exception as e:
        return f"Exception occurred when submitting job. Exception: {e}"

    if status.get("success") is True:
        job_id = status.get("job_id", status.get("pbs_job_id", "UNKNOWN"))
        rundir = status.get("rundir", "UNKNOWN")
        matched = status.get("matched_jobs")
        resolved = status.get("resolved_tags")
        respawn_template = status.get("respawn_template_file")
        lines = [
            f"Job was submitted successfully.",
            f"PBS job ID: {job_id}",
            f"Run directory: {rundir}",
        ]
        if matched is not None:
            lines.append(f"Matched {matched} runnable job(s) in DB.")
        if resolved:
            lines.append(f"Resolved tags: {', '.join(resolved)}")
        if respawn_template:
            lines.append(f"Respawn template: {respawn_template} (chain auto-resubmits at walltime)")
        lines += _remote_lines(status)
        lines += _app_summary_lines(status, "qdel")
        return "\n".join(lines)
    else:
        error_msg = status.get("error", "Unknown error")
        rundir = status.get("rundir", "UNKNOWN")
        return (
            "Failed to submit job.\n"
            f"Error: {error_msg}\n"
            f"Run directory (if created): {rundir}"
        )


@mcp.tool(
    name="submit_slurm_job",
    description=(
        "Submit a SLURM job via ParslBox.sbatch.\n\n"
        "Takes configuration (system config name, job name, partition, nodes, walltime, project) "
        "plus optional run directory, apps, tags, retries, and sched_opts for extra SLURM directives. "
        "Walltime defaults to minutes; supports h/d suffixes (e.g., 90, 4.25h, 3.5d). "
        "Set respawn=N to enable a self-respawn chain: at walltime, in-flight jobs are "
        "marked Restart and the next link is auto-submitted, until all jobs finish or "
        "respawn reaches 0. Returns a short message describing whether submission succeeded, "
        "including an `apps:` line resolving each app in the submission to its executable "
        "and env block on the target system (env:configured / NO SYSTEM ENTRY / NO APP BLOCK). "
        "Check that line: a missing block still submits and still runs, with an empty "
        "executable. Cancel with the command given there if it looks wrong."
    ),
)
@_with_db_line
def submit_slurm_job(params: SBatchSchema) -> str:
    input_dict = params.model_dump()

    try:
        status = pbx.sbatch(**input_dict)
    except Exception as e:
        return f"Exception occurred when submitting SLURM job. Exception: {e}"

    if status.get("success") is True:
        job_id = status.get("job_id", status.get("slurm_job_id", "UNKNOWN"))
        rundir = status.get("rundir", "UNKNOWN")
        matched = status.get("matched_jobs")
        resolved = status.get("resolved_tags")
        respawn_template = status.get("respawn_template_file")
        lines = [
            f"SLURM job was submitted successfully.",
            f"SLURM job ID: {job_id}",
            f"Run directory: {rundir}",
        ]
        if matched is not None:
            lines.append(f"Matched {matched} runnable job(s) in DB.")
        if resolved:
            lines.append(f"Resolved tags: {', '.join(resolved)}")
        if respawn_template:
            lines.append(f"Respawn template: {respawn_template} (chain auto-resubmits at walltime)")
        lines += _remote_lines(status)
        lines += _app_summary_lines(status, "scancel")
        return "\n".join(lines)
    else:
        error_msg = status.get("error", "Unknown error")
        rundir = status.get("rundir", "UNKNOWN")
        return (
            "Failed to submit SLURM job.\n"
            f"Error: {error_msg}\n"
            f"Run directory (if created): {rundir}"
        )


@mcp.tool(
    name="cancel_pbs_job",
    description=(
        "Gracefully cancel a running ParslBox PBS batch job.\n\n"
        "Sends SIGTERM via qsig, waits `grace` seconds (default 30) so the orchestrator "
        "can reconcile in-flight jobs in the database, then runs qdel to terminate. "
        "After the scheduler kill, pbx reconciles the DB per state for any jobs still "
        "non-terminal under this batch (matched by sched_job_id): Running→Killed, "
        "Submitted→Ready, Resubmitted→Restart, so the DB ends up consistent even when "
        "the orchestrator's signal handler doesn't complete cleanly. Scoped by "
        "sched_job_id, so concurrent batch jobs are unaffected. "
        "Always prefer this over a raw qdel for ParslBox jobs — raw qdel only gives the "
        "orchestrator the cluster's default kill grace (often ~2s) and does no DB cleanup."
    ),
)
@_with_db_line
def cancel_pbs_job(params: CancelJobSchema) -> str:
    input_dict = params.model_dump()
    try:
        result = pbx.qdel(**input_dict)
    except Exception as e:
        return f"Exception occurred when cancelling PBS job. Exception: {e}"

    reconciled = result.get("reconciled_count", 0)
    recon_suffix = (
        f" Reconciled {reconciled} non-terminal job(s) (Running→Killed, Submitted→Ready, Resubmitted→Restart)."
        if reconciled else ""
    )

    if result.get("success"):
        return (f"PBS job {result['jobid']} cancelled cleanly "
                f"(grace: {result.get('grace', '?')}s).{recon_suffix}")
    return (f"Failed to cancel PBS job {result.get('jobid', '?')} "
            f"at stage '{result.get('stage', '?')}': {result.get('error', 'unknown')}."
            f"{recon_suffix}")


@mcp.tool(
    name="cancel_slurm_job",
    description=(
        "Gracefully cancel a running ParslBox SLURM batch job.\n\n"
        "Sends SIGTERM to the batch script via `scancel --signal=TERM --batch`, waits "
        "`grace` seconds (default 30) so the orchestrator can reconcile in-flight jobs "
        "in the database, then runs scancel to terminate. After the scheduler kill, pbx "
        "reconciles the DB per state for any jobs still non-terminal under this batch "
        "(matched by sched_job_id): Running→Killed, Submitted→Ready, Resubmitted→Restart, "
        "so the DB ends up consistent even when the orchestrator's signal handler doesn't "
        "complete cleanly. Scoped by sched_job_id, so concurrent batch jobs are "
        "unaffected. Always prefer this over a raw scancel for ParslBox jobs — raw "
        "scancel does no DB cleanup."
    ),
)
@_with_db_line
def cancel_slurm_job(params: CancelJobSchema) -> str:
    input_dict = params.model_dump()
    try:
        result = pbx.scancel(**input_dict)
    except Exception as e:
        return f"Exception occurred when cancelling SLURM job. Exception: {e}"

    reconciled = result.get("reconciled_count", 0)
    recon_suffix = (
        f" Reconciled {reconciled} non-terminal job(s) (Running→Killed, Submitted→Ready, Resubmitted→Restart)."
        if reconciled else ""
    )

    if result.get("success"):
        return (f"SLURM job {result['jobid']} cancelled cleanly "
                f"(grace: {result.get('grace', '?')}s).{recon_suffix}")
    return (f"Failed to cancel SLURM job {result.get('jobid', '?')} "
            f"at stage '{result.get('stage', '?')}': {result.get('error', 'unknown')}."
            f"{recon_suffix}")


@mcp.tool(
    name="remove_jobs",
    description="Remove jobs from ParslBox",
)
@_with_db_line
def remove_jobs(params: RemoveJobsSchema) -> str:
    try:
        removed_count = pbx.remove_jobs(**_resolved_params(params))
    except Exception as e:
        return f"Exception occurred when removing jobs. Exception: {e}"
    
    if removed_count > 0:
        return f"Successfully removed {removed_count} job(s)."
    else:
        return "No jobs were removed (jobs may not exist)."


@mcp.tool(
    name="update_jobs",
    description=("Update fields for one or more existing ParslBox jobs.\n\n"),
)
@_with_db_line
def update_jobs(params: UpdateJobSchema) -> str:
    try:
        input_dict = _resolved_params(params)
        job_ids = input_dict["job_ids"]
        updated_job_ids, failed_jobs, msg_log = pbx.update_jobs(**input_dict)
    except Exception as e:
        return f"Exception occurred when updating jobs. Exception: {e}"

    response_parts = []

    if updated_job_ids:
        response_parts.append(
            f"Successfully updated {len(updated_job_ids)} job(s): "
            f"{format_job_ids(updated_job_ids)}."
        )

    for failed_job_id, error_msg in failed_jobs:
        response_parts.append(f"Failed to update job {failed_job_id}: {error_msg}")

    untouched = sorted(set(job_ids) - set(updated_job_ids)
                       - {fid for fid, _ in failed_jobs})
    if untouched:
        response_parts.append(
            f"Not found, or no fields changed: {format_job_ids(untouched)}."
        )

    if not response_parts:
        response_parts.append("No jobs were updated.")

    for warning in msg_log.get('warnings', []):
        response_parts.append(f"Warning: {warning}")

    for info in msg_log.get('info', []):
        response_parts.append(f"Info: {info}")

    return "\n".join(response_parts)


@mcp.tool(
    name="get_resource_requirements",
    description=(
        "How many nodes of a target system these jobs need. Read-only: nothing "
        "is submitted and nothing in the database changes.\n\n"
        "Counts only Ready/Restart jobs and reports the rest as skipped. "
        "'simultaneous' is every job running at once; 'optimal' is what pbx "
        "needs once it queues jobs behind each other."
    ),
)
@_with_db_line
def get_resource_requirements(params: ResourceRequirementsSchema) -> str:
    try:
        req = pbx.resource_requirements(**_resolved_params(params))
    except Exception as e:
        return f"Exception occurred when sizing jobs. Exception: {e}"

    lines = [f"{req['system']}: {req['gpus_per_node']} GPUs/node, "
             f"{req['cores_per_node']} cores/node",
             f"schedulable: {req['schedulable']}"]
    if req["skipped"]:
        lines.append("skipped: " + ", ".join(f"{c} {s}" for s, c in req["skipped"].items()))
    if not req["schedulable"]:
        lines.append("No Ready/Restart jobs to size.")
        return "\n".join(lines)

    gpu = req["gpu"]
    if gpu:
        if not gpu["system_has_gpus"]:
            lines.append(f"gpu: {gpu['jobs']} job(s) need {gpu['total_gpus']} GPUs, "
                         f"but {req['system']} has none -- they cannot run here")
        else:
            lines.append(f"gpu: {gpu['jobs']} job(s), {gpu['total_gpus']} GPUs -> "
                         f"{gpu['simultaneous']} nodes simultaneous / "
                         f"{gpu['optimal']} optimal")
            for j in gpu["oversized"]:
                lines.append(f"  job {j['job_id']} wants {j['ngpus']} GPUs on one node, "
                             f"more than the {gpu['gpus_per_node']} available")
    cpu = req["cpu"]
    if cpu:
        lines.append(f"cpu: {cpu['jobs']} job(s), occupancy {cpu['total_occupancy']:.1f} -> "
                     f"{cpu['simultaneous']} nodes simultaneous / {cpu['optimal']} optimal")
    if req["total"]:
        lines.append(f"total: {req['total']['simultaneous']} nodes simultaneous / "
                     f"{req['total']['optimal']} optimal")
    return "\n".join(lines)


@mcp.tool(
    name="get_command_lines",
    description=(
        "The MPI/srun line each job would be launched with on a target system. "
        "A preview only -- nothing runs, and hostnames are placeholders.\n\n"
        "Returns one line per job, so name the jobs you care about rather than "
        "asking for a whole large database."
    ),
)
@_with_db_line
def get_command_lines(params: CommandLinesSchema) -> str:
    try:
        preview = pbx.command_lines(**_resolved_params(params))
    except Exception as e:
        return f"Exception occurred when building command lines. Exception: {e}"

    rows = preview["jobs"]
    if not rows:
        return f"{preview['system']}: no jobs to preview."
    lines = [f"{preview['system']} (hostnames are placeholders):"]
    for r in rows:
        if r["error"]:
            lines.append(f"{r['job_id']}: error: {r['error']}")
        else:
            lines.append(f"{r['job_id']} [{r['resources']}] {r['command']}")
    return "\n".join(lines)


@mcp.tool(
    name="filter_jobs",
    description=("Filter jobs in ParslBox and return their IDs. "),
)
@_with_db_line
def filter_jobs(params: FilterJobsSchema) -> str:
    input_dict = params.model_dump()
    try:
        job_ids: list[int] = pbx.filter_jobs(**input_dict)
    except Exception as e:
        return f"Exception occurred when filtering jobs. Exception: {e}"

    if not job_ids:
        return "No jobs matched the given filters."

    # Space-separated IDs like your CLI filter
    return " ".join(str(jid) for jid in job_ids)


@mcp.tool(
    name="list_jobs",
    description=(
        "List jobs as one compact line per job (id, app, status, resources, "
        "tag, sched_job_id, input/env files, parents, timestamp, path). "
        "Supports the same filters as filter_jobs (tag accepts `*` globs). "
        "For full per-field details on specific jobs, use get_job / get_jobs."
    ),
)
@_with_db_line
def list_jobs(params: ListJobsSchema) -> str:
    input_dict = params.model_dump()
    try:
        jobs = pbx.list_jobs(**input_dict)
    except Exception as e:
        return f"Exception occurred when listing jobs. Exception: {e}"

    if not jobs:
        return "No jobs found matching the given filters."

    lines = [f"Found {len(jobs)} job(s):"]
    lines.extend(_job_one_line(j) for j in jobs)
    return "\n".join(lines)


@mcp.tool(
    name="get_job",
    description="Get a single job's full details by its ID.",
)
@_with_db_line
def get_job(params: GetJobSchema) -> str:
    try:
        job = pbx.get_job(params.job_id)
    except Exception as e:
        return f"Exception occurred when getting job {params.job_id}. Exception: {e}"

    # Format job as a readable block
    parts = [f"Job {job['job_id']}:"]
    for key, value in job.items():
        if key != "job_id":
            parts.append(f"  {key}: {value}")

    return "\n".join(parts)


@mcp.tool(
    name="get_jobs",
    description="Get multiple jobs' full details by their IDs.",
)
@_with_db_line
def get_jobs(params: GetJobsByIdsSchema) -> str:
    try:
        jobs = pbx.get_jobs_by_ids(params.job_ids)
    except Exception as e:
        return f"Exception occurred when getting jobs. Exception: {e}"

    if not jobs:
        return "No jobs found for the given IDs."

    response_parts = [f"Found {len(jobs)} job(s):\n"]
    for job in jobs:
        parts = [f"  Job {job['job_id']}:"]
        for key, value in job.items():
            if key != "job_id":
                parts.append(f"    {key}: {value}")
        response_parts.append("\n".join(parts))

    return "\n".join(response_parts)


@mcp.tool(
    name="local_init",
    description=(
        "Create a local project: a directory whose ParslBox database stores remote paths, "
        "so jobs can be authored on this machine and run on a remote one.\n\n"
        "Writes a database and a .pbxlocal.yaml, touches no network. Refuses if the "
        "directory already holds a populated database with no project file, or if a "
        "project exists in a parent directory (pass nested_ok to go ahead). Re-running "
        "on an existing project reports it and changes nothing."
    ),
)
def local_init(params: LocalInitSchema) -> str:
    try:
        result = pbx.local_init(**params.model_dump())
    except Exception as e:
        return f"Exception occurred when creating the local project. Exception: {e}"
    lines = [
        f"{result['action']}: {result['local_root']} -> {result['remote_root']}",
        f"db: {result['db_path']}",
        f"endpoint: {result['compute_endpoint'] or 'not configured'}",
    ]
    check = result.get("endpoint_check")
    if check is not None:
        lines.append(
            f"endpoint reachable: {check.get('host')}, python {check.get('python')}, "
            f"parslbox {check.get('parslbox')}"
            if check.get("reachable")
            else f"endpoint did not answer: {check.get('error')}"
        )
    lines += result.get("notes", [])
    lines.append(result["export_line"])
    return "\n".join(lines)


@mcp.tool(
    name="local_status",
    description=(
        "Report on a local project: one whose database stores remote paths so jobs can be "
        "authored here and run on a remote machine.\n\n"
        "Says which side has changed since the last sync, whether the Globus Compute "
        "endpoint answers, and what PBX_DB_PATH resolves to. Read-only. "
        "If the current database is not a local project, says so -- that is not an error."
    ),
)
def local_status(params: LocalStatusSchema) -> str:
    try:
        report = pbx.local_status(check_remote=params.check_remote)
    except Exception as e:
        return f"Exception occurred when reading local project status. Exception: {e}"

    if not report.get("is_local_project"):
        if report.get("misconfigured"):
            return f"Local project misconfigured: {report['message']}"
        return f"Not a local project. PBX_DB_PATH resolves to {report['db_path']}."

    if report.get("error"):
        return (f"local project at {report.get('local_root')}, but its database "
                f"could not be read: {report['error']}")

    remote = report.get("remote", {})
    if remote.get("reachable"):
        endpoint_state = "reachable"
    elif remote.get("checked"):
        endpoint_state = "no answer"
    else:
        endpoint_state = remote.get("error", "not checked")

    last = report.get("last_sync")
    lines = [
        f"local: {report['local_root']} | remote: {report['remote_root']}",
        f"db: {report['db_path']} | {report.get('job_count')} jobs",
        f"endpoint: {report.get('compute_endpoint') or 'none'} ({endpoint_state})",
        f"last sync: {last['at']} ({last.get('direction')})" if last else "last sync: never",
        f"local drift: {report['local']['drift']}",
    ]
    if remote.get("fingerprint"):
        lines.append(f"remote drift: {remote['drift']}")
    elif remote.get("checked") and remote.get("reachable"):
        lines.append("remote drift: no database there yet")
    if report.get("transfer_remote"):
        root = report.get("transfer_remote_root")
        lines.append(f"transfer collection: {report['transfer_remote']} "
                     + (f"rooted at {root}" if root else "root not detected yet"))
    if remote.get("identity_ok") is False:
        lines.append("WARNING: the remote database belongs to a different project.")
    if report.get("moved"):
        lines.append(f"NOTE: project was created at {report['recorded_local_root']} and moved.")
    if report.get("recommendation"):
        lines.append(f"-> {report['recommendation']}")
    lines.append(report["export_line"])
    return "\n".join(lines)


@mcp.tool(
    name="local_push",
    description=(
        "Send a local project's database up to the remote machine.\n\n"
        "Refuses if the remote changed since the last sync -- that is usually a running "
        "allocation's progress, and pushing over it would lose the result. Pull first, or "
        "pass force to overwrite deliberately. Optionally sends job directories too."
    ),
)
def local_push(params: LocalPushSchema) -> str:
    try:
        result = pbx.local_push(
            force=params.force,
            with_dirs=params.with_dirs,
            apps=params.apps,
            tags=params.tags,
            wait_for_dirs=params.wait_for_dirs,
            sync_level=params.sync_level,
        )
    except Exception as e:
        return f"Exception occurred when pushing. Exception: {e}"

    lines = [
        f"Pushed {result['bytes'] / 1024:.0f} KB to {result['remote_db_path']}",
        f"remote now: {result['fingerprint'].get('count')} jobs",
    ]
    dirs = result.get("dirs")
    if dirs:
        statuses = "/".join(dirs.get("statuses") or []) or "any status"
        lines.append(
            f"directories: {len(dirs['present'])} to send "
            f"({dirs.get('matched')} of {dirs.get('total')} jobs are {statuses}"
            f"; {len(dirs['missing'])} missing locally, "
            f"{len(dirs['outside'])} outside the project)")
    if result.get("transfer_remote_root_detected"):
        lines.append(f"collection root detected: "
                     f"{result['transfer_remote_root']} (saved to .pbxlocal.yaml)")
    transfer = result.get("transfer") or {}
    if transfer.get("submitted"):
        lines.append(f"transfer: {transfer['count']} director(ies) in "
                     f"{transfer['batches']} task(s): {', '.join(transfer['task_ids'])}")
        outcome = transfer.get("outcome")
        if outcome:
            note = outcome.get("nice_status")
            lines.append(f"transfer {outcome['status']}, "
                         f"{outcome.get('files_transferred')} files, "
                         f"{(outcome.get('bytes_transferred') or 0) / 1e6:.1f} MB"
                         + (f" ({note})" if note else ""))
        else:
            lines.append("not waiting; watch at https://app.globus.org/activity")
    elif transfer.get("reason"):
        lines.append(f"no directory transfer: {transfer['reason']}")
    return "\n".join(lines)


@mcp.tool(
    name="local_pull",
    description=(
        "Bring a local project's remote database back down over the local one.\n\n"
        "Refuses if the local database changed since the last sync, since those changes "
        "would be lost and there is no merge. Pass force to overwrite deliberately."
    ),
)
def local_pull(params: LocalPullSchema) -> str:
    try:
        result = pbx.local_pull(force=params.force)
    except Exception as e:
        return f"Exception occurred when pulling. Exception: {e}"
    return (f"Pulled {result['bytes'] / 1024:.0f} KB, "
            f"local now: {result['fingerprint'].get('count')} jobs")



def _append_schema_blocks() -> None:
    """Restate each tool's arguments at the end of its description.

    Done by introspection rather than by hand at 16 call sites, so adding a
    field to a schema updates the description with no further edit.
    """
    import inspect
    from pydantic import BaseModel as _BM

    for tool in mcp._tool_manager._tools.values():
        try:
            params = inspect.signature(tool.fn).parameters
        except (TypeError, ValueError):
            continue
        for p in params.values():
            ann = p.annotation
            if isinstance(ann, type) and issubclass(ann, _BM):
                tool.description = (tool.description or "") + schema_text_block(ann)
                break


_append_schema_blocks()


# Start MCP server
app = mcp.streamable_http_app()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ParslBox MCP server")
    parser.add_argument("--stdio", action="store_true", help="Run in stdio mode (the harness launches and manages the process)")
    parser.add_argument("--port", type=int, default=9795, help="Port for HTTP mode (default: 9795)")
    args = parser.parse_args()

    if args.stdio:
        # stdio mode: the harness launches and manages the process
        mcp.run(transport="stdio")
    else:
        # HTTP mode: run as a standalone server
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=args.port)
