import sys

from mcp.server.fastmcp import FastMCP

from parslbox.api import ParslBox
from parslbox.local.project import LocalProjectError
from parslbox.utils.path_utils import PbxPathError
from parslbox.commands.helpers.job_id_parser import format_job_ids, group_failures
from parslbox.mcp.schemas import (
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
        "Available capabilities:\n"
        "- add_jobs: create new jobs and add them to the ParslBox database.\n"
        "- submit_pbs_job: submit PBS jobs to the queue using a qsub-style configuration.\n"
        "- submit_slurm_job: submit SLURM jobs to the queue using an sbatch-style configuration.\n"
        "- cancel_pbs_job: gracefully cancel a running PBS batch job (preferred over raw qdel — keeps DB accurate).\n"
        "- cancel_slurm_job: gracefully cancel a running SLURM batch job (preferred over raw scancel — keeps DB accurate).\n"
        "- remove_jobs: remove one or more jobs from the database.\n"
        "- update_job: modify fields of an existing job (status, tag, input file, resources, dependencies, etc.). Note: app cannot be changed after job creation.\n"
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
    name="add_jobs",
    description="Add jobs to ParslBox",
)
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


def _remote_lines(status: dict) -> list:
    """Extra lines when the submission went to another machine.

    Without these an agent cannot tell a remote submission from a local one:
    the run directory it is shown does not exist here, and a failed post-submit
    pull -- which means the local database is now behind -- is invisible.
    """
    if not status.get("remote"):
        return []
    lines = [f"Submitted on remote Compute endpoint {status.get('endpoint')}.",
             f"Remote run directory: {status.get('remote_run_dir')}"]
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
        "respawn reaches 0. Returns a short message describing whether submission succeeded."
    ),
)
def submit_pbs_job(params: QSubSchema) -> str:
    input_dict = params.model_dump()

    try:
        status = pbx.qsub(**input_dict)
    except Exception as e:
        return f"Exception occurred when submitting job. Exception: {e}"

    if status.get("success") is True:
        job_id = status.get("job_id", status.get("pbs_job_id", "UNKNOWN"))
        run_dir = status.get("run_dir", "UNKNOWN")
        matched = status.get("matched_jobs")
        resolved = status.get("resolved_tags")
        respawn_template = status.get("respawn_template_file")
        lines = [
            f"Job was submitted successfully.",
            f"PBS job ID: {job_id}",
            f"Run directory: {run_dir}",
        ]
        if matched is not None:
            lines.append(f"Matched {matched} runnable job(s) in DB.")
        if resolved:
            lines.append(f"Resolved tags: {', '.join(resolved)}")
        if respawn_template:
            lines.append(f"Respawn template: {respawn_template} (chain auto-resubmits at walltime)")
        lines += _remote_lines(status)
        return "\n".join(lines)
    else:
        error_msg = status.get("error", "Unknown error")
        run_dir = status.get("run_dir", "UNKNOWN")
        return (
            "Failed to submit job.\n"
            f"Error: {error_msg}\n"
            f"Run directory (if created): {run_dir}"
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
        "respawn reaches 0. Returns a short message describing whether submission succeeded."
    ),
)
def submit_slurm_job(params: SBatchSchema) -> str:
    input_dict = params.model_dump()

    try:
        status = pbx.sbatch(**input_dict)
    except Exception as e:
        return f"Exception occurred when submitting SLURM job. Exception: {e}"

    if status.get("success") is True:
        job_id = status.get("job_id", status.get("slurm_job_id", "UNKNOWN"))
        run_dir = status.get("run_dir", "UNKNOWN")
        matched = status.get("matched_jobs")
        resolved = status.get("resolved_tags")
        respawn_template = status.get("respawn_template_file")
        lines = [
            f"SLURM job was submitted successfully.",
            f"SLURM job ID: {job_id}",
            f"Run directory: {run_dir}",
        ]
        if matched is not None:
            lines.append(f"Matched {matched} runnable job(s) in DB.")
        if resolved:
            lines.append(f"Resolved tags: {', '.join(resolved)}")
        if respawn_template:
            lines.append(f"Respawn template: {respawn_template} (chain auto-resubmits at walltime)")
        lines += _remote_lines(status)
        return "\n".join(lines)
    else:
        error_msg = status.get("error", "Unknown error")
        run_dir = status.get("run_dir", "UNKNOWN")
        return (
            "Failed to submit SLURM job.\n"
            f"Error: {error_msg}\n"
            f"Run directory (if created): {run_dir}"
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
    name="remove_job",
    description="Remove jobs from ParslBox",
)
def remove_jobs(params: RemoveJobsSchema) -> str:
    input_dict = params.model_dump()
    try:
        removed_count = pbx.remove_jobs(**input_dict)
    except Exception as e:
        return f"Exception occurred when removing jobs. Exception: {e}"
    
    if removed_count > 0:
        return f"Successfully removed {removed_count} job(s)."
    else:
        return "No jobs were removed (jobs may not exist)."


@mcp.tool(
    name="update_job",
    description=("Update fields for an existing ParslBox job.\n\n"),
)
def update_job(params: UpdateJobSchema) -> str:
    input_dict = params.model_dump()
    
    # Extract job_id and convert to list for the API call
    job_id = input_dict.pop('job_id')
    input_dict['job_ids'] = [job_id]

    try:
        updated_job_ids, failed_jobs, msg_log = pbx.update_jobs(**input_dict)
    except Exception as e:
        return f"Exception occurred when updating job {job_id}. Exception: {e}"

    # Build response message
    response_parts = []
    
    if job_id in updated_job_ids:
        response_parts.append(f"Successfully updated job {job_id}.")
    elif failed_jobs:
        # Find the specific error for this job
        for failed_job_id, error_msg in failed_jobs:
            if failed_job_id == job_id:
                response_parts.append(f"Failed to update job {job_id}: {error_msg}")
                break
        else:
            response_parts.append(f"Job {job_id} was not found or no fields were updated.")
    else:
        response_parts.append(f"Job {job_id} was not found or no fields were updated.")
    
    # Add warnings and info messages
    for warning in msg_log.get('warnings', []):
        response_parts.append(f"Warning: {warning}")
    
    for info in msg_log.get('info', []):
        response_parts.append(f"Info: {info}")
    
    return "\n".join(response_parts)


@mcp.tool(
    name="filter_jobs",
    description=("Filter jobs in ParslBox and return their IDs. "),
)
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
