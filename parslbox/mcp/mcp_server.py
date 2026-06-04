from mcp.server.fastmcp import FastMCP

from parslbox.api import ParslBox
from parslbox.mcp.schemas import (
    AddJobSchema,
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
pbx = ParslBox()


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
        "- get_jobs: get multiple jobs' full details by their IDs.\n\n"
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
        response_parts.append(f"Successfully added {len(successful_job_ids)} job(s) with IDs: {', '.join(map(str, successful_job_ids))}")
    
    if failed_jobs:
        response_parts.append(f"Failed to add {len(failed_jobs)} job(s):")
        for path, error in failed_jobs:
            response_parts.append(f"  - {path}: {error}")
    
    # Add warnings and info messages
    for warning in msg_log.get('warnings', []):
        response_parts.append(f"Warning: {warning}")
    
    for info in msg_log.get('info', []):
        response_parts.append(f"Info: {info}")
    
    if not successful_job_ids and not failed_jobs:
        response_parts.append("No jobs were processed.")
    
    return "\n".join(response_parts)


@mcp.tool(
    name="submit_pbs_job",
    description=(
        "Submit a PBS job via ParslBox.qsub.\n\n"
        "Takes configuration (system config name, job name, queue, select, walltime, project) "
        "plus optional run directory, apps, tags, retries, and sched_opts for extra PBS directives. "
        "Walltime defaults to minutes; supports h/d suffixes (e.g., 90, 4.25h, 3.5d). "
        "Set restart=True with max_restarts=N to enable a self-restart chain: at walltime, "
        "in-flight jobs are marked Restart and a new allocation is auto-submitted, up to N times. "
        "Returns a short message describing whether submission succeeded."
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
        lines = [
            f"Job was submitted successfully.",
            f"PBS job ID: {job_id}",
            f"Run directory: {run_dir}",
        ]
        if matched is not None:
            lines.append(f"Matched {matched} runnable job(s) in DB.")
        if resolved:
            lines.append(f"Resolved tags: {', '.join(resolved)}")
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
        "Set restart=True with max_restarts=N to enable a self-restart chain: at walltime, "
        "in-flight jobs are marked Restart and a new allocation is auto-submitted, up to N times. "
        "Returns a short message describing whether submission succeeded."
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
        lines = [
            f"SLURM job was submitted successfully.",
            f"SLURM job ID: {job_id}",
            f"Run directory: {run_dir}",
        ]
        if matched is not None:
            lines.append(f"Matched {matched} runnable job(s) in DB.")
        if resolved:
            lines.append(f"Resolved tags: {', '.join(resolved)}")
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
        "can mark in-flight jobs as Killed in the database, then runs qdel to terminate. "
        "Always prefer this over a raw qdel for ParslBox jobs — raw qdel only gives the "
        "orchestrator the cluster's default kill grace (often ~2s), which may leave jobs "
        "stuck in 'Running' state in the database."
    ),
)
def cancel_pbs_job(params: CancelJobSchema) -> str:
    input_dict = params.model_dump()
    try:
        result = pbx.qdel(**input_dict)
    except Exception as e:
        return f"Exception occurred when cancelling PBS job. Exception: {e}"

    if result.get("success"):
        return (f"PBS job {result['jobid']} cancelled cleanly "
                f"(grace: {result.get('grace', '?')}s).")
    return (f"Failed to cancel PBS job {result.get('jobid', '?')} "
            f"at stage '{result.get('stage', '?')}': {result.get('error', 'unknown')}")


@mcp.tool(
    name="cancel_slurm_job",
    description=(
        "Gracefully cancel a running ParslBox SLURM batch job.\n\n"
        "Sends SIGTERM to the batch script via `scancel --signal=TERM --batch`, waits "
        "`grace` seconds (default 30) so the orchestrator can mark in-flight jobs as Killed "
        "in the database, then runs scancel to terminate. Always prefer this over a raw "
        "scancel for ParslBox jobs — raw scancel only gives the orchestrator the cluster's "
        "default kill grace, which may leave jobs stuck in 'Running' state in the database."
    ),
)
def cancel_slurm_job(params: CancelJobSchema) -> str:
    input_dict = params.model_dump()
    try:
        result = pbx.scancel(**input_dict)
    except Exception as e:
        return f"Exception occurred when cancelling SLURM job. Exception: {e}"

    if result.get("success"):
        return (f"SLURM job {result['jobid']} cancelled cleanly "
                f"(grace: {result.get('grace', '?')}s).")
    return (f"Failed to cancel SLURM job {result.get('jobid', '?')} "
            f"at stage '{result.get('stage', '?')}': {result.get('error', 'unknown')}")


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


# Start MCP server
app = mcp.streamable_http_app()

if __name__ == "__main__":
    import sys

    if "--stdio" in sys.argv:
        # stdio mode: Claude Code launches and manages the process
        mcp.run(transport="stdio")
    else:
        # HTTP mode: run as a standalone server
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=9795)
