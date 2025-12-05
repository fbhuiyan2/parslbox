import uvicorn

from mcp.server.fastmcp import FastMCP

from parslbox.api import ParslBox
from parslbox.mcp.schemas import (
    AddJobSchema,
    FilterJobsSchema,
    QSubSchema,
    RemoveJobsSchema,
    UpdateJobSchema,
)

# Start ParslBox and MCP.
pbx = ParslBox()

mcp = FastMCP(
    name="ParslBox Tool MCP",
    instructions=(
        "You expose tools for managing and running ParslBox jobs.\n\n"
        "Available capabilities:\n"
        "- add_jobs: create new jobs and add them to the ParslBox database.\n"
        "- submit_job: submit PBS jobs to the queue using a qsub-style configuration.\n"
        "- remove_jobs: remove one or more jobs from the database.\n"
        "- update_job: modify fields of an existing job (status, app, tag, resources, dependencies, etc.).\n"
        "- filter_jobs: filter jobs by status, app, tag, path, or input file and return their IDs.\n\n"
    ),
)


@mcp.tool(
    name="add_jobs",
    description="Add jobs to ParslBox",
)
def add_jobs(params: AddJobSchema):
    input_dict = params.model_dump()
    try:
        job_id = pbx.add_jobs(**input_dict)
    except Exception as e:
        message = f"Exception occured when submitting job. Exception: {e}"
    message = f"JobID {job_id} added to database"
    return message


@mcp.tool(
    name="submit_job",
    description=(
        "Submit a PBS job via ParslBox.qsub.\n\n"
        "Takes configuration (system config name, job name, queue, select, walltime, project) "
        "plus optional filesystems, run directory, apps, tags, and retries. "
        "Returns a short message describing whether submission succeeded."
    ),
)
def submit_job(params: QSubSchema) -> str:
    input_dict = params.model_dump()

    try:
        status = pbx.qsub(**input_dict)
    except Exception as e:
        return f"Exception occurred when submitting job. Exception: {e}"

    if status.get("success") is True:
        job_id = status.get("pbs_job_id", "UNKNOWN")
        run_dir = status.get("run_dir", "UNKNOWN")
        return (
            f"Job was submitted successfully.\n"
            f"PBS job ID: {job_id}\n"
            f"Run directory: {run_dir}"
        )
    else:
        error_msg = status.get("error", "Unknown error")
        run_dir = status.get("run_dir", "UNKNOWN")
        return (
            "Failed to submit job.\n"
            f"Error: {error_msg}\n"
            f"Run directory (if created): {run_dir}"
        )


@mcp.tool(
    name="remove_job",
    description="Remove a job from ParslBox",
)
def remove_jobs(params: RemoveJobsSchema):
    input_dict = params.model_dump()
    try:
        removed_count = pbx.remove_jobs(**input_dict)
    except Exception as e:
        return f"Exception occured when removing job. Exception: {e}"
    return f"Removed {removed_count} job(s)."


@mcp.tool(
    name="update_job",
    description=("Update fields for an existing ParslBox job.\n\n"),
)
def update_job(params: UpdateJobSchema) -> str:
    input_dict = params.model_dump()

    try:
        success = pbx.update_job(**input_dict)
    except Exception as e:
        return f"Exception occurred when updating job {params.job_id}. Exception: {e}"

    if success:
        return f"Successfully updated job {params.job_id}."
    else:
        return f"Job {params.job_id} was not found or no fields were updated."


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


# Start MCP server
app = mcp.streamable_http_app()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9005)
