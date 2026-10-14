from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import List, Optional, Union


class McpSchema(BaseModel):
    """Reject unknown arguments, and say what the valid ones are.

    Pydantic's default silently drops an unrecognised key, so a call with
    `nodes=4` instead of `nnodes=4` stores the default and reports success.
    Rejecting alone is not enough either: "Extra inputs are not permitted"
    names the wrong key without naming the right one, and a caller then
    guesses again. Listing the accepted arguments makes one retry sufficient.
    """

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_arguments(cls, data):
        if isinstance(data, dict):
            unknown = sorted(set(data) - set(cls.model_fields))
            if unknown:
                raise ValueError(
                    f"Unknown argument(s) {unknown}. "
                    f"Valid arguments: {sorted(cls.model_fields)}"
                )
        return data


JobIdTokens = List[Union[int, str]]
"""Job ids as the CLI accepts them: plain ids, ranges, or a mix."""


def _check_job_id_tokens(value: JobIdTokens) -> List[str]:
    """Check ['1-5', 8] for syntax and hand the tokens on unexpanded.

    Ranges spare an agent from transcribing a hundred integers out of a
    previous tool's output, where one corrupted element removes the wrong
    job. Expanding them here would be the wrong place: a validator runs
    before the tool body, so it has no database to size '1-100000000'
    against and would build the hundred million integers before anything
    could object. The tool bodies call `resolve_job_ids` instead.
    """
    from parslbox.commands.helpers.job_id_parser import validate_job_id_tokens

    tokens = [str(v).strip() for v in value]
    validate_job_id_tokens(tokens)
    return tokens


def _check_optional_job_id_tokens(value):
    """As _check_job_id_tokens, but None means "every job" rather than "none"."""
    return None if value is None else _check_job_id_tokens(value)


def schema_text_block(model: type[BaseModel]) -> str:
    """Render a schema's arguments as text, to append to a tool description.

    A tool declared as `def tool(params: XSchema)` reaches the client as a
    single `params` property behind a `$ref` into `$defs`. Not every model
    follows the reference; some invent a container instead and the call is
    malformed before any validation runs. Restating the field names where
    there is no reference to follow costs a few dozen tokens and removes
    that failure. Generated from the model, so it cannot drift.
    """
    js = model.model_json_schema()
    required = set(js.get("required", []))
    lines = []
    for name, spec in js.get("properties", {}).items():
        kind = spec.get("type") or "/".join(
            x.get("type", "?") for x in spec.get("anyOf", [])
            if x.get("type") != "null"
        ) or "any"
        note = ("required" if name in required
                else f"optional, default {spec.get('default')!r}")
        lines.append(f"  {name} ({kind}, {note})")
    if not lines:
        return ""
    return ("\n\nArguments, all inside the single `params` object:\n"
            + "\n".join(lines))


class AddJobSchema(McpSchema):
    """Schema for adding job(s)"""

    paths: List[str] = Field(
        description=(
            "One or more paths to job directories. Use 'all:<dir>' to add every subdirectory of <dir> "
            "(pass an absolute <dir>). Bare 'all' resolves against the MCP server's working directory, "
            "which is usually not the caller's location, so prefer 'all:<dir>'. "
            "In a local project (see local_status), paths under the project root must exist here and "
            "are stored with the remote machine's prefix instead; paths outside it must be absolute "
            "and are stored unchanged, since they already describe the remote filesystem."
        ),
    )
    app: str = Field(
        description="The application type. Built-in options are 'lammps-kk', 'vasp', 'python' and 'julia'. Custom apps may also be available.",
    )
    config: str = Field(
        description="The name of the configuration to use. Options are 'polaris', 'sophia', 'crux', 'aurora-tile', 'aurora-gpu', 'lcrc-swing', 'lcrc-improv', 'pinnacles-cenvalarc', 'perlmutter-gpu', 'perlmutter-cpu' and 'perlmutter-gpu-srun'",
    )
    tag: Optional[str] = Field(
        default=None,
        description="An optional tag to categorize the job(s).",
    )
    input_file: Optional[str] = Field(
        default=None,
        description="Input filename for the job(s).",
    )
    ngpus: int = Field(
        default=0,
        ge=0,
        description="Number of GPUs required for the job(s). Must be non-negative.",
    )
    nnodes: int = Field(
        default=1,
        ge=1,
        description="Number of nodes required for the job(s). Must be at least 1.",
    )
    node_occupancy: Optional[float] = Field(
        default=None,
        description="Node occupancy fraction for CPU-only jobs. Must be between 0.0 (exclusive) and 1.0 (inclusive).",
        gt=0.0,
        le=1.0,
    )
    ranks_per_node: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Number of MPI ranks per node. If not specified, uses the app's default "
            "(e.g., 1 rank per GPU for LAMMPS, 1 rank per node for Python). Must be a positive integer."
        ),
    )
    mpi_opts: Optional[str] = Field(
        default=None,
        description="Additional MPI options to append to the MPI command.",
    )
    env_file: Optional[str] = Field(
        default=None,
        description="Path to environment setup file (relative or absolute).",
    )
    parents: Optional[List[int]] = Field(
        default=None,
        description="List of parent job IDs.",
    )
    parent_tag: Optional[str] = Field(
        default=None,
        description="Wait for all jobs with this tag to complete.",
    )
    status: str = Field(
        default="Ready",
        description="Initial status for the job(s). Valid statuses: Ready, Done, Failed, Killed, Restart, Running, Submitted, Resubmitted, Warning.",
    )
    app_args: Optional[str] = Field(
        default=None,
        description="Extra arguments appended to the application command (e.g. '-var T 300'). Requires input_file — the job's input file becomes '<input_file> <app_args>'.",
    )


class FilterJobsSchema(McpSchema):
    """Schema for filtering jobs and returning their IDs."""

    status: Optional[str] = Field(
        default=None,
        description="Filter jobs by status.",
    )
    app: Optional[str] = Field(
        default=None,
        description="Filter jobs by application.",
    )
    tag: Optional[str] = Field(
        default=None,
        description="Filter jobs by tag. Supports `*` glob: '*prod' (suffix), 'run*' (prefix), '*3c*' (substring). Plain strings match exactly.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Filter jobs by path (partial match).",
    )
    in_file: Optional[str] = Field(
        default=None,
        description="Filter jobs by input file (partial match).",
    )
    num_nodes: Optional[int] = Field(
        default=None,
        description="Filter jobs by number of nodes required (exact match).",
    )
    exclude_status: Optional[str] = Field(
        default=None,
        description="Drop jobs with this status.",
    )
    exclude_app: Optional[str] = Field(
        default=None,
        description="Drop jobs with this app.",
    )
    exclude_tag: Optional[str] = Field(
        default=None,
        description="Drop jobs with this tag. Supports `*` glob (same syntax as the include `tag` filter).",
    )


class ListJobsSchema(McpSchema):
    """Schema for listing jobs with full details and optional filters."""

    status: Optional[str] = Field(
        default=None,
        description="Filter jobs by status.",
    )
    app: Optional[str] = Field(
        default=None,
        description="Filter jobs by app.",
    )
    tag: Optional[str] = Field(
        default=None,
        description="Filter jobs by tag. Supports `*` glob: '*prod' (suffix), 'run*' (prefix), '*3c*' (substring). Plain strings match exactly.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Filter jobs by path (partial match).",
    )
    in_file: Optional[str] = Field(
        default=None,
        description="Filter jobs by input file (partial match).",
    )
    num_nodes: Optional[int] = Field(
        default=None,
        description="Filter jobs by number of nodes required (exact match).",
    )


    limit: Optional[int] = Field(
        default=None,
        description=(
            "Return only the first N jobs, or the last N for a negative N. "
            "Omit for every match. A large database answers with one line per "
            "job, so set this unless you need the whole list."
        ),
    )


class GetJobSchema(McpSchema):
    """Schema for getting a single job's full details by ID."""

    job_id: int = Field(
        description="Job ID to retrieve.",
    )


class GetJobsByIdsSchema(McpSchema):
    """Schema for getting multiple jobs' full details by their IDs."""

    job_ids: List[int] = Field(
        description="List of job IDs to retrieve.",
    )


class ResourceRequirementsSchema(McpSchema):
    """Schema for sizing jobs against a target system."""

    system: str = Field(
        description=(
            "Target system. Options are 'polaris', 'sophia', 'crux', "
            "'aurora-tile', 'aurora-gpu', 'lcrc-swing', 'lcrc-improv', "
            "'pinnacles-cenvalarc', 'perlmutter-gpu', 'perlmutter-cpu' and "
            "'perlmutter-gpu-srun'."
        ),
    )

    job_ids: Optional[JobIdTokens] = Field(
        default=None,
        description=(
            "Jobs to size. Accepts plain ids and ranges in any mix: "
            "[1, 2, 3], ['1-50'], or [1, '5-10']; a range may not reach past "
            "the last job in the database. Omit to size every job, which is "
            "usually the question worth asking."
        ),
    )

    _check = field_validator("job_ids")(_check_optional_job_id_tokens)


class CommandLinesSchema(McpSchema):
    """Schema for previewing the launch command of jobs on a target system."""

    system: str = Field(
        description=(
            "Target system, e.g. 'aurora-tile' or 'polaris'. Same set as "
            "get_resource_requirements."
        ),
    )

    job_ids: Optional[JobIdTokens] = Field(
        default=None,
        description=(
            "Jobs to preview. Accepts plain ids and ranges in any mix; a "
            "range may not reach past the last job in the database. Omit "
            "for every job, though one line comes back per job."
        ),
    )

    _check = field_validator("job_ids")(_check_optional_job_id_tokens)


class QSubSchema(McpSchema):
    """Schema for generating and submitting a PBS job via ParslBox."""

    config: str = Field(
        description="The name of the configuration to use. Options are 'polaris', 'sophia', 'crux', 'aurora-tile', 'aurora-gpu', 'lcrc-swing', 'lcrc-improv', 'pinnacles-cenvalarc', 'perlmutter-gpu', 'perlmutter-cpu' and 'perlmutter-gpu-srun'",
    )

    job_name: str = Field(
        description="PBS job name.",
    )

    queue: str = Field(
        description="PBS queue name.",
    )

    select: str = Field(
        description="PBS select specification. Can be a simple node count (e.g., '4') or a complex spec (e.g., '2:ncpus=32:ngpus=4').",
    )

    walltime: Union[int, float, str] = Field(
        description="Wall time (default: minutes). Supports h/d suffixes (e.g., 90, 4.25h, 3.5d).",
    )

    project: Optional[str] = Field(
        default=None,
        description="Project/account name. Optional — omit if your cluster does not require one.",
    )

    rundir: Optional[Path] = Field(
        default=None,
        description="Custom run directory (default: timestamped directory under ~/.parslbox/runs).",
    )

    apps: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of apps to run. Will be rendered as a comma-separated list in the submit script."
        ),
    )

    tags: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of tags to run. Each entry may be a literal tag or a `*` glob "
            "('*prod', 'run*', '*3c*'). Globs are resolved against the DB before "
            "submission; every entry must match at least one existing tag or the "
            "call will fail. The final literal list is rendered into the submit script."
        ),
    )

    retries: int = Field(
        default=0,
        ge=0,
        description="Number of retries for failed tasks (must be >= 0).",
    )

    loglevel: str = Field(
        default="info",
        description="Logging level for the orchestrator (e.g. 'debug', 'info', 'warning').",
    )

    dynamic: bool = Field(
        default=True,
        description=(
            "Keep discovering jobs added to the database after the run starts. "
            "Set false to pin the run to the jobs matched at submit time."
        ),
    )

    sched_opts: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of extra PBS scheduler directives (e.g., ['#PBS -l filesystems=home:eagle', '#PBS -l place=scatter']). "
            "Directives matching a template key override it; new directives are appended."
        ),
    )

    respawn: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Enable the self-respawn chain. The integer is the number of remaining "
            "auto-resubmissions; decremented each link. At walltime the orchestrator "
            "marks in-flight jobs Restart and auto-submits the next link, continuing "
            "until all jobs reach Done/Failed or respawn reaches 0. Pass None "
            "(default) to disable the chain entirely."
        ),
    )

    no_local: bool = Field(
        default=False,
        description=(
            "Submit from this machine even when the database belongs to a local project. "
            "By default, in a local project the database is pushed to the remote machine, "
            "the submission runs there, and the result is pulled back."
        ),
    )

    push_dirs: bool = Field(
        default=False,
        description=(
            "In a local project, also send the matching jobs' directories to the remote "
            "machine over Globus Transfer before submitting. Requires Transfer collections "
            "in .pbxlocal.yaml. Off by default: the directories are usually already there."
        ),
    )


class SBatchSchema(McpSchema):
    """Schema for generating and submitting a SLURM job via ParslBox."""

    config: str = Field(
        description="The name of the configuration to use.",
    )

    job_name: str = Field(
        description="SLURM job name.",
    )

    queue: str = Field(
        description="SLURM partition name.",
    )

    select: str = Field(
        description="Number of nodes to request (e.g., '4').",
    )

    walltime: Union[int, float, str] = Field(
        description="Wall time (default: minutes). Supports h/d suffixes (e.g., 90, 4.25h, 3.5d).",
    )

    project: Optional[str] = Field(
        default=None,
        description="Project/account name. Optional — omit if your cluster does not require one.",
    )

    rundir: Optional[Path] = Field(
        default=None,
        description="Custom run directory (default: timestamped directory under ~/.parslbox/runs).",
    )

    apps: Optional[List[str]] = Field(
        default=None,
        description="List of apps to run.",
    )

    tags: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of tags to run. Each entry may be a literal tag or a `*` glob "
            "('*prod', 'run*', '*3c*'). Globs are resolved against the DB before "
            "submission; every entry must match at least one existing tag or the "
            "call will fail. The final literal list is rendered into the submit script."
        ),
    )

    retries: int = Field(
        default=0,
        ge=0,
        description="Number of retries for failed tasks (must be >= 0).",
    )

    loglevel: str = Field(
        default="info",
        description="Logging level for the orchestrator (e.g. 'debug', 'info', 'warning').",
    )

    dynamic: bool = Field(
        default=True,
        description=(
            "Keep discovering jobs added to the database after the run starts. "
            "Set false to pin the run to the jobs matched at submit time."
        ),
    )

    sched_opts: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of extra SLURM scheduler directives (e.g., ['#SBATCH --mem=64G', '#SBATCH -C gpu&hbm80g']). "
            "Directives matching a template key override it; new directives are appended."
        ),
    )

    respawn: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Enable the self-respawn chain. The integer is the number of remaining "
            "auto-resubmissions; decremented each link. At walltime the orchestrator "
            "marks in-flight jobs Restart and auto-submits the next link, continuing "
            "until all jobs reach Done/Failed or respawn reaches 0. Pass None "
            "(default) to disable the chain entirely."
        ),
    )

    no_local: bool = Field(
        default=False,
        description=(
            "Submit from this machine even when the database belongs to a local project. "
            "By default, in a local project the database is pushed to the remote machine, "
            "the submission runs there, and the result is pulled back."
        ),
    )

    push_dirs: bool = Field(
        default=False,
        description=(
            "In a local project, also send the matching jobs' directories to the remote "
            "machine over Globus Transfer before submitting. Requires Transfer collections "
            "in .pbxlocal.yaml. Off by default: the directories are usually already there."
        ),
    )


class CancelJobSchema(McpSchema):
    """Schema for gracefully cancelling a running ParslBox batch job."""

    jobid: str = Field(
        description="Scheduler job ID (PBS or SLURM) to cancel.",
    )

    grace: int = Field(
        default=30,
        ge=0,
        description=(
            "Seconds between SIGTERM (sent via qsig / scancel --signal=TERM) "
            "and the hard kill. The grace period lets the running `pbx run` "
            "orchestrator reconcile its in-flight jobs in the database "
            "(Running→Killed, Submitted→Ready, Resubmitted→Restart) before "
            "SIGKILL. Default: 30."
        ),
    )


class RemoveJobsSchema(McpSchema):
    """Schema for removing existing job in the database"""

    job_ids: JobIdTokens = Field(
        description=(
            "Job IDs to remove. Accepts plain ids and ranges in any mix: "
            "[1, 2, 3], ['1-5'], or [1, '5-10', 12]. Use a range rather than "
            "listing every id. All or nothing: if any id does not exist "
            "nothing is removed, and a range may not reach past the last job "
            "in the database."
        ),
    )

    _check = field_validator("job_ids")(_check_job_id_tokens)


class UpdateJobSchema(McpSchema):
    """Schema for updating an existing job in the database."""

    job_ids: JobIdTokens = Field(
        description=(
            "Job IDs to update, all receiving the same changes. Accepts plain "
            "ids and ranges in any mix: [1, 2, 3], ['1-5'], or [1, '5-10', 12]. "
            "A range may not reach past the last job in the database."
        ),
    )

    _check = field_validator("job_ids")(_check_job_id_tokens)

    status: Optional[str] = Field(
        default=None,
        description="New status for the job. Valid statuses: Ready, Done, Failed, Killed, Restart, Running, Submitted, Resubmitted, Warning.",
    )

    tag: Optional[str] = Field(
        default=None,
        description="New tag for the job.",
    )

    input_file: Optional[str] = Field(
        default=None,
        description="New input file path for the job.",
    )

    ngpus: Optional[int] = Field(
        default=None,
        ge=0,
        description="New number of GPUs. Must be non-negative.",
    )

    env_file: Optional[str] = Field(
        default=None,
        description="New environment file path (will be resolved to an absolute path).",
    )

    nnodes: Optional[int] = Field(
        default=None,
        ge=1,
        description="New number of nodes. Must be at least 1.",
    )

    node_occupancy: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="New node occupancy. Must be between 0.0 (exclusive) and 1.0 (inclusive).",
    )

    ranks_per_node: Optional[int] = Field(
        default=None,
        ge=1,
        description="New ranks per node. Must be a positive integer.",
    )

    add_deps: Optional[List[int]] = Field(
        default=None,
        description="Parent job IDs to add as dependencies.",
    )

    rm_deps: Optional[List[int]] = Field(
        default=None,
        description="Parent job IDs to remove from dependencies.",
    )

    app_args: Optional[str] = Field(
        default=None,
        description="Extra arguments appended to the application command (e.g. '-var T 300'). Rebuilds the job's input file as '<base script> <app_args>', replacing any args already set. Combine with input_file to change the base script at the same time.",
    )


class LocalStatusSchema(McpSchema):
    """Schema for reporting on a local project."""

    check_remote: bool = Field(
        default=True,
        description=(
            "Ping the Globus Compute endpoint and fingerprint the remote database. "
            "This can take up to 120 seconds if the endpoint is down. Set False "
            "to report only what is known locally, which needs no network."
        ),
    )


class LocalPushSchema(McpSchema):
    """Schema for sending a local project's database to the remote machine."""

    force: bool = Field(
        default=False,
        description=(
            "Overwrite the remote database even though it changed since the last sync. "
            "Refused by default: the change is usually a running allocation's progress, "
            "and there is no merge. Pull first unless you mean to discard it."
        ),
    )

    with_dirs: bool = Field(
        default=False,
        description=(
            "Also send the matching jobs' directories over Globus Transfer. "
            "Requires transfer_local and transfer_remote in .pbxlocal.yaml, and "
            "the PBX_GLOBUS_CLIENT_ID environment variable set to a registered "
            "Globus Native App client id -- without it the push fails at login. "
            "The database itself syncs over the Compute endpoint and needs "
            "none of this."
        ),
    )

    apps: Optional[List[str]] = Field(
        default=None,
        description="Only send directories for jobs with these apps. Used with with_dirs.",
    )

    tags: Optional[List[str]] = Field(
        default=None,
        description=(
            "Only send directories for jobs with these tags. Accepts '*' globs, "
            "like --tags elsewhere in pbx. A tag (or glob) matching nothing in "
            "the database is an error, not an empty push. Used with with_dirs."
        ),
    )

    sync_level: str = Field(
        default="checksum",
        description=(
            "How Globus decides a file is already at the destination: 'exists', "
            "'size', 'mtime' or 'checksum'. checksum is safest but reads every file "
            "on both ends; drop to 'mtime' when the directories hold large output "
            "files the remote wrote itself."
        ),
    )

    wait_for_dirs: bool = Field(
        default=True,
        description=(
            "Block until the directory transfer finishes before returning. Used with "
            "with_dirs. Set False for a large transfer that would outlast the MCP "
            "call timeout; the task keeps running and can be watched in the Globus "
            "web app."
        ),
    )


class LocalPullSchema(McpSchema):
    """Schema for bringing a local project's remote database back down."""

    force: bool = Field(
        default=False,
        description=(
            "Overwrite the local database even though it changed since the last sync. "
            "Refused by default; the local changes would be lost, and there is no merge."
        ),
    )


class LocalInitSchema(McpSchema):
    """Schema for creating a local project."""

    directory: str = Field(
        description=(
            "Absolute path to the directory that becomes the local project root. "
            "Give an absolute path: the MCP server's working directory is usually "
            "not the caller's."
        ),
    )

    remote_root: str = Field(
        description=(
            "Absolute path to the project root on the remote machine. Every job path "
            "stored in this project's database is written under this root."
        ),
    )

    compute_endpoint: Optional[str] = Field(
        default=None,
        description="Globus Compute endpoint UUID running on the remote machine.",
    )

    transfer_local: Optional[str] = Field(
        default=None,
        description="Globus Transfer collection UUID for this machine. Only needed to move job directories.",
    )

    transfer_remote: Optional[str] = Field(
        default=None,
        description="Globus Transfer collection UUID for the remote machine. Only needed to move job directories.",
    )
    transfer_remote_root: Optional[str] = Field(
        default=None,
        description=(
            "Override for the path that the remote Transfer collection exposes "
            "as its own root, e.g. '/lus/flare/projects'. Normally leave unset: "
            "the first directory push finds it by listing the collection and "
            "matching the database it just wrote there, then saves the answer. "
            "Only set it by hand if that detection cannot see the collection."
        ),
    )

    remote_config: Optional[str] = Field(
        default=None,
        description="Path to config.yaml on the remote machine. Defaults to its own default location.",
    )

    nested_ok: bool = Field(
        default=False,
        description=(
            "Allow creating this project inside an existing one. False by default: "
            "nesting works, since PBX_DB_PATH decides which database is in use, but "
            "it is far more often an accident than an intention."
        ),
    )

    verify: bool = Field(
        default=False,
        description=(
            "Ping the Compute endpoint once the project exists and report the "
            "result under 'endpoint_check'. Skipped when no endpoint is "
            "configured yet. Can take up to 60 seconds if the endpoint is down."
        ),
    )


class UseDatabaseSchema(McpSchema):
    """Schema for rebinding the server to a different ParslBox database"""

    path: str = Field(
        description=(
            "Absolute path to the project directory, or directly to a .db file. "
            "A directory holding a .pbxlocal.yaml resolves to its "
            "job_database_pbx-local.db; any other directory resolves to "
            "job_database_pbx.db. Relative paths are rejected -- they would resolve "
            "against the server's working directory, not yours."
        ),
    )
    config_path: Optional[str] = Field(
        default=None,
        description=(
            "Absolute path to a config.yaml, or a directory holding one. Omit to keep "
            "the config currently in use, which is the usual case when projects share "
            "one config. Pass it when the project declares its own custom apps or "
            "executables."
        ),
    )
    create: bool = Field(
        default=False,
        description=(
            "Create the database if it does not exist. Default false: a path that is "
            "not already a database is refused, because a typo would otherwise bind "
            "the session to a new empty database that looks like a project with no "
            "jobs. Set true only when deliberately starting a new project here."
        ),
    )
