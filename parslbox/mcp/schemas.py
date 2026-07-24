from pathlib import Path

from pydantic import BaseModel, Field
from typing import List, Optional, Union


class AddJobSchema(BaseModel):
    """Schema for adding job(s)"""

    paths: List[str] = Field(
        description=(
            "One or more paths to job directories. Use 'all:<dir>' to add every subdirectory of <dir> "
            "(pass an absolute <dir>). Bare 'all' resolves against the MCP server's working directory, "
            "which is usually not the caller's location, so prefer 'all:<dir>'."
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


class FilterJobsSchema(BaseModel):
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


class ListJobsSchema(BaseModel):
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


class GetJobSchema(BaseModel):
    """Schema for getting a single job's full details by ID."""

    job_id: int = Field(
        description="Job ID to retrieve.",
    )


class GetJobsByIdsSchema(BaseModel):
    """Schema for getting multiple jobs' full details by their IDs."""

    job_ids: List[int] = Field(
        description="List of job IDs to retrieve.",
    )


class QSubSchema(BaseModel):
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

    run_dir: Optional[Path] = Field(
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


class SBatchSchema(BaseModel):
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

    run_dir: Optional[Path] = Field(
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


class CancelJobSchema(BaseModel):
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


class RemoveJobsSchema(BaseModel):
    """Schema for removing existing job in the database"""

    job_ids: List[int] = Field(
        description="List of job IDs to remove",
    )


class UpdateJobSchema(BaseModel):
    """Schema for updating an existing job in the database."""

    job_id: int = Field(
        description="Job ID to update.",
    )

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
