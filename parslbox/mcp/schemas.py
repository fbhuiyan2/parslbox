from pathlib import Path

from pydantic import BaseModel, Field
from typing import List, Optional


class AddJobSchema(BaseModel):
    """Schema for adding job(s)"""

    paths: List[str] = Field(
        description="One or more paths to job directories, or 'all' to add all subdirectories in the current location.",
    )
    app: str = Field(
        description="The application type. Options are 'lammps', 'vasp' and 'python'.",
    )
    config: str = Field(
        description="The name of the configuration to use. Options are 'crux', 'polaris', 'sophia', 'aurora_gpu' and 'aurora_tile'",
    )
    tag: str = Field(
        default="test",
        description="An optional tag to categorize the job(s).",
    )
    input_file: Optional[str] = Field(
        default=None,
        description="Input filename for the job(s).",
    )
    ngpus: int = Field(
        default=0,
        description="Number of GPUs required for the job(s).",
    )
    nnodes: int = Field(
        default=1,
        description="Number of nodes required for the job(s).",
    )
    node_occupancy: Optional[float] = Field(
        default=None,
        description="Node occupancy fraction for CPU-only jobs (0.0-1.0).",
        ge=0.0,
        le=1.0,
    )
    ranks_per_node: Optional[int] = Field(
        default=None,
        description=(
            "Number of MPI ranks per node. For CPU jobs only; ignored for GPU jobs. "
            "If not specified, defaults to cores_per_node * node_occupancy."
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
        description="Initial status for the job(s).",
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
        description="Filter jobs by tag.",
    )
    path: Optional[str] = Field(
        default=None,
        description="Filter jobs by path (partial match).",
    )
    in_file: Optional[str] = Field(
        default=None,
        description="Filter jobs by input file (partial match).",
    )


class ListJobsSchema(BaseModel):
    """Schema for listing jobs with optional filters and limits."""

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
        description="Filter jobs by tag.",
    )
    all_jobs: bool = Field(
        default=False,
        description="Show all jobs regardless of count.",
    )
    n: Optional[int] = Field(
        default=None,
        description="Number of jobs to show. Negative for last N jobs, 0 for all.",
    )


class QSubSchema(BaseModel):
    """Schema for generating and submitting a PBS job via ParslBox."""

    config: str = Field(
        description="The name of the configuration to use. Options are 'crux', 'polaris', 'sophia', 'aurora_gpu' and 'aurora_tile'",
    )

    job_name: str = Field(
        description="PBS job name.",
    )

    queue: str = Field(
        description="PBS queue name.",
    )

    select: int = Field(
        ge=1,
        description="Number of nodes to request (must be >= 1).",
    )

    walltime: int = Field(
        gt=0,
        description="Wall time in minutes (e.g., 90 for 1.5 hours).",
    )

    project: str = Field(
        description="Project/account name.",
    )

    filesystems: Optional[str] = Field(
        default=None,
        description="Comma-separated list of filesystems (e.g., 'home:eagle').",
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
            "List of tags to run. Will be rendered as a comma-separated list in the submit script."
        ),
    )

    retries: int = Field(
        default=0,
        ge=0,
        description="Number of retries for failed tasks (must be >= 0).",
    )


class RemoveJobsSchema(BaseModel):
    """Schema for removing existing job in the database"""

    job_ids: list[int] = Field(
        description="List of job IDs to remove",
    )


class UpdateJobSchema(BaseModel):
    """Schema for updating an existing job in the database."""

    job_id: int = Field(
        description="Job ID to update.",
    )

    status: Optional[str] = Field(
        default=None,
        description="New status for the job.",
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
