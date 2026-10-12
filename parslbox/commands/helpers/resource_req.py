"""Resource-requirement and command-line analysis, with no printing in it.

The arithmetic used to sit inside `info.py`'s console.print calls, so only
the CLI could reach it: the API had no equivalent and an MCP caller could not
compute it either, because the system constants are not exposed over MCP.
These functions return plain dicts. `info.py` renders them with Rich, and the
API and MCP layers hand them back as data.
"""
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from parslbox.system_configs.loader import get_available_systems, get_system_config

SCHEDULABLE_STATUSES = ("Ready", "Restart")


class UnknownSystemError(ValueError):
    """Named system is not one parslbox knows about."""

    def __init__(self, system_name: str):
        self.system_name = system_name
        self.available = get_available_systems()
        super().__init__(
            f"Unknown system '{system_name}'. Available: {', '.join(self.available)}"
        )


def _require_system(system_name: str):
    if system_name not in get_available_systems():
        raise UnknownSystemError(system_name)
    return get_system_config(system_name)


def _gpu_nodes(gpu_jobs: List[dict], gpus_per_node: int) -> tuple:
    """(simultaneous, optimal) node counts for GPU jobs."""
    if gpus_per_node == 0:
        return (0, 0)

    multinode_nodes = sum(j['num_nodes'] for j in gpu_jobs if j['num_nodes'] > 1)
    singlenode_gpus = sum(j['ngpus'] for j in gpu_jobs
                          if j['num_nodes'] == 1 and j['ngpus'] <= gpus_per_node)

    sim = multinode_nodes + math.ceil(singlenode_gpus / gpus_per_node) if singlenode_gpus else multinode_nodes
    opt = multinode_nodes + (singlenode_gpus // gpus_per_node) if singlenode_gpus else multinode_nodes

    if singlenode_gpus % gpus_per_node > 0 and opt == multinode_nodes:
        opt += 1

    return (sim, opt)


def _cpu_nodes(cpu_jobs: List[dict]) -> tuple:
    """(simultaneous, optimal) node counts for CPU jobs.

    Equal by construction: occupancy is already the tightest packing.
    """
    total_occupancy = sum(j['node_occupancy'] for j in cpu_jobs if j['num_nodes'] == 1)
    multinode_nodes = sum(j['num_nodes'] for j in cpu_jobs if j['num_nodes'] > 1)
    nodes = multinode_nodes + math.ceil(total_occupancy)
    return (nodes, nodes)


def compute_requirements(jobs: List[dict], system_name: str) -> Dict[str, Any]:
    """How many nodes of `system_name` these jobs need.

    Only Ready/Restart jobs count; the rest are reported under 'skipped' so a
    caller can say why a total looks small.

    Raises:
        UnknownSystemError: `system_name` is not a configured system.
    """
    sys_config = _require_system(system_name)
    gpus_per_node = sys_config.GPUS_PER_NODE
    cores_per_node = sys_config.CORES_PER_NODE

    schedulable = [j for j in jobs if j['status'] in SCHEDULABLE_STATUSES]
    skipped: Dict[str, int] = {}
    for j in jobs:
        if j['status'] not in SCHEDULABLE_STATUSES:
            skipped[j['status']] = skipped.get(j['status'], 0) + 1

    out: Dict[str, Any] = {
        "system": system_name,
        "gpus_per_node": gpus_per_node,
        "cores_per_node": cores_per_node,
        "schedulable": len(schedulable),
        "skipped": skipped,
        "gpu": None,
        "cpu": None,
        "total": None,
    }
    if not schedulable:
        return out

    gpu_jobs = [j for j in schedulable if j['ngpus'] > 0]
    cpu_jobs = [j for j in schedulable if j['ngpus'] == 0]

    if gpu_jobs:
        total_gpus = sum(j['ngpus'] for j in gpu_jobs)
        oversized = [{"job_id": j['job_id'], "ngpus": j['ngpus']} for j in gpu_jobs
                     if j['num_nodes'] == 1 and j['ngpus'] > gpus_per_node]
        block: Dict[str, Any] = {
            "jobs": len(gpu_jobs),
            "total_gpus": total_gpus,
            "gpus_per_node": gpus_per_node,
            "system_has_gpus": gpus_per_node > 0,
            "oversized": oversized,
        }
        if gpus_per_node > 0:
            multinode = [j for j in gpu_jobs if j['num_nodes'] > 1]
            sim, opt = _gpu_nodes(gpu_jobs, gpus_per_node)
            block.update({
                "multinode_jobs": len(multinode),
                "multinode_nodes": sum(j['num_nodes'] for j in multinode),
                "simultaneous": sim,
                "optimal": opt,
                "idle_slots": max(0, sim * gpus_per_node - total_gpus),
                "queued_gpus": (total_gpus - opt * gpus_per_node) if opt < sim else 0,
                "min_viable_nodes": max((j['num_nodes'] for j in gpu_jobs), default=1),
            })
        out["gpu"] = block

    if cpu_jobs:
        singlenode_occupancy = sum(j['node_occupancy'] for j in cpu_jobs
                                   if j['num_nodes'] == 1)
        multinode = [j for j in cpu_jobs if j['num_nodes'] > 1]
        sim, opt = _cpu_nodes(cpu_jobs)
        full = int(singlenode_occupancy)
        out["cpu"] = {
            "jobs": len(cpu_jobs),
            "total_occupancy": singlenode_occupancy,
            "multinode_nodes": sum(j['num_nodes'] for j in multinode),
            "simultaneous": sim,
            "optimal": opt,
            "full_nodes": full,
            "fraction": singlenode_occupancy - full,
            "gpus_would_idle": gpus_per_node > 0,
        }

    if gpu_jobs and cpu_jobs and gpus_per_node > 0:
        gpu_sim, gpu_opt = _gpu_nodes(gpu_jobs, gpus_per_node)
        cpu_sim, cpu_opt = _cpu_nodes(cpu_jobs)
        out["total"] = {"simultaneous": gpu_sim + cpu_sim,
                        "optimal": gpu_opt + cpu_opt}
    elif gpu_jobs and cpu_jobs:
        cpu_sim, cpu_opt = _cpu_nodes(cpu_jobs)
        out["total"] = {"simultaneous": cpu_sim, "optimal": cpu_opt}

    return out


def _synthetic_assignment(job_spec, gpus_per_node: int, effective_cores: int):
    """A ResourceAssignment with placeholder hostnames, for previewing only."""
    from parslbox.resource_manager.models import ResourceAssignment

    num_nodes = job_spec.num_nodes
    ranks_per_node = job_spec.ranks_per_node
    cores_per_rank = effective_cores // ranks_per_node if ranks_per_node > 0 else effective_cores

    if job_spec.is_gpu_job():
        gpus_on_node = gpus_per_node if num_nodes > 1 else job_spec.ngpus
        gpus_per_rank = max(1, gpus_on_node // ranks_per_node)
    else:
        gpus_per_rank = 0

    gpu_assignments, cpu_assignments = [], []
    for _ in range(num_nodes):
        node_gpus, node_cpus = {}, {}
        for rank in range(ranks_per_node):
            if job_spec.is_gpu_job():
                node_gpus[rank] = list(range(rank * gpus_per_rank, (rank + 1) * gpus_per_rank))
            start_core = rank * cores_per_rank
            node_cpus[rank] = list(range(start_core, start_core + cores_per_rank))
        gpu_assignments.append(node_gpus)
        cpu_assignments.append(node_cpus)

    return ResourceAssignment(
        job_id=job_spec.job_id,
        node_ids=[f"node-{i}" for i in range(num_nodes)],
        hostnames=[f"host-{i}" for i in range(num_nodes)],
        gpu_assignments=gpu_assignments,
        cpu_assignments=cpu_assignments,
        node_occupancy=job_spec.node_occupancy,
    )


def build_command_lines(jobs: List[dict], system_name: str,
                        config_path: Optional[Path] = None) -> Dict[str, Any]:
    """The MPI/srun line each job would be launched with on `system_name`.

    Nothing is run and nothing is scheduled; hostnames are placeholders. Any
    rankfiles or wrapper scripts the command refers to are generated into a
    temporary directory that is removed before returning, so the paths in the
    returned commands do not outlive the call.

    `config_path` picks the config.yaml whose per-app MPI settings are used.
    Omit it only from a single-shot CLI process: a server rebound by
    use_database must pass its own, or it previews another project's settings.

    Raises:
        UnknownSystemError: `system_name` is not a configured system.
    """
    from parslbox.resource_manager.mpi_command_builder import MPICommandBuilder
    from parslbox.resource_manager.mpi_config import load_mpi_config
    from parslbox.resource_manager.models import create_job_resource_spec
    from parslbox.utils.pbx_config_utils import load_full_config

    sys_config = _require_system(system_name)
    gpus_per_node = sys_config.GPUS_PER_NODE
    excluded = getattr(sys_config, 'EXCLUDE_CORES', None) or []
    effective_cores = sys_config.CORES_PER_NODE - len(excluded)

    full_config = load_full_config(config_path)
    mpi_configs: Dict[str, Any] = {}
    rows = []

    tmp_dir = tempfile.mkdtemp(prefix="pbx_cmdline_preview_")
    try:
        for job in jobs:
            job_spec = create_job_resource_spec(job)
            app_name = job['app']

            if app_name not in mpi_configs:
                try:
                    mpi_configs[app_name] = load_mpi_config(
                        system_name=system_name, app_name=app_name,
                        system_config_class=sys_config, yaml_config=full_config,
                    )
                except Exception as e:
                    mpi_configs[app_name] = None
                    rows.append({"job_id": job['job_id'], "app": app_name,
                                 "resources": None, "command": None,
                                 "error": f"Error loading MPI config for app '{app_name}': {e}"})
                    continue

            mpi_cfg = mpi_configs[app_name]
            if mpi_cfg is None:
                rows.append({"job_id": job['job_id'], "app": app_name,
                             "resources": None, "command": None,
                             "error": f"MPI config unavailable for app '{app_name}'"})
                continue

            if job_spec.ngpus > 0:
                res_str = f"{job_spec.num_nodes}N/{job_spec.ngpus}G"
            elif job_spec.node_occupancy < 1.0:
                res_str = f"{job_spec.num_nodes}N/{job_spec.node_occupancy}occ"
            else:
                res_str = f"{job_spec.num_nodes}N/CPU"

            assignment = _synthetic_assignment(job_spec, gpus_per_node, effective_cores)
            builder = MPICommandBuilder(mpi_cfg, sys_config)
            try:
                command, error = builder.build_command(assignment, job_spec,
                                                       job_path=tmp_dir), None
            except Exception as e:
                command, error = None, str(e)

            rows.append({"job_id": job['job_id'], "app": app_name,
                         "resources": res_str, "command": command, "error": error})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {"system": system_name, "jobs": rows}
