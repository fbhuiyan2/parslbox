"""
Resource estimation helpers shared by `pbx info --req` and the restart-mode
orchestrator's per-link resubmission step.

The logic of "what's the smallest node count that can run this set of jobs"
already lived inside `info.py`'s display routines. Extracted here as
pure-data functions so the orchestrator can reuse it without dragging in
Rich printing.
"""

import math
from typing import List, Dict, Any


def _gpu_optimal_nodes(gpu_jobs: List[Dict[str, Any]], gpus_per_node: int) -> int:
    """Optimal (tightest-pack) node count for the GPU jobs in this set."""
    if gpus_per_node == 0:
        return 0

    multinode_nodes = sum(j['num_nodes'] for j in gpu_jobs if j['num_nodes'] > 1)
    singlenode_gpus = sum(
        j['ngpus'] for j in gpu_jobs
        if j['num_nodes'] == 1 and j['ngpus'] <= gpus_per_node
    )

    opt = multinode_nodes + (singlenode_gpus // gpus_per_node) if singlenode_gpus else multinode_nodes
    if singlenode_gpus % gpus_per_node > 0 and opt == multinode_nodes:
        opt += 1
    return opt


def _cpu_optimal_nodes(cpu_jobs: List[Dict[str, Any]]) -> int:
    """Optimal node count for the CPU jobs in this set."""
    total_occupancy = sum(j['node_occupancy'] for j in cpu_jobs if j['num_nodes'] == 1)
    multinode_nodes = sum(j['num_nodes'] for j in cpu_jobs if j['num_nodes'] > 1)
    return multinode_nodes + math.ceil(total_occupancy)


def compute_required_nodes(jobs: List[Dict[str, Any]], system_config) -> int:
    """
    Return the optimal (tightest-pack) node count needed to run all schedulable
    jobs (Ready or Restart status) in `jobs` on the target system.

    GPU and CPU jobs are estimated separately and summed. Returns at least 1 if
    there are any schedulable jobs.

    Args:
        jobs: Job dicts as returned by `database.get_jobs()`.
        system_config: System config object (must expose GPUS_PER_NODE).

    Returns:
        int: optimal node count. 0 if no schedulable jobs.
    """
    schedulable = [j for j in jobs if j.get('status') in ('Ready', 'Restart')]
    if not schedulable:
        return 0

    gpus_per_node = system_config.GPUS_PER_NODE
    gpu_jobs = [j for j in schedulable if j['ngpus'] > 0]
    cpu_jobs = [j for j in schedulable if j['ngpus'] == 0]

    total = _gpu_optimal_nodes(gpu_jobs, gpus_per_node) + _cpu_optimal_nodes(cpu_jobs)
    return max(1, total)
