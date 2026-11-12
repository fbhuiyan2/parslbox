"""Helpers shared by the GROMACS, LAMMPS and Quantum ESPRESSO apps."""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def split_input(in_file: str) -> tuple[str, str]:
    """Split a job's `in_file` into (input file, extra args).

    `pbx add --app-args` is appended to the input file name, so `in_file` can
    be `"md.tpr -nsteps 5000"`. Apps that derive file names from the input
    must use only its first token.
    """
    parts = (in_file or "").split(None, 1)
    if not parts:
        return "", ""
    return parts[0], parts[1] if len(parts) > 1 else ""


def read_text_safe(path: Path) -> Optional[str]:
    """File contents, or None when it is missing or unreadable.

    Undecodable bytes are replaced rather than raised on: a program can echo
    raw bytes into its log, and a strict decode would hide a finished run.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        logger.warning(f"Could not read {path}: {e}")
        return None


def last_segment(text: str, start_marker: str) -> str:
    """Text from the last occurrence of `start_marker`, or all of it if absent.

    Appended logs hold one segment per restart; only the last one says whether
    the most recent run finished.
    """
    idx = text.rfind(start_marker)
    return text if idx < 0 else text[idx:]


def job_input_file(job_id: int, db_path) -> Optional[str]:
    """The input file the job was added with, without any `--app-args`.

    `check_success` is not given the input file, so read it from the job's DB
    row. Returns None when the row cannot be read.
    """
    if db_path is None:
        return None
    try:
        from parslbox.database import database
        rows = database.get_jobs_by_ids(db_path, [job_id])
    except Exception as e:
        logger.warning(f"Job {job_id}: could not read the job row: {e}")
        return None
    if not rows:
        return None
    first, _ = split_input(rows[0].get("in_file") or "")
    return first or None


def newest_file(job_path: Path, pattern: str) -> Optional[Path]:
    """Newest file in the job directory matching `pattern`, pbx's own
    `pbx_job_*` files excluded."""
    candidates = [p for p in Path(job_path).glob(pattern)
                  if not p.name.startswith("pbx_job_")]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def num_nodes(mpi_commands: dict) -> int:
    hostnames = mpi_commands.get("PBX_HOSTNAMES", "")
    return max(1, len([h for h in hostnames.split(",") if h]))


def ranks_per_node(mpi_commands: dict) -> int:
    return max(1, int(mpi_commands.get("PBX_RANKS_PER_NODE") or 1))


def threads_per_rank(system_config, total_gpus: int, mpi_commands: dict) -> int:
    """OpenMP threads per rank, from the cores the job holds on a node.

    Same split the resource manager makes: a GPU job holds
    `usable_cores // GPUS_PER_NODE` cores per GPU, so a sub-node GPU job holds
    only its GPUs' share. A CPU job is taken to hold the whole node; pbx does
    not pass node occupancy to apps, so a CPU job at occupancy < 1 gets more
    threads than cores. Override with the program's own thread flag in
    `--app-args`.
    """
    excluded = getattr(system_config, "EXCLUDE_CORES", None) or []
    usable = system_config.CORES_PER_NODE - len(excluded)
    held = usable
    if total_gpus > 0 and system_config.GPUS_PER_NODE > 0:
        gpus_held = min(max(1, total_gpus // num_nodes(mpi_commands)),
                        system_config.GPUS_PER_NODE)
        cores_per_gpu = max(1, usable // system_config.GPUS_PER_NODE)
        held = min(usable, gpus_held * cores_per_gpu)
    return max(1, held // ranks_per_node(mpi_commands))


def gpus_visible_per_rank(mpi_prefix: str, total_gpus: int, gpus_per_node: int,
                          mpi_commands: dict) -> int:
    """GPUs one rank can see, read from the GPU binding in the MPI prefix.

    Same rule `lammps-kk` uses for `-k on g N`: one for srun `map_gpu` or a
    per-rank wrapper script, the per-rank share for `mask_gpu`, and every GPU
    on the node when nothing isolates them.
    """
    if total_gpus > 0 and ("map_gpu" in mpi_prefix
                           or (".sh" in mpi_prefix and "wrapper" in mpi_prefix)):
        return 1
    if total_gpus > 0 and "mask_gpu" in mpi_prefix:
        rpn = mpi_commands.get("PBX_RANKS_PER_NODE")
        if rpn:
            return max(1, min(total_gpus, gpus_per_node) // int(rpn))
    return min(total_gpus, gpus_per_node)
