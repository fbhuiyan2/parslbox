"""Splitting a node's GPUs and CPU cores across the ranks placed on it.

Ranks may outnumber the resources the job holds -- several MPI ranks sharing
one GPU is a supported configuration (LAMMPS and GROMACS both want it). The
two functions here are the single definition of how that split is done; the
scheduler and the `pbx info` command preview both call them, so a preview
cannot describe a placement the scheduler would not produce.

The shared rule: while there are enough resources to go round, ranks get
disjoint blocks, which is what makes pinning worth doing. Once ranks
outnumber resources that trade inverts -- locality is lost anyway because the
resource is shared -- so ranks stop being pinned individually and are confined
to the block they share, leaving the kernel free to balance them.
"""
from typing import List


def _split_into_blocks(items: List[int], num_ranks: int) -> List[List[int]]:
    """Contiguous blocks, longer ones first so nothing is left over."""
    base, extra = divmod(len(items), num_ranks)
    blocks, start = [], 0
    for rank in range(num_ranks):
        size = base + (1 if rank < extra else 0)
        blocks.append(items[start:start + size])
        start += size
    return blocks


def distribute_gpus(gpu_ids: List[int], num_ranks: int) -> List[List[int]]:
    """Map each rank to the GPU ids it owns.

    With at least as many ranks as GPUs, ranks round-robin over the GPUs and
    each gets exactly one, so ranks past the GPU count share rather than
    coming away empty. With fewer ranks than GPUs the GPUs are blocked out
    between them, remainder first -- every allocated GPU reaches a rank, since
    a GPU the job holds but never maps is one no other job can use either.
    """
    if not gpu_ids or num_ranks <= 0:
        return [[] for _ in range(max(0, num_ranks))]

    if num_ranks >= len(gpu_ids):
        return [[gpu_ids[rank % len(gpu_ids)]] for rank in range(num_ranks)]

    return _split_into_blocks(gpu_ids, num_ranks)


def distribute_cores(core_ids: List[int], num_ranks: int) -> List[List[int]]:
    """Map each rank to the CPU core ids it may run on.

    Oversubscribed, every rank gets the whole set: pinning each to a single
    shared core would stop the kernel moving a rank to a core its neighbours
    are not using. The set is still only what the job owns, so concurrent pbx
    jobs on the same node stay isolated from each other.

    This is the assignment. Whether a launcher can express a set depends on
    its cpu_bind_method -- rankfile, list and srun's mask_cpu can, while
    depth/cores/threads carry one number and end up pinning one core per
    rank. See resource_manager/README.md.
    """
    if not core_ids or num_ranks <= 0:
        return [[] for _ in range(max(0, num_ranks))]

    if num_ranks > len(core_ids):
        return [list(core_ids) for _ in range(num_ranks)]

    return _split_into_blocks(core_ids, num_ranks)
