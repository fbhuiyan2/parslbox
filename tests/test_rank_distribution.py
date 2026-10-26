"""More MPI ranks than the job's GPUs or cores.

pbx was built on one rank per GPU. 0.9.3 made `ranks_per_node` configurable
but left the allocator slicing a resource list by rank index, so asking for
more ranks than resources produced ranks with *empty* GPU and core lists --
silently. The scheduler allocated the nodes, the job launched, and those ranks
ran with no GPU mask and no CPU binding at all.

Sharing is now supported: ranks round-robin over the GPUs and share the cores
of the GPU they land on; oversubscribed CPU jobs bind every rank to the whole
owned set rather than pinning each to one shared core. These tests pin both
the split itself and the launch artifacts it feeds, and -- just as important
-- that nothing changes when ranks do fit.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from parslbox.resource_manager import ResourceManager, create_job_resource_spec
from parslbox.resource_manager.mpi_command_builder import build_mpi_command
from parslbox.resource_manager.mpi_config import MPIBackend, MPIConfig
from parslbox.resource_manager.rank_distribution import (
    distribute_cores,
    distribute_gpus,
)
from test_comprehensive_resource_manager import MockSystemConfig


def _manager(**kwargs):
    return ResourceManager(MockSystemConfig(**kwargs), job_tracker=None)


def _assign(rm, **job):
    job.setdefault("job_id", 1)
    job.setdefault("num_nodes", 1)
    job.setdefault("ngpus", 0)
    job.setdefault("node_occupancy", 1.0)
    return rm.assign_resources(job)


class TestDistributeGpus:
    def test_one_each_when_counts_match(self):
        assert distribute_gpus([0, 1, 2, 3], 4) == [[0], [1], [2], [3]]

    def test_round_robin_when_ranks_outnumber_gpus(self):
        assert distribute_gpus([0, 1], 5) == [[0], [1], [0], [1], [0]]

    def test_every_rank_gets_a_gpu(self):
        for num_ranks in range(1, 20):
            blocks = distribute_gpus([0, 1, 2], num_ranks)
            assert len(blocks) == num_ranks
            assert all(b for b in blocks)

    def test_fewer_ranks_get_contiguous_blocks(self):
        assert distribute_gpus([0, 1, 2, 3, 4, 5], 2) == [[0, 1, 2], [3, 4, 5]]

    def test_no_allocated_gpu_is_stranded(self):
        """A GPU the job holds but never maps to a rank is one no other job
        can use either -- `-g 6 -rpn 4` used to leave GPUs 4 and 5 idle."""
        blocks = distribute_gpus([0, 1, 2, 3, 4, 5], 4)
        assert blocks == [[0, 1], [2, 3], [4], [5]]
        assert sorted(g for b in blocks for g in b) == [0, 1, 2, 3, 4, 5]

    def test_no_gpus_yields_empty_lists_not_a_crash(self):
        assert distribute_gpus([], 3) == [[], [], []]


class TestDistributeCores:
    def test_disjoint_blocks_when_cores_suffice(self):
        blocks = distribute_cores(list(range(16)), 2)
        assert blocks == [list(range(8)), list(range(8, 16))]
        assert set(blocks[0]).isdisjoint(blocks[1])

    def test_remainder_goes_to_the_leading_ranks(self):
        """Matches the old `+ (1 if rank < remaining)` pass exactly."""
        assert [len(b) for b in distribute_cores(list(range(14)), 4)] == [4, 4, 3, 3]

    def test_oversubscribed_ranks_share_the_whole_set(self):
        blocks = distribute_cores([5, 6, 7], 7)
        assert blocks == [[5, 6, 7]] * 7

    def test_shared_blocks_are_independent_objects(self):
        blocks = distribute_cores([1, 2], 3)
        blocks[0].append(99)
        assert blocks[1] == [1, 2]

    def test_every_rank_gets_cores(self):
        for num_ranks in range(1, 40):
            blocks = distribute_cores(list(range(8)), num_ranks)
            assert len(blocks) == num_ranks
            assert all(b for b in blocks)

    def test_exactly_one_core_each_at_the_boundary(self):
        """ranks == cores is the last fitting case, so it must still pin one
        core per rank. Sliding the comparison to `>=` would hand every rank
        the whole set here and nothing else in the suite notices."""
        assert distribute_cores([0, 1, 2, 3], 4) == [[0], [1], [2], [3]]


class TestGpuJobsShareGpus:
    """4 GPUs, 32 cores per node -> 8 cores per GPU."""

    def test_subnode_ranks_round_robin_over_the_allocated_gpus(self):
        a = _assign(_manager(), ngpus=2, ranks_per_node=6)
        assert [a.get_gpu_assignments_for_rank(r) for r in range(6)] == [
            [0], [1], [0], [1], [0], [1],
        ]

    def test_ranks_sharing_a_gpu_share_its_cores(self):
        a = _assign(_manager(), ngpus=1, ranks_per_node=4)
        blocks = [a.get_cpu_assignments_for_rank(r) for r in range(4)]
        assert all(len(b) == 8 for b in blocks)
        assert all(b == blocks[0] for b in blocks)

    def test_fullnode_shares_too(self):
        a = _assign(_manager(), ngpus=4, ranks_per_node=8)
        assert [a.get_gpu_assignments_for_rank(r) for r in range(8)] == [
            [0], [1], [2], [3], [0], [1], [2], [3],
        ]

    def test_multinode_shares_per_node(self):
        a = _assign(_manager(), num_nodes=2, ngpus=8, ranks_per_node=6)
        for node_idx in range(2):
            ranks = sorted(a.get_ranks_for_node(node_idx))
            assert len(ranks) == 6
            assert [a.get_gpu_assignments_for_rank(r) for r in ranks] == [
                [0], [1], [2], [3], [0], [1],
            ]

    def test_no_rank_is_left_without_resources(self):
        """The original defect: ranks past the GPU count got `[]` for both."""
        a = _assign(_manager(), ngpus=1, ranks_per_node=8)
        for rank in range(8):
            assert a.get_gpu_assignments_for_rank(rank), rank
            assert a.get_cpu_assignments_for_rank(rank), rank

    def test_leftover_gpus_are_no_longer_stranded(self):
        a = _assign(_manager(), ngpus=4, ranks_per_node=3)
        mapped = {g for r in range(3) for g in a.get_gpu_assignments_for_rank(r)}
        assert mapped == {0, 1, 2, 3}


class TestCpuJobsShareCores:
    def test_oversubscribed_ranks_get_the_whole_owned_set(self):
        a = _assign(_manager(), node_occupancy=0.5, ranks_per_node=40)
        blocks = [a.get_cpu_assignments_for_rank(r) for r in range(40)]
        assert all(len(b) == 16 for b in blocks)
        assert all(b == blocks[0] for b in blocks)

    def test_the_shared_set_is_only_what_the_job_owns(self):
        """Confinement is the point -- a concurrent pbx job on the same node
        must still get cores this one cannot touch."""
        rm = _manager()
        first = _assign(rm, job_id=1, node_occupancy=0.5, ranks_per_node=40)
        second = _assign(rm, job_id=2, node_occupancy=0.5, ranks_per_node=40)
        a = set(first.get_cpu_assignments_for_rank(0))
        b = set(second.get_cpu_assignments_for_rank(0))
        assert a and b and a.isdisjoint(b)

    def test_fullnode_oversubscription(self):
        a = _assign(_manager(), node_occupancy=1.0, ranks_per_node=100)
        assert all(len(a.get_cpu_assignments_for_rank(r)) == 32 for r in range(100))


class TestFittingJobsAreUnchanged:
    """The split only changes once ranks outnumber resources. Everything that
    fitted before must come out bit-for-bit the same."""

    def test_one_rank_per_gpu_still_disjoint(self):
        a = _assign(_manager(), ngpus=4, ranks_per_node=4)
        assert [a.get_gpu_assignments_for_rank(r) for r in range(4)] == [[0], [1], [2], [3]]
        blocks = [set(a.get_cpu_assignments_for_rank(r)) for r in range(4)]
        assert all(len(b) == 8 for b in blocks)
        assert len(set.union(*blocks)) == 32

    def test_multi_gpu_per_rank_still_blocks(self):
        a = _assign(_manager(), ngpus=4, ranks_per_node=2)
        assert a.get_gpu_assignments_for_rank(0) == [0, 1]
        assert a.get_gpu_assignments_for_rank(1) == [2, 3]

    def test_a_rank_holding_two_gpus_gets_both_core_blocks(self):
        """A sub-node rank with several GPUs must own every one of their core
        blocks. Taking only the first GPU's block halves the rank's cores and
        every other assertion here still passes."""
        a = _assign(_manager(), ngpus=2, ranks_per_node=1)
        assert a.get_gpu_assignments_for_rank(0) == [0, 1]
        assert len(a.get_cpu_assignments_for_rank(0)) == 16

    def test_uneven_subnode_blocks_each_carry_their_own_cores(self):
        a = _assign(_manager(), ngpus=3, ranks_per_node=2)
        assert a.get_gpu_assignments_for_rank(0) == [0, 1]
        assert a.get_gpu_assignments_for_rank(1) == [2]
        assert len(a.get_cpu_assignments_for_rank(0)) == 16
        assert len(a.get_cpu_assignments_for_rank(1)) == 8

    def test_cpu_ranks_still_pinned_disjointly(self):
        a = _assign(_manager(), node_occupancy=0.5, ranks_per_node=4)
        blocks = [set(a.get_cpu_assignments_for_rank(r)) for r in range(4)]
        assert all(len(b) == 4 for b in blocks)
        assert len(set.union(*blocks)) == 16


class TestTotalGpusCountsDevicesNotRanks:
    def test_a_shared_gpu_counts_once(self):
        a = _assign(_manager(), ngpus=1, ranks_per_node=8)
        assert a.get_total_gpus() == 1

    def test_multinode_does_not_collapse_matching_ids_across_nodes(self):
        """GPU ids restart at 0 on each node, so a flat set() would report 4
        for an 8-GPU two-node job."""
        a = _assign(_manager(), num_nodes=2, ngpus=8, ranks_per_node=6)
        assert a.get_total_gpus() == 8


class TestLaunchArtifacts:
    """What the ranks actually receive at launch."""

    def _mpich(self, **job):
        rm = _manager(gpu_type="nvidia")
        assignment = _assign(rm, **job)
        job.setdefault("job_id", 1)
        job.setdefault("num_nodes", 1)
        job.setdefault("node_occupancy", 1.0)
        spec = create_job_resource_spec(job)
        tmp = tempfile.mkdtemp()
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            config = MPIConfig(backend=MPIBackend.MPICH, use_gpu_wrapper=True,
                               cpu_bind_method="rankfile")
            cmd = build_mpi_command(config, rm.system_config, assignment, spec, job_path=tmp)
            files = {name: Path(tmp, name).read_text() for name in os.listdir(tmp)}
        finally:
            os.chdir(cwd)
        return cmd, files

    def _only(self, files, needle):
        return next(text for name, text in files.items() if needle in name)

    def test_wrapper_covers_every_rank(self):
        """Ranks with no GPU got no `if` block at all, so they inherited the
        environment's mask and saw every GPU on the node."""
        _, files = self._mpich(ngpus=1, ranks_per_node=8)
        wrapper = self._only(files, "gpu_wrapper")
        for rank in range(8):
            assert f'"$GLOBAL_RANK" -eq {rank}' in wrapper, rank

    def test_ranks_sharing_a_gpu_get_the_same_mask(self):
        _, files = self._mpich(ngpus=2, ranks_per_node=4)
        wrapper = self._only(files, "gpu_wrapper")
        assert wrapper.count("CUDA_VISIBLE_DEVICES=0") == 2
        assert wrapper.count("CUDA_VISIBLE_DEVICES=1") == 2

    def test_rankfile_has_a_line_per_rank(self):
        """Empty ranks were skipped, so the file was short and the ranks it
        omitted launched unbound."""
        _, files = self._mpich(ngpus=1, ranks_per_node=8)
        rankfile = self._only(files, "rankfile")
        lines = [ln for ln in rankfile.splitlines() if ln.strip()]
        assert [ln.split()[0] for ln in lines] == [str(r) for r in range(8)]

    def test_cores_per_rank_does_not_claim_shared_cores_twice(self):
        """Each of the 4 ranks is given the GPU's whole 8-core block, but
        --depth means exclusive cores, so it must report the divided share."""
        rm = _manager()
        assignment = _assign(rm, ngpus=1, ranks_per_node=4)
        spec = create_job_resource_spec(
            {"job_id": 1, "num_nodes": 1, "ngpus": 1, "node_occupancy": 1.0,
             "ranks_per_node": 4}
        )
        config = MPIConfig(backend=MPIBackend.MPICH, cpu_bind_method="depth",
                           use_gpu_wrapper=False)
        cmd = build_mpi_command(config, rm.system_config, assignment, spec)
        assert "--depth 2" in cmd["PBX_MPI_PREFIX"]


class TestPreviewMatchesTheScheduler:
    """`pbx info` previews launch commands without scheduling. It kept its own
    copy of the split and invented GPU ids the node does not have."""

    @pytest.mark.parametrize("ngpus,nocc,ranks", [
        (2, 1.0, 7), (1, 1.0, 8), (4, 1.0, 4), (4, 1.0, 2), (0, 0.5, 40), (0, 0.5, 4),
    ])
    def test_preview_agrees_with_the_real_assignment(self, ngpus, nocc, ranks):
        from parslbox.commands.helpers.resource_req import _synthetic_assignment

        rm = _manager()
        job = {"job_id": 1, "num_nodes": 1, "ngpus": ngpus,
               "node_occupancy": nocc, "ranks_per_node": ranks}
        real = rm.assign_resources(dict(job))
        spec = create_job_resource_spec(job)
        preview = _synthetic_assignment(spec, 4, 32)

        for rank in range(ranks):
            assert (preview.get_gpu_assignments_for_rank(rank)
                    == real.get_gpu_assignments_for_rank(rank)), rank
            # Preview numbers cores from 0 rather than from the node's free
            # pool, so the ids differ by an offset; the shape must not.
            assert (len(preview.get_cpu_assignments_for_rank(rank))
                    == len(real.get_cpu_assignments_for_rank(rank))), rank

    def test_preview_agrees_on_a_multinode_job(self):
        from parslbox.commands.helpers.resource_req import _synthetic_assignment

        rm = _manager()
        job = {"job_id": 1, "num_nodes": 2, "ngpus": 8,
               "node_occupancy": 1.0, "ranks_per_node": 6}
        real = rm.assign_resources(dict(job))
        preview = _synthetic_assignment(create_job_resource_spec(job), 4, 32)
        for node_idx in range(2):
            ranks = sorted(real.get_ranks_for_node(node_idx))
            assert sorted(preview.get_ranks_for_node(node_idx)) == ranks
            for rank in ranks:
                assert (preview.get_gpu_assignments_for_rank(rank)
                        == real.get_gpu_assignments_for_rank(rank)), rank


class TestSrunReadsTheAssignment:
    """The srun full/multi-node branch recomputed `GPUS_PER_NODE //
    ranks_per_node` instead of reading the assignment, so it stranded the
    remainder GPUs under srun even after the allocator stopped stranding them
    everywhere else -- the two backends disagreed about the same job."""

    def _gpu_bind(self, **job):
        rm = ResourceManager(
            MockSystemConfig(cores_per_node=48, gpus_per_node=6, num_nodes=2),
            job_tracker=None,
        )
        job.setdefault("job_id", 1)
        job.setdefault("node_occupancy", 1.0)
        assignment = rm.assign_resources(dict(job))
        spec = create_job_resource_spec(job)
        cmd = build_mpi_command(MPIConfig(backend=MPIBackend.SRUN),
                                rm.system_config, assignment, spec)
        return next(f for f in cmd["PBX_MPI_PREFIX"].split() if "gpu-bind" in f)

    def test_uneven_blocks_bind_every_gpu(self):
        # [0,1] [2,3] [4] [5] -> 0x3, 0xc, 0x10, 0x20. It emitted
        # 0x1,0x2,0x4,0x8, leaving GPUs 4 and 5 held but bound to no task.
        assert self._gpu_bind(num_nodes=1, ngpus=6, ranks_per_node=4) == (
            "--gpu-bind=mask_gpu:0x3,0xc,0x10,0x20")

    def test_uneven_blocks_bind_every_gpu_multinode(self):
        assert self._gpu_bind(num_nodes=2, ngpus=12, ranks_per_node=4) == (
            "--gpu-bind=mask_gpu:0x3,0xc,0x10,0x20")

    def test_one_rank_per_gpu_is_unchanged(self):
        assert self._gpu_bind(num_nodes=1, ngpus=6, ranks_per_node=6) == (
            "--gpu-bind=map_gpu:0,1,2,3,4,5")

    def test_shared_gpus_are_mapped_explicitly(self):
        assert self._gpu_bind(num_nodes=1, ngpus=6, ranks_per_node=8) == (
            "--gpu-bind=map_gpu:0,1,2,3,4,5,0,1")

    def test_the_map_describes_one_node_not_all_of_them(self):
        """SLURM indexes the map by local task id, so a multi-node job must
        not repeat the layout once per node."""
        assert self._gpu_bind(num_nodes=2, ngpus=12, ranks_per_node=6) == (
            "--gpu-bind=map_gpu:0,1,2,3,4,5")


class TestSrunHasNoListBinding:
    """srun's only per-task CPU forms are map_cpu (one cpu id per task) and
    mask_cpu (one bitmask per task) -- `list` is MPICH vocabulary. pbx used to
    flatten every rank's core set into map_cpu, so SLURM read the second core
    of rank 0's block as rank 1's only CPU. Reached by `cpu_bind_method: list`
    and, more easily, by `rankfile`, which falls through to the same builder."""

    def _cpu_bind(self, method, **job):
        rm = ResourceManager(
            MockSystemConfig(cores_per_node=16, gpus_per_node=4, num_nodes=2),
            job_tracker=None,
        )
        job.setdefault("job_id", 1)
        job.setdefault("node_occupancy", 1.0)
        assignment = rm.assign_resources(dict(job))
        spec = create_job_resource_spec(job)
        cmd = build_mpi_command(
            MPIConfig(backend=MPIBackend.SRUN, cpu_bind_method=method),
            rm.system_config, assignment, spec)
        flag = next(f for f in cmd["PBX_MPI_PREFIX"].split() if "cpu-bind" in f)
        return flag, assignment

    @pytest.mark.parametrize("method", ["list", "rankfile"])
    def test_masks_match_the_per_rank_core_sets(self, method):
        flag, a = self._cpu_bind(method, num_nodes=2, ngpus=8, ranks_per_node=4)
        assert flag == "--cpu-bind=mask_cpu:0xf,0xf0,0xf00,0xf000"
        assert [a.get_cpu_assignments_for_rank(r) for r in a.get_ranks_for_node(0)] == [
            [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

    @pytest.mark.parametrize("method", ["list", "rankfile"])
    def test_never_emits_map_cpu(self, method):
        flag, _ = self._cpu_bind(method, num_nodes=2, ngpus=8, ranks_per_node=4)
        assert "map_cpu" not in flag

    def test_one_mask_per_local_task_not_per_global_rank(self):
        """mask_cpu is "specified for a node and identical mapping is applied
        to the tasks on every node" (srun(1)), so a 2-node job must emit
        ranks_per_node masks, not twice that."""
        flag, _ = self._cpu_bind("list", num_nodes=2, ngpus=8, ranks_per_node=4)
        assert len(flag.split(":")[1].split(",")) == 4

    def test_ranks_sharing_a_gpu_share_its_mask(self):
        flag, _ = self._cpu_bind("list", num_nodes=2, ngpus=8, ranks_per_node=6)
        assert flag == "--cpu-bind=mask_cpu:0xf,0xf0,0xf00,0xf000,0xf,0xf0"


class TestDepthNeverExceedsTheSmallestBlock:
    """`--depth` / `--cpus-per-task` is one number for every rank, so it is a
    claim about exclusive cores that the shortest rank has to be able to meet.
    An uneven split leaves ranks with different block sizes, and both the old
    code (rank 0's length) and the first version of this change (the node
    average) over-claimed for the ranks below that."""

    def _depth(self, backend, flag, **job):
        rm = _manager()
        assignment = _assign(rm, **job)
        job.setdefault("job_id", 1)
        job.setdefault("num_nodes", 1)
        job.setdefault("node_occupancy", 1.0)
        spec = create_job_resource_spec(job)
        config = MPIConfig(backend=backend, cpu_bind_method="depth")
        cmd = build_mpi_command(config, rm.system_config, assignment, spec,
                                job_path=None)["PBX_MPI_PREFIX"]
        tokens = cmd.split()
        return int(tokens[tokens.index(flag) + 1]), assignment

    def test_uneven_gpu_split_uses_the_shortest_rank(self):
        depth, a = self._depth(MPIBackend.MPICH, "--depth", ngpus=3, ranks_per_node=2)
        smallest = min(len(a.get_cpu_assignments_for_rank(r))
                       for r in a.get_ranks_for_node(0))
        assert smallest == 8
        assert depth == smallest

    def test_even_splits_are_unchanged(self):
        depth, _ = self._depth(MPIBackend.MPICH, "--depth", ngpus=4, ranks_per_node=4)
        assert depth == 8

    def test_oversubscribed_ranks_claim_one_core(self):
        """Ranks outnumber cores, so no rank can hold one exclusively."""
        depth, _ = self._depth(MPIBackend.MPICH, "--depth",
                               node_occupancy=0.5, ranks_per_node=40)
        assert depth == 1

    def test_no_rank_is_ever_promised_more_than_it_holds(self):
        for ngpus in range(1, 5):
            for ranks in range(1, 9):
                depth, a = self._depth(MPIBackend.MPICH, "--depth",
                                       ngpus=ngpus, ranks_per_node=ranks)
                smallest = min(len(a.get_cpu_assignments_for_rank(r))
                               for r in a.get_ranks_for_node(0))
                assert depth <= smallest, (ngpus, ranks, depth, smallest)


class TestCoresPerGpuNeverFloorsToZero:
    def test_a_config_with_fewer_cores_than_gpus_fails_instead_of_assigning_nothing(self):
        """`cores_per_gpu` floored to 0 on a node with fewer usable cores than
        GPUs, so the job was accepted and every rank got an empty core list.
        The config is impossible either way; the point is that it now says so
        rather than launching ranks with no CPU binding at all."""
        rm = ResourceManager(MockSystemConfig(cores_per_node=4, gpus_per_node=8),
                             job_tracker=None)
        with pytest.raises(Exception):
            _assign(rm, ngpus=8, ranks_per_node=4)

    def test_a_sane_config_is_untouched(self):
        a = _assign(_manager(), ngpus=4, ranks_per_node=4)
        assert all(len(a.get_cpu_assignments_for_rank(r)) == 8 for r in range(4))


class TestSrunSubnodeAsksForItsCores:
    """A sub-node srun step gets `--cpus-per-task N` alongside the mask_cpu
    list. Nothing asserted the flag was emitted at all, so deleting it left
    the whole suite green -- and a step that does not ask for its CPUs can be
    given one per task by SLURM regardless of the mask."""

    def _srun(self, method, **job):
        rm = ResourceManager(
            MockSystemConfig(cores_per_node=32, gpus_per_node=4, num_nodes=2),
            job_tracker=None)
        job.setdefault("job_id", 1)
        job.setdefault("num_nodes", 1)
        job.setdefault("node_occupancy", 1.0)
        assignment = rm.assign_resources(dict(job))
        spec = create_job_resource_spec(job)
        cmd = build_mpi_command(
            MPIConfig(backend=MPIBackend.SRUN, cpu_bind_method=method),
            rm.system_config, assignment, spec)["PBX_MPI_PREFIX"]
        return cmd.split(), assignment

    @pytest.mark.parametrize("method", ["cores", "rankfile", "depth", "none"])
    def test_subnode_gpu_job_asks_for_cpus_per_task(self, method):
        """Sub-node srun ignores cpu_bind_method and uses mask_cpu, but it must
        still request the cores: the flag is emitted on that path, not by the
        binding builder."""
        tokens, a = self._srun(method, ngpus=2, ranks_per_node=2)
        assert "--cpus-per-task" in tokens
        n = int(tokens[tokens.index("--cpus-per-task") + 1])
        smallest = min(len(a.get_cpu_assignments_for_rank(r))
                       for r in a.get_ranks_for_node(0))
        assert n == smallest == 8
        assert any(t.startswith("--cpu-bind=mask_cpu:") for t in tokens)

    def test_subnode_cpu_job_asks_for_cpus_per_task(self):
        tokens, _ = self._srun("cores", node_occupancy=0.5, ranks_per_node=4)
        assert "--cpus-per-task" in tokens
        assert int(tokens[tokens.index("--cpus-per-task") + 1]) == 4

    def test_oversubscribed_subnode_still_asks_for_one(self):
        tokens, _ = self._srun("cores", node_occupancy=0.5, ranks_per_node=30)
        assert int(tokens[tokens.index("--cpus-per-task") + 1]) == 1
