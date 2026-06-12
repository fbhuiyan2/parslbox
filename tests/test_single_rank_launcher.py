"""
Tests for build_single_rank_launcher().

Verifies the minimal per-backend command emitter that lands hook subprocesses
on the FIRST assigned compute node. Asserts:
  - Per-backend command shape (mpiexec / mpirun / srun flags)
  - Only the first hostname is used (multi-node assignments collapse to first)
  - use_short_hostnames strips the domain
  - No rankfile is generated as a side effect (the original collision bug)
  - srun gets --overlap and --cpu-bind=none for coexistence with the bash_app
  - Unknown backend raises ValueError
"""

import pytest

from parslbox.resource_manager.mpi_command_builder import build_single_rank_launcher
from parslbox.resource_manager.mpi_config import MPIConfig, MPIBackend
from parslbox.resource_manager.models import ResourceAssignment


def _make_assignment(hostnames, job_id=1):
    """Minimal ResourceAssignment — only hostnames and job_id are consulted."""
    n = len(hostnames)
    return ResourceAssignment(
        job_id=job_id,
        node_ids=[f"n{i}" for i in range(n)],
        hostnames=list(hostnames),
        cpu_assignments=[{} for _ in range(n)],
        gpu_assignments=[{} for _ in range(n)],
        node_occupancy=1.0,
    )


# ============================================================================
# Per-backend command shape
# ============================================================================

class TestPerBackendCommandShape:
    def test_mpich_minimal_command(self):
        config = MPIConfig(backend=MPIBackend.MPICH)
        assignment = _make_assignment(["host0"])

        launcher = build_single_rank_launcher(config, assignment)

        assert launcher == "mpiexec -n 1 --ppn 1 -hosts host0"

    def test_openmpi_minimal_command(self):
        config = MPIConfig(backend=MPIBackend.OPENMPI)
        assignment = _make_assignment(["host0"])

        launcher = build_single_rank_launcher(config, assignment)

        assert launcher == "mpirun -np 1 -H host0"

    def test_srun_minimal_command(self):
        config = MPIConfig(backend=MPIBackend.SRUN)
        assignment = _make_assignment(["host0"])

        launcher = build_single_rank_launcher(config, assignment)

        assert launcher == "srun --overlap --cpu-bind=none -N 1 -n 1 -w host0"

    def test_srun_includes_overlap_and_cpu_bind_none(self):
        """--overlap so srun coexists with the bash_app step; --cpu-bind=none
        so slurm.conf defaults can't grab specific cores."""
        config = MPIConfig(backend=MPIBackend.SRUN)
        assignment = _make_assignment(["host0"])

        launcher = build_single_rank_launcher(config, assignment)

        assert "--overlap" in launcher
        assert "--cpu-bind=none" in launcher


# ============================================================================
# Multi-node collapse: only the FIRST hostname is used
# ============================================================================

class TestFirstHostnameOnly:
    @pytest.mark.parametrize("backend,expected_host_part", [
        (MPIBackend.MPICH, "-hosts host0"),
        (MPIBackend.OPENMPI, "-H host0"),
        (MPIBackend.SRUN, "-w host0"),
    ])
    def test_multi_node_assignment_uses_first_host_only(self, backend, expected_host_part):
        config = MPIConfig(backend=backend)
        assignment = _make_assignment(["host0", "host1", "host2", "host3", "host4"])

        launcher = build_single_rank_launcher(config, assignment)

        assert expected_host_part in launcher
        # Other hosts must NOT appear (no comma-separated host list)
        for stray in ("host1", "host2", "host3", "host4"):
            assert stray not in launcher


# ============================================================================
# Short-hostname handling
# ============================================================================

class TestShortHostnames:
    @pytest.mark.parametrize("backend,host_flag", [
        (MPIBackend.MPICH, "-hosts"),
        (MPIBackend.OPENMPI, "-H"),
        (MPIBackend.SRUN, "-w"),
    ])
    def test_short_hostnames_strips_domain(self, backend, host_flag):
        config = MPIConfig(backend=backend, use_short_hostnames=True)
        assignment = _make_assignment([
            "x4703c0s0b0n0.hsn.cm.aurora.alcf.anl.gov",
            "x4703c0s1b0n0.hsn.cm.aurora.alcf.anl.gov",
        ])

        launcher = build_single_rank_launcher(config, assignment)

        assert f"{host_flag} x4703c0s0b0n0" in launcher
        assert "aurora" not in launcher  # domain stripped

    @pytest.mark.parametrize("backend,host_flag", [
        (MPIBackend.MPICH, "-hosts"),
        (MPIBackend.OPENMPI, "-H"),
        (MPIBackend.SRUN, "-w"),
    ])
    def test_short_hostnames_off_keeps_full_hostname(self, backend, host_flag):
        config = MPIConfig(backend=backend, use_short_hostnames=False)
        assignment = _make_assignment(["host0.domain.example.com"])

        launcher = build_single_rank_launcher(config, assignment)

        assert f"{host_flag} host0.domain.example.com" in launcher


# ============================================================================
# No rankfile / GPU wrapper is generated as a side effect
# (regression for the filename-collision bug)
# ============================================================================

class TestNoSideEffectFiles:
    @pytest.mark.parametrize("backend", [MPIBackend.MPICH, MPIBackend.OPENMPI, MPIBackend.SRUN])
    def test_no_rankfile_written(self, backend, tmp_path, monkeypatch):
        """Building the launcher must not write rankfile_pbx_*.txt or
        gpu_wrapper_pbx_*.sh — those would collide with the main bash_app's
        artifacts on disk. Function takes no job_path arg and shouldn't touch
        the filesystem at all; assert tmp_path stays empty as proof."""
        monkeypatch.chdir(tmp_path)
        config = MPIConfig(backend=backend)
        assignment = _make_assignment(["host0", "host1"])

        build_single_rank_launcher(config, assignment)

        leftovers = list(tmp_path.iterdir())
        assert not leftovers, f"launcher leaked artifacts into tmp_path: {leftovers}"

    @pytest.mark.parametrize("backend", [MPIBackend.MPICH, MPIBackend.OPENMPI, MPIBackend.SRUN])
    def test_launcher_string_has_no_rankfile_flag(self, backend):
        config = MPIConfig(backend=backend)
        assignment = _make_assignment(["host0", "host1"])

        launcher = build_single_rank_launcher(config, assignment)

        assert "--rankfile" not in launcher
        assert "rankfile_pbx_" not in launcher
        assert "gpu_wrapper_pbx_" not in launcher


# ============================================================================
# Sub-node job assignments: launcher still emits single-host single-rank
# (the launcher doesn't care about per-rank CPU/GPU specifics)
# ============================================================================

class TestSubNodeJobAssignments:
    @pytest.mark.parametrize("backend,host_flag", [
        (MPIBackend.MPICH, "-hosts"),
        (MPIBackend.OPENMPI, "-H"),
        (MPIBackend.SRUN, "-w"),
    ])
    def test_subnode_job_still_minimal(self, backend, host_flag):
        """A sub-node job (e.g. 4 cores on 1 node) still gets the same
        minimal launcher — no --cpu-bind=mask_cpu, no --cpus-per-task,
        no rankfile."""
        config = MPIConfig(backend=backend)
        assignment = ResourceAssignment(
            job_id=42,
            node_ids=["n0"],
            hostnames=["host0"],
            cpu_assignments=[{0: [0, 1, 2, 3]}],
            gpu_assignments=[{0: [0]}],
            node_occupancy=0.25,
        )

        launcher = build_single_rank_launcher(config, assignment)

        assert f"{host_flag} host0" in launcher
        assert "--cpus-per-task" not in launcher
        assert "--cpu-bind=mask_cpu" not in launcher
        assert "--rankfile" not in launcher


# ============================================================================
# Unknown backend
# ============================================================================

class TestUnknownBackend:
    def test_raises_on_unknown_backend(self, monkeypatch):
        """If a new backend ever appears, fail loudly rather than emit garbage."""
        config = MPIConfig(backend=MPIBackend.MPICH)
        # Forge an unknown backend value
        from enum import Enum

        class FakeBackend(Enum):
            UNKNOWN = "unknown"

        monkeypatch.setattr(config, "backend", FakeBackend.UNKNOWN)
        assignment = _make_assignment(["host0"])

        with pytest.raises(ValueError, match="Unsupported MPI backend"):
            build_single_rank_launcher(config, assignment)
