"""
Tests for build_resource_launcher() and the USES_MPI app attribute.

Verifies that the resource launcher correctly generates single-process MPI
commands for non-MPI apps, ensuring they run on assigned resources rather
than the head node.
"""

import pytest
from unittest.mock import Mock

from parslbox.resource_manager.mpi_command_builder import build_resource_launcher
from parslbox.resource_manager.mpi_config import MPIConfig, MPIBackend
from parslbox.resource_manager.models import JobResourceSpec, ResourceAssignment


class TestBuildResourceLauncher:
    """Tests for build_resource_launcher()."""

    def _make_system_config(self, cores_per_node=64, exclude_cores=None):
        """Create a mock system config."""
        sc = Mock()
        sc.CORES_PER_NODE = cores_per_node
        sc.EXCLUDE_CORES = exclude_cores or []
        return sc

    def _make_assignment(self, node_ids, hostnames, cpu_assignments=None,
                         gpu_assignments=None, node_occupancy=1.0, job_id=1):
        """Create a real ResourceAssignment."""
        return ResourceAssignment(
            job_id=job_id,
            node_ids=node_ids,
            hostnames=hostnames,
            cpu_assignments=cpu_assignments or [{} for _ in node_ids],
            gpu_assignments=gpu_assignments or [{} for _ in node_ids],
            node_occupancy=node_occupancy,
        )

    def _make_job_spec(self, job_id=1, num_nodes=1, ngpus=0,
                       node_occupancy=1.0, ranks_per_node=1):
        """Create a real JobResourceSpec."""
        return JobResourceSpec(
            job_id=job_id,
            num_nodes=num_nodes,
            ngpus=ngpus,
            node_occupancy=node_occupancy,
            ranks_per_node=ranks_per_node,
        )

    # --- Always -n 1 tests across backends ---

    def test_mpich_always_n1_ppn1(self):
        """MPICH produces -n 1 --ppn 1 even for multi-GPU/multi-node job specs."""
        config = MPIConfig(backend=MPIBackend.MPICH)
        sys_config = self._make_system_config()

        # Original job is 4-GPU, 2-node — resource launcher should still be -n 1
        job_spec = self._make_job_spec(num_nodes=2, ngpus=8, ranks_per_node=4)
        assignment = self._make_assignment(
            node_ids=["n0", "n1"],
            hostnames=["node0", "node1"],
            cpu_assignments=[
                {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]},
                {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]},
            ],
            gpu_assignments=[
                {0: [0], 1: [1], 2: [2], 3: [3]},
                {0: [0], 1: [1], 2: [2], 3: [3]},
            ],
        )

        launcher = build_resource_launcher(config, sys_config, assignment, job_spec)

        assert "-n 1" in launcher
        assert "--ppn 1" in launcher
        assert "-n 2" not in launcher
        assert "-n 8" not in launcher

    def test_openmpi_always_np1(self):
        """OpenMPI produces -np 1 even for multi-rank job specs."""
        config = MPIConfig(backend=MPIBackend.OPENMPI)
        sys_config = self._make_system_config()

        job_spec = self._make_job_spec(num_nodes=1, ngpus=4, ranks_per_node=4)
        assignment = self._make_assignment(
            node_ids=["n0"],
            hostnames=["node0"],
        )

        launcher = build_resource_launcher(config, sys_config, assignment, job_spec)

        assert "-np 1" in launcher
        assert "-np 4" not in launcher

    def test_srun_single_node_n1(self):
        """SRUN single-node produces -n 1 --ntasks-per-node 1."""
        config = MPIConfig(backend=MPIBackend.SRUN)
        sys_config = self._make_system_config()

        job_spec = self._make_job_spec(num_nodes=1, ranks_per_node=4)
        assignment = self._make_assignment(
            node_ids=["n0"],
            hostnames=["node0"],
        )

        launcher = build_resource_launcher(config, sys_config, assignment, job_spec)

        assert "-n 1" in launcher
        assert "--ntasks-per-node 1" in launcher

    def test_srun_multinode_uses_nodes_flag(self):
        """SRUN multi-node non-MPI: -n 1 --nodes=N, no --ntasks-per-node."""
        config = MPIConfig(backend=MPIBackend.SRUN)
        sys_config = self._make_system_config()

        job_spec = self._make_job_spec(num_nodes=3, ranks_per_node=8)
        assignment = self._make_assignment(
            node_ids=["n0", "n1", "n2"],
            hostnames=["node0", "node1", "node2"],
        )

        launcher = build_resource_launcher(config, sys_config, assignment, job_spec)

        assert "-n 1" in launcher
        assert "--nodes=3" in launcher
        assert "--ntasks-per-node" not in launcher

    # --- CPU core consolidation ---

    def test_consolidates_cpu_cores(self):
        """All CPU cores from multiple ranks on node 0 are merged into one binding."""
        config = MPIConfig(backend=MPIBackend.MPICH, cpu_bind_method="list")
        sys_config = self._make_system_config()

        # 4 ranks on node 0, each with different cores
        assignment = self._make_assignment(
            node_ids=["n0"],
            hostnames=["node0"],
            cpu_assignments=[{0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7]}],
        )
        job_spec = self._make_job_spec(ngpus=4, ranks_per_node=4)

        launcher = build_resource_launcher(config, sys_config, assignment, job_spec)

        # All 8 cores should be in a single binding for rank 0
        assert "--cpu-bind" in launcher
        assert "list:" in launcher
        # The consolidated cores should appear (0,1,2,3,4,5,6,7)
        assert "0,1,2,3,4,5,6,7" in launcher

    # --- Hostlist tests ---

    def test_hostlist_includes_all_nodes(self):
        """With use_hostlist=true, ALL assigned hostnames appear."""
        config = MPIConfig(backend=MPIBackend.MPICH, use_hostlist=True)
        sys_config = self._make_system_config()

        assignment = self._make_assignment(
            node_ids=["n0", "n1", "n2"],
            hostnames=["node0", "node1", "node2"],
        )
        job_spec = self._make_job_spec(num_nodes=3)

        launcher = build_resource_launcher(config, sys_config, assignment, job_spec)

        assert "node0" in launcher
        assert "node1" in launcher
        assert "node2" in launcher

    def test_hostlist_respects_config_disabled(self):
        """With use_hostlist=false (default), no hostlist flags appear."""
        config = MPIConfig(backend=MPIBackend.MPICH, use_hostlist=False)
        sys_config = self._make_system_config()

        assignment = self._make_assignment(
            node_ids=["n0"],
            hostnames=["node0"],
        )
        job_spec = self._make_job_spec()

        launcher = build_resource_launcher(config, sys_config, assignment, job_spec)

        assert "-host" not in launcher
        assert "-hosts" not in launcher
        assert "node0" not in launcher

    # --- GPU wrapper ---

    def test_no_gpu_wrapper(self):
        """No GPU wrapper even when use_gpu_wrapper=true and original job has GPUs."""
        config = MPIConfig(backend=MPIBackend.MPICH, use_gpu_wrapper=True)
        sys_config = self._make_system_config()
        sys_config.GPU_TYPE = "cuda"

        # Original job has 4 GPUs
        assignment = self._make_assignment(
            node_ids=["n0"],
            hostnames=["node0"],
            gpu_assignments=[{0: [0], 1: [1], 2: [2], 3: [3]}],
        )
        job_spec = self._make_job_spec(ngpus=4)

        launcher = build_resource_launcher(config, sys_config, assignment, job_spec)

        # No GPU wrapper script reference
        assert "gpu_wrapper" not in launcher
        assert "gpu-wrapper" not in launcher
        # Still -n 1
        assert "-n 1" in launcher

    # --- Short hostnames ---

    def test_short_hostnames(self):
        """With use_short_hostnames=true, domains are stripped."""
        config = MPIConfig(
            backend=MPIBackend.MPICH,
            use_hostlist=True,
            use_short_hostnames=True,
        )
        sys_config = self._make_system_config()

        assignment = self._make_assignment(
            node_ids=["n0"],
            hostnames=["node0.example.com"],
        )
        job_spec = self._make_job_spec()

        launcher = build_resource_launcher(config, sys_config, assignment, job_spec)

        assert "node0" in launcher
        assert "example.com" not in launcher

    # --- No CPU cores ---

    def test_no_cpu_cores_no_binding(self):
        """When assignment has no CPU cores, no CPU binding flags appear."""
        config = MPIConfig(backend=MPIBackend.MPICH, cpu_bind_method="list")
        sys_config = self._make_system_config()

        assignment = self._make_assignment(
            node_ids=["n0"],
            hostnames=["node0"],
            cpu_assignments=[{}],  # No CPU assignments
        )
        job_spec = self._make_job_spec()

        launcher = build_resource_launcher(config, sys_config, assignment, job_spec)

        assert "--cpu-bind" not in launcher


class TestUsesMPIAttribute:
    """Tests for the USES_MPI class attribute on app classes."""

    def test_appbase_default_true(self):
        """AppBase.USES_MPI defaults to True."""
        from parslbox.apps.appbase import AppBase
        assert AppBase.USES_MPI is True

    def test_python_app_true(self):
        """PythonApp.USES_MPI is True."""
        from parslbox.apps.python import PythonApp
        assert PythonApp.USES_MPI is True

    def test_vasp_app_true(self):
        """VaspApp.USES_MPI is True (inherited)."""
        from parslbox.apps.vasp import VaspApp
        assert VaspApp.USES_MPI is True

    def test_lammps_app_true(self):
        """LammpsKokkosApp.USES_MPI is True (inherited)."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp
        assert LammpsKokkosApp.USES_MPI is True
