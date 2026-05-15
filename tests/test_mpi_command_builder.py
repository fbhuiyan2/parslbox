"""
Unit tests for MPICommandBuilder and build_mpi_command.

Tests the simplified MPI command builder's ability to generate correct MPI commands
across different backends (MPICH, OpenMPI, SRUN) with various configurations.
"""

import pytest
from unittest.mock import Mock, patch
from parslbox.resource_manager.mpi_command_builder import MPICommandBuilder, build_mpi_command
from parslbox.resource_manager.mpi_config import MPIConfig, MPIBackend


class TestMPICommandBuilder:
    """Test cases for the MPICommandBuilder class."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock assignment object
        self.mock_assignment = Mock()
        self.mock_assignment.hostnames = ["node1", "node2"]
        self.mock_assignment.node_ids = [1, 2]
        self.mock_assignment.job_id = 123
        self.mock_assignment.get_ranks_for_node.return_value = [0, 1]
        self.mock_assignment.get_cpu_assignments_for_rank.return_value = [0, 1, 2, 3]
        self.mock_assignment.get_gpu_assignments_for_rank.return_value = [0]

        # Mock job spec
        self.mock_job_spec = Mock()
        self.mock_job_spec.get_total_ranks.return_value = 4
        self.mock_job_spec.ranks_per_node = 2
        self.mock_job_spec.ngpus = 4
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.is_gpu_job.return_value = False

        # Mock system config
        self.mock_system_config = Mock()
        self.mock_system_config.CORES_PER_NODE = 64
        self.mock_system_config.GPUS_PER_NODE = 4
        self.mock_system_config.EXCLUDE_CORES = []
        self.mock_system_config.DRAM_PER_NODE = 256

        # Default MPI config (MPICH, minimal)
        self.mpi_config = MPIConfig(backend=MPIBackend.MPICH)

    # --- Initialization tests ---

    def test_init(self):
        """Test builder stores config and system_config."""
        builder = MPICommandBuilder(self.mpi_config, self.mock_system_config)
        assert builder.config is self.mpi_config
        assert builder.system_config is self.mock_system_config

    # --- _calculate_ranks_per_node tests ---

    def test_ranks_per_node_single_node_gpu(self):
        """Single-node GPU job: ngpus=4, num_nodes=1 → returns 4."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 4
        self.mock_job_spec.num_nodes = 1

        builder = MPICommandBuilder(self.mpi_config, self.mock_system_config)
        result = builder._calculate_ranks_per_node(self.mock_job_spec, self.mock_assignment)
        assert result == 4

    def test_ranks_per_node_multinode_gpu(self):
        """Multi-node GPU job: ngpus=1200, num_nodes=100 → returns 12 (THE BUG FIX)."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 1200
        self.mock_job_spec.num_nodes = 100

        builder = MPICommandBuilder(self.mpi_config, self.mock_system_config)
        result = builder._calculate_ranks_per_node(self.mock_job_spec, self.mock_assignment)
        assert result == 12

    def test_ranks_per_node_cpu_job(self):
        """CPU job: ranks_per_node=8, is_gpu_job=False → returns 8."""
        self.mock_job_spec.is_gpu_job.return_value = False
        self.mock_job_spec.ranks_per_node = 8

        builder = MPICommandBuilder(self.mpi_config, self.mock_system_config)
        result = builder._calculate_ranks_per_node(self.mock_job_spec, self.mock_assignment)
        assert result == 8

    # --- MPICH backend tests (--ppn) ---

    def test_build_mpich_single_node_gpu(self):
        """MPICH single-node GPU: assert -n 4 and --ppn 4."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 4
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.get_total_ranks.return_value = 4

        config = MPIConfig(backend=MPIBackend.MPICH)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-n 4" in command
        assert "--ppn 4" in command

    def test_build_mpich_multinode_gpu(self):
        """MPICH multi-node GPU: assert -n 1200 and --ppn 12 (not 1200)."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 1200
        self.mock_job_spec.num_nodes = 100
        self.mock_job_spec.get_total_ranks.return_value = 1200
        self.mock_assignment.hostnames = [f"node{i}" for i in range(100)]

        config = MPIConfig(backend=MPIBackend.MPICH)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-n 1200" in command
        assert "--ppn 12" in command
        assert "--ppn 1200" not in command

    def test_build_mpich_single_node_cpu(self):
        """MPICH single-node CPU: assert -n 8 and --ppn 8."""
        self.mock_job_spec.is_gpu_job.return_value = False
        self.mock_job_spec.ranks_per_node = 8
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.get_total_ranks.return_value = 8

        config = MPIConfig(backend=MPIBackend.MPICH)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-n 8" in command
        assert "--ppn 8" in command

    def test_build_mpich_multinode_cpu(self):
        """MPICH multi-node CPU: assert correct -n and --ppn."""
        self.mock_job_spec.is_gpu_job.return_value = False
        self.mock_job_spec.ranks_per_node = 32
        self.mock_job_spec.num_nodes = 2
        self.mock_job_spec.get_total_ranks.return_value = 64

        config = MPIConfig(backend=MPIBackend.MPICH)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-n 64" in command
        assert "--ppn 32" in command

    # --- OpenMPI backend tests (ppr:N:node) ---

    def test_build_openmpi_single_node_gpu(self):
        """OpenMPI single-node GPU: assert -np N and ppr:N:node."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 4
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.get_total_ranks.return_value = 4

        config = MPIConfig(backend=MPIBackend.OPENMPI)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-np 4" in command
        assert "ppr:4:node" in command

    def test_build_openmpi_multinode_gpu(self):
        """OpenMPI multi-node GPU: assert correct per-node ppr value."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 1200
        self.mock_job_spec.num_nodes = 100
        self.mock_job_spec.get_total_ranks.return_value = 1200
        self.mock_assignment.hostnames = [f"node{i}" for i in range(100)]

        config = MPIConfig(backend=MPIBackend.OPENMPI)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-np 1200" in command
        assert "ppr:12:node" in command
        assert "ppr:1200:node" not in command

    def test_build_openmpi_cpu(self):
        """OpenMPI CPU job: assert correct ppr value."""
        self.mock_job_spec.is_gpu_job.return_value = False
        self.mock_job_spec.ranks_per_node = 8
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.get_total_ranks.return_value = 8

        config = MPIConfig(backend=MPIBackend.OPENMPI)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-np 8" in command
        assert "ppr:8:node" in command

    # --- SRUN backend tests (--ntasks-per-node) ---

    def test_build_srun_single_node_gpu(self):
        """SRUN single-node GPU: assert -n N and --ntasks-per-node N."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 4
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.get_total_ranks.return_value = 4

        config = MPIConfig(backend=MPIBackend.SRUN)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-n 4" in command
        assert "--ntasks-per-node 4" in command

    def test_build_srun_multinode_gpu(self):
        """SRUN multi-node GPU: assert correct per-node ntasks value."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 1200
        self.mock_job_spec.num_nodes = 100
        self.mock_job_spec.get_total_ranks.return_value = 1200
        self.mock_assignment.hostnames = [f"node{i}" for i in range(100)]

        config = MPIConfig(backend=MPIBackend.SRUN)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-n 1200" in command
        assert "--ntasks-per-node 12" in command
        assert "--ntasks-per-node 1200" not in command

    # --- SRUN GPU gres tests ---

    def test_srun_subnode_gpu_job_native_flags(self):
        """SRUN sub-node GPU job: --gpus-per-task, --mem-per-gpu, --exact, -u."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 2
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.get_total_ranks.return_value = 2
        self.mock_job_spec.detect_job_type.return_value = "subnode_gpu"

        config = MPIConfig(backend=MPIBackend.SRUN)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--gpus-per-task=1" in command
        assert "--mem-per-gpu=64G" in command  # 256 // 4
        assert "--exact" in command
        assert " -u" in command
        assert "--gres" not in command
        assert "--gpu-bind=none" not in command

    def test_srun_fullnode_gpu_job_native_flags(self):
        """SRUN full-node GPU job: --gpus-per-node and --gpu-bind=map_gpu."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 4
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.get_total_ranks.return_value = 4
        self.mock_job_spec.detect_job_type.return_value = "fullnode_gpu"

        config = MPIConfig(backend=MPIBackend.SRUN)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--gpus-per-node=4" in command
        assert "--gpu-bind=map_gpu:0,1,2,3" in command
        assert "--exact" not in command
        assert "--gpus-per-task" not in command

    def test_srun_multinode_gpu_job_native_flags(self):
        """SRUN multi-node GPU job: --gpus-per-node and --gpu-bind=map_gpu."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 8
        self.mock_job_spec.num_nodes = 2
        self.mock_job_spec.get_total_ranks.return_value = 8
        self.mock_job_spec.detect_job_type.return_value = "multinode_gpu"

        config = MPIConfig(backend=MPIBackend.SRUN)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--gpus-per-node=4" in command
        assert "--gpu-bind=map_gpu:0,1,2,3" in command
        assert "--exact" not in command

    def test_srun_cpu_job_no_gpu_flags(self):
        """SRUN CPU job: no GPU-related flags."""
        self.mock_job_spec.is_gpu_job.return_value = False

        config = MPIConfig(backend=MPIBackend.SRUN)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--gres" not in command
        assert "--gpu-bind" not in command
        assert "--gpus-per-node" not in command
        assert "--gpus-per-task" not in command

    def test_openmpi_gpu_job_no_slurm_gpu_flags(self):
        """OpenMPI GPU job: no SLURM GPU flags."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 2
        self.mock_job_spec.get_total_ranks.return_value = 2

        config = MPIConfig(backend=MPIBackend.OPENMPI)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--gres" not in command
        assert "--gpu-bind" not in command
        assert "--gpus-per-node" not in command

    # --- Hostlist tests ---

    def test_hostlist_mpich(self):
        """MPICH with use_hostlist=True: assert -hosts node1,node2."""
        config = MPIConfig(backend=MPIBackend.MPICH, use_hostlist=True)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-hosts node1,node2" in command

    def test_hostlist_openmpi(self):
        """OpenMPI with use_hostlist=True: assert -H node1,node2."""
        config = MPIConfig(backend=MPIBackend.OPENMPI, use_hostlist=True)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-H node1,node2" in command

    def test_hostlist_srun(self):
        """SRUN with use_hostlist=True: assert --nodelist node1,node2."""
        config = MPIConfig(backend=MPIBackend.SRUN, use_hostlist=True)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--nodelist node1,node2" in command

    def test_short_hostnames(self):
        """With use_short_hostnames=True: assert domain stripped."""
        self.mock_assignment.hostnames = ["node1.example.com", "node2.example.com"]
        config = MPIConfig(backend=MPIBackend.MPICH, use_hostlist=True, use_short_hostnames=True)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-hosts node1,node2" in command
        assert "example.com" not in command

    # --- CPU binding tests ---

    def test_cpu_bind_none(self):
        """cpu_bind_method=none: no binding flags added."""
        config = MPIConfig(backend=MPIBackend.MPICH, cpu_bind_method="none")
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--cpu-bind" not in command
        assert "--rankfile" not in command
        assert "--map-by" not in command

    @patch('parslbox.resource_manager.mpi_command_builder.generate_mpich_rankfile')
    def test_cpu_bind_rankfile_mpich(self, mock_rankfile):
        """MPICH rankfile binding: assert --rankfile in output."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        config = MPIConfig(backend=MPIBackend.MPICH, cpu_bind_method="rankfile")
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--rankfile" in command
        assert "/tmp/rankfile.txt" in command

    @patch('parslbox.resource_manager.mpi_command_builder.generate_openmpi_rankfile')
    def test_cpu_bind_rankfile_openmpi(self, mock_rankfile):
        """OpenMPI rankfile binding: assert --map-by rankfile:file= in output."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        config = MPIConfig(backend=MPIBackend.OPENMPI, cpu_bind_method="rankfile")
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--map-by rankfile:file=" in command

    def test_cpu_bind_list_mpich(self):
        """MPICH list binding: assert --cpu-bind list: in output."""
        config = MPIConfig(backend=MPIBackend.MPICH, cpu_bind_method="list")
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--cpu-bind list:" in command

    def test_cpu_bind_depth_mpich(self):
        """MPICH depth binding: assert --cpu-bind depth --depth N."""
        config = MPIConfig(backend=MPIBackend.MPICH, cpu_bind_method="depth")
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--cpu-bind depth" in command
        assert "--depth" in command

    def test_cpu_bind_depth_openmpi(self):
        """OpenMPI depth binding: assert --map-by core:PE=N --bind-to core."""
        config = MPIConfig(backend=MPIBackend.OPENMPI, cpu_bind_method="depth")
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--map-by core:PE=" in command
        assert "--bind-to core" in command

    def test_cpu_bind_cores_srun(self):
        """SRUN cores binding: --cpus-per-task N --cpu-bind=cores."""
        config = MPIConfig(backend=MPIBackend.SRUN, cpu_bind_method="cores")
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--cpus-per-task" in command
        assert "--cpu-bind=cores" in command

    def test_cpu_bind_threads_srun(self):
        """SRUN threads binding: --cpus-per-task N --cpu-bind=threads."""
        config = MPIConfig(backend=MPIBackend.SRUN, cpu_bind_method="threads")
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--cpus-per-task" in command
        assert "--cpu-bind=threads" in command

    # --- GPU wrapper tests ---

    @patch('parslbox.resource_manager.mpi_command_builder.generate_mpich_gpu_wrapper')
    def test_gpu_wrapper_enabled(self, mock_wrapper):
        """With use_gpu_wrapper=True and GPU job, wrapper in command."""
        mock_wrapper.return_value = "/tmp/gpu_wrapper.sh"
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 4
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.get_total_ranks.return_value = 4

        config = MPIConfig(backend=MPIBackend.MPICH, use_gpu_wrapper=True)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "gpu_wrapper.sh" in command
        mock_wrapper.assert_called_once()

    @patch('parslbox.resource_manager.mpi_command_builder.generate_mpich_gpu_wrapper')
    def test_gpu_wrapper_disabled_via_config(self, mock_wrapper):
        """With 'gpu-wrapper' in disable list, no wrapper."""
        mock_wrapper.return_value = "/tmp/gpu_wrapper.sh"
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 4
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.get_total_ranks.return_value = 4

        config = MPIConfig(backend=MPIBackend.MPICH, use_gpu_wrapper=True, disable=["gpu-wrapper"])
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "gpu_wrapper" not in command

    def test_gpu_wrapper_not_added_for_cpu_job(self):
        """CPU job gets no wrapper even with use_gpu_wrapper=True."""
        self.mock_job_spec.is_gpu_job.return_value = False

        config = MPIConfig(backend=MPIBackend.MPICH, use_gpu_wrapper=True)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "wrapper" not in command

    def test_srun_skips_gpu_wrapper(self):
        """SRUN backend skips GPU wrapper — uses native SLURM GPU binding."""
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 4
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.get_total_ranks.return_value = 4
        self.mock_job_spec.detect_job_type.return_value = "fullnode_gpu"

        config = MPIConfig(backend=MPIBackend.SRUN, use_gpu_wrapper=True)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "wrapper" not in command
        assert "--gpus-per-node=4" in command

    # --- Disable/Add override tests ---

    def test_disable_exact_flag(self):
        """Disable ['-hosts'] removes flag + value."""
        config = MPIConfig(backend=MPIBackend.MPICH, use_hostlist=True, disable=["-hosts"])
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-hosts" not in command
        assert "node1,node2" not in command

    @patch('parslbox.resource_manager.mpi_command_builder.generate_mpich_rankfile')
    def test_disable_substring(self, mock_rankfile):
        """Disable ['rankfile'] removes matching flags."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        config = MPIConfig(backend=MPIBackend.MPICH, cpu_bind_method="rankfile", disable=["rankfile"])
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "rankfile" not in command

    @patch('parslbox.resource_manager.mpi_command_builder.generate_mpich_rankfile')
    def test_disable_multiple_rules(self, mock_rankfile):
        """Multiple disable rules work together."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        config = MPIConfig(
            backend=MPIBackend.MPICH,
            use_hostlist=True,
            cpu_bind_method="rankfile",
            disable=["-hosts", "rankfile"]
        )
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-hosts" not in command
        assert "rankfile" not in command

    def test_add_flags(self):
        """Add ['--mca btl ^openib'] appends flags."""
        config = MPIConfig(backend=MPIBackend.MPICH, add=["--mca btl ^openib"])
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "--mca" in command
        assert "btl" in command
        assert "^openib" in command

    def test_add_with_template(self):
        """Add ['-hosts {hostlist}'] substitutes template."""
        config = MPIConfig(backend=MPIBackend.MPICH, add=["-hosts {hostlist}"])
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-hosts node1,node2" in command

    @patch('parslbox.resource_manager.mpi_command_builder.generate_mpich_rankfile')
    def test_disable_and_add_combined(self, mock_rankfile):
        """Both disable and add work together."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        config = MPIConfig(
            backend=MPIBackend.MPICH,
            use_hostlist=True,
            cpu_bind_method="rankfile",
            disable=["-hosts", "rankfile"],
            add=["--verbose"]
        )
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(self.mock_assignment, self.mock_job_spec)

        assert "-hosts" not in command
        assert "rankfile" not in command
        assert "--verbose" in command


class TestBuildMPICommandFunction:
    """Tests for the convenience build_mpi_command() function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_assignment = Mock()
        self.mock_assignment.hostnames = ["node1"]
        self.mock_assignment.node_ids = [1]
        self.mock_assignment.get_ranks_for_node.return_value = [0, 1]
        self.mock_assignment.get_cpu_assignments_for_rank.return_value = [0, 1]

        self.mock_job_spec = Mock()
        self.mock_job_spec.get_total_ranks.return_value = 4
        self.mock_job_spec.ranks_per_node = 4
        self.mock_job_spec.ngpus = 0
        self.mock_job_spec.num_nodes = 1
        self.mock_job_spec.is_gpu_job.return_value = False

        self.mock_system_config = Mock()
        self.mock_system_config.CORES_PER_NODE = 64
        self.mock_system_config.EXCLUDE_CORES = []

    def test_returns_dict_with_prefix_keys(self):
        """Returns dict with PBX_MPI_PREFIX and backend-specific key."""
        config = MPIConfig(backend=MPIBackend.MPICH)
        result = build_mpi_command(config, self.mock_system_config, self.mock_assignment, self.mock_job_spec)

        assert "PBX_MPI_PREFIX" in result
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_mpich_key_name(self):
        """Returns PBX_MPIEXEC_PREFIX for MPICH backend."""
        config = MPIConfig(backend=MPIBackend.MPICH)
        result = build_mpi_command(config, self.mock_system_config, self.mock_assignment, self.mock_job_spec)

        assert "PBX_MPIEXEC_PREFIX" in result
        assert result["PBX_MPIEXEC_PREFIX"] == result["PBX_MPI_PREFIX"]

    def test_openmpi_key_name(self):
        """Returns PBX_MPIRUN_PREFIX for OpenMPI backend."""
        config = MPIConfig(backend=MPIBackend.OPENMPI)
        result = build_mpi_command(config, self.mock_system_config, self.mock_assignment, self.mock_job_spec)

        assert "PBX_MPIRUN_PREFIX" in result
        assert result["PBX_MPIRUN_PREFIX"] == result["PBX_MPI_PREFIX"]

    def test_srun_key_name(self):
        """Returns PBX_SRUN_PREFIX for SRUN backend."""
        config = MPIConfig(backend=MPIBackend.SRUN)
        result = build_mpi_command(config, self.mock_system_config, self.mock_assignment, self.mock_job_spec)

        assert "PBX_SRUN_PREFIX" in result
        assert result["PBX_SRUN_PREFIX"] == result["PBX_MPI_PREFIX"]


class TestIntegrationScenarios:
    """Real-world integration scenarios."""

    def setup_method(self):
        """Set up test fixtures for integration tests."""
        self.mock_system_config = Mock()
        self.mock_system_config.CORES_PER_NODE = 52
        self.mock_system_config.EXCLUDE_CORES = []

    def test_aurora_100_node_gpu(self):
        """Aurora: 100 nodes, 12 GPUs/node, MPICH → --ppn 12."""
        assignment = Mock()
        assignment.hostnames = [f"x4712c{i}s{j}b0n0" for i in range(25) for j in range(4)]
        assignment.node_ids = list(range(100))
        assignment.get_ranks_for_node.return_value = list(range(12))
        assignment.get_cpu_assignments_for_rank.return_value = [0, 1, 2, 3]

        job_spec = Mock()
        job_spec.get_total_ranks.return_value = 1200
        job_spec.ngpus = 1200
        job_spec.num_nodes = 100
        job_spec.ranks_per_node = 12
        job_spec.is_gpu_job.return_value = True

        config = MPIConfig(backend=MPIBackend.MPICH)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(assignment, job_spec)

        assert "-n 1200" in command
        assert "--ppn 12" in command
        assert "--ppn 1200" not in command

    def test_polaris_single_node_4gpu(self):
        """Polaris: 1 node, 4 GPUs, MPICH → --ppn 4."""
        assignment = Mock()
        assignment.hostnames = ["x3005c0s7b0n0"]
        assignment.node_ids = [0]
        assignment.get_ranks_for_node.return_value = [0, 1, 2, 3]
        assignment.get_cpu_assignments_for_rank.return_value = [0, 1, 2, 3]

        job_spec = Mock()
        job_spec.get_total_ranks.return_value = 4
        job_spec.ngpus = 4
        job_spec.num_nodes = 1
        job_spec.ranks_per_node = 4
        job_spec.is_gpu_job.return_value = True

        config = MPIConfig(backend=MPIBackend.MPICH)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(assignment, job_spec)

        assert "-n 4" in command
        assert "--ppn 4" in command

    def test_polaris_multinode_cpu(self):
        """Polaris: 2 nodes, 32 ranks/node, MPICH → --ppn 32."""
        assignment = Mock()
        assignment.hostnames = ["x3005c0s7b0n0", "x3005c0s7b1n0"]
        assignment.node_ids = [0, 1]
        assignment.get_ranks_for_node.return_value = list(range(32))
        assignment.get_cpu_assignments_for_rank.return_value = [0, 1]

        job_spec = Mock()
        job_spec.get_total_ranks.return_value = 64
        job_spec.ngpus = 0
        job_spec.num_nodes = 2
        job_spec.ranks_per_node = 32
        job_spec.is_gpu_job.return_value = False

        config = MPIConfig(backend=MPIBackend.MPICH)
        builder = MPICommandBuilder(config, self.mock_system_config)
        command = builder.build_command(assignment, job_spec)

        assert "-n 64" in command
        assert "--ppn 32" in command


if __name__ == "__main__":
    pytest.main([__file__])
