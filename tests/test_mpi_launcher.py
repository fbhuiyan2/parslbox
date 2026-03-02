"""
Unit tests for MPI Command Launcher functionality.

Tests the MPICommandBuilder class and its ability to generate correct MPI commands
with and without overrides in various scenarios.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from parslbox.resource_manager.mpi_launcher import MPICommandBuilder, compose_mpi_command


class TestMPICommandBuilder:
    """Test cases for the MPICommandBuilder class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock assignment object
        self.mock_assignment = Mock()
        self.mock_assignment.hostnames = ["node1", "node2"]
        self.mock_assignment.node_ids = [1, 2]
        self.mock_assignment.job_id = 123
        self.mock_assignment.is_single_node.return_value = False
        self.mock_assignment.get_ranks_for_node.return_value = [0, 1]
        self.mock_assignment.get_cpu_assignments_for_rank.return_value = [0, 1, 2, 3]
        self.mock_assignment.get_gpu_assignments_for_rank.return_value = [0]
        
        # Mock system config
        self.mock_system_config = Mock()
        self.mock_system_config.CORES_PER_NODE = 64
        self.mock_system_config.MPI_CMD_TO_USE = "mpirun"
        self.mock_system_config.EXCLUDE_CORES = []  # Empty list for excluded cores
        
        # Mock job spec
        self.mock_job_spec = Mock()
        self.mock_job_spec.get_total_ranks.return_value = 4
        self.mock_job_spec.ranks_per_node = 2
        self.mock_job_spec.ngpus = 2
        self.mock_job_spec.detect_job_type.return_value = "subnode_cpu"
        self.mock_job_spec.is_gpu_job.return_value = False
    
    def test_init_valid_launcher(self):
        """Test initialization with valid launcher types."""
        for launcher in ['mpirun', 'mpiexec', 'srun']:
            builder = MPICommandBuilder(launcher)
            assert builder.launcher_type == launcher
            assert builder.overrides == {}
    
    def test_init_invalid_launcher(self):
        """Test initialization with invalid launcher type."""
        with pytest.raises(ValueError, match="Invalid launcher type"):
            MPICommandBuilder("invalid_launcher")
    
    def test_init_with_overrides(self):
        """Test initialization with overrides."""
        overrides = {"disable": ["--rankfile"], "add": ["--mca btl ^openib"]}
        builder = MPICommandBuilder("mpirun", overrides)
        assert builder.overrides == overrides
    
    def test_init_with_none_overrides(self):
        """Test initialization with None overrides."""
        builder = MPICommandBuilder("mpirun", None)
        assert builder.overrides == {}
    
    @patch('parslbox.resource_manager.mpi_launcher.generate_openmpi_rankfile')
    def test_build_mpirun_command_no_overrides(self, mock_rankfile):
        """Test building mpirun command without overrides."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        
        builder = MPICommandBuilder("mpirun")
        command = builder.build_command(
            self.mock_assignment, 
            self.mock_system_config, 
            self.mock_job_spec
        )
        
        # Check that the command contains the expected components
        assert "mpirun" in command
        assert "-np 4" in command
        assert "-H node1,node2" in command
        assert "--map-by rankfile:file=" in command
        # Check that rankfile was called and some rankfile path is in command
        mock_rankfile.assert_called_once()
        assert "rankfile" in command
    
    @patch('parslbox.resource_manager.mpi_launcher.generate_openmpi_rankfile')
    def test_build_mpirun_command_with_disable_overrides(self, mock_rankfile):
        """Test building mpirun command with disable overrides."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        
        overrides = {"disable": ["-H", "rankfile"]}
        builder = MPICommandBuilder("mpirun", overrides)
        command = builder.build_command(
            self.mock_assignment, 
            self.mock_system_config, 
            self.mock_job_spec
        )
        
        # Should remove -H flag and its argument, and rankfile-related flags
        expected_command = "mpirun -np 4"
        assert command == expected_command
    
    @patch('parslbox.resource_manager.mpi_launcher.generate_openmpi_rankfile')
    def test_build_mpirun_command_with_add_overrides(self, mock_rankfile):
        """Test building mpirun command with add overrides."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        
        overrides = {"add": ["--mca btl ^openib", "--verbose"]}
        builder = MPICommandBuilder("mpirun", overrides)
        command = builder.build_command(
            self.mock_assignment, 
            self.mock_system_config, 
            self.mock_job_spec
        )
        
        # Should include additional flags
        assert "--mca" in command
        assert "btl" in command
        assert "^openib" in command
        assert "--verbose" in command
    
    @patch('parslbox.resource_manager.mpi_launcher.generate_openmpi_rankfile')
    def test_build_mpirun_command_with_both_overrides(self, mock_rankfile):
        """Test building mpirun command with both disable and add overrides."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        
        overrides = {
            "disable": ["-H"],
            "add": ["--mca btl ^openib"]
        }
        builder = MPICommandBuilder("mpirun", overrides)
        command = builder.build_command(
            self.mock_assignment, 
            self.mock_system_config, 
            self.mock_job_spec
        )
        
        # Should remove -H but keep other flags and add new ones
        assert "-H" not in command
        assert "node1,node2" not in command
        assert "--mca" in command
        assert "btl" in command
        assert "^openib" in command
        assert "-np 4" in command
        assert "rankfile" in command
    
    def test_build_mpirun_fullnode_cpu(self):
        """Test building mpirun command for full-node CPU job."""
        self.mock_job_spec.detect_job_type.return_value = "fullnode_cpu"
        
        builder = MPICommandBuilder("mpirun")
        command = builder.build_command(
            self.mock_assignment, 
            self.mock_system_config, 
            self.mock_job_spec
        )
        
        # Should use core:PE mapping instead of rankfile
        assert "--map-by" in command
        assert "core:PE=" in command
        assert "--bind-to core" in command
        assert "rankfile" not in command
    
    @patch('parslbox.resource_manager.mpi_launcher.generate_openmpi_gpu_wrapper')
    @patch('parslbox.resource_manager.mpi_launcher.generate_openmpi_rankfile')
    def test_build_mpirun_gpu_job(self, mock_rankfile, mock_wrapper):
        """Test building mpirun command for GPU job."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        mock_wrapper.return_value = "/tmp/wrapper.sh"
        self.mock_job_spec.is_gpu_job.return_value = True
        
        builder = MPICommandBuilder("mpirun")
        command = builder.build_command(
            self.mock_assignment, 
            self.mock_system_config, 
            self.mock_job_spec
        )
        
        # Should include wrapper script and rankfile
        mock_wrapper.assert_called_once()
        mock_rankfile.assert_called_once()
        assert "rankfile" in command
        assert "wrapper" in command
    
    def test_build_mpiexec_single_node_fullnode_cpu(self):
        """Test building mpiexec command for single-node full-node CPU job."""
        self.mock_assignment.is_single_node.return_value = True
        self.mock_assignment.hostnames = ["node1"]
        self.mock_job_spec.detect_job_type.return_value = "fullnode_cpu"
        
        builder = MPICommandBuilder("mpiexec")
        command = builder.build_command(
            self.mock_assignment, 
            self.mock_system_config, 
            self.mock_job_spec
        )
        
        # Should use ppn and depth for full-node CPU
        assert "-n 4" in command
        assert "-host node1" in command
        assert "--ppn 2" in command
        assert "--depth" in command
        assert "--cpu-bind depth" in command
    
    def test_build_mpiexec_single_node_subnode(self):
        """Test building mpiexec command for single-node sub-node job."""
        self.mock_assignment.is_single_node.return_value = True
        self.mock_assignment.hostnames = ["node1"]
        self.mock_assignment.get_ranks_for_node.return_value = [0, 1]
        
        builder = MPICommandBuilder("mpiexec")
        command = builder.build_command(
            self.mock_assignment, 
            self.mock_system_config, 
            self.mock_job_spec
        )
        
        # Should use cpu-bind list for sub-node
        assert "-n 4" in command
        assert "-host node1" in command
        assert "--cpu-bind list:" in command
    
    @patch('parslbox.resource_manager.mpi_launcher.generate_mpich_rankfile')
    def test_build_mpiexec_multinode(self, mock_rankfile):
        """Test building mpiexec command for multi-node job."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        
        builder = MPICommandBuilder("mpiexec")
        command = builder.build_command(
            self.mock_assignment, 
            self.mock_system_config, 
            self.mock_job_spec
        )
        
        # Should use rankfile for multi-node
        assert "-n 4" in command
        assert "-hosts node1,node2" in command
        assert "--rankfile" in command
        # Check that rankfile was called
        mock_rankfile.assert_called_once()
    
    def test_build_srun_command(self):
        """Test building srun command."""
        self.mock_job_spec.detect_job_type.return_value = "fullnode_cpu"
        builder = MPICommandBuilder("srun")
        command = builder.build_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )

        # Should include SLURM-specific flags
        assert "srun" in command
        assert "--ntasks 4" in command
        assert "--ntasks-per-node 2" in command
        assert "--nodelist node1,node2" in command
        assert "--nodes 2" in command

    @patch('parslbox.resource_manager.mpi_launcher.generate_srun_gpu_wrapper')
    def test_build_srun_fullnode_cpu(self, mock_wrapper):
        """Test building srun command for fullnode CPU job."""
        self.mock_job_spec.detect_job_type.return_value = "fullnode_cpu"

        builder = MPICommandBuilder("srun")
        command = builder.build_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )

        # Should include CPU binding flags
        assert "--ntasks 4" in command
        assert "--nodes 2" in command
        assert "--ntasks-per-node 2" in command
        assert "--cpus-per-task 32" in command  # 64 cores / 2 ranks_per_node
        assert "--cpu-bind=cores" in command
        # No wrapper for CPU-only job
        mock_wrapper.assert_not_called()
        assert "wrapper" not in command

    @patch('parslbox.resource_manager.mpi_launcher.generate_srun_gpu_wrapper')
    def test_build_srun_subnode_gpu(self, mock_wrapper):
        """Test building srun command for subnode GPU job."""
        mock_wrapper.return_value = "/tmp/gpu_wrapper.sh"
        self.mock_assignment.is_single_node.return_value = True
        self.mock_assignment.hostnames = ["node1"]
        self.mock_assignment.node_ids = [1]
        self.mock_assignment.get_ranks_for_node.return_value = [0, 1]
        self.mock_assignment.get_cpu_assignments_for_rank.return_value = [0, 1, 2, 3]
        self.mock_job_spec.detect_job_type.return_value = "subnode_gpu"
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.get_total_ranks.return_value = 2
        self.mock_job_spec.ngpus = 2

        builder = MPICommandBuilder("srun")
        command = builder.build_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )

        # Should have subnode CPU flags + GPU wrapper
        assert "--ntasks 2" in command
        assert "--nodes 1" in command
        assert "--cpus-per-task 4" in command
        assert "--cpu-bind=cores" in command
        assert "--exact" in command
        mock_wrapper.assert_called_once()
        assert "gpu_wrapper.sh" in command

    @patch('parslbox.resource_manager.mpi_launcher.generate_srun_gpu_wrapper')
    def test_build_srun_fullnode_gpu(self, mock_wrapper):
        """Test building srun command for fullnode GPU job."""
        mock_wrapper.return_value = "/tmp/gpu_wrapper.sh"
        self.mock_job_spec.detect_job_type.return_value = "fullnode_gpu"
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.ngpus = 2
        self.mock_job_spec.ranks_per_node = 2

        builder = MPICommandBuilder("srun")
        command = builder.build_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )

        # Should have fullnode CPU flags + GPU wrapper
        assert "--ntasks 4" in command
        assert "--nodes 2" in command
        assert "--ntasks-per-node 2" in command
        assert "--cpus-per-task 32" in command  # 64 cores / 2 gpus_per_node
        assert "--cpu-bind=cores" in command
        mock_wrapper.assert_called_once()
        assert "gpu_wrapper.sh" in command
        # No --exact for fullnode
        assert "--exact" not in command

    @patch('parslbox.resource_manager.mpi_launcher.generate_srun_gpu_wrapper')
    def test_build_srun_multinode_gpu(self, mock_wrapper):
        """Test building srun command for multinode GPU job."""
        mock_wrapper.return_value = "/tmp/gpu_wrapper.sh"
        self.mock_assignment.hostnames = ["node1", "node2"]
        self.mock_assignment.node_ids = [1, 2]
        self.mock_job_spec.detect_job_type.return_value = "fullnode_gpu"
        self.mock_job_spec.is_gpu_job.return_value = True
        self.mock_job_spec.get_total_ranks.return_value = 8
        self.mock_job_spec.ngpus = 4
        self.mock_job_spec.ranks_per_node = 4

        builder = MPICommandBuilder("srun")
        command = builder.build_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )

        # Should have multinode flags
        assert "--ntasks 8" in command
        assert "--nodes 2" in command
        assert "--ntasks-per-node 4" in command
        assert "--cpus-per-task 16" in command  # 64 cores / 4 gpus_per_node
        assert "--cpu-bind=cores" in command
        mock_wrapper.assert_called_once()
        assert "gpu_wrapper.sh" in command
    
    def test_filter_disabled_flags_exact_match(self):
        """Test filtering flags with exact flag matching."""
        builder = MPICommandBuilder("mpirun")
        flags = ["-np", "4", "-H", "node1,node2", "--verbose"]
        disable_list = ["-H"]
        
        result = builder._filter_disabled_flags(flags, disable_list)
        
        # Should remove -H and its argument
        expected = ["-np", "4", "--verbose"]
        assert result == expected
    
    def test_filter_disabled_flags_substring_match(self):
        """Test filtering flags with substring matching."""
        builder = MPICommandBuilder("mpirun")
        flags = ["-np", "4", "--map-by", "rankfile:file=/tmp/file", "--verbose"]
        disable_list = ["rankfile"]
        
        result = builder._filter_disabled_flags(flags, disable_list)
        
        # Should remove --map-by and its rankfile argument
        expected = ["-np", "4", "--verbose"]
        assert result == expected
    
    def test_filter_disabled_flags_multiple_rules(self):
        """Test filtering flags with multiple disable rules."""
        builder = MPICommandBuilder("mpirun")
        flags = ["-np", "4", "-H", "node1", "--map-by", "rankfile:file=/tmp/file", "--verbose"]
        disable_list = ["-H", "rankfile"]
        
        result = builder._filter_disabled_flags(flags, disable_list)
        
        # Should remove both -H and rankfile-related flags
        expected = ["-np", "4", "--verbose"]
        assert result == expected
    
    def test_filter_disabled_flags_no_matches(self):
        """Test filtering flags with no matches."""
        builder = MPICommandBuilder("mpirun")
        flags = ["-np", "4", "--verbose"]
        disable_list = ["--nonexistent"]
        
        result = builder._filter_disabled_flags(flags, disable_list)
        
        # Should return original flags
        assert result == flags
    
    def test_filter_disabled_flags_empty_list(self):
        """Test filtering flags with empty disable list."""
        builder = MPICommandBuilder("mpirun")
        flags = ["-np", "4", "--verbose"]
        disable_list = []
        
        result = builder._filter_disabled_flags(flags, disable_list)
        
        # Should return original flags
        assert result == flags
    
    def test_apply_overrides_no_overrides(self):
        """Test applying overrides when no overrides are specified."""
        builder = MPICommandBuilder("mpirun")
        flags = ["-np", "4", "--verbose"]
        
        result = builder._apply_overrides(flags)
        
        # Should return original flags
        assert result == flags
    
    def test_apply_overrides_only_add(self):
        """Test applying overrides with only add rules."""
        overrides = {"add": ["--mca btl ^openib"]}
        builder = MPICommandBuilder("mpirun", overrides)
        flags = ["-np", "4"]
        
        result = builder._apply_overrides(flags)
        
        # Should add new flags
        expected = ["-np", "4", "--mca", "btl", "^openib"]
        assert result == expected
    
    def test_apply_overrides_multiword_add(self):
        """Test applying overrides with multi-word add flags."""
        overrides = {"add": ["--bind-to core", "--verbose"]}
        builder = MPICommandBuilder("mpirun", overrides)
        flags = ["-np", "4"]
        
        result = builder._apply_overrides(flags)
        
        # Should split multi-word flags correctly
        expected = ["-np", "4", "--bind-to", "core", "--verbose"]
        assert result == expected


class TestComposeMPICommand:
    """Test cases for the compose_mpi_command function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock assignment object
        self.mock_assignment = Mock()
        self.mock_assignment.job_id = 123
        
        # Mock system config
        self.mock_system_config = Mock()
        self.mock_system_config.MPI_CMD_TO_USE = "mpirun"
        
        # Mock job spec
        self.mock_job_spec = Mock()
    
    @patch('parslbox.resource_manager.mpi_launcher.MPICommandBuilder')
    def test_compose_mpi_command_no_overrides(self, mock_builder_class):
        """Test compose_mpi_command without overrides."""
        mock_builder = Mock()
        mock_builder.build_command.return_value = "mpirun -np 4 -H node1"
        mock_builder_class.return_value = mock_builder
        
        result = compose_mpi_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )
        
        # Should create builder with no overrides
        mock_builder_class.assert_called_once_with("mpirun", None)
        
        # Should return correct dictionary
        expected = {
            "PBX_MPIRUN_PREFIX": "mpirun -np 4 -H node1",
            "PBX_MPI_PREFIX": "mpirun -np 4 -H node1"
        }
        assert result == expected
    
    @patch('parslbox.resource_manager.mpi_launcher.MPICommandBuilder')
    def test_compose_mpi_command_with_overrides(self, mock_builder_class):
        """Test compose_mpi_command with overrides."""
        mock_builder = Mock()
        mock_builder.build_command.return_value = "mpirun -np 4 --mca btl ^openib"
        mock_builder_class.return_value = mock_builder
        
        overrides = {"disable": ["-H"], "add": ["--mca btl ^openib"]}
        result = compose_mpi_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec,
            mpi_overrides=overrides
        )
        
        # Should create builder with overrides
        mock_builder_class.assert_called_once_with("mpirun", overrides)
        
        # Should return correct dictionary
        expected = {
            "PBX_MPIRUN_PREFIX": "mpirun -np 4 --mca btl ^openib",
            "PBX_MPI_PREFIX": "mpirun -np 4 --mca btl ^openib"
        }
        assert result == expected
    
    @patch('parslbox.resource_manager.mpi_launcher.MPICommandBuilder')
    def test_compose_mpi_command_mpiexec(self, mock_builder_class):
        """Test compose_mpi_command with mpiexec launcher."""
        mock_builder = Mock()
        mock_builder.build_command.return_value = "mpiexec -n 4 -host node1"
        mock_builder_class.return_value = mock_builder
        
        self.mock_system_config.MPI_CMD_TO_USE = "mpiexec"
        
        result = compose_mpi_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )
        
        # Should create builder with mpiexec
        mock_builder_class.assert_called_once_with("mpiexec", None)
        
        # Should return correct dictionary with mpiexec key
        expected = {
            "PBX_MPIEXEC_PREFIX": "mpiexec -n 4 -host node1",
            "PBX_MPI_PREFIX": "mpiexec -n 4 -host node1"
        }
        assert result == expected
    
    @patch('parslbox.resource_manager.mpi_launcher.MPICommandBuilder')
    def test_compose_mpi_command_srun(self, mock_builder_class):
        """Test compose_mpi_command with srun launcher."""
        mock_builder = Mock()
        mock_builder.build_command.return_value = "srun --ntasks 4 --nodes 1"
        mock_builder_class.return_value = mock_builder
        
        self.mock_system_config.MPI_CMD_TO_USE = "srun"
        
        result = compose_mpi_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )
        
        # Should create builder with srun
        mock_builder_class.assert_called_once_with("srun", None)
        
        # Should return correct dictionary with srun key
        expected = {
            "PBX_SRUN_PREFIX": "srun --ntasks 4 --nodes 1",
            "PBX_MPI_PREFIX": "srun --ntasks 4 --nodes 1"
        }
        assert result == expected


class TestIntegrationScenarios:
    """Integration test scenarios for real-world use cases."""
    
    def setup_method(self):
        """Set up test fixtures for integration tests."""
        # Create more realistic mock objects
        self.mock_assignment = Mock()
        self.mock_assignment.hostnames = ["node1", "node2"]
        self.mock_assignment.node_ids = [1, 2]
        self.mock_assignment.job_id = 123
        self.mock_assignment.is_single_node.return_value = False
        self.mock_assignment.get_ranks_for_node.side_effect = lambda idx: [idx*2, idx*2+1]
        self.mock_assignment.get_cpu_assignments_for_rank.side_effect = lambda rank: [rank*4, rank*4+1, rank*4+2, rank*4+3]
        self.mock_assignment.get_gpu_assignments_for_rank.side_effect = lambda rank: [rank % 4]
        
        self.mock_system_config = Mock()
        self.mock_system_config.CORES_PER_NODE = 64
        self.mock_system_config.MPI_CMD_TO_USE = "mpirun"
        
        self.mock_job_spec = Mock()
        self.mock_job_spec.get_total_ranks.return_value = 4
        self.mock_job_spec.ranks_per_node = 2
        self.mock_job_spec.ngpus = 2
        self.mock_job_spec.detect_job_type.return_value = "subnode_cpu"
        self.mock_job_spec.is_gpu_job.return_value = False
    
    @patch('parslbox.resource_manager.mpi_launcher.generate_openmpi_rankfile')
    def test_vasp_sophia_scenario(self, mock_rankfile):
        """Test the VASP on Sophia scenario that motivated this feature."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        
        # Simulate VASP on Sophia configuration
        vasp_sophia_overrides = {
            "disable": ["rankfile", "-H"]
        }
        
        builder = MPICommandBuilder("mpirun", vasp_sophia_overrides)
        command = builder.build_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )
        
        # Should remove problematic flags for VASP on Sophia
        assert "rankfile" not in command
        assert "-H" not in command
        assert "node1,node2" not in command
        assert "-np 4" in command  # Should keep essential flags
    
    @patch('parslbox.resource_manager.helpers.mpi_launcher_helpers.generate_openmpi_rankfile')
    def test_custom_mpi_tuning_scenario(self, mock_rankfile):
        """Test scenario with custom MPI tuning flags."""
        mock_rankfile.return_value = "/tmp/rankfile.txt"
        
        # Simulate custom MPI tuning
        custom_overrides = {
            "disable": ["--bind-to"],
            "add": ["--mca btl ^openib", "--mca pml ob1", "--verbose"]
        }
        
        builder = MPICommandBuilder("mpirun", custom_overrides)
        command = builder.build_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )
        
        # Should remove binding and add custom MCA parameters
        assert "--bind-to" not in command
        assert "--mca btl ^openib" in command
        assert "--mca pml ob1" in command
        assert "--verbose" in command
    
    def test_no_overrides_backward_compatibility(self):
        """Test that the system works without any overrides (backward compatibility)."""
        # Test with no overrides - should work exactly as before
        result = compose_mpi_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec
        )
        
        # Should return valid MPI command dictionary
        assert "PBX_MPIRUN_PREFIX" in result
        assert "PBX_MPI_PREFIX" in result
        assert result["PBX_MPIRUN_PREFIX"] == result["PBX_MPI_PREFIX"]
        assert "mpirun" in result["PBX_MPI_PREFIX"]
    
    def test_empty_overrides_dict(self):
        """Test with empty overrides dictionary."""
        empty_overrides = {}
        
        result = compose_mpi_command(
            self.mock_assignment,
            self.mock_system_config,
            self.mock_job_spec,
            mpi_overrides=empty_overrides
        )
        
        # Should work the same as no overrides
        assert "PBX_MPIRUN_PREFIX" in result
        assert "PBX_MPI_PREFIX" in result
        assert "mpirun" in result["PBX_MPI_PREFIX"]


if __name__ == "__main__":
    pytest.main([__file__])
