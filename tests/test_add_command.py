import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from parslbox.commands.add import app as add_app
from parslbox.commands.helpers.job_info_validator import validate_resource_parameters
from parslbox.database import database


class TestValidateResourceParameters:
    """Unit tests for validate_resource_parameters — ranks_per_node + node_occupancy."""

    @pytest.fixture
    def sys_config(self):
        mock = MagicMock()
        mock.GPUS_PER_NODE = 4
        mock.CORES_PER_NODE = 64
        mock.EXCLUDE_CORES = None
        return mock

    @pytest.fixture
    def sys_config_excluded(self):
        mock = MagicMock()
        mock.GPUS_PER_NODE = 4
        mock.CORES_PER_NODE = 64
        mock.EXCLUDE_CORES = [0, 1, 2, 3]
        return mock

    def test_lammps_cpu_full_node(self, sys_config):
        """LAMMPS CPU job, full node → ranks = CORES_PER_NODE."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=1.0,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 64

    def test_lammps_cpu_half_node(self, sys_config):
        """LAMMPS CPU job, nocc=0.5 → ranks = 32."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.5,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 32

    def test_lammps_cpu_quarter_node(self, sys_config):
        """LAMMPS CPU job, nocc=0.25 → ranks = 16."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.25,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 16

    def test_lammps_cpu_half_node_with_excluded(self, sys_config_excluded):
        """LAMMPS CPU, nocc=0.5, 4 excluded cores → ranks = (64-4)*0.5 = 30."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.5,
            system_config=sys_config_excluded, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 30

    def test_python_cpu_half_node(self, sys_config):
        """Python CPU job, nocc=0.5 → ranks stays 1 (Python override)."""
        from parslbox.apps.python import PythonApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.5,
            system_config=sys_config, app_class=PythonApp,
        )
        assert params['final_ranks_per_node'] == 1

    def test_lammps_gpu_job_ignores_occupancy(self, sys_config):
        """GPU jobs should not scale ranks by occupancy."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=2, nnodes=1,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 2

    def test_no_app_class_cpu_half_node(self, sys_config):
        """Fallback (no app class), nocc=0.5 → ranks = 32."""
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.5,
            system_config=sys_config, app_class=None,
        )
        assert params['final_ranks_per_node'] == 32

    def test_user_override_not_scaled(self, sys_config):
        """User-specified ranks_per_node should NOT be scaled by occupancy."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.5, ranks_per_node=8,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 8

    # --- Multi-node on GPU system: CPU vs GPU ---

    def test_multinode_gpu_system_no_flags(self, sys_config):
        """Multi-node on GPU system, no -g or -o → auto-assign all GPUs."""
        params, info, _ = validate_resource_parameters(
            ngpus=0, nnodes=2,
            system_config=sys_config,
        )
        assert params['final_num_nodes'] == 2
        assert params['final_ngpus'] == 8  # 2 nodes * 4 GPUs
        assert params['final_node_occupancy'] == 1.0

    def test_multinode_gpu_system_with_nocc(self, sys_config):
        """Multi-node on GPU system with -o → CPU-only, no GPUs."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=2, node_occupancy=1.0,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_num_nodes'] == 2
        assert params['final_ngpus'] == 0
        assert params['final_node_occupancy'] == 1.0
        assert params['final_ranks_per_node'] == 64

    def test_multinode_gpu_system_explicit_gpus(self, sys_config):
        """Multi-node with explicit -g → use specified GPU count."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=4, nnodes=2,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_num_nodes'] == 2
        assert params['final_ngpus'] == 4
        assert params['final_ranks_per_node'] == 4  # GPUS_PER_NODE for multi-node

    def test_multinode_cpu_system(self, sys_config):
        """Multi-node on CPU-only system (no GPUs) → CPU job."""
        sys_config.GPUS_PER_NODE = 0
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=2,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_num_nodes'] == 2
        assert params['final_ngpus'] == 0
        assert params['final_ranks_per_node'] == 64

    def test_multinode_python_cpu_with_nocc(self, sys_config):
        """Multi-node Python on GPU system with -o → CPU-only, 1 rank."""
        from parslbox.apps.python import PythonApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=2, node_occupancy=1.0,
            system_config=sys_config, app_class=PythonApp,
        )
        assert params['final_num_nodes'] == 2
        assert params['final_ngpus'] == 0
        assert params['final_ranks_per_node'] == 1


class TestAddCommand:
    """Test suite for the add command functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = Path(tmp.name)
            database.initialize_database(db_path)
            yield db_path
            # Cleanup
            if db_path.exists():
                db_path.unlink()
    
    @pytest.fixture
    def temp_job_dirs(self):
        """Create temporary job directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test job directories
            job1_dir = temp_path / "job1"
            job2_dir = temp_path / "job2"
            job1_dir.mkdir()
            job2_dir.mkdir()
            
            # Create input files
            (job1_dir / "in.lammps").write_text("# LAMMPS input file")
            (job2_dir / "input.vasp").write_text("# VASP input file")
            
            yield {
                "job1": job1_dir,
                "job2": job2_dir,
                "temp_dir": temp_path
            }
    
    @pytest.fixture
    def mock_system_config(self):
        """Mock system configuration."""
        mock_config = MagicMock()
        mock_config.GPUS_PER_NODE = 4
        mock_config.CORES_PER_NODE = 64
        return mock_config
    
    def test_single_node_gpu_job(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding a single-node GPU job."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True,
                 'DFLT_INPUT': 'in.lammps'
             }):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk",
                "--config", "polaris",
                "--ngpus", "2",
                "--tag", "test-gpu"
            ])
            
            assert result.exit_code == 0
            assert "✅ Added 1 job(s) with IDs:" in result.stdout
            assert "Resource specification: n:1-r:2-g:2-nocc:NA" in result.stdout
            
            # Verify database entry
            jobs = database.get_jobs(temp_db)
            assert len(jobs) == 1
            job = jobs[0]
            assert job['app'] == 'lammps-kk'
            assert job['num_nodes'] == 1
            assert job['ngpus'] == 2
            assert job['node_occupancy'] == 1.0
            assert job['tag'] == 'test-gpu'
    
    def test_single_node_cpu_job(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding a single-node CPU job with node allocation."""
        runner = CliRunner()
        
        # Create a temporary environment file for Python app
        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('typer.confirm', return_value=True):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "python",
                "--config", "polaris",
                "--nocc", "0.5",
                "--tag", "test-cpu",
                "--envfile", str(env_file)
            ])
            
            assert result.exit_code == 0
            assert "✅ Added 1 job(s) with IDs:" in result.stdout
            assert "Resource specification: n:1-r:1-g:0-nocc:0.5" in result.stdout

            # Verify database entry
            jobs = database.get_jobs(temp_db)
            assert len(jobs) == 1
            job = jobs[0]
            assert job['app'] == 'python'
            assert job['num_nodes'] == 1
            assert job['ngpus'] == 0
            assert job['node_occupancy'] == 0.5
            assert job['tag'] == 'test-cpu'
    
    def test_multi_node_job(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding a multi-node job."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True,
                 'DFLT_INPUT': 'in.lammps'
             }):

            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk",
                "--config", "polaris",
                "--nnodes", "2"
            ])
            
            assert result.exit_code == 0
            assert "✅ Added 1 job(s) with IDs:" in result.stdout
            assert "Multi-node job will use 8 total GPUs (4 per node)" in result.stdout
            assert "Resource specification: n:2-r:8-g:8-nocc:NA" in result.stdout
            
            # Verify database entry
            jobs = database.get_jobs(temp_db)
            assert len(jobs) == 1
            job = jobs[0]
            assert job['app'] == 'lammps-kk'
            assert job['num_nodes'] == 2
            assert job['node_occupancy'] == 1.0
    
    def test_conflicting_parameters(self, temp_job_dirs, mock_system_config):
        """Test handling of conflicting parameters."""
        runner = CliRunner()
        
        # Create a temporary environment file
        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            temp_db = Path(tmp.name)
            database.initialize_database(temp_db)
            
            with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
                 patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
                 patch('parslbox.commands.add.is_app_registered', return_value=True), \
                 patch('parslbox.commands.add.get_app_config', return_value={
                     'INPUT_REQUIRED': False,
                     'DFLT_INPUT': None
                 }), \
                 patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
                 patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
                 patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                     'INPUT_REQUIRED': False,
                     'DFLT_INPUT': None
                 }), \
                 patch('typer.confirm', return_value=False):
                
                # Test CLI interactive handling of conflicting parameters
                result = runner.invoke(add_app, [
                    str(temp_job_dirs["job1"]),
                    "--app", "python",
                    "--config", "polaris",
                    "--ngpus", "2",
                    "--nocc", "0.5",
                    "--envfile", str(env_file)
                ])
                
                assert result.exit_code == 0
                assert "⚠️  Warning: Both --ngpus and --nocc specified" in result.stdout
                assert "Setting ngpus=0 for CPU-only job" in result.stdout
                assert "Resource specification: n:1-r:1-g:0-nocc:0.5" in result.stdout

            # Cleanup
            if temp_db.exists():
                temp_db.unlink()
    
    def test_gpu_limit_validation(self, temp_db, temp_job_dirs, mock_system_config):
        """Test validation of GPU limits."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk",
                "--config", "polaris",
                "--ngpus", "8"  # More than 4 GPUs per node
            ])
            
            assert result.exit_code == 1
            assert "❌ Error: Requested 8 GPUs but only 4 available per node" in result.stdout
    
    def test_invalid_nocc(self, temp_db, temp_job_dirs, mock_system_config):
        """Test validation of node occupancy range."""
        runner = CliRunner()
        
        # Create a temporary environment file
        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "python",
                "--config", "polaris",
                "--nocc", "1.5",  # Invalid range
                "--envfile", str(env_file)
            ])
            
            # Accept either exit code 1 (application error) or 2 (typer validation error)
            assert result.exit_code in [1, 2]
            assert "❌ Error: --nocc must be between 0.0 and 1.0" in result.stdout
    
    def test_invalid_nnodes(self, temp_db, temp_job_dirs, mock_system_config):
        """Test validation of node count."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk",
                "--config", "polaris",
                "--nnodes", "0"  # Invalid node count
            ])
            
            assert result.exit_code == 1
            assert "❌ Error: --nnodes must be at least 1" in result.stdout
    
    def test_unknown_app(self, temp_db, temp_job_dirs):
        """Test handling of unknown application."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.is_app_registered', return_value=False), \
             patch('parslbox.apps.app_registry.get_registered_apps', return_value=['lammps-kk', 'vasp', 'python']):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "unknown_app",
                "--config", "polaris"
            ])
            
            assert result.exit_code == 1
            assert "❌ Error: Unknown application: 'unknown_app'." in result.stdout
            assert "Available built-in apps:" in result.stdout
    
    def test_nonexistent_path(self, temp_db, mock_system_config):
        """Test handling of nonexistent path."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True):
            
            result = runner.invoke(add_app, [
                "/nonexistent/path",
                "--app", "lammps-kk",
                "--config", "polaris",
            ])
            
            assert result.exit_code == 1
            assert "❌ Failed to add 1 job(s):" in result.stdout
            assert "- /nonexistent/path: Path '/nonexistent/path' does not exist" in result.stdout
    
    def test_duplicate_job_path(self, temp_db, temp_job_dirs, mock_system_config):
        """Test handling of duplicate job paths."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }):
            
            # Add job first time
            result1 = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--config", "polaris",
                "--app", "lammps-kk"
            ])
            assert result1.exit_code == 0

            # Try to add same job again
            result2 = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--config", "polaris",
                "--app", "lammps-kk"
            ])
            assert result2.exit_code == 1
            assert "❌ Failed to add 1 job(s):" in result2.stdout
            assert "already exists in the database" in result2.stdout
    
    def test_add_all_subdirectories(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding all subdirectories with 'all' argument."""
        runner = CliRunner()
        
        # Create a temporary environment file for Python app
        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('typer.confirm', return_value=True):
            
            # Change to temp directory
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_job_dirs["temp_dir"])
                
                result = runner.invoke(add_app, [
                    "all",
                    "--config", "polaris",
                    "--app", "python",
                    "--envfile", str(env_file)
                ])
                
                assert result.exit_code == 0
                assert "Successfully added 2 job(s)" in result.stdout
                
                # Verify both jobs were added
                jobs = database.get_jobs(temp_db)
                assert len(jobs) == 2
                
            finally:
                os.chdir(original_cwd)
    
    def test_input_file_handling(self, temp_db, temp_job_dirs, mock_system_config):
        """Test input file handling for different app configurations."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True):
            
            # Test app with required input and default
            with patch('parslbox.apps.app_registry.get_app_config', return_value={
                'INPUT_REQUIRED': True,
                'DFLT_INPUT': 'in.lammps'
            }):
                result = runner.invoke(add_app, [
                    str(temp_job_dirs["job1"]),
                    "--config", "polaris",
                    "--app", "lammps-kk"
                ])
                
                assert result.exit_code == 0
                assert "Using default input file 'in.lammps'" in result.stdout
                
                jobs = database.get_jobs(temp_db)
                assert jobs[0]['in_file'] == 'in.lammps'
    
    def test_default_parameters(self, temp_db, temp_job_dirs, mock_system_config):
        """Test default parameter values."""
        runner = CliRunner()
        
        # Create a temporary environment file for Python app
        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('typer.confirm', return_value=True):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--config", "polaris",
                "--app", "python",
                "--envfile", str(env_file)
            ])
            
            assert result.exit_code == 0
            assert "Single-node job on GPU system: auto-assigned 4 GPUs" in result.stdout
            assert "Resource specification: n:1-r:4-g:4-nocc:NA" in result.stdout

            # Verify default values - GPU system auto-assigns GPUs
            jobs = database.get_jobs(temp_db)
            job = jobs[0]
            assert job['num_nodes'] == 1
            assert job['ngpus'] == 4
            assert job['node_occupancy'] == 1.0
            assert job['status'] == 'Ready'

    def test_add_with_app_args(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding a job with --args appends to in_file."""
        runner = CliRunner()

        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")

        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': True,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': True,
                 'DFLT_INPUT': None
             }), \
             patch('typer.confirm', return_value=True):

            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--config", "polaris",
                "--app", "python",
                "--input", "script.py",
                "--args", "--file afile -o 8 bfile",
                "--envfile", str(env_file)
            ])

            assert result.exit_code == 0
            assert "✅ Added 1 job(s) with IDs:" in result.stdout

            # Verify in_file contains script name + args
            jobs = database.get_jobs(temp_db)
            assert len(jobs) == 1
            assert jobs[0]['in_file'] == "script.py --file afile -o 8 bfile"

    def _seed_parents(self, temp_db, n):
        """Add `n` simple parent jobs and return their IDs."""
        ids = []
        for i in range(n):
            ids.append(database.add_job(
                db_path=temp_db, path=f"/parent_{i}", app="lammps-kk",
                num_nodes=1, ngpus=0, node_occupancy=1.0, tag="parent",
            ))
        return ids

    def test_parents_supports_range(self, temp_db, temp_job_dirs, mock_system_config):
        """--parents '1-3 5' should expand to [1,2,3,5]."""
        self._seed_parents(temp_db, 5)
        runner = CliRunner()
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True, 'DFLT_INPUT': 'in.lammps',
             }):
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk", "--config", "polaris",
                "--ngpus", "2", "--tag", "child",
                "--parents", "1-3 5",
            ])
        assert result.exit_code == 0, result.stdout
        # The new child is job_id 6 (parents seeded 1-5)
        child = [j for j in database.get_jobs(temp_db) if j['tag'] == 'child'][0]
        import json
        assert [int(x) for x in json.loads(child['parents'])] == [1, 2, 3, 5]

    def test_parents_mixed_ranges_and_singles(self, temp_db, temp_job_dirs, mock_system_config):
        """--parents '1-2 4 6-7' should expand to [1,2,4,6,7]."""
        self._seed_parents(temp_db, 7)
        runner = CliRunner()
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True, 'DFLT_INPUT': 'in.lammps',
             }):
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk", "--config", "polaris",
                "--ngpus", "2", "--tag", "child",
                "--parents", "1-2 4 6-7",
            ])
        assert result.exit_code == 0, result.stdout
        child = [j for j in database.get_jobs(temp_db) if j['tag'] == 'child'][0]
        import json
        assert [int(x) for x in json.loads(child['parents'])] == [1, 2, 4, 6, 7]

    def test_parents_invalid_range_errors(self, temp_db, temp_job_dirs, mock_system_config):
        """Backwards range '5-3' should produce a clear error and not add anything."""
        self._seed_parents(temp_db, 5)
        runner = CliRunner()
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True, 'DFLT_INPUT': 'in.lammps',
             }):
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk", "--config", "polaris",
                "--ngpus", "2",
                "--parents", "5-3",
            ])
        assert result.exit_code != 0
        assert "Invalid parent job IDs" in result.stdout
