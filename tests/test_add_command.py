import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from parslbox.commands.add import app as add_app
from parslbox.database import database


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
