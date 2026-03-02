"""
Unit tests for ParslBox update command

Tests the CLI update command functionality for updating job parameters.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import typer
from typer.testing import CliRunner

from parslbox.commands.update import update_jobs, ValidationError, app as update_app
from parslbox.database import database
from parslbox.utils import path_utils


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield db_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def setup_test_jobs(temp_db):
    """Set up test jobs in the database."""
    # Initialize database
    database.initialize_database(temp_db)
    
    # Create temporary job directories
    temp_dir = temp_db.parent
    job_dirs = {}
    
    for i, app_name in enumerate(['lammps-kk', 'vasp', 'python'], 1):
        job_dir = temp_dir / f"job_{i}_{app_name}"
        job_dir.mkdir()
        job_dirs[app_name] = str(job_dir)
        
        # Add job to database
        database.add_job(
            db_path=temp_db,
            path=str(job_dir),
            app=app_name,
            num_nodes=1,
            ngpus=0,
            node_occupancy=1.0,
            tag=f"test_{app_name}",
            in_file=None,
            status="Ready"
        )
    
    return temp_db, job_dirs


class TestUpdateJobsCore:
    """Test the core update_jobs function."""

    def test_update_jobs_input_file_validation_success(self, setup_test_jobs):
        """Test updating input file with validation against existing job app."""
        db_path, job_dirs = setup_test_jobs
        
        # Get lammps job (requires input)
        jobs = database.get_jobs(db_path, app="lammps-kk")
        job_id = jobs[0]['job_id']
        
        # Update input file for lammps job (should succeed)
        updated_ids, failed_jobs, msg_log = update_jobs(
            job_ids=[job_id],
            input_file="in.test",
            db_path=db_path
        )
        
        # Should succeed
        assert job_id in updated_ids
        assert len(failed_jobs) == 0
        
        # Verify job was updated
        updated_job = database.get_jobs_by_ids(db_path, [job_id])[0]
        assert updated_job['in_file'] == 'in.test'

    def test_update_jobs_input_file_validation_failure(self, setup_test_jobs):
        """Test updating input file with validation failure."""
        db_path, job_dirs = setup_test_jobs
        
        # Manually insert a job with an invalid app
        database.add_job(
            db_path=db_path,
            path="/fake/path",
            app="invalid_app",
            num_nodes=1,
            ngpus=0,
            node_occupancy=1.0,
            tag="test",
            status="Ready"
        )
        
        # Get the job with invalid app
        jobs = database.get_jobs(db_path, app="invalid_app")
        job_id = jobs[0]['job_id']
        
        # Try to update input file (should fail due to invalid app)
        updated_ids, failed_jobs, msg_log = update_jobs(
            job_ids=[job_id],
            input_file="test.input",
            db_path=db_path
        )
        
        # Should fail
        assert job_id not in updated_ids
        assert len(failed_jobs) == 1
        assert "not compatible" in failed_jobs[0][1]

    def test_update_jobs_multiple_jobs_mixed_results(self, setup_test_jobs):
        """Test updating multiple jobs where some succeed and some fail."""
        db_path, job_dirs = setup_test_jobs
        
        # Get valid jobs
        lammps_jobs = database.get_jobs(db_path, app="lammps-kk")
        vasp_jobs = database.get_jobs(db_path, app="vasp")
        lammps_id = lammps_jobs[0]['job_id']
        vasp_id = vasp_jobs[0]['job_id']
        
        # Add an invalid job
        database.add_job(
            db_path=db_path,
            path="/fake/path",
            app="invalid_app",
            num_nodes=1,
            ngpus=0,
            node_occupancy=1.0,
            tag="test",
            status="Ready"
        )
        invalid_jobs = database.get_jobs(db_path, app="invalid_app")
        invalid_id = invalid_jobs[0]['job_id']
        
        # Try to update input file for all jobs
        # lammps requires input (should succeed)
        # vasp doesn't require input (should fail with warning)
        # invalid_app doesn't exist (should fail with error)
        updated_ids, failed_jobs, msg_log = update_jobs(
            job_ids=[lammps_id, vasp_id, invalid_id],
            input_file="test.input",
            db_path=db_path
        )
        
        # Only lammps should succeed (it requires input files)
        assert lammps_id in updated_ids
        # vasp and invalid should fail (vasp doesn't need input, invalid app doesn't exist)
        assert vasp_id not in updated_ids
        assert invalid_id not in updated_ids
        assert len(failed_jobs) == 2
        failed_job_ids = [job_id for job_id, _ in failed_jobs]
        assert vasp_id in failed_job_ids
        assert invalid_id in failed_job_ids


class TestUpdateCommandCLI:
    """Test the CLI update command."""

    def test_cli_update_status_basic(self, setup_test_jobs):
        """Test basic CLI update command for status change."""
        db_path, job_dirs = setup_test_jobs
        
        # Get any job
        jobs = database.get_jobs(db_path)
        job_id = jobs[0]['job_id']
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, [str(job_id), '--status', 'Running'])
        
        # Should succeed
        assert result.exit_code == 0
        assert "Successfully updated 1 job(s)" in result.output
        
        # Verify job was updated
        updated_job = database.get_jobs_by_ids(db_path, [job_id])[0]
        assert updated_job['status'] == 'Running'

    def test_cli_update_input_file(self, setup_test_jobs):
        """Test CLI update command for input file change."""
        db_path, job_dirs = setup_test_jobs
        
        # Get lammps job
        jobs = database.get_jobs(db_path, app="lammps-kk")
        job_id = jobs[0]['job_id']
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, [str(job_id), '--input', 'in.test'])
        
        # Should succeed
        assert result.exit_code == 0
        assert "Successfully updated 1 job(s)" in result.output
        
        # Verify job was updated
        updated_job = database.get_jobs_by_ids(db_path, [job_id])[0]
        assert updated_job['in_file'] == 'in.test'

    def test_cli_update_ngpus(self, setup_test_jobs):
        """Test CLI update command for ngpus change."""
        db_path, job_dirs = setup_test_jobs
        
        # Get any job
        jobs = database.get_jobs(db_path)
        job_id = jobs[0]['job_id']
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, [str(job_id), '--ngpus', '2'])
        
        # Should succeed
        assert result.exit_code == 0
        assert "Successfully updated 1 job(s)" in result.output
        
        # Verify job was updated
        updated_job = database.get_jobs_by_ids(db_path, [job_id])[0]
        assert updated_job['ngpus'] == 2

    def test_cli_update_tag(self, setup_test_jobs):
        """Test CLI update command for tag change."""
        db_path, job_dirs = setup_test_jobs
        
        # Get any job
        jobs = database.get_jobs(db_path)
        job_id = jobs[0]['job_id']
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, [str(job_id), '--tag', 'new_tag'])
        
        # Should succeed
        assert result.exit_code == 0
        assert "Successfully updated 1 job(s)" in result.output
        
        # Verify job was updated
        updated_job = database.get_jobs_by_ids(db_path, [job_id])[0]
        assert updated_job['tag'] == 'new_tag'

    def test_cli_update_invalid_job_id(self, setup_test_jobs):
        """Test CLI update command with invalid job ID."""
        db_path, job_dirs = setup_test_jobs
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, ['99999', '--status', 'Running'])
        
        # Should succeed but show no updates
        assert result.exit_code == 0
        assert "No jobs were updated" in result.output

    def test_cli_update_no_options_provided(self, setup_test_jobs):
        """Test CLI update command with no update options provided."""
        db_path, job_dirs = setup_test_jobs
        
        # Get any job
        jobs = database.get_jobs(db_path)
        job_id = jobs[0]['job_id']
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, [str(job_id)])
        
        # Should fail with validation error
        assert result.exit_code == 1
        assert "You must provide at least one field to update" in result.output

    def test_cli_update_multiple_jobs(self, setup_test_jobs):
        """Test CLI update command with multiple job IDs."""
        db_path, job_dirs = setup_test_jobs
        
        # Get multiple jobs
        jobs = database.get_jobs(db_path)
        job_ids = [str(job['job_id']) for job in jobs[:2]]
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, job_ids + ['--status', 'Running'])
        
        # Should succeed
        assert result.exit_code == 0
        assert "Successfully updated 2 job(s)" in result.output


class TestUpdateJobsValidation:
    """Test validation logic in update_jobs function."""

    def test_validation_no_options_provided(self, temp_db):
        """Test validation when no update options are provided."""
        with pytest.raises(ValidationError, match="You must provide at least one field to update"):
            update_jobs(job_ids=[1], db_path=temp_db)

    def test_validation_invalid_nnodes(self, temp_db):
        """Test validation for invalid nnodes."""
        with pytest.raises(ValidationError, match="--nnodes must be at least 1"):
            update_jobs(job_ids=[1], nnodes=0, db_path=temp_db)

    def test_validation_invalid_ngpus(self, temp_db):
        """Test validation for invalid ngpus."""
        with pytest.raises(ValidationError, match="--ngpus must be non-negative"):
            update_jobs(job_ids=[1], ngpus=-1, db_path=temp_db)

    def test_validation_invalid_node_occupancy(self, temp_db):
        """Test validation for invalid node occupancy."""
        with pytest.raises(ValidationError, match="--nocc must be between 0.0 and 1.0"):
            update_jobs(job_ids=[1], node_occupancy=1.5, db_path=temp_db)

    def test_validation_invalid_ranks_per_node(self, temp_db):
        """Test validation for invalid ranks per node."""
        with pytest.raises(ValidationError, match="--ranks-per-node must be a positive integer"):
            update_jobs(job_ids=[1], ranks_per_node=0, db_path=temp_db)

    def test_validation_gpu_cpu_conflict(self, temp_db):
        """Test validation for GPU/CPU parameter conflict."""
        with pytest.raises(ValidationError, match="Cannot specify both ngpus and node_occupancy to be > 0"):
            update_jobs(job_ids=[1], ngpus=2, node_occupancy=0.5, db_path=temp_db)


class TestUpdateJobsEdgeCases:
    """Test edge cases and error conditions."""

    def test_update_nonexistent_job_ids(self, temp_db):
        """Test updating non-existent job IDs."""
        database.initialize_database(temp_db)
        
        # Try to update non-existent jobs
        updated_ids, failed_jobs, msg_log = update_jobs(
            job_ids=[999, 1000],
            status="Running",
            db_path=temp_db
        )
        
        # Should return empty results (no jobs found)
        assert len(updated_ids) == 0
        assert len(failed_jobs) == 0
        assert len(msg_log['warnings']) == 0

    def test_update_mixed_existing_nonexistent_jobs(self, setup_test_jobs):
        """Test updating mix of existing and non-existent job IDs."""
        db_path, job_dirs = setup_test_jobs
        
        # Get one existing job
        jobs = database.get_jobs(db_path)
        existing_job_id = jobs[0]['job_id']
        
        # Try to update existing + non-existent jobs
        updated_ids, failed_jobs, msg_log = update_jobs(
            job_ids=[existing_job_id, 999],
            status="Running",
            db_path=db_path
        )
        
        # Should update only the existing job
        assert existing_job_id in updated_ids
        assert len(updated_ids) == 1
        assert len(failed_jobs) == 0

    def test_update_dependencies(self, setup_test_jobs):
        """Test updating job dependencies."""
        db_path, job_dirs = setup_test_jobs
        
        # Get jobs
        jobs = database.get_jobs(db_path)
        parent_id = jobs[0]['job_id']
        child_id = jobs[1]['job_id']
        
        # Add dependency
        updated_ids, failed_jobs, msg_log = update_jobs(
            job_ids=[child_id],
            add_deps=[parent_id],
            db_path=db_path
        )
        
        # Should succeed
        assert child_id in updated_ids
        assert len(failed_jobs) == 0
        
        # Verify dependency was added
        updated_job = database.get_jobs_by_ids(db_path, [child_id])[0]
        parents = database.parse_existing_parents(updated_job['parents'])
        assert parent_id in parents

    def test_update_environment_file(self, setup_test_jobs):
        """Test updating environment file."""
        db_path, job_dirs = setup_test_jobs
        
        # Create a temporary env file
        temp_dir = db_path.parent
        env_file = temp_dir / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test env'")
        
        # Get any job
        jobs = database.get_jobs(db_path)
        job_id = jobs[0]['job_id']
        
        # Update environment file
        updated_ids, failed_jobs, msg_log = update_jobs(
            job_ids=[job_id],
            env_file=str(env_file),
            db_path=db_path
        )
        
        # Should succeed
        assert job_id in updated_ids
        assert len(failed_jobs) == 0
        
        # Verify env file was updated
        updated_job = database.get_jobs_by_ids(db_path, [job_id])[0]
        assert str(env_file.resolve()) in updated_job['env_file']
