"""
Unit tests for ParslBox update command

Tests the CLI update command functionality, especially the app update scenarios
with input file validation.
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
from parslbox.apps.app_registry import get_app_config


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
    
    for i, app_name in enumerate(['lammps', 'vasp', 'python'], 1):
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

    def test_update_jobs_app_scenario_1_input_required_to_not_required(self, setup_test_jobs):
        """Test scenario 1: input-required app to non-input-required app."""
        db_path, job_dirs = setup_test_jobs
        
        # Get lammps job (requires input)
        jobs = database.get_jobs(db_path, app="lammps")
        job_id = jobs[0]['job_id']
        
        # Update lammps -> vasp (input required -> not required)
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[job_id],
            app="vasp",
            db_path=db_path,
            interactive_prompts=False
        )
        
        # Should succeed with warning
        assert job_id in updated_ids
        assert len(failed_jobs) == 0
        assert len(warnings) == 1
        assert "Input file set to None" in warnings[0]
        
        # Verify job was updated
        updated_job = database.get_jobs_by_ids(db_path, [job_id])[0]
        assert updated_job['app'] == 'vasp'
        assert updated_job['in_file'] == ''  # Should be empty string (None converted)

    def test_update_jobs_app_scenario_2_not_required_to_required_fail(self, setup_test_jobs):
        """Test scenario 2: non-input-required app to input-required app (should fail)."""
        db_path, job_dirs = setup_test_jobs
        
        # Get vasp job (doesn't require input)
        jobs = database.get_jobs(db_path, app="vasp")
        job_id = jobs[0]['job_id']
        
        # Update vasp -> lammps without providing input (should fail)
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[job_id],
            app="lammps",
            db_path=db_path,
            interactive_prompts=False
        )
        
        # Should fail
        assert job_id not in updated_ids
        assert len(failed_jobs) == 1
        assert job_id == failed_jobs[0][0]
        assert "new app requires input file but none provided" in failed_jobs[0][1]

    def test_update_jobs_app_scenario_2_not_required_to_required_success(self, setup_test_jobs):
        """Test scenario 2b: non-input-required app to input-required app with input (should succeed)."""
        db_path, job_dirs = setup_test_jobs
        
        # Get vasp job (doesn't require input)
        jobs = database.get_jobs(db_path, app="vasp")
        job_id = jobs[0]['job_id']
        
        # Update vasp -> lammps with input provided (should succeed)
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[job_id],
            app="lammps",
            input_file="in.test",
            db_path=db_path,
            interactive_prompts=False
        )
        
        # Should succeed with warning
        assert job_id in updated_ids
        assert len(failed_jobs) == 0
        assert len(warnings) == 1
        assert "User-specified input file" in warnings[0]
        
        # Verify job was updated
        updated_job = database.get_jobs_by_ids(db_path, [job_id])[0]
        assert updated_job['app'] == 'lammps'
        assert updated_job['in_file'] == 'in.test'

    def test_update_jobs_app_scenario_3_both_required_different_defaults(self, setup_test_jobs):
        """Test scenario 3: both apps require input with different defaults (should fail)."""
        db_path, job_dirs = setup_test_jobs
        
        # Get lammps job (requires input, default='in.lammps')
        jobs = database.get_jobs(db_path, app="lammps")
        job_id = jobs[0]['job_id']
        
        # Update lammps -> python (both require input, different defaults)
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[job_id],
            app="python",
            db_path=db_path,
            interactive_prompts=False
        )
        
        # Should fail
        assert job_id not in updated_ids
        assert len(failed_jobs) == 1
        assert "apps have different default input files" in failed_jobs[0][1]

    def test_update_jobs_app_scenario_3_both_required_with_input(self, setup_test_jobs):
        """Test scenario 3b: both apps require input with user-provided input (should succeed)."""
        db_path, job_dirs = setup_test_jobs
        
        # Get lammps job (requires input)
        jobs = database.get_jobs(db_path, app="lammps")
        job_id = jobs[0]['job_id']
        
        # Update lammps -> python with input provided
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[job_id],
            app="python",
            input_file="script.py",
            db_path=db_path,
            interactive_prompts=False
        )
        
        # Should succeed with warning
        assert job_id in updated_ids
        assert len(failed_jobs) == 0
        assert len(warnings) == 1
        assert "User-specified input file" in warnings[0]

    def test_update_jobs_app_scenario_4_both_not_required(self, setup_test_jobs):
        """Test scenario 4: both apps don't require input (should succeed with warning)."""
        db_path, job_dirs = setup_test_jobs
        
        # Get vasp job (doesn't require input)
        jobs = database.get_jobs(db_path, app="vasp")
        job_id = jobs[0]['job_id']
        
        # Update vasp -> vasp (both don't require input)
        # Note: This is a bit artificial since we're updating to the same app,
        # but it tests the logic for both apps not requiring input
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[job_id],
            app="vasp",
            db_path=db_path,
            interactive_prompts=False
        )
        
        # Should succeed with warning
        assert job_id in updated_ids
        assert len(failed_jobs) == 0
        assert len(warnings) == 1
        assert "user should verify input file compatibility" in warnings[0]

    def test_update_jobs_multiple_jobs_partial_failure(self, setup_test_jobs):
        """Test updating multiple jobs where some succeed and some fail."""
        db_path, job_dirs = setup_test_jobs
        
        # Get vasp and lammps jobs
        vasp_jobs = database.get_jobs(db_path, app="vasp")
        lammps_jobs = database.get_jobs(db_path, app="lammps")
        vasp_id = vasp_jobs[0]['job_id']
        lammps_id = lammps_jobs[0]['job_id']
        
        # Try to update both to python without providing input
        # vasp -> python should fail (no input provided)
        # lammps -> python should fail (different defaults)
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[vasp_id, lammps_id],
            app="python",
            db_path=db_path,
            interactive_prompts=False
        )
        
        # Both should fail
        assert len(updated_ids) == 0
        assert len(failed_jobs) == 2
        failed_job_ids = [job_id for job_id, _ in failed_jobs]
        assert vasp_id in failed_job_ids
        assert lammps_id in failed_job_ids

    def test_update_jobs_invalid_app(self, setup_test_jobs):
        """Test updating to an invalid app."""
        db_path, job_dirs = setup_test_jobs
        
        # Get any job
        jobs = database.get_jobs(db_path)
        job_id = jobs[0]['job_id']
        
        # Try to update to invalid app
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[job_id],
            app="invalid_app",
            db_path=db_path,
            interactive_prompts=False
        )
        
        # Should fail
        assert job_id not in updated_ids
        assert len(failed_jobs) == 1
        assert "Unknown application" in failed_jobs[0][1]


class TestUpdateCommandCLI:
    """Test the CLI update command."""

    def test_cli_update_app_scenario_1_success(self, setup_test_jobs):
        """Test CLI update command for scenario 1 (input required -> not required)."""
        db_path, job_dirs = setup_test_jobs
        
        # Get lammps job
        jobs = database.get_jobs(db_path, app="lammps")
        job_id = jobs[0]['job_id']
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, [str(job_id), '--app', 'vasp'])
        
        # Should succeed
        assert result.exit_code == 0
        assert "Successfully updated 1 job(s)" in result.output
        assert "Input file set to None" in result.output

    def test_cli_update_app_scenario_2_failure(self, setup_test_jobs):
        """Test CLI update command for scenario 2 (not required -> required, should fail)."""
        db_path, job_dirs = setup_test_jobs
        
        # Get vasp job
        jobs = database.get_jobs(db_path, app="vasp")
        job_id = jobs[0]['job_id']
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, [str(job_id), '--app', 'lammps'])
        
        # Should succeed (no exit code 1) but show error for the job
        assert result.exit_code == 0
        assert "No jobs were updated" in result.output
        assert "new app requires input file but none provided" in result.output

    def test_cli_update_app_scenario_2_success_with_input(self, setup_test_jobs):
        """Test CLI update command for scenario 2 with input provided (should succeed)."""
        db_path, job_dirs = setup_test_jobs
        
        # Get vasp job
        jobs = database.get_jobs(db_path, app="vasp")
        job_id = jobs[0]['job_id']
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, [str(job_id), '--app', 'lammps', '--input', 'in.test'])
        
        # Should succeed
        assert result.exit_code == 0
        assert "Successfully updated 1 job(s)" in result.output
        assert "User-specified input file" in result.output

    def test_cli_update_multiple_jobs_partial_failure(self, setup_test_jobs):
        """Test CLI update command with multiple jobs where some fail."""
        db_path, job_dirs = setup_test_jobs
        
        # Get vasp and lammps jobs
        vasp_jobs = database.get_jobs(db_path, app="vasp")
        lammps_jobs = database.get_jobs(db_path, app="lammps")
        vasp_id = vasp_jobs[0]['job_id']
        lammps_id = lammps_jobs[0]['job_id']
        
        runner = CliRunner()
        
        # Mock the database path
        with patch('parslbox.commands.update.path_utils.DB_FILE', db_path):
            result = runner.invoke(update_app, [str(vasp_id), str(lammps_id), '--app', 'python'])
        
        # Should succeed (no exit code 1) but show errors and no updates
        assert result.exit_code == 0
        assert "No jobs were updated" in result.output
        assert "Cannot update job" in result.output

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

    def test_validation_invalid_node_occupancy(self, temp_db):
        """Test validation for invalid node occupancy."""
        with pytest.raises(ValidationError, match="--nocc must be between 0.0 and 1.0"):
            update_jobs(job_ids=[1], node_occupancy=1.5, db_path=temp_db)

    def test_validation_invalid_ranks_per_node(self, temp_db):
        """Test validation for invalid ranks per node."""
        with pytest.raises(ValidationError, match="--ranks-per-node must be a positive integer"):
            update_jobs(job_ids=[1], ranks_per_node=0, db_path=temp_db)


class TestUpdateJobsEdgeCases:
    """Test edge cases and error conditions."""

    def test_update_nonexistent_job_ids(self, temp_db):
        """Test updating non-existent job IDs."""
        database.initialize_database(temp_db)
        
        # Try to update non-existent jobs
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[999, 1000],
            status="Running",
            db_path=temp_db,
            interactive_prompts=False
        )
        
        # Should return empty results (no jobs found)
        assert len(updated_ids) == 0
        assert len(failed_jobs) == 0
        assert len(warnings) == 0

    def test_update_mixed_existing_nonexistent_jobs(self, setup_test_jobs):
        """Test updating mix of existing and non-existent job IDs."""
        db_path, job_dirs = setup_test_jobs
        
        # Get one existing job
        jobs = database.get_jobs(db_path)
        existing_job_id = jobs[0]['job_id']
        
        # Try to update existing + non-existent jobs
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[existing_job_id, 999],
            status="Running",
            db_path=db_path,
            interactive_prompts=False
        )
        
        # Should update only the existing job
        assert existing_job_id in updated_ids
        assert len(updated_ids) == 1
        assert len(failed_jobs) == 0

    def test_update_app_with_original_app_not_found(self, setup_test_jobs):
        """Test updating app when original app is not in registry (edge case)."""
        db_path, job_dirs = setup_test_jobs
        
        # Manually insert a job with an invalid app
        database.add_job(
            db_path=db_path,
            path="/fake/path",
            app="invalid_original_app",
            num_nodes=1,
            ngpus=0,
            node_occupancy=1.0,
            tag="test",
            status="Ready"
        )
        
        # Get the job with invalid app
        jobs = database.get_jobs(db_path, app="invalid_original_app")
        job_id = jobs[0]['job_id']
        
        # Try to update to valid app
        updated_ids, failed_jobs, warnings = update_jobs(
            job_ids=[job_id],
            app="vasp",
            db_path=db_path,
            interactive_prompts=False
        )
        
        # Should fail due to original app not being found
        assert job_id not in updated_ids
        assert len(failed_jobs) == 1
        assert "Unknown application" in failed_jobs[0][1]
