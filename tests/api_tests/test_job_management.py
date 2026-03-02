"""
Unit tests for ParslBox job management operations (add, update).
"""

import pytest
from parslbox.api import ValidationError


class TestAddJobs:
    """Test add_jobs functionality."""

    def test_add_jobs_single_basic(self, pbx, tmp_path):
        """Test adding a single basic job."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        
        # Create a test environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")

        job_ids, failures, msg_log = pbx.add_jobs(
            paths=[str(job_dir)], app="python", config="polaris", tag="test", input_file="script.py", env_file=str(env_file)
        )

        assert len(job_ids) == 1
        assert len(failures) == 0
        assert isinstance(job_ids[0], int)
        assert job_ids[0] > 0

        # Verify job was added
        job = pbx.get_job(job_ids[0])
        assert job["app"] == "python"
        assert job["tag"] == "test"
        assert job["status"] == "Ready"

    def test_add_jobs_invalid_app(self, pbx, tmp_path):
        """Test adding job with invalid app."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()

        with pytest.raises(ValidationError, match="Unknown application"):
            pbx.add_jobs(paths=[str(job_dir)], app="invalid_app", config="polaris")

    def test_add_jobs_invalid_path(self, pbx, tmp_path):
        """Test adding job with invalid path."""
        # Create environment file for the test
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        job_ids, failures, msg_log = pbx.add_jobs(paths=["/nonexistent/path"], app="python", config="polaris", input_file="script.py", env_file=str(env_file))
        
        assert len(job_ids) == 0
        assert len(failures) == 1
        assert failures[0][0] == "/nonexistent/path"
        assert "does not exist" in failures[0][1]

    def test_add_jobs_with_resources(self, pbx, tmp_path):
        """Test adding job with resource specifications."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")

        job_ids, failures, msg_log = pbx.add_jobs(
            paths=[str(job_dir)],
            app="python",
            config="polaris",
            ngpus=2,
            nnodes=1,
            tag="gpu_job",
            input_file="script.py",
            env_file=str(env_file)
        )

        assert len(job_ids) == 1
        assert len(failures) == 0
        
        job = pbx.get_job(job_ids[0])
        assert job["ngpus"] == 2
        assert job["num_nodes"] == 1

    def test_add_jobs_with_parents(self, pbx, tmp_path):
        """Test adding job with parent dependencies."""
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        # Add parent job
        parent_dir = tmp_path / "parent_job"
        parent_dir.mkdir()
        parent_ids, _, _ = pbx.add_jobs(paths=[str(parent_dir)], app="python", config="polaris", input_file="script.py", env_file=str(env_file))
        parent_id = parent_ids[0]

        # Add child job
        child_dir = tmp_path / "child_job"
        child_dir.mkdir()
        child_ids, failures, msg_log = pbx.add_jobs(
            paths=[str(child_dir)], app="python", config="polaris", parents=[parent_id], input_file="script.py", env_file=str(env_file)
        )

        assert len(child_ids) == 1
        assert len(failures) == 0
        
        child_job = pbx.get_job(child_ids[0])
        # Check that parents are stored (as JSON string)
        assert child_job["parents"] is not None

    def test_add_jobs_with_input_file(self, pbx, tmp_path):
        """Test adding job with input file specified."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")

        job_ids, failures, msg_log = pbx.add_jobs(
            paths=[str(job_dir)], app="python", config="polaris", input_file="script.py", env_file=str(env_file)
        )

        assert len(job_ids) == 1
        assert len(failures) == 0
        
        job = pbx.get_job(job_ids[0])
        assert job["in_file"] == "script.py"

    def test_add_jobs_python_without_env_file(self, pbx, tmp_path):
        """Test adding Python job without environment file (should fail)."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()

        with pytest.raises(ValidationError, match="Python app requires an environment file"):
            pbx.add_jobs(paths=[str(job_dir)], app="python", config="polaris")

    def test_add_jobs_multiple(self, pbx, tmp_path):
        """Test adding multiple jobs."""
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        paths = []
        for i in range(3):
            job_dir = tmp_path / f"job_{i}"
            job_dir.mkdir()
            paths.append(str(job_dir))

        job_ids, failures, msg_log = pbx.add_jobs(paths=paths, app="python", config="polaris", input_file="script.py", env_file=str(env_file))

        assert len(job_ids) == 3
        assert len(failures) == 0
        # All should succeed
        for job_id in job_ids:
            assert job_id is not None
            assert isinstance(job_id, int)

    def test_add_jobs_partial_failure(self, pbx, tmp_path):
        """Test adding jobs when some fail."""
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        paths = [
            str(tmp_path / "valid_job"),  # Will be created
            "/nonexistent/path",  # Will fail
        ]
        (tmp_path / "valid_job").mkdir()

        job_ids, failures, msg_log = pbx.add_jobs(paths=paths, app="python", config="polaris", input_file="script.py", env_file=str(env_file))

        assert len(job_ids) == 1  # One successful job
        assert len(failures) == 1  # One failed job
        
        # First should succeed
        assert job_ids[0] is not None
        assert isinstance(job_ids[0], int)
        
        # Second should fail
        failed_path, error_msg = failures[0]
        assert failed_path == "/nonexistent/path"
        assert "does not exist" in error_msg


class TestUpdateJobs:
    """Test update_jobs functionality."""

    def test_update_jobs_status(self, pbx, tmp_path):
        """Test updating job status."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        job_ids, _, _ = pbx.add_jobs(
            paths=[str(job_dir)], app="python", config="polaris", status="Ready", input_file="script.py", env_file=str(env_file)
        )
        job_id = job_ids[0]

        updated_ids, failed_jobs, msg_log = pbx.update_jobs([job_id], status="Running")
        assert job_id in updated_ids
        assert len(failed_jobs) == 0
        assert len(msg_log["warnings"]) == 0

        job = pbx.get_job(job_id)
        assert job["status"] == "Running"

    def test_update_jobs_tag(self, pbx, tmp_path):
        """Test updating job tag."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        job_ids, _, _ = pbx.add_jobs(
            paths=[str(job_dir)], app="python", config="polaris", tag="old_tag", input_file="script.py", env_file=str(env_file)
        )
        job_id = job_ids[0]

        updated_ids, failed_jobs, msg_log = pbx.update_jobs([job_id], tag="new_tag")
        assert job_id in updated_ids

        job = pbx.get_job(job_id)
        assert job["tag"] == "new_tag"

    def test_update_jobs_invalid_nnodes(self, pbx, tmp_path):
        """Test updating job with invalid nnodes."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        job_ids, _, _ = pbx.add_jobs(paths=[str(job_dir)], app="python", config="polaris", input_file="script.py", env_file=str(env_file))
        job_id = job_ids[0]

        with pytest.raises(ValidationError):
            pbx.update_jobs([job_id], nnodes=0)

    def test_update_jobs_add_dependencies(self, pbx, tmp_path):
        """Test adding dependencies to a job."""
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        # Add parent job
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        parent_ids, _, _ = pbx.add_jobs(paths=[str(parent_dir)], app="python", config="polaris", input_file="script.py", env_file=str(env_file))
        parent_id = parent_ids[0]

        # Add child job
        child_dir = tmp_path / "child"
        child_dir.mkdir()
        child_ids, _, _ = pbx.add_jobs(paths=[str(child_dir)], app="python", config="polaris", input_file="script.py", env_file=str(env_file))
        child_id = child_ids[0]

        # Add dependency
        updated_ids, failed_jobs, msg_log = pbx.update_jobs([child_id], add_deps=[parent_id])
        assert child_id in updated_ids

        child_job = pbx.get_job(child_id)
        assert child_job["parents"] is not None

    def test_update_jobs_circular_dependency(self, pbx, tmp_path):
        """Test that circular dependencies are prevented."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        job_ids, _, _ = pbx.add_jobs(paths=[str(job_dir)], app="python", config="polaris", input_file="script.py", env_file=str(env_file))
        job_id = job_ids[0]

        with pytest.raises(ValidationError, match="cannot be parents of themselves"):
            pbx.update_jobs([job_id], add_deps=[job_id])

    def test_update_jobs_input_file_validation(self, pbx, tmp_path):
        """Test input file validation during updates."""
        # Create a lammps job (requires input)
        job_dir = tmp_path / "job1"
        job_dir.mkdir()
        job_ids, _, _ = pbx.add_jobs(paths=[str(job_dir)], app="lammps-kk", config="polaris", input_file="in.lammps")
        job_id = job_ids[0]
        
        # Update input file (should succeed for lammps)
        updated_ids, failed_jobs, msg_log = pbx.update_jobs([job_id], input_file="in.new")
        assert job_id in updated_ids
        assert len(failed_jobs) == 0
        
        # Verify the input file was updated
        job = pbx.get_job(job_id)
        assert job["in_file"] == "in.new"

    def test_update_jobs_input_file_warning_failure(self, pbx, tmp_path):
        """Test that input file warnings cause job update failures."""
        # Create a vasp job (doesn't require input)
        job_dir = tmp_path / "job1"
        job_dir.mkdir()
        job_ids, _, _ = pbx.add_jobs(paths=[str(job_dir)], app="vasp", config="polaris")
        job_id = job_ids[0]
        
        # Try to set input file for vasp (should fail with warning)
        updated_ids, failed_jobs, msg_log = pbx.update_jobs([job_id], input_file="unnecessary.input")
        
        # Should fail because vasp doesn't need input files
        assert job_id not in updated_ids
        assert len(failed_jobs) == 1
        assert failed_jobs[0][0] == job_id
        assert "Input file ignored" in failed_jobs[0][1]
