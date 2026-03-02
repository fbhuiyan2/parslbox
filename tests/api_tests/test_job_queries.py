"""
Unit tests for ParslBox job query operations (list, get, remove, filter).
"""

import pytest
from parslbox.api import JobNotFoundError


class TestListJobs:
    """Test list_jobs functionality."""

    def test_list_jobs_empty(self, pbx):
        """Test listing jobs when database is empty."""
        jobs = pbx.list_jobs()
        assert jobs == []

    def test_list_jobs_filter_by_status(self, pbx, tmp_path):
        """Test filtering jobs by status."""
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        # Add jobs with different statuses
        for i, status in enumerate(["Ready", "Running", "Done"]):
            job_dir = tmp_path / f"job_{i}"
            job_dir.mkdir()
        job_ids, _, _ = pbx.add_jobs(
            paths=[str(job_dir)], app="python", config="polaris", status="Ready", input_file="script.py", env_file=str(env_file)
        )

        ready_jobs = pbx.list_jobs(status="Ready")
        assert len(ready_jobs) == 1
        assert ready_jobs[0]["status"] == "Ready"

    def test_list_jobs_filter_by_app(self, pbx, tmp_path):
        """Test filtering jobs by app."""
        # Create environment file for python jobs
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        # Add jobs with different apps
        for app in ["python", "lammps-kk"]:
            job_dir = tmp_path / f"job_{app}"
            job_dir.mkdir()
            if app == "python":
                pbx.add_jobs(paths=[str(job_dir)], app=app, config="polaris", input_file="script.py", env_file=str(env_file))
            else:
                pbx.add_jobs(paths=[str(job_dir)], app=app, config="polaris", input_file="in.test")

        python_jobs = pbx.list_jobs(app="python")
        assert len(python_jobs) == 1
        assert python_jobs[0]["app"] == "python"

    def test_list_jobs_filter_by_tag(self, pbx, tmp_path):
        """Test filtering jobs by tag."""
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        # Add jobs with different tags
        for tag in ["tag1", "tag2"]:
            job_dir = tmp_path / f"job_{tag}"
            job_dir.mkdir()
            pbx.add_jobs(paths=[str(job_dir)], app="python", config="polaris", tag=tag, input_file="script.py", env_file=str(env_file))

        tagged_jobs = pbx.list_jobs(tag="tag1")
        assert len(tagged_jobs) == 1
        assert tagged_jobs[0]["tag"] == "tag1"


class TestGetJob:
    """Test get_job functionality."""

    def test_get_job_exists(self, pbx, tmp_path):
        """Test getting an existing job."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        job_ids, _, _ = pbx.add_jobs(paths=[str(job_dir)], app="python", config="polaris", input_file="script.py", env_file=str(env_file))
        job_id = job_ids[0]

        job = pbx.get_job(job_id)
        assert job["job_id"] == job_id
        assert job["app"] == "python"

    def test_get_job_not_found(self, pbx):
        """Test getting a non-existent job."""
        with pytest.raises(JobNotFoundError):
            pbx.get_job(99999)


class TestRemoveJob:
    """Test remove_job functionality."""

    def test_remove_job(self, pbx, tmp_path):
        """Test removing a job."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        job_ids, _, _ = pbx.add_jobs(paths=[str(job_dir)], app="python", config="polaris", input_file="script.py", env_file=str(env_file))
        job_id = job_ids[0]

        # Remove job
        result = pbx.remove_job(job_id)
        assert result is True

        # Verify job is gone
        with pytest.raises(JobNotFoundError):
            pbx.get_job(job_id)

    def test_remove_job_not_found(self, pbx):
        """Test removing a non-existent job."""
        result = pbx.remove_job(99999)
        assert result is False

    def test_remove_jobs_multiple(self, pbx, tmp_path):
        """Test removing multiple jobs."""
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        job_ids = []
        for i in range(3):
            job_dir = tmp_path / f"job_{i}"
            job_dir.mkdir()
            ids, _, _ = pbx.add_jobs(paths=[str(job_dir)], app="python", config="polaris", input_file="script.py", env_file=str(env_file))
            job_ids.append(ids[0])

        count = pbx.remove_jobs(job_ids)
        assert count == 3

        # Verify all jobs are gone
        for job_id in job_ids:
            with pytest.raises(JobNotFoundError):
                pbx.get_job(job_id)

    def test_remove_all_jobs(self, pbx, tmp_path):
        """Test removing all jobs."""
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        # Add multiple jobs
        for i in range(3):
            job_dir = tmp_path / f"job_{i}"
            job_dir.mkdir()
            pbx.add_jobs(paths=[str(job_dir)], app="python", config="polaris", input_file="script.py", env_file=str(env_file))

        count = pbx.remove_all_jobs()
        assert count == 3

        # Verify all jobs are gone
        jobs = pbx.list_jobs()
        assert len(jobs) == 0


class TestFilterJobs:
    """Test filter_jobs functionality."""

    def test_filter_jobs_by_status(self, pbx, tmp_path):
        """Test filtering jobs by status."""
        # Create environment file
        env_file = tmp_path / "test.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        # Add jobs with different statuses
        for i, status in enumerate(["Ready", "Running", "Done"]):
            job_dir = tmp_path / f"job_{i}"
            job_dir.mkdir()
            pbx.add_jobs(
                paths=[str(job_dir)], app="python", config="polaris", status=status, input_file="script.py", env_file=str(env_file)
            )

        ready_ids = pbx.filter_jobs(status="Ready")
        assert len(ready_ids) == 1

    def test_filter_jobs_empty_result(self, pbx):
        """Test filtering when no jobs match."""
        ids = pbx.filter_jobs(status="Done")
        assert ids == []
