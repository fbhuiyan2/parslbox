"""
Unit tests for ParslBox API

Tests the programmatic API to ensure it behaves consistently with the CLI.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from parslbox.api import ParslBox, ValidationError, JobNotFoundError


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    yield db_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def pbx(temp_db):
    """Create a ParslBox instance with temporary database."""
    return ParslBox(db_path=temp_db)


class TestParslBoxInit:
    """Test ParslBox initialization."""

    def test_init_default(self):
        """Test initialization with default paths."""
        pbx = ParslBox()
        assert pbx.db_path is not None
        assert pbx.config_path is not None

    def test_init_custom_paths(self, temp_db):
        """Test initialization with custom paths."""
        config_path = temp_db.parent / "config.yaml"
        pbx = ParslBox(db_path=temp_db, config_path=config_path)
        assert pbx.db_path == temp_db
        assert pbx.config_path == config_path


class TestAddJob:
    """Test add_job functionality."""

    def test_add_job_basic(self, pbx, tmp_path):
        """Test adding a basic job."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()

        job_id = pbx.add_job(
            path=str(job_dir), app="python", config="polaris", tag="test"
        )

        assert isinstance(job_id, int)
        assert job_id > 0

        # Verify job was added
        job = pbx.get_job(job_id)
        assert job["app"] == "python"
        assert job["tag"] == "test"
        assert job["status"] == "Ready"

    def test_add_job_invalid_app(self, pbx, tmp_path):
        """Test adding job with invalid app."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()

        with pytest.raises(ValidationError, match="Unknown application"):
            pbx.add_job(path=str(job_dir), app="invalid_app", config="polaris")

    def test_add_job_invalid_path(self, pbx):
        """Test adding job with invalid path."""
        with pytest.raises(ValidationError):
            pbx.add_job(path="/nonexistent/path", app="python", config="polaris")

    def test_add_job_with_resources(self, pbx, tmp_path):
        """Test adding job with resource specifications."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()

        job_id = pbx.add_job(
            path=str(job_dir),
            app="python",
            config="polaris",
            ngpus=2,
            nnodes=1,
            tag="gpu_job",
        )

        job = pbx.get_job(job_id)
        assert job["ngpus"] == 2
        assert job["num_nodes"] == 1

    def test_add_job_with_parents(self, pbx, tmp_path):
        """Test adding job with parent dependencies."""
        # Add parent job
        parent_dir = tmp_path / "parent_job"
        parent_dir.mkdir()
        parent_id = pbx.add_job(path=str(parent_dir), app="python", config="polaris")

        # Add child job
        child_dir = tmp_path / "child_job"
        child_dir.mkdir()
        child_id = pbx.add_job(
            path=str(child_dir), app="python", config="polaris", parents=[parent_id]
        )

        child_job = pbx.get_job(child_id)
        # Check that parents are stored (as JSON string)
        assert child_job["parents"] is not None

    def test_add_job_with_input_file(self, pbx, tmp_path):
        """Test adding job with input file specified."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()

        job_id = pbx.add_job(
            path=str(job_dir), app="python", config="polaris", input_file="script.py"
        )

        job = pbx.get_job(job_id)
        assert job["in_file"] == "script.py"


class TestListJobs:
    """Test list_jobs functionality."""

    def test_list_jobs_empty(self, pbx):
        """Test listing jobs when database is empty."""
        jobs = pbx.list_jobs()
        assert jobs == []

    def test_list_jobs_filter_by_status(self, pbx, tmp_path):
        """Test filtering jobs by status."""
        # Add jobs with different statuses
        for i, status in enumerate(["Ready", "Running", "Done"]):
            job_dir = tmp_path / f"job_{i}"
            job_dir.mkdir()
            job_id = pbx.add_job(
                path=str(job_dir), app="python", config="polaris", status=status
            )

        ready_jobs = pbx.list_jobs(status="Ready")
        assert len(ready_jobs) == 1
        assert ready_jobs[0]["status"] == "Ready"

    def test_list_jobs_filter_by_app(self, pbx, tmp_path):
        """Test filtering jobs by app."""
        # Add jobs with different apps
        for app in ["python", "lammps"]:
            job_dir = tmp_path / f"job_{app}"
            job_dir.mkdir()
            pbx.add_job(path=str(job_dir), app=app, config="polaris")

        python_jobs = pbx.list_jobs(app="python")
        assert len(python_jobs) == 1
        assert python_jobs[0]["app"] == "python"

    def test_list_jobs_filter_by_tag(self, pbx, tmp_path):
        """Test filtering jobs by tag."""
        # Add jobs with different tags
        for tag in ["tag1", "tag2"]:
            job_dir = tmp_path / f"job_{tag}"
            job_dir.mkdir()
            pbx.add_job(path=str(job_dir), app="python", config="polaris", tag=tag)

        tagged_jobs = pbx.list_jobs(tag="tag1")
        assert len(tagged_jobs) == 1
        assert tagged_jobs[0]["tag"] == "tag1"


class TestGetJob:
    """Test get_job functionality."""

    def test_get_job_exists(self, pbx, tmp_path):
        """Test getting an existing job."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        job_id = pbx.add_job(path=str(job_dir), app="python", config="polaris")

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
        job_id = pbx.add_job(path=str(job_dir), app="python", config="polaris")

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
        job_ids = []
        for i in range(3):
            job_dir = tmp_path / f"job_{i}"
            job_dir.mkdir()
            job_id = pbx.add_job(path=str(job_dir), app="python", config="polaris")
            job_ids.append(job_id)

        count = pbx.remove_jobs(job_ids)
        assert count == 3

        # Verify all jobs are gone
        for job_id in job_ids:
            with pytest.raises(JobNotFoundError):
                pbx.get_job(job_id)

    def test_remove_all_jobs(self, pbx, tmp_path):
        """Test removing all jobs."""
        # Add multiple jobs
        for i in range(3):
            job_dir = tmp_path / f"job_{i}"
            job_dir.mkdir()
            pbx.add_job(path=str(job_dir), app="python", config="polaris")

        count = pbx.remove_all_jobs()
        assert count == 3

        # Verify all jobs are gone
        jobs = pbx.list_jobs()
        assert len(jobs) == 0


class TestUpdateJob:
    """Test update_job functionality."""

    def test_update_job_status(self, pbx, tmp_path):
        """Test updating job status."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        job_id = pbx.add_job(
            path=str(job_dir), app="python", config="polaris", status="Ready"
        )

        result = pbx.update_job(job_id, status="Running")
        assert result is True

        job = pbx.get_job(job_id)
        assert job["status"] == "Running"

    def test_update_job_tag(self, pbx, tmp_path):
        """Test updating job tag."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        job_id = pbx.add_job(
            path=str(job_dir), app="python", config="polaris", tag="old_tag"
        )

        pbx.update_job(job_id, tag="new_tag")

        job = pbx.get_job(job_id)
        assert job["tag"] == "new_tag"

    def test_update_job_invalid_nnodes(self, pbx, tmp_path):
        """Test updating job with invalid nnodes."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        job_id = pbx.add_job(path=str(job_dir), app="python", config="polaris")

        with pytest.raises(ValidationError):
            pbx.update_job(job_id, nnodes=0)

    def test_update_job_add_dependencies(self, pbx, tmp_path):
        """Test adding dependencies to a job."""
        # Add parent job
        parent_dir = tmp_path / "parent"
        parent_dir.mkdir()
        parent_id = pbx.add_job(path=str(parent_dir), app="python", config="polaris")

        # Add child job
        child_dir = tmp_path / "child"
        child_dir.mkdir()
        child_id = pbx.add_job(path=str(child_dir), app="python", config="polaris")

        # Add dependency
        pbx.update_job(child_id, add_deps=[parent_id])

        child_job = pbx.get_job(child_id)
        assert child_job["parents"] is not None

    def test_update_job_circular_dependency(self, pbx, tmp_path):
        """Test that circular dependencies are prevented."""
        job_dir = tmp_path / "test_job"
        job_dir.mkdir()
        job_id = pbx.add_job(path=str(job_dir), app="python", config="polaris")

        with pytest.raises(ValidationError, match="cannot be parents of themselves"):
            pbx.update_job(job_id, add_deps=[job_id])


class TestFilterJobs:
    """Test filter_jobs functionality."""

    def test_filter_jobs_by_status(self, pbx, tmp_path):
        """Test filtering jobs by status."""
        # Add jobs with different statuses
        for i, status in enumerate(["Ready", "Running", "Done"]):
            job_dir = tmp_path / f"job_{i}"
            job_dir.mkdir()
            pbx.add_job(
                path=str(job_dir), app="python", config="polaris", status=status
            )

        ready_ids = pbx.filter_jobs(status="Ready")
        assert len(ready_ids) == 1

    def test_filter_jobs_empty_result(self, pbx):
        """Test filtering when no jobs match."""
        ids = pbx.filter_jobs(status="Done")
        assert ids == []


class TestAddJobs:
    """Test add_jobs (multiple) functionality."""

    def test_add_jobs_multiple(self, pbx, tmp_path):
        """Test adding multiple jobs."""
        paths = []
        for i in range(3):
            job_dir = tmp_path / f"job_{i}"
            job_dir.mkdir()
            paths.append(str(job_dir))

        job_ids, failures = pbx.add_jobs(paths=paths, app="python", config="polaris")

        assert len(job_ids) == 3
        assert len(failures) == 0
        # All should succeed
        for job_id in job_ids:
            assert job_id is not None
            assert isinstance(job_id, int)

    def test_add_jobs_partial_failure(self, pbx, tmp_path):
        """Test adding jobs when some fail."""
        paths = [
            str(tmp_path / "valid_job"),  # Will be created
            "/nonexistent/path",  # Will fail
        ]
        (tmp_path / "valid_job").mkdir()

        job_ids, failures = pbx.add_jobs(paths=paths, app="python", config="polaris")

        assert len(job_ids) == 1  # One successful job
        assert len(failures) == 1  # One failed job
        
        # First should succeed
        assert job_ids[0] is not None
        assert isinstance(job_ids[0], int)
        
        # Second should fail
        failed_path, error_msg = failures[0]
        assert failed_path == "/nonexistent/path"
        assert "does not exist" in error_msg
