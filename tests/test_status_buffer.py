"""
Tests for StatusBuffer functionality.
"""

import pytest
import tempfile
import sqlite3
from pathlib import Path

from parslbox.database.status_buffer import StatusBuffer
from parslbox.database import database


class TestStatusBuffer:
    """Test cases for StatusBuffer class."""
    
    def setup_method(self):
        """Set up test database and StatusBuffer."""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.db_path = Path(self.temp_db.name)
        self.temp_db.close()
        
        # Initialize database
        database.initialize_database(self.db_path)
        
        # Add some test jobs
        database.add_job(self.db_path, "/test/path1", "test_app", 1, 0, 1.0, "test_tag", status="Ready")
        database.add_job(self.db_path, "/test/path2", "test_app", 1, 0, 1.0, "test_tag", status="Ready")
        database.add_job(self.db_path, "/test/path3", "test_app", 1, 0, 1.0, "test_tag", status="Ready")
        
        # Create StatusBuffer
        self.status_buffer = StatusBuffer(self.db_path)
    
    def teardown_method(self):
        """Clean up test database."""
        self.db_path.unlink(missing_ok=True)
    
    def test_buffer_single_status_update(self):
        """Test buffering and flushing a single status update."""
        # Buffer a status update
        self.status_buffer.add_status_update(1, status="Running")
        
        # Verify it's buffered but not yet in database
        assert self.status_buffer.get_buffer_size() == 1
        jobs = database.get_jobs_by_ids(self.db_path, [1])
        assert jobs[0]['status'] == "Ready"  # Still original status
        
        # Flush and verify database is updated
        updated_count = self.status_buffer.flush_all()
        assert updated_count == 1
        assert self.status_buffer.get_buffer_size() == 0
        
        jobs = database.get_jobs_by_ids(self.db_path, [1])
        assert jobs[0]['status'] == "Running"
    
    def test_buffer_multiple_status_updates(self):
        """Test buffering and flushing multiple status updates."""
        # Buffer multiple status updates
        self.status_buffer.add_status_update(1, status="Running")
        self.status_buffer.add_status_update(2, status="Running")
        self.status_buffer.add_status_update(3, status="Failed")
        
        # Verify they're buffered
        assert self.status_buffer.get_buffer_size() == 3
        
        # Flush and verify database is updated
        updated_count = self.status_buffer.flush_all()
        assert updated_count == 3
        assert self.status_buffer.get_buffer_size() == 0
        
        # Verify all jobs were updated correctly
        jobs = database.get_jobs_by_ids(self.db_path, [1, 2, 3])
        assert jobs[0]['status'] == "Running"
        assert jobs[1]['status'] == "Running"
        assert jobs[2]['status'] == "Failed"
    
    def test_buffer_overwrite_status(self):
        """Test that newer status updates overwrite older ones for the same job."""
        # Buffer multiple updates for the same job
        self.status_buffer.add_status_update(1, status="Running")
        self.status_buffer.add_status_update(1, status="Done")
        
        # Should only have one job in buffer
        assert self.status_buffer.get_buffer_size() == 1
        
        # Flush and verify only the latest status is applied
        updated_count = self.status_buffer.flush_all()
        assert updated_count == 1
        
        jobs = database.get_jobs_by_ids(self.db_path, [1])
        assert jobs[0]['status'] == "Done"
    
    def test_buffer_mixed_field_updates(self):
        """Test buffering updates with different field combinations."""
        # Buffer updates with different fields
        self.status_buffer.add_status_update(1, status="Running")
        self.status_buffer.add_status_update(2, status="Submitted", sched_job_id="job123")
        self.status_buffer.add_status_update(3, status="Failed")
        
        # Flush and verify all updates are applied
        updated_count = self.status_buffer.flush_all()
        assert updated_count == 3
        
        # Verify updates
        jobs = database.get_jobs_by_ids(self.db_path, [1, 2, 3])
        assert jobs[0]['status'] == "Running"
        assert jobs[1]['status'] == "Submitted"
        assert jobs[1]['sched_job_id'] == "job123"
        assert jobs[2]['status'] == "Failed"
    
    def test_empty_buffer_flush(self):
        """Test flushing an empty buffer."""
        updated_count = self.status_buffer.flush_all()
        assert updated_count == 0
        assert self.status_buffer.get_buffer_size() == 0
    
    def test_clear_buffer(self):
        """Test clearing buffer without flushing."""
        # Buffer some updates
        self.status_buffer.add_status_update(1, status="Running")
        self.status_buffer.add_status_update(2, status="Failed")
        assert self.status_buffer.get_buffer_size() == 2
        
        # Clear buffer
        self.status_buffer.clear_buffer()
        assert self.status_buffer.get_buffer_size() == 0
        
        # Verify database was not updated
        jobs = database.get_jobs_by_ids(self.db_path, [1, 2])
        assert jobs[0]['status'] == "Ready"
        assert jobs[1]['status'] == "Ready"


if __name__ == "__main__":
    pytest.main([__file__])
