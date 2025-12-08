"""
Shared fixtures for API tests.
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
