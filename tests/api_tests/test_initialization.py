"""
Unit tests for ParslBox initialization.
"""

import pytest
from pathlib import Path
from parslbox.api import ParslBox


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
        # Create a minimal config file for the test
        config_path.write_text("# Minimal test config\nsystems: {}\napps: {}")
        pbx = ParslBox(db_path=temp_db, config_path=config_path)
        assert pbx.db_path == temp_db
        assert pbx.config_path == config_path
