import os
from pathlib import Path

def get_db_path():
    """Get the database file path from environment variable or default location.
    
    Environment variable PBX_DB_PATH can point to either:
    - A directory (e.g., '/custom/path/') - will append 'job_database_pbx.db'
    - A full file path (e.g., '/custom/path/my_database.db') - will use as-is
    """
    env_path = os.getenv("PBX_DB_PATH")
    if env_path:
        path = Path(env_path)
        # Check if it's a file path (has .db extension)
        if path.suffix == ".db":
            return path
        # Otherwise treat it as a directory
        return path / "job_database_pbx.db"
    return Path.home() / ".parslbox" / "job_database_pbx.db"

def get_config_path():
    """Get the config file path from environment variable or default location.
    
    Environment variable PBX_CONFIG_PATH can point to either:
    - A directory (e.g., '/custom/path/') - will append 'config.yaml'
    - A full file path (e.g., '/custom/path/my_config.yaml') - will use as-is
    """
    env_path = os.getenv("PBX_CONFIG_PATH")
    if env_path:
        path = Path(env_path)
        # Check if it's a file path (has .yaml or .yml extension)
        if path.suffix in [".yaml", ".yml"]:
            return path
        # Otherwise treat it as a directory
        return path / "config.yaml"
    return Path.home() / ".parslbox" / "config.yaml"

# Define the path to the SQLite database file
DB_FILE = get_db_path()

# package config file
PBX_CONFIG_FILE = get_config_path()






