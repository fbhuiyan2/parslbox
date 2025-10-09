from pathlib import Path

# Define the root directory for the application's data
APP_DIR = Path.home() / ".parslbox"

# Define the path to the SQLite database file
DB_FILE = APP_DIR / "job_database.db"

# package config file
PBX_CONFIG_FILE = APP_DIR / "config.yaml"
