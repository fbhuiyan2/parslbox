import sqlite3
import yaml
import typer
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional


def get_current_db_schema(schema_file_path: Path) -> Dict[str, str]:
    """Read current schema from YAML file"""
    if not schema_file_path.exists():
        return {}  # No schema file = first run
    
    try:
        with open(schema_file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        typer.secho(f"⚠️  Warning: Could not read schema file: {e}", fg=typer.colors.YELLOW)
        return {}


def save_current_db_schema(schema_file_path: Path, create_table_sql: str, create_trigger_sql: str):
    """Save current schema to YAML file"""
    schema_data = {
        'create_table_sql': create_table_sql.strip(),
        'create_trigger_sql': create_trigger_sql.strip(),
        'last_updated': datetime.now().isoformat()
    }
    
    try:
        with open(schema_file_path, 'w') as f:
            yaml.dump(schema_data, f, default_flow_style=False)
    except Exception as e:
        typer.secho(f"⚠️  Warning: Could not save schema file: {e}", fg=typer.colors.YELLOW)


def needs_migration(schema_file_path: Path, current_create_table: str, current_create_trigger: str) -> bool:
    """Simple string comparison to detect schema changes"""
    saved_schema = get_current_db_schema(schema_file_path)
    
    if not saved_schema:
        return False  # First run, no migration needed
    
    return (saved_schema.get('create_table_sql', '').strip() != current_create_table.strip() or
            saved_schema.get('create_trigger_sql', '').strip() != current_create_trigger.strip())


def get_column_default(column_name: str):
    """Get appropriate default value for new columns"""
    defaults = {
        'num_nodes': 1,
        'ngpus': 0,
        'node_occupancy': 1.0,
        'in_file': '',
        'mpi_opts': None,
        'env_file': None,
        'parents': None,
        'tag': None,
        'sched_job_id': None,
        'status': 'Ready'
    }
    
    if column_name not in defaults:
        typer.secho(f"⚠️  Warning: No default value found for new column '{column_name}', using None as default.", 
                   fg=typer.colors.YELLOW)
    
    return defaults.get(column_name, None)


def migrate_database(db_path: Path, new_create_table_sql: str, new_create_trigger_sql: str):
    """
    Recreate database table with new schema.
    Saves existing jobs, creates new table, transfers data.
    Handles new columns by using their DEFAULT values or appropriate fallbacks.
    """
    typer.secho(f"🔄 Migrating database schema...", fg=typer.colors.BLUE)
    
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        
        # 1. Save existing data
        cur.execute("SELECT * FROM jobs")
        existing_jobs = cur.fetchall()
        
        # Get old column info
        cur.execute("PRAGMA table_info(jobs)")
        old_columns = [row[1] for row in cur.fetchall()]
        
        # 2. Drop old table and create new one
        cur.execute("DROP TABLE IF EXISTS jobs")
        cur.execute(new_create_table_sql)
        cur.execute(new_create_trigger_sql)
        
        # 3. Get new column info with defaults
        cur.execute("PRAGMA table_info(jobs)")
        new_column_info = {row[1]: row[4] for row in cur.fetchall()}  # {name: default_value}
        new_columns = list(new_column_info.keys())
        
        # 4. Transfer data
        if existing_jobs:
            for job_row in existing_jobs:
                job_dict = dict(zip(old_columns, job_row))
                
                # Handle specific data transformations
                if 'in_file' in job_dict and job_dict['in_file'] is None:
                    job_dict['in_file'] = ''
                
                # Build complete row for new schema
                new_row_data = {}
                for col in new_columns:
                    if col in job_dict:
                        # Use existing data (INCLUDING job_id to preserve IDs)
                        new_row_data[col] = job_dict[col]
                    else:
                        # New column - use DEFAULT value or appropriate fallback
                        default_val = new_column_info[col]
                        if default_val is not None:
                            new_row_data[col] = default_val
                        else:
                            new_row_data[col] = get_column_default(col)
                
                # Insert the row WITH explicit job_id to preserve original IDs
                columns_str = ','.join(new_row_data.keys())
                placeholders = ','.join(['?' for _ in new_row_data.values()])
                cur.execute(f"INSERT INTO jobs ({columns_str}) VALUES ({placeholders})", 
                           list(new_row_data.values()))
    
    typer.secho(f"✅ Database migration completed!", fg=typer.colors.GREEN)
