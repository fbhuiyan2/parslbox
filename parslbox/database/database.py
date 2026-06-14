import sqlite3
import typer
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from .database_migrate import needs_migration, migrate_database
import yaml
from datetime import datetime

# Updated schema with individual resource columns, env_file support, and parent dependencies
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY,
    app TEXT NOT NULL,
    path TEXT NOT NULL,
    status TEXT NOT NULL,
    num_nodes INTEGER DEFAULT 1,
    ngpus INTEGER DEFAULT 0,
    node_occupancy REAL DEFAULT 1.0,
    ranks_per_node INTEGER DEFAULT 1,
    sched_job_id TEXT,
    tag TEXT,
    in_file TEXT,
    mpi_opts TEXT,
    env_file TEXT,
    parents TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(path, in_file)
);
"""

CREATE_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS update_jobs_timestamp
AFTER UPDATE ON jobs
FOR EACH ROW
BEGIN
    UPDATE jobs
    SET timestamp = CURRENT_TIMESTAMP
    WHERE job_id = OLD.job_id;
END;
"""


def configure_connection(conn: sqlite3.Connection) -> None:
    """
    Configure SQLite connection for better concurrency and performance.
    
    Args:
        conn: SQLite connection to configure
    """
    # Enable WAL mode for better concurrency and reduced locking
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")  # Faster than FULL, still safe
    conn.execute("PRAGMA cache_size=10000;")    # 10MB cache for better performance
    conn.execute("PRAGMA temp_store=memory;")   # Use RAM for temporary tables
    conn.execute("PRAGMA busy_timeout=30000;")  # Wait 30 seconds for locks before failing


def get_configured_connection(db_path: Path) -> sqlite3.Connection:
    """
    Get a SQLite connection with WAL mode and performance optimizations.
    
    Args:
        db_path: Path to the SQLite database file
        
    Returns:
        Configured SQLite connection
    """
    conn = sqlite3.connect(db_path)
    configure_connection(conn)
    return conn





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



def initialize_database(db_path: Path):
    """
    Ensures the database directory and file exist, creating them if necessary.
    Runs migrations to update schema if needed.
    Provides clear error handling if directory creation fails.
    """
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        typer.secho(f"❌ Error: Permission denied to create directory: {db_path.parent}", fg=typer.colors.RED, err=True)
        typer.secho(f"Please check permissions or create the directory manually: mkdir -p {db_path.parent}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)
    
    try:
        schema_file_path = db_path.parent / "db_schema_reference.yaml"
        
        # Check if migration is needed
        if needs_migration(schema_file_path, CREATE_TABLE_SQL, CREATE_TRIGGER_SQL):
            migrate_database(db_path, CREATE_TABLE_SQL, CREATE_TRIGGER_SQL)
        
        # Create/update database
        with get_configured_connection(db_path) as con:
            cur = con.cursor()
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(CREATE_TRIGGER_SQL)
        
        # Always update schema file after successful initialization
        save_current_db_schema(schema_file_path, CREATE_TABLE_SQL, CREATE_TRIGGER_SQL)
        
    except sqlite3.OperationalError as e:
        typer.secho(f"❌ A database error occurred: {e}", fg=typer.colors.RED, err=True)
        typer.secho(f"Failed to open or initialize the database at: {db_path}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

def add_job(db_path: Path, path: str, app: str, num_nodes: int, ngpus: int, node_occupancy: float, tag: Optional[str], in_file: Optional[str] = None, mpi_opts: Optional[str] = None, env_file: Optional[str] = None, parents: Optional[List[int]] = None, ranks_per_node: int = 1, status: str = 'Ready') -> int:
    """Adds a new job to the database with app, tag, input file info, resource specification, MPI options, environment file, and parent dependencies."""
    
    # Convert None to empty string for in_file to ensure consistent UNIQUE constraint behavior
    in_file_value = in_file if in_file is not None else ""
    
    # Convert parent list to JSON string
    parents_str = None
    if parents:
        import json
        parents_str = json.dumps([str(p) for p in parents])
    
    with get_configured_connection(db_path) as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO jobs (path, app, tag, in_file, mpi_opts, env_file, parents, status, num_nodes, ngpus, node_occupancy, ranks_per_node) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (path, app, tag, in_file_value, mpi_opts, env_file, parents_str, status, num_nodes, ngpus, node_occupancy, ranks_per_node)
        )
        return cur.lastrowid


def get_jobs(db_path: Path, status: Optional[str] = None, app: Optional[str] = None, tag: Optional[str] = None, path: Optional[str] = None, in_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves jobs from the database, allowing for filtering by status, app, tag, path, and in_file.
    Filters are combined with AND logic. Path and in_file use pattern matching (LIKE).
    """
    with get_configured_connection(db_path) as con:
        con.row_factory = sqlite3.Row  # Access columns by name
        cur = con.cursor()
        
        # Start with the base query and empty lists for conditions and parameters
        base_query = "SELECT * FROM jobs"
        conditions = []
        params = []
        
        # Dynamically add conditions and parameters for each filter if it's provided
        if status:
            conditions.append("status = ?")
            params.append(status.capitalize())
        
        if app:
            conditions.append("app = ?")
            params.append(app)

        if tag:
            from parslbox.utils.tag_match import has_glob, to_sql_like
            if has_glob(tag):
                conditions.append("tag LIKE ? ESCAPE '\\'")
                params.append(to_sql_like(tag))
            else:
                conditions.append("tag = ?")
                params.append(tag)
        
        if path:
            conditions.append("path LIKE ?")
            params.append(f"%{path}%")
        
        if in_file:
            conditions.append("in_file LIKE ?")
            params.append(f"%{in_file}%")
        
        # If any conditions were added, join them with "AND" and append to the query
        if conditions:
            query = f"{base_query} WHERE {' AND '.join(conditions)}"
        else:
            query = base_query
        
        # Maintain a consistent order
        query += " ORDER BY job_id ASC"
        
        results = cur.execute(query, params).fetchall()
        # Convert sqlite3.Row objects to plain dictionaries
        return [dict(row) for row in results]

def remove_jobs_by_id(db_path: Path, job_ids: List[int]) -> int:
    """Removes jobs by their IDs and returns the number of rows deleted."""
    with get_configured_connection(db_path) as con:
        cur = con.cursor()
        cur.execute(f"DELETE FROM jobs WHERE job_id IN ({','.join('?' for _ in job_ids)})", job_ids)
        return cur.rowcount

def remove_all_jobs(db_path: Path) -> int:
    """Removes all jobs from the database."""
    with get_configured_connection(db_path) as con:
        cur = con.cursor()
        cur.execute("DELETE FROM jobs")
        return cur.rowcount

def get_jobs_by_ids(db_path: Path, job_ids: List[int]) -> List[Dict[str, Any]]:
    """
    Retrieves specific jobs from the database by their IDs.
    Returns jobs in the same order as the provided job_ids list.
    """
    if not job_ids:
        return []
    
    with get_configured_connection(db_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        
        # Create placeholders for the IN clause
        placeholders = ','.join('?' for _ in job_ids)
        query = f"SELECT * FROM jobs WHERE job_id IN ({placeholders}) ORDER BY job_id ASC"
        
        results = cur.execute(query, job_ids).fetchall()
        return [dict(row) for row in results]

def get_jobs_by_sched_id(
    db_path: Path,
    sched_job_id: str,
    statuses: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch jobs whose `sched_job_id` matches, optionally filtered by status.

    Used by `pbx qdel` / `pbx scancel` reconciliation: after killing a batch
    job, any jobs the orchestrator's signal handler didn't manage to flip to
    `Killed` (Step 2 of perform_shutdown) are still sitting in `Running` or
    `Submitted`. They are uniquely identifiable by their `sched_job_id`
    (overwritten on every claim, so only stuck jobs from the killed batch
    still carry its id in a non-terminal status).
    """
    if not sched_job_id:
        return []
    with get_configured_connection(db_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        if statuses:
            placeholders = ','.join('?' for _ in statuses)
            query = (
                f"SELECT * FROM jobs WHERE sched_job_id = ? "
                f"AND status IN ({placeholders}) ORDER BY job_id ASC"
            )
            params = [sched_job_id, *statuses]
        else:
            query = "SELECT * FROM jobs WHERE sched_job_id = ? ORDER BY job_id ASC"
            params = [sched_job_id]
        results = cur.execute(query, params).fetchall()
        return [dict(row) for row in results]


def parse_existing_parents(parents_str: Optional[str]) -> List[int]:
    """
    Parse existing parents from JSON string to list of integers.
    
    Args:
        parents_str: JSON string of parent IDs (e.g., '["1", "2", "3"]')
        
    Returns:
        List of parent job IDs as integers
    """
    if not parents_str:
        return []
    
    try:
        import json
        parent_strings = json.loads(parents_str)
        return [int(p) for p in parent_strings]
    except (json.JSONDecodeError, ValueError, TypeError):
        return []


def add_dependencies(existing_parents: List[int], new_parents: List[int]) -> List[int]:
    """
    Add new parent dependencies to existing ones, avoiding duplicates.
    
    Args:
        existing_parents: Current list of parent job IDs
        new_parents: New parent job IDs to add
        
    Returns:
        Updated list of parent job IDs (sorted, no duplicates)
    """
    # Combine lists and remove duplicates
    combined = set(existing_parents + new_parents)
    return sorted(list(combined))


def remove_dependencies(existing_parents: List[int], remove_parents: List[int]) -> tuple[List[int], List[int]]:
    """
    Remove parent dependencies from existing ones.
    
    Args:
        existing_parents: Current list of parent job IDs
        remove_parents: Parent job IDs to remove
        
    Returns:
        Tuple of (updated_parents_list, not_found_parents_list)
    """
    existing_set = set(existing_parents)
    remove_set = set(remove_parents)
    
    # Find parents that don't exist in current list
    not_found = sorted(list(remove_set - existing_set))
    
    # Remove existing parents
    updated = sorted(list(existing_set - remove_set))
    
    return updated, not_found


def validate_parent_job_ids(db_path: Path, parent_ids: List[int]) -> tuple[List[int], List[int]]:
    """
    Validate that parent job IDs exist in the database.
    
    Args:
        db_path: Path to database file
        parent_ids: List of parent job IDs to validate
        
    Returns:
        Tuple of (valid_ids, invalid_ids)
    """
    if not parent_ids:
        return [], []
    
    existing_jobs = get_jobs_by_ids(db_path, parent_ids)
    existing_ids = {job['job_id'] for job in existing_jobs}
    
    #valid_ids = [pid for pid in parent_ids if pid in existing_ids]
    invalid_ids = [pid for pid in parent_ids if pid not in existing_ids]
    
    return invalid_ids


def update_jobs(
    db_path: Path,
    job_ids: List[int],
    status: Optional[str] = None,
    sched_job_id: Optional[str] = None,
    app: Optional[str] = None,
    tag: Optional[str] = None,
    num_nodes: Optional[int] = None,
    ngpus: Optional[int] = None,
    node_occupancy: Optional[float] = None,
    ranks_per_node: Optional[int] = None,
    in_file: Optional[str] = None,
    mpi_opts: Optional[str] = None,
    env_file: Optional[str] = None,
    parents: Optional[List[int]] = None
) -> int:
    """
    Updates jobs with the given IDs. Only fields that are not None will be updated.
    """
    if not job_ids:
        return 0

    set_clauses = []
    params = []

    # Dynamically build the SET part of the query
    if status is not None:
        set_clauses.append("status = ?")
        params.append(status.capitalize())
    
    if app is not None:
        set_clauses.append("app = ?")
        params.append(app)

    if tag is not None:
        set_clauses.append("tag = ?")
        params.append(tag)

    if num_nodes is not None:
        set_clauses.append("num_nodes = ?")
        params.append(num_nodes)

    if ngpus is not None:
        set_clauses.append("ngpus = ?")
        params.append(ngpus)

    if node_occupancy is not None:
        set_clauses.append("node_occupancy = ?")
        params.append(node_occupancy)

    if ranks_per_node is not None:
        set_clauses.append("ranks_per_node = ?")
        params.append(ranks_per_node)

    if sched_job_id is not None:
        set_clauses.append("sched_job_id = ?")
        params.append(sched_job_id)

    if in_file is not None:
        set_clauses.append("in_file = ?")
        params.append(in_file)

    if mpi_opts is not None:
        set_clauses.append("mpi_opts = ?")
        params.append(mpi_opts)

    if env_file is not None:
        set_clauses.append("env_file = ?")
        params.append(env_file)

    if parents is not None:
        set_clauses.append("parents = ?")
        # Convert parent list to JSON string
        import json
        parents_str = json.dumps([str(p) for p in parents]) if parents else None
        params.append(parents_str)

    # If no fields to update were provided, do nothing.
    if not set_clauses:
        return 0

    # Join the SET clauses with a comma
    set_statement = ", ".join(set_clauses)
    
    # Final parameters list includes the update values and the job IDs for the WHERE clause
    final_params = params + job_ids
    
    # Build the final query
    query = f"""
        UPDATE jobs
        SET {set_statement}
        WHERE job_id IN ({','.join('?' for _ in job_ids)})
    """

    with get_configured_connection(db_path) as con:
        cur = con.cursor()
        cur.execute(query, final_params)
        return cur.rowcount
