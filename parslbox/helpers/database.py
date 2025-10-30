import sqlite3
import typer
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# Updated schema with individual resource columns and env_file support
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY,
    app TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    num_nodes INTEGER DEFAULT 1,
    ngpus INTEGER DEFAULT 0,
    node_occupancy REAL DEFAULT 1.0,
    sched_job_id TEXT,
    tag TEXT,
    in_file TEXT,
    mpi_opts TEXT,
    env_file TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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

def get_expected_schema() -> Dict[str, str]:
    """
    Parse the CREATE_TABLE_SQL to extract expected columns and their types.
    Returns a dict like: {'column_name': 'column_type', ...}
    """
    # Extract the content between parentheses in CREATE TABLE
    table_content = re.search(r'CREATE TABLE.*?\((.*)\)', CREATE_TABLE_SQL, re.DOTALL)
    if not table_content:
        return {}
    
    content = table_content.group(1)
    schema = {}
    
    # Split by commas and process each line
    for line in content.split(','):
        line = line.strip()
        if not line or line.upper().startswith(('PRIMARY KEY', 'FOREIGN KEY', 'UNIQUE', 'CHECK', 'CONSTRAINT')):
            continue
            
        # Extract column name and type
        parts = line.split()
        if len(parts) >= 2:
            column_name = parts[0].strip()
            column_type = parts[1].strip()
            
            # Handle DEFAULT values and constraints
            if 'DEFAULT' in line.upper():
                # Find the DEFAULT part and include it in the type
                default_match = re.search(r'DEFAULT\s+([^\s,]+(?:\s+[^\s,]+)*)', line, re.IGNORECASE)
                if default_match:
                    column_type += f" DEFAULT {default_match.group(1)}"
            
            # Handle NOT NULL
            if 'NOT NULL' in line.upper():
                column_type += " NOT NULL"
                
            schema[column_name] = column_type
    
    return schema

def get_current_schema(db_path: Path) -> Dict[str, str]:
    """
    Get the current database schema for the jobs table.
    Returns a dict like: {'column_name': 'column_type', ...}
    """
    try:
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute("PRAGMA table_info(jobs)")
            rows = cur.fetchall()
            
            schema = {}
            for row in rows:
                # row format: (cid, name, type, notnull, dflt_value, pk)
                column_name = row[1]
                column_type = row[2]
                
                # Add NOT NULL if applicable
                if row[3]:  # notnull
                    column_type += " NOT NULL"
                
                # Add DEFAULT if applicable
                if row[4] is not None:  # dflt_value
                    default_value = row[4]
                    # Handle string defaults
                    if isinstance(default_value, str) and not default_value.upper().startswith('CURRENT_'):
                        default_value = f"'{default_value}'"
                    column_type += f" DEFAULT {default_value}"
                
                schema[column_name] = column_type
            
            return schema
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return {}

def migrate_database(db_path: Path):
    """
    Automatically migrate database to match expected schema.
    Compares current schema with expected schema and adds missing columns.
    """
    # Skip migration if database doesn't exist yet
    if not db_path.exists():
        return
    
    try:
        expected_schema = get_expected_schema()
        current_schema = get_current_schema(db_path)
        
        # Find missing columns
        missing_columns = set(expected_schema.keys()) - set(current_schema.keys())
        
        if missing_columns:
            typer.secho(f"🔄 Migrating database schema...", fg=typer.colors.BLUE)
            
            with sqlite3.connect(db_path) as con:
                cur = con.cursor()
                for column in sorted(missing_columns):  # Sort for consistent order
                    column_type = expected_schema[column]
                    try:
                        cur.execute(f"ALTER TABLE jobs ADD COLUMN {column} {column_type}")
                        typer.secho(f"✅ Added column: {column} {column_type}", fg=typer.colors.GREEN)
                    except sqlite3.OperationalError as e:
                        typer.secho(f"⚠️  Warning: Could not add column {column}: {e}", fg=typer.colors.YELLOW)
            
            typer.secho(f"🎉 Database migration completed!", fg=typer.colors.GREEN)
        
    except Exception as e:
        typer.secho(f"⚠️  Warning: Database migration failed: {e}", fg=typer.colors.YELLOW)
        typer.secho("Database will continue to work, but some features may not be available.", fg=typer.colors.YELLOW)

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
        # Run migration first (for existing databases)
        migrate_database(db_path)
        
        # Create table and trigger (for new databases or if migration didn't cover everything)
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(CREATE_TRIGGER_SQL)
    except sqlite3.OperationalError as e:
        typer.secho(f"❌ A database error occurred: {e}", fg=typer.colors.RED, err=True)
        typer.secho(f"Failed to open or initialize the database at: {db_path}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

def add_job(db_path: Path, path: str, app: str, num_nodes: int, ngpus: int, node_occupancy: float, tag: Optional[str], in_file: Optional[str] = None, mpi_opts: Optional[str] = None, env_file: Optional[str] = None, status: str = 'Ready') -> int:
    """Adds a new job to the database with app, tag, input file info, resource specification, MPI options, and environment file."""
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO jobs (path, app, tag, in_file, mpi_opts, env_file, status, num_nodes, ngpus, node_occupancy) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (path, app, tag, in_file, mpi_opts, env_file, status, num_nodes, ngpus, node_occupancy)
        )
        return cur.lastrowid

'''
def get_jobs(db_path: Path, status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieves jobs from the database, optionally filtering by status."""
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        query = "SELECT * FROM jobs"
        params = []
        if status_filter:
            query += " WHERE status = ?"
            params.append(status_filter.capitalize())
        query += " ORDER BY job_id ASC"
        results = cur.execute(query, params).fetchall()
        return [dict(row) for row in results]
'''

def get_jobs(db_path: Path, status: Optional[str] = None, app: Optional[str] = None, tag: Optional[str] = None, path: Optional[str] = None, in_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves jobs from the database, allowing for filtering by status, app, tag, path, and in_file.
    Filters are combined with AND logic. Path and in_file use pattern matching (LIKE).
    """
    with sqlite3.connect(db_path) as con:
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
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(f"DELETE FROM jobs WHERE job_id IN ({','.join('?' for _ in job_ids)})", job_ids)
        return cur.rowcount

def remove_all_jobs(db_path: Path) -> int:
    """Removes all jobs from the database."""
    with sqlite3.connect(db_path) as con:
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
    
    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        
        # Create placeholders for the IN clause
        placeholders = ','.join('?' for _ in job_ids)
        query = f"SELECT * FROM jobs WHERE job_id IN ({placeholders}) ORDER BY job_id ASC"
        
        results = cur.execute(query, job_ids).fetchall()
        return [dict(row) for row in results]

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
    in_file: Optional[str] = None,
    mpi_opts: Optional[str] = None,
    env_file: Optional[str] = None
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

    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(query, final_params)
        return cur.rowcount
