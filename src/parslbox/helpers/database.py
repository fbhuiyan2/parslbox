import sqlite3
import typer
from pathlib import Path
from typing import List, Dict, Any, Optional

# Updated schema with 'app' and 'tag' columns
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id INTEGER PRIMARY KEY,
    app TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    ngpus INTEGER NOT NULL DEFAULT 1,
    sched_job_id TEXT,
    tag TEXT,
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

def initialize_database(db_path: Path):
    """
    Ensures the database directory and file exist, creating them if necessary.
    Provides clear error handling if directory creation fails.
    """
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        typer.secho(f"❌ Error: Permission denied to create directory: {db_path.parent}", fg=typer.colors.RED, err=True)
        typer.secho(f"Please check permissions or create the directory manually: mkdir -p {db_path.parent}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)
    
    try:
        with sqlite3.connect(db_path) as con:
            cur = con.cursor()
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(CREATE_TRIGGER_SQL)
    except sqlite3.OperationalError as e:
        typer.secho(f"❌ A database error occurred: {e}", fg=typer.colors.RED, err=True)
        typer.secho(f"Failed to open or initialize the database at: {db_path}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)

def add_job(db_path: Path, path: str, app: str, ngpus: int, tag: Optional[str], status: str = 'Created') -> int:
    """Adds a new job to the database with app and tag info."""
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        cur.execute(
            "INSERT INTO jobs (path, app, tag, status, ngpus) VALUES (?, ?, ?, ?, ?)",
            (path, app, tag, status, ngpus)
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

def get_jobs(db_path: Path, status: Optional[str] = None, app: Optional[str] = None, tag: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Retrieves jobs from the database, allowing for filtering by status, app, and tag.
    Filters are combined with AND logic.
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
    ngpus: Optional[int] = None
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

    if ngpus is not None:
        set_clauses.append("ngpus = ?")
        params.append(ngpus)

    if sched_job_id is not None:
        set_clauses.append("sched_job_id = ?")
        params.append(status.capitalize())

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
