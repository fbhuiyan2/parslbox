"""Change detection for the two copies of a local project's database.

File mtime is no use here. database.py:53 puts the database in WAL mode, so a
commit is appended to the -wal sidecar and the .db file itself is untouched
until a checkpoint -- the remote could run a hundred jobs with the mtime
unmoved. Ask SQLite instead:

  MAX(timestamp)  catches inserts and updates (the update_jobs_timestamp
                  trigger at database.py:33-42 bumps it on every row update)
  COUNT(*)        catches deletes
  user_version    identifies the project, so a replaced database is not
                  mistaken for an unchanged one

CURRENT_TIMESTAMP has one-second resolution, so two changes inside the same
second look like one. That does not matter for what is being guarded here:
"the remote ran jobs while I was away" is minutes or hours apart.
"""

import sqlite3
import zlib
from pathlib import Path
from typing import Any, Dict, Optional


class SyncConflict(Exception):
    """The side about to be overwritten has changed since the last sync.

    `problem` is what is wrong and `remedy` is what to do about it, kept
    apart because the way out depends on the command: `pbx local push` has a
    --force flag of its own, `pbx qsub` does not.
    """

    def __init__(self, problem: str, remedy: str = ""):
        self.problem = problem
        self.remedy = remedy
        super().__init__(f"{problem}\n{remedy}" if remedy else problem)


def identity_of(created: str) -> int:
    """A stable 31-bit id for a project, derived from its creation stamp."""
    return zlib.crc32(str(created).encode()) & 0x7FFFFFFF


def fingerprint(db_path) -> Dict[str, Any]:
    """Read a database's change fingerprint. Cheap: one query, one pragma."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"No database at {db_path}")
    con = sqlite3.connect(str(db_path))
    try:
        ts, count = con.execute(
            "SELECT MAX(timestamp), COUNT(*) FROM jobs"
        ).fetchone()
        identity = con.execute("PRAGMA user_version").fetchone()[0]
    finally:
        con.close()
    return {"max_timestamp": ts, "count": count, "identity": identity}


def stamp_identity(db_path, created: str) -> int:
    """Write the project id into the database header.

    PRAGMA user_version lives in the file header, so it needs no schema
    change, it travels with the file through a transfer, and VACUUM INTO
    preserves it -- which is how the remote copy stays identifiable without
    a .pbxlocal.yaml of its own.
    """
    value = identity_of(created)
    con = sqlite3.connect(str(db_path))
    try:
        con.execute(f"PRAGMA user_version = {value}")
        con.commit()
    finally:
        con.close()
    return value


def unchanged(current: Dict[str, Any], recorded: Optional[Dict[str, Any]]) -> bool:
    """True when `current` matches what was recorded at the last sync.

    With nothing recorded, an empty database still counts as unchanged: there
    is no work in it to lose, so there is nothing to refuse over.
    """
    if not recorded:
        return not current.get("count") and current.get("max_timestamp") is None
    return (
        current.get("max_timestamp") == recorded.get("max_timestamp")
        and current.get("count") == recorded.get("count")
    )


def describe_drift(current: Dict[str, Any], recorded: Optional[Dict[str, Any]]) -> str:
    """One phrase saying how far a side has moved since the last sync."""
    if not recorded:
        count = current.get("count") or 0
        return "empty, never synced" if not count else f"{count} jobs, never synced"
    if unchanged(current, recorded):
        return "unchanged"
    delta = (current.get("count") or 0) - (recorded.get("count") or 0)
    if delta > 0:
        return f"+{delta} jobs added since"
    if delta < 0:
        return f"{delta} jobs removed since"
    return f"{current.get('count') or 0} rows changed since"


def check_unchanged(side: str, current: Dict[str, Any],
                    recorded: Optional[Dict[str, Any]]) -> None:
    """Refuse to overwrite a side that moved since the last sync."""
    if unchanged(current, recorded):
        return
    if not recorded:
        raise SyncConflict(
            f"The {side} database holds {current.get('count')} job(s) and this "
            f"project has never been synced, so there is no way to tell whether "
            f"they are accounted for on the other side. Overwriting would lose "
            f"them.",
            "Pass --force to overwrite deliberately.",
        )
    raise SyncConflict(
        f"The {side} database has changed since the last sync "
        f"({describe_drift(current, recorded)}). Overwriting it would lose "
        f"that work.",
        "Run the opposite direction first, or pass --force to overwrite "
        "deliberately.",
    )


def check_identity(local_created: str, remote: Dict[str, Any]) -> None:
    """Refuse to sync against a database belonging to a different project."""
    expected = identity_of(local_created)
    found = remote.get("identity")
    if found in (None, 0):
        return
    if found != expected:
        raise SyncConflict(
            "The remote database is not this project's database "
            f"(id {found}, expected {expected}). Something else is using that "
            "remote root."
        )
