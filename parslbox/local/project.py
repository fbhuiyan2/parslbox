"""Local project metadata: the .pbxlocal.yaml file beside a local database.

A local project's database stores remote (HPC) paths from its first row, so
nothing has to be rewritten when the work is handed over. The .pbxlocal.yaml
sitting next to the database is what marks it as such, and it records the two
roots that the path translation swaps between.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional

import yaml

LOCAL_DB_NAME = "job_database_pbx-local.db"
PROJECT_FILE = ".pbxlocal.yaml"

REQUIRED_KEYS = ("created", "local_root", "remote_root")


class LocalProjectError(Exception):
    """Base class for local-project problems."""


class ProjectFileError(LocalProjectError):
    """.pbxlocal.yaml is missing, unreadable, or incomplete."""


class WrongDatabaseName(LocalProjectError):
    """PBX_DB_PATH resolved to a non-local database inside a local project."""


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class LocalProject:
    local_root: Path
    created: str
    remote_root: str
    remote_config: Optional[str] = None
    compute_endpoint: Optional[str] = None
    transfer_local: Optional[str] = None
    transfer_remote: Optional[str] = None
    transfer_remote_root: Optional[str] = None
    last_sync: Optional[Dict[str, Any]] = None
    recorded_local_root: Optional[str] = None

    def __post_init__(self):
        self.local_root = Path(self.local_root)
        self.remote_root = str(self.remote_root).rstrip("/") or "/"

    @property
    def db_path(self) -> Path:
        return self.local_root / LOCAL_DB_NAME

    @property
    def yaml_path(self) -> Path:
        return self.local_root / PROJECT_FILE

    @property
    def remote_db_path(self) -> str:
        return str(PurePosixPath(self.remote_root) / LOCAL_DB_NAME)

    @property
    def moved(self) -> bool:
        """True when the project directory is not where the YAML says it is."""
        if not self.recorded_local_root:
            return False
        return Path(self.recorded_local_root) != self.local_root

    @property
    def export_line(self) -> str:
        return f'export PBX_DB_PATH="{self.db_path}"'

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "created": self.created,
            "local_root": str(self.local_root),
            "remote_root": self.remote_root,
            "remote_config": self.remote_config,
            "compute_endpoint": self.compute_endpoint,
            "transfer_local": self.transfer_local,
            "transfer_remote": self.transfer_remote,
            "transfer_remote_root": self.transfer_remote_root,
        }
        if self.last_sync is not None:
            data["last_sync"] = self.last_sync
        return data


def load_project(directory) -> LocalProject:
    """Read .pbxlocal.yaml from `directory`.

    The directory itself is the local root, not whatever the file records --
    that way a project that was moved keeps translating paths correctly. The
    recorded value is kept only so `pbx local status` can point out the move.
    """
    directory = Path(directory)
    yaml_path = directory / PROJECT_FILE
    try:
        raw = yaml.safe_load(yaml_path.read_text())
    except FileNotFoundError:
        raise ProjectFileError(f"No {PROJECT_FILE} in {directory}")
    except yaml.YAMLError as e:
        raise ProjectFileError(f"{yaml_path} is not valid YAML: {e}")

    if not isinstance(raw, dict):
        raise ProjectFileError(f"{yaml_path} is empty or is not a mapping")

    missing = [k for k in REQUIRED_KEYS if not raw.get(k)]
    if missing:
        raise ProjectFileError(
            f"{yaml_path} is missing required key(s): {', '.join(missing)}.\n"
            f"Recreate the project, or add the keys by hand."
        )

    remote_root = str(raw["remote_root"]).rstrip("/") or "/"
    if not remote_root.startswith("/"):
        raise ProjectFileError(
            f"{yaml_path}: remote_root must be an absolute path, got {remote_root!r}"
        )

    return LocalProject(
        local_root=directory,
        created=str(raw["created"]),
        remote_root=remote_root,
        remote_config=raw.get("remote_config"),
        compute_endpoint=raw.get("compute_endpoint"),
        transfer_local=raw.get("transfer_local"),
        transfer_remote=raw.get("transfer_remote"),
        transfer_remote_root=raw.get("transfer_remote_root"),
        last_sync=raw.get("last_sync"),
        recorded_local_root=str(raw["local_root"]),
    )


def save_project(project: LocalProject) -> Path:
    """Write .pbxlocal.yaml, quoting values so they reload as plain strings."""
    project.yaml_path.write_text(
        yaml.safe_dump(project.to_dict(), sort_keys=False, default_flow_style=False)
    )
    return project.yaml_path


def check_db_name(db_path) -> bool:
    """True if `db_path` is a local project's database, False if no project.

    One stat: .pbxlocal.yaml either sits beside the database or it does not.
    No directory walk -- PBX_DB_PATH already says which database is in use.
    The YAML is not read, so every command can afford this before it opens
    or creates anything.

    Raises:
        WrongDatabaseName: the directory is a local project but the resolved
            database is not the -local one. This is what exporting the
            directory instead of the file looks like: get_db_path() appends
            job_database_pbx.db, and pbx would otherwise create that database
            and report an empty project.
    """
    if db_path is None:
        return False
    db_path = Path(db_path)
    directory = db_path.parent
    if not (directory / PROJECT_FILE).is_file():
        return False
    if db_path.name != LOCAL_DB_NAME:
        raise WrongDatabaseName(
            f"{directory} is a local project, but PBX_DB_PATH resolves to\n"
            f"    {db_path}\n"
            f"A local project's database is {LOCAL_DB_NAME}. Inside a local "
            f"project, PBX_DB_PATH must name the file:\n"
            f'    export PBX_DB_PATH="{directory / LOCAL_DB_NAME}"'
        )
    return True


def find_project(db_path) -> Optional[LocalProject]:
    """Return the local project owning `db_path`, or None.

    Raises:
        WrongDatabaseName: see check_db_name().
        ProjectFileError: the project file is there but unreadable.
    """
    if not check_db_name(db_path):
        return None
    return load_project(Path(db_path).parent)


def find_parent_project(directory) -> Optional[Path]:
    """Look for a project in the directories above `directory`.

    Only `pbx local init` calls this, and only to point out nesting before
    creating a project inside another one. It stats one file per ancestor and
    never descends, so the cost is the depth of the path regardless of how
    many sibling directories exist.
    """
    directory = Path(directory).absolute()
    for parent in directory.parents:
        if (parent / PROJECT_FILE).is_file():
            return parent
    return None


class NestedProject(LocalProjectError):
    """A project already exists in a directory above this one."""

    def __init__(self, parent: Path):
        self.parent = parent
        super().__init__(
            f"A local project already exists at {parent}.\n"
            f"Creating one inside it is allowed -- PBX_DB_PATH decides which "
            f"database you are on -- but it is more often an accident. Pass "
            f"nested_ok to go ahead."
        )


class PopulatedDatabase(LocalProjectError):
    """A -local database is present with rows in it but no project file."""


def count_rows(db_path) -> int:
    import sqlite3
    con = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        return con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
    except sqlite3.DatabaseError:
        return 0
    finally:
        con.close()


def inspect_directory(directory) -> Dict[str, Any]:
    """Report what `pbx local init` would find, changing nothing.

    Everything reported is about this directory alone. Parent directories are
    checked for a project file and nothing else -- a database sitting in a
    parent belongs to somebody else's project.
    """
    directory = Path(directory).absolute()
    db_path = directory / LOCAL_DB_NAME
    yaml_path = directory / PROJECT_FILE
    db_exists = db_path.is_file()
    return {
        "directory": directory,
        "yaml_exists": yaml_path.is_file(),
        "db_exists": db_exists,
        "plain_db_exists": (directory / "job_database_pbx.db").is_file(),
        "row_count": count_rows(db_path) if db_exists else None,
        "parent_project": find_parent_project(directory),
    }


def init_project(
    directory,
    remote_root: Optional[str] = None,
    compute_endpoint: Optional[str] = None,
    transfer_local: Optional[str] = None,
    transfer_remote: Optional[str] = None,
    transfer_remote_root: Optional[str] = None,
    remote_config: Optional[str] = None,
    nested_ok: bool = False,
) -> Dict[str, Any]:
    """Create, adopt, or report an existing local project in `directory`.

    Writes files, touches no network. Callers that can prompt should run
    inspect_directory() first: the refusal cases below are cheaper to hit
    before asking the user for a remote root and three UUIDs.

    Returns a dict with 'action' ('created', 'adopted' or 'existing'), the
    LocalProject, and any 'notes' worth showing.
    """
    from parslbox.database import database
    from parslbox.local.guard import stamp_identity

    state = inspect_directory(directory)
    directory = state["directory"]
    notes = []

    if state["yaml_exists"]:
        if not state["db_exists"]:
            raise ProjectFileError(
                f"{directory / PROJECT_FILE} exists but {LOCAL_DB_NAME} does not.\n"
                f"The project is half-deleted. Remove the project file and run "
                f"'pbx local init' again:\n    rm {directory / PROJECT_FILE}"
            )
        return {
            "action": "existing",
            "project": load_project(directory),
            "notes": notes,
        }

    if state["parent_project"] is not None and not nested_ok:
        raise NestedProject(state["parent_project"])

    if state["db_exists"] and state["row_count"]:
        raise PopulatedDatabase(
            f"{directory / LOCAL_DB_NAME} holds {state['row_count']} job(s) but "
            f"{PROJECT_FILE} is gone.\nThose rows are already remote paths, so "
            f"the remote root and endpoint they belong to cannot be recovered "
            f"from them. Restore {PROJECT_FILE} from a backup, or write it by "
            f"hand, or move the database aside and start over."
        )

    if not remote_root:
        raise LocalProjectError("remote_root is required to create a local project")
    remote_root = str(remote_root).rstrip("/") or "/"
    if not remote_root.startswith("/"):
        raise LocalProjectError(
            f"remote_root must be an absolute path on the remote machine, "
            f"got {remote_root!r}"
        )

    if state["plain_db_exists"]:
        notes.append(
            f"An ordinary job_database_pbx.db is also in {directory}. It is a "
            f"different file and is never touched by the local project."
        )
    if state["parent_project"] is not None:
        notes.append(f"Nested inside the project at {state['parent_project']}.")

    action = "adopted" if state["db_exists"] else "created"
    project = LocalProject(
        local_root=directory,
        created=utc_stamp(),
        remote_root=remote_root,
        remote_config=remote_config,
        compute_endpoint=compute_endpoint,
        transfer_local=transfer_local,
        transfer_remote=transfer_remote,
        transfer_remote_root=transfer_remote_root,
    )
    database.initialize_database(project.db_path)
    stamp_identity(project.db_path, project.created)
    save_project(project)

    return {"action": action, "project": project, "notes": notes}
