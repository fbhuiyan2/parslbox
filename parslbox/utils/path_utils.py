import os
from pathlib import Path


class PbxPathError(ValueError):
    """Raised when a pbx database or config path is not absolute."""


def _relative_path_message(label: str, value: str) -> str:
    msg = (
        f"{label} must be an absolute path, got: {value}\n"
        "A relative path is resolved against the current working directory, so "
        "every directory you run pbx from gets its own file, and a batch job "
        "resolves it against the allocation's working directory instead of yours."
    )
    if label.startswith("PBX_"):
        rel = value[2:] if value.startswith("./") else value
        msg += f'\nTry: export {label}="$PWD/{rel}"'
    return msg


def require_absolute(label: str, path) -> Path:
    """Expand ~ in path and return it, raising PbxPathError if it is not absolute."""
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        raise PbxPathError(_relative_path_message(label, str(path)))
    return expanded


def _resolve_env_path(var_name: str, default_filename: str, suffixes: tuple,
                      default_path: Path, strict: bool) -> Path:
    env_path = os.getenv(var_name)
    if not env_path:
        return default_path
    if strict:
        path = require_absolute(var_name, env_path)
    else:
        path = Path(env_path).expanduser()
    if path.suffix in suffixes:
        return path
    return path / default_filename


def get_db_path(strict: bool = True) -> Path:
    """Get the database file path from environment variable or default location.

    Environment variable PBX_DB_PATH must be an absolute path pointing to either:
    - A directory (e.g., '/custom/path/') - will append 'job_database_pbx.db'
    - A full file path (e.g., '/custom/path/my_database.db') - will use as-is

    Raises:
        PbxPathError: If PBX_DB_PATH is set to a relative path (unless strict=False,
            used only for the module-level default so that importing this module
            never raises; callers validate via validate_env_paths()).
    """
    return _resolve_env_path(
        "PBX_DB_PATH",
        "job_database_pbx.db",
        (".db",),
        Path.home() / ".parslbox" / "job_database_pbx.db",
        strict,
    )


def get_config_path(strict: bool = True) -> Path:
    """Get the config file path from environment variable or default location.

    Environment variable PBX_CONFIG_PATH must be an absolute path pointing to either:
    - A directory (e.g., '/custom/path/') - will append 'config.yaml'
    - A full file path (e.g., '/custom/path/my_config.yaml') - will use as-is

    Raises:
        PbxPathError: If PBX_CONFIG_PATH is set to a relative path (unless
            strict=False; see get_db_path()).
    """
    return _resolve_env_path(
        "PBX_CONFIG_PATH",
        "config.yaml",
        (".yaml", ".yml"),
        Path.home() / ".parslbox" / "config.yaml",
        strict,
    )


def validate_env_paths() -> None:
    """Raise PbxPathError if PBX_DB_PATH or PBX_CONFIG_PATH is a relative path."""
    get_db_path()
    get_config_path()


# Define the path to the SQLite database file
DB_FILE = get_db_path(strict=False)

# package config file
PBX_CONFIG_FILE = get_config_path(strict=False)
