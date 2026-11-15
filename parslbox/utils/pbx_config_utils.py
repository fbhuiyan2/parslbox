from pathlib import Path
from typing import Optional

import yaml

from parslbox.utils import path_utils


def _read_config(config_path: Optional[Path] = None) -> dict:
    """Load the config the caller means, not the one the process started with.

    A caller handed a path uses it; everyone else falls back to the default
    resolved at import. Shared helpers have to accept the parameter: one
    server process can be rebound to another project by use_database, and
    only the caller knows which project a given call belongs to. CLI command
    modules are the exception -- a fresh process per command means the
    default is already their context.
    """
    resolved = Path(config_path) if config_path else path_utils.PBX_CONFIG_FILE
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {resolved}")

    with open(resolved, 'r') as f:
        return yaml.safe_load(f)


def load_app_config(app_name: str, system_name: str,
                    config_path: Optional[Path] = None) -> dict:
    """
    Loads the configuration for a specific application on a specific system.

    Returns an empty dictionary if the app or system configuration is not found,
    allowing apps to work without explicit configuration.

    Args:
        app_name (str): Name of the application
        system_name (str): Name of the system/configuration
        config_path: Config file to read. Defaults to the process-wide one.

    Returns:
        dict: Application configuration for the system, or empty dict if not configured

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    full_config = _read_config(config_path)

    app_configs = full_config.get(app_name)
    if not app_configs:
        # Return empty dict instead of raising error - allows unconfigured apps
        return {}

    system_specific_config = app_configs.get(system_name)
    if not system_specific_config:
        # Return empty dict instead of raising error - allows partial configuration
        return {}

    return system_specific_config


def is_app_configured(app_name: str, system_name: str = None,
                      config_path: Optional[Path] = None) -> bool:
    """
    Check if an application has any configuration defined.

    Args:
        app_name (str): Name of the application
        system_name (str, optional): Name of the system. If provided, checks for
                                   system-specific config. If None, checks if app
                                   has any configuration at all.
        config_path: Config file to read. Defaults to the process-wide one.

    Returns:
        bool: True if configuration exists, False otherwise
    """
    try:
        full_config = _read_config(config_path)

        app_configs = full_config.get(app_name)
        if not app_configs:
            return False

        if system_name is None:
            # Check if app has any configuration
            return bool(app_configs)
        else:
            # Check if app has configuration for specific system
            return bool(app_configs.get(system_name))

    except (FileNotFoundError, yaml.YAMLError):
        return False


def load_full_config(config_path: Optional[Path] = None) -> dict:
    """
    Load the full YAML configuration file.

    Args:
        config_path: Config file to read. Defaults to the process-wide one.

    Returns:
        dict: Full configuration dictionary

    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    return _read_config(config_path)
