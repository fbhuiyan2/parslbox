"""
Application Registry for ParslBox

This module defines the APP_FACTORY which contains the built-in application classes
and manages custom applications loaded from user configuration.
"""

from pathlib import Path
from typing import Optional

from parslbox.apps.appbase import AppBase
from parslbox.apps.lammps_kk import LammpsKokkosApp
from parslbox.apps.vasp import VaspApp
from parslbox.apps.python import PythonApp
from parslbox.apps.julia import JuliaApp
from parslbox.apps.orca import OrcaApp

# Built-in application factory (always available, no loading needed)
APP_FACTORY = {
    "lammps-kk": LammpsKokkosApp,
    "vasp": VaspApp,
    "python": PythonApp,
    "julia": JuliaApp,
    "orca": OrcaApp,
}

# Custom applications, loaded lazily and cached per config file. Keying by
# path rather than a single slot is what lets ParslBox(config_path=...) see the
# apps declared in the config it was handed: two configs in one process each
# get their own entry instead of the first one winning.
_CUSTOM_APPS: dict = {}


def _custom_apps(config_path: Optional[Path] = None) -> dict:
    """Custom app classes declared by `config_path`, loading them once."""
    from parslbox.utils import path_utils

    resolved = Path(config_path) if config_path else path_utils.PBX_CONFIG_FILE
    key = str(resolved)
    if key not in _CUSTOM_APPS:
        from parslbox.apps.custom_app_loader import load_custom_apps
        _CUSTOM_APPS[key] = load_custom_apps(resolved)
    return _CUSTOM_APPS[key]


def get_app_class(app_name: str, config_path: Optional[Path] = None) -> type[AppBase]:
    """
    Get application class for a specific application.

    Priority order (highest to lowest):
    1. Built-in apps (APP_FACTORY) - lammps-kk, vasp, python
    2. Custom apps (loaded from config) - user-defined apps

    This ensures built-in apps cannot be overridden by custom apps.
    Custom apps are loaded lazily only when needed.

    Args:
        app_name (str): Name of the application
        config_path: Config declaring the custom apps. Defaults to the
            process-wide one.

    Returns:
        type[AppBase]: Application class

    Raises:
        ValueError: If the application is not registered
    """
    # Priority 1: Check built-in apps first (instant lookup)
    if app_name in APP_FACTORY:
        return APP_FACTORY[app_name]

    # Priority 2: Load custom apps lazily if needed
    custom_apps = _custom_apps(config_path)
    if app_name in custom_apps:
        return custom_apps[app_name]

    # Not found - provide helpful error message
    builtin_apps = sorted(APP_FACTORY.keys())
    custom = sorted(custom_apps.keys())

    error_msg = f"Unknown application: '{app_name}'.\n"
    error_msg += f"Available built-in apps: {', '.join(builtin_apps)}"
    if custom:
        error_msg += f"\nAvailable custom apps: {', '.join(custom)}"
    else:
        error_msg += "\nNo custom apps registered. See documentation for how to add custom apps."

    raise ValueError(error_msg)


def get_app_instance(app_name: str, config_path: Optional[Path] = None) -> AppBase:
    """
    Get application instance for a specific application.

    Args:
        app_name (str): Name of the application
        config_path: Config declaring the custom apps.

    Returns:
        AppBase: Application instance

    Raises:
        ValueError: If the application is not registered
    """
    return get_app_class(app_name, config_path)()


def get_app_config(app_name: str, config_path: Optional[Path] = None) -> dict:
    """
    Get configuration for a specific application.

    Args:
        app_name (str): Name of the application
        config_path: Config declaring the custom apps.

    Returns:
        dict: Application configuration containing INPUT_REQUIRED and DFLT_INPUT

    Raises:
        ValueError: If the application is not registered
    """
    app_class = get_app_class(app_name, config_path)
    return {
        "INPUT_REQUIRED": app_class.INPUT_REQUIRED,
        "DFLT_INPUT": app_class.DFLT_INPUT
    }


def is_app_registered(app_name: str, config_path: Optional[Path] = None) -> bool:
    """
    Check if an application is registered (built-in or custom).

    Args:
        app_name (str): Name of the application
        config_path: Config declaring the custom apps.

    Returns:
        bool: True if the application is registered, False otherwise
    """
    if app_name in APP_FACTORY:
        return True
    return app_name in _custom_apps(config_path)


def get_registered_apps(config_path: Optional[Path] = None) -> list[str]:
    """
    Get list of all registered application names (built-in + custom).

    Args:
        config_path: Config declaring the custom apps.

    Returns:
        list[str]: List of registered application names
    """
    return list(APP_FACTORY.keys()) + list(_custom_apps(config_path).keys())
