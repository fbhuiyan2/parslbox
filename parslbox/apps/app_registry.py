"""
Application Registry for ParslBox

This module defines the APP_FACTORY which contains the built-in application classes
and manages custom applications loaded from user configuration.
"""

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

# Custom applications (loaded lazily from config)
_CUSTOM_APPS = None


def get_app_class(app_name: str) -> type[AppBase]:
    """
    Get application class for a specific application.
    
    Priority order (highest to lowest):
    1. Built-in apps (APP_FACTORY) - lammps-kk, vasp, python
    2. Custom apps (loaded from config) - user-defined apps
    
    This ensures built-in apps cannot be overridden by custom apps.
    Custom apps are loaded lazily only when needed.
    
    Args:
        app_name (str): Name of the application
        
    Returns:
        type[AppBase]: Application class
        
    Raises:
        ValueError: If the application is not registered
    """
    global _CUSTOM_APPS
    
    # Priority 1: Check built-in apps first (instant lookup)
    if app_name in APP_FACTORY:
        return APP_FACTORY[app_name]
    
    # Priority 2: Load custom apps lazily if needed
    if _CUSTOM_APPS is None:
        from parslbox.apps.custom_app_loader import load_custom_apps
        _CUSTOM_APPS = load_custom_apps()
    
    # Check custom apps
    if app_name in _CUSTOM_APPS:
        return _CUSTOM_APPS[app_name]
    
    # Not found - provide helpful error message
    builtin_apps = sorted(APP_FACTORY.keys())
    custom_apps = sorted(_CUSTOM_APPS.keys()) if _CUSTOM_APPS else []
    
    error_msg = f"Unknown application: '{app_name}'.\n"
    error_msg += f"Available built-in apps: {', '.join(builtin_apps)}"
    if custom_apps:
        error_msg += f"\nAvailable custom apps: {', '.join(custom_apps)}"
    else:
        error_msg += "\nNo custom apps registered. See documentation for how to add custom apps."
    
    raise ValueError(error_msg)


def get_app_instance(app_name: str) -> AppBase:
    """
    Get application instance for a specific application.
    
    Args:
        app_name (str): Name of the application
        
    Returns:
        AppBase: Application instance
        
    Raises:
        ValueError: If the application is not registered
    """
    app_class = get_app_class(app_name)
    return app_class()


def get_app_config(app_name: str) -> dict:
    """
    Get configuration for a specific application.
    
    Args:
        app_name (str): Name of the application
        
    Returns:
        dict: Application configuration containing INPUT_REQUIRED and DFLT_INPUT
        
    Raises:
        ValueError: If the application is not registered
    """
    app_class = get_app_class(app_name)
    return {
        "INPUT_REQUIRED": app_class.INPUT_REQUIRED,
        "DFLT_INPUT": app_class.DFLT_INPUT
    }


def is_app_registered(app_name: str) -> bool:
    """
    Check if an application is registered (built-in or custom).
    
    Args:
        app_name (str): Name of the application
        
    Returns:
        bool: True if the application is registered, False otherwise
    """
    global _CUSTOM_APPS
    
    # Check built-in apps first
    if app_name in APP_FACTORY:
        return True
    
    # Load custom apps if needed
    if _CUSTOM_APPS is None:
        from parslbox.apps.custom_app_loader import load_custom_apps
        _CUSTOM_APPS = load_custom_apps()
    
    return app_name in _CUSTOM_APPS


def get_registered_apps() -> list[str]:
    """
    Get list of all registered application names (built-in + custom).
    
    Returns:
        list[str]: List of registered application names
    """
    global _CUSTOM_APPS
    
    # Load custom apps if needed
    if _CUSTOM_APPS is None:
        from parslbox.apps.custom_app_loader import load_custom_apps
        _CUSTOM_APPS = load_custom_apps()
    
    return list(APP_FACTORY.keys()) + list(_CUSTOM_APPS.keys())
