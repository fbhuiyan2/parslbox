"""
Application Registry for ParslBox

This module defines the APP_FACTORY which contains the application classes
for each supported application.
"""

from parslbox.apps.base import AppBase
from parslbox.apps.lammps import LammpsApp
from parslbox.apps.vasp import VaspApp
from parslbox.apps.python import PythonApp

# Application factory containing app classes
APP_FACTORY = {
    "lammps": LammpsApp,
    "vasp": VaspApp,
    "python": PythonApp,
}

def get_app_class(app_name: str) -> type[AppBase]:
    """
    Get application class for a specific application.
    
    Args:
        app_name (str): Name of the application
        
    Returns:
        type[AppBase]: Application class
        
    Raises:
        ValueError: If the application is not registered
    """
    app_class = APP_FACTORY.get(app_name)
    
    if app_class is None:
        available_apps = ", ".join(APP_FACTORY.keys())
        raise ValueError(
            f"Unknown application: '{app_name}'. "
            f"Available applications are: {available_apps}"
        )
    
    return app_class

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
    Check if an application is registered in the factory.
    
    Args:
        app_name (str): Name of the application
        
    Returns:
        bool: True if the application is registered, False otherwise
    """
    return app_name in APP_FACTORY

def get_registered_apps() -> list[str]:
    """
    Get list of all registered application names.
    
    Returns:
        list[str]: List of registered application names
    """
    return list(APP_FACTORY.keys())
