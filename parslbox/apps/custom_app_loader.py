"""
Custom App Loader for ParslBox

Loads custom applications defined in pbx_config.yaml and validates them.
Uses lazy loading to minimize overhead for commands that don't need custom apps.
"""

import importlib.util
import logging
import yaml
from pathlib import Path
from typing import Dict, Optional, Type

from parslbox.apps.appbase import AppBase
from parslbox.utils import path_utils

logger = logging.getLogger(__name__)


def load_custom_apps() -> Dict[str, Type[AppBase]]:
    """
    Load custom applications from pbx_config.yaml.
    
    Reads the 'custom_apps' section from the config file and imports
    the specified app classes. Validates that each class properly
    implements the AppBase interface.
    
    Returns:
        Dict mapping app names to app classes
        
    Performance:
        - No custom apps: ~5-10ms (just YAML read/parse)
        - With custom apps: ~40ms (read + import + validate)
    """
    custom_apps = {}
    
    try:
        config_path = path_utils.PBX_CONFIG_FILE
        
        if not config_path.exists():
            logger.debug("Config file not found, no custom apps to load")
            return custom_apps
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        custom_app_defs = config.get('custom_apps', {})
        
        if not custom_app_defs:
            logger.debug("No custom apps defined in config")
            return custom_apps
        
        logger.info(f"Loading {len(custom_app_defs)} custom app(s)...")
        
        for app_name, app_def in custom_app_defs.items():
            try:
                app_class = _load_single_app(app_name, app_def)
                if app_class:
                    custom_apps[app_name] = app_class
                    logger.info(f"✓ Loaded custom app: {app_name}")
            except Exception as e:
                logger.warning(f"✗ Failed to load custom app '{app_name}': {e}")
                # Continue loading other apps
        
        if custom_apps:
            logger.info(f"Successfully loaded {len(custom_apps)} custom app(s)")
        
    except Exception as e:
        logger.warning(f"Error loading custom apps from config: {e}")
    
    return custom_apps


def _load_single_app(app_name: str, app_def: dict) -> Optional[Type[AppBase]]:
    """
    Load a single custom app from its definition.
    
    Args:
        app_name: Name of the app being loaded
        app_def: App definition dict with 'module' and 'class' keys
        
    Returns:
        App class if successful, None if should be skipped
        
    Raises:
        ValueError: If definition is invalid
        ImportError: If module/class cannot be imported
    """
    # Import built-in apps to check for conflicts
    from parslbox.apps.app_registry import APP_FACTORY
    
    # Check for name conflicts with built-in apps
    if app_name in APP_FACTORY:
        logger.warning(
            f"Custom app '{app_name}' has the same name as a built-in app. "
            f"The built-in app will always be used. "
            f"Please choose a different name for your custom app."
        )
        return None  # Don't load conflicting custom app
    
    # Validate definition
    module_path = app_def.get('module')
    class_name = app_def.get('class')
    
    if not module_path:
        raise ValueError("Missing 'module' field")
    if not class_name:
        raise ValueError("Missing 'class' field")
    
    # Import the class
    app_class = import_app_class(module_path, class_name)
    
    # Validate it's a proper AppBase subclass
    if not validate_app_class(app_class):
        raise ValueError(f"Class {class_name} does not properly implement AppBase")
    
    return app_class


def import_app_class(module_path: str, class_name: str) -> Type[AppBase]:
    """
    Import an app class from a file path or module name.
    
    Args:
        module_path: File path (e.g., "/path/to/app.py") or module name (e.g., "my_package.my_app")
        class_name: Name of the class to import
        
    Returns:
        The imported class
        
    Raises:
        ImportError: If module or class cannot be imported
    """
    # Expand paths like ~/my_app.py
    expanded_path = expand_path(module_path)
    
    # Check if it's a file path
    if expanded_path.exists() and expanded_path.suffix == '.py':
        # Import from file
        module = _import_from_file(expanded_path)
    else:
        # Try importing as module name
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            raise ImportError(
                f"Could not import module '{module_path}'. "
                f"Make sure it's either a valid file path or an installed Python module. "
                f"Error: {e}"
            )
    
    # Get the class from the module
    if not hasattr(module, class_name):
        raise ImportError(
            f"Module '{module_path}' does not have a class named '{class_name}'"
        )
    
    app_class = getattr(module, class_name)
    
    # Verify it's actually a class
    if not isinstance(app_class, type):
        raise ImportError(f"{class_name} is not a class")
    
    # Verify it inherits from AppBase
    if not issubclass(app_class, AppBase):
        raise ImportError(f"{class_name} does not inherit from AppBase")
    
    return app_class


def _import_from_file(file_path: Path):
    """
    Import a module from a file path.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        The imported module
        
    Raises:
        ImportError: If file cannot be imported
    """
    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_app_class(cls: Type[AppBase]) -> bool:
    """
    Validate that a class properly implements AppBase interface.
    
    Args:
        cls: The class to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Check required class attributes
    required_attrs = ['INPUT_REQUIRED', 'DFLT_INPUT']
    for attr in required_attrs:
        if not hasattr(cls, attr):
            logger.error(f"{cls.__name__} missing required attribute: {attr}")
            return False
    
    # Check required methods
    required_methods = ['get_command_template']
    for method in required_methods:
        if not hasattr(cls, method) or not callable(getattr(cls, method)):
            logger.error(f"{cls.__name__} missing required method: {method}")
            return False
    
    return True


def expand_path(path_str: str) -> Path:
    """
    Expand a path string, handling ~ and relative paths.
    
    Args:
        path_str: Path string to expand
        
    Returns:
        Expanded Path object
    """
    path = Path(path_str).expanduser()
    
    # If it's not absolute, try relative to config directory
    if not path.is_absolute():
        config_dir = path_utils.PBX_CONFIG_FILE.parent
        path = (config_dir / path).resolve()
    
    return path
