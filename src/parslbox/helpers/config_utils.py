import yaml
import typer
from pathlib import Path

from parslbox.helpers import path_utils
from parslbox.default_template import DEFAULT_CONFIG_YAML

def initialize_config_file():
    """
    Checks if the config.yaml file exists, and creates a default
    template if it does not.
    """
    config_path = path_utils.PBX_CONFIG_FILE
    
    if not config_path.exists():
        typer.secho(
            f"Note: Configuration file not found. Creating a new template...",
            fg=typer.colors.YELLOW
        )
        
        # Ensure the ~/.parslbox directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            f.write(DEFAULT_CONFIG_YAML)

        typer.secho(f"✅ Default configuration file created at: {config_path}", fg=typer.colors.GREEN)
        typer.secho(
            "👉 IMPORTANT: You must edit the config file to set the correct executable paths before running.",
            bold=True,
            fg=typer.colors.RED
        )


def load_app_config(app_name: str, system_name: str) -> dict:
    """
    Loads the configuration for a specific application on a specific system.
    
    Returns an empty dictionary if the app or system configuration is not found,
    allowing apps to work without explicit configuration.
    
    Args:
        app_name (str): Name of the application
        system_name (str): Name of the system/configuration
        
    Returns:
        dict: Application configuration for the system, or empty dict if not configured
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
    """
    config_path = path_utils.PBX_CONFIG_FILE
    if not config_path.is_file():
        # This error will now only be hit if the file was deleted after the program started.
        # The callback should prevent this from being the user's first experience.
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    app_configs = full_config.get(app_name)
    if not app_configs:
        # Return empty dict instead of raising error - allows unconfigured apps
        return {}

    system_specific_config = app_configs.get(system_name)
    if not system_specific_config:
        # Return empty dict instead of raising error - allows partial configuration
        return {}

    return system_specific_config


def is_app_configured(app_name: str, system_name: str = None) -> bool:
    """
    Check if an application has any configuration defined.
    
    Args:
        app_name (str): Name of the application
        system_name (str, optional): Name of the system. If provided, checks for 
                                   system-specific config. If None, checks if app 
                                   has any configuration at all.
        
    Returns:
        bool: True if configuration exists, False otherwise
    """
    try:
        config_path = path_utils.PBX_CONFIG_FILE
        if not config_path.is_file():
            return False

        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

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


def get_configured_systems_for_app(app_name: str) -> list[str]:
    """
    Get list of systems that have configuration for the specified app.
    
    Args:
        app_name (str): Name of the application
        
    Returns:
        list[str]: List of system names that have configuration for this app
    """
    try:
        config_path = path_utils.PBX_CONFIG_FILE
        if not config_path.is_file():
            return []

        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)

        app_configs = full_config.get(app_name, {})
        return list(app_configs.keys())
        
    except (FileNotFoundError, yaml.YAMLError):
        return []
