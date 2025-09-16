import yaml
import typer
from pathlib import Path

from parslbox.helpers import path_utils
from parslbox.default_template import DEFAULT_CONFIG_YAML

def initialize_config_file():
    """
    Checks if the flow_config.yaml file exists, and creates a default
    template if it does not.
    """
    config_path = path_utils.FLOW_CONFIG_FILE
    
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
            "👉 IMPORTANT: You must edit this file to set the correct executable paths before running.",
            bold=True,
            fg=typer.colors.RED
        )


def load_app_config(app_name: str, system_name: str) -> dict:
    """
    Loads the configuration for a specific application on a specific system.
    """
    config_path = path_utils.FLOW_CONFIG_FILE
    if not config_path.is_file():
        # This error will now only be hit if the file was deleted after the program started.
        # The callback should prevent this from being the user's first experience.
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    app_configs = full_config.get(app_name)
    if not app_configs:
        raise ValueError(f"Application '{app_name}' not found in {config_path}")

    system_specific_config = app_configs.get(system_name)
    if not system_specific_config:
        raise ValueError(
            f"Configuration for system '{system_name}' not found for application '{app_name}'."
        )

    return system_specific_config
