from pathlib import Path
from parsl.config import Config
from parslbox.configs.base import SystemConfig

# Import the system configuration classes
from parslbox.configs.polaris import PolarisConfig
from parslbox.configs.sophia import SophiaConfig

# Create a dictionary that maps the config name to its configuration class
CONFIG_FACTORIES = {
    "polaris": PolarisConfig,
    "sophia": SophiaConfig,
    # To add a new system, create its module and add it here.
}

def load_config(name: str, run_dir: Path, retries: int) -> tuple[Config, str]:
    """
    Loads a Parsl configuration by name.

    This function acts as a dispatcher, finding the correct configuration
    class based on the provided name and calling it with the given
    runtime parameters.

    Args:
        name (str): The name of the configuration to load (e.g., "polaris").
        run_dir (Path): The path for Parsl's run directory.
        retries (int): The number of retries for failed Parsl apps.

    Raises:
        ValueError: If the requested configuration name is not found.

    Returns:
        tuple: A tuple of (Config, scheduler_name)
    """
    config_class = CONFIG_FACTORIES.get(name)

    if config_class is None:
        available = ", ".join(CONFIG_FACTORIES.keys())
        raise ValueError(
            f"Unknown configuration name: '{name}'. "
            f"Available configurations are: {available}"
        )

    config_instance = config_class()
    parsl_config = config_instance.get_config(run_dir=run_dir, retries=retries)
    return parsl_config, config_instance.SCHEDULER

def get_system_config(name: str) -> SystemConfig:
    """
    Get system configuration class instance by name.
    
    This function provides access to system specifications and methods
    for any system configuration.
    
    Args:
        name (str): The name of the system configuration (e.g., "polaris").
        
    Raises:
        ValueError: If the requested configuration name is not found.
        
    Returns:
        SystemConfig: An instance of the system configuration class.
    """
    config_class = CONFIG_FACTORIES.get(name)
    
    if config_class is None:
        available = ", ".join(CONFIG_FACTORIES.keys())
        raise ValueError(
            f"Unknown configuration name: '{name}'. "
            f"Available configurations are: {available}"
        )
    
    return config_class()

def get_available_systems() -> list[str]:
    """
    Get list of all available system configuration names.
    
    Returns:
        list[str]: List of available system names.
    """
    return list(CONFIG_FACTORIES.keys())
