from pathlib import Path
from parsl.config import Config

# Import the get_config function from each system module with a unique alias
from parslbox.configs.polaris import get_config as polaris_config
from parslbox.configs.sophia import get_config as sophia_config

# Create a dictionary that maps the config name to its factory function
# Scheduler options = "PBS", "SLURM"
CONFIG_FACTORIES = {
    "polaris": [polaris_config, "PBS"],
    "sophia": [sophia_config, "PBS"],
    # To add a new system, create its module and add it here.
}

def load_config(name: str, run_dir: Path, retries: int) -> Config:
    """
    Loads a Parsl configuration by name.

    This function acts as a dispatcher, finding the correct configuration
    factory based on the provided name and calling it with the given
    runtime parameters.

    Args:
        name (str): The name of the configuration to load (e.g., "polaris").
        run_dir (Path): The path for Parsl's run directory.
        retries (int): The number of retries for failed Parsl apps.

    Raises:
        ValueError: If the requested configuration name is not found.

    Returns:
        Config: A fully instantiated Parsl configuration object.
    """
    factory = CONFIG_FACTORIES.get(name)

    if factory is None:
        available = ", ".join(CONFIG_FACTORIES.keys())
        raise ValueError(
            f"Unknown configuration name: '{name}'. "
            f"Available configurations are: {available}"
        )

    conf_loader, sched = factory[0], factory[1]
    return conf_loader(run_dir=run_dir, retries=retries), sched
