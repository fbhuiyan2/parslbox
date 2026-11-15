from pathlib import Path
from parsl.config import Config
from parslbox.system_configs.base_sysconf import SystemConfig

# Import the system configuration classes
from parslbox.system_configs.polaris import PolarisConfig
from parslbox.system_configs.sophia import SophiaConfig
from parslbox.system_configs.crux import CruxConfig
from parslbox.system_configs.aurora_tile import AuroraTileConfig
from parslbox.system_configs.aurora_gpu import AuroraGpuConfig
from parslbox.system_configs.lcrc_swing import LcrcSwingConfig
from parslbox.system_configs.lcrc_improv import LcrcImprovConfig
from parslbox.system_configs.pinnacles_cenvalarc import PinnaclesCenvalarcConfig
from parslbox.system_configs.perlmutter_gpu import PerlmutterGpuConfig
from parslbox.system_configs.perlmutter_cpu import PerlmutterCpuConfig
from parslbox.system_configs.perlmutter_gpu_srun import PerlmutterGpuSrunConfig
from parslbox.system_configs.kestrel_cpu import KestrelCpuConfig
from parslbox.system_configs.kestrel_gpu import KestrelGpuConfig
from parslbox.system_configs.delta_cpu import DeltaCpuConfig
from parslbox.system_configs.delta_gpu import DeltaGpuConfig
from parslbox.system_configs.delta_h200 import DeltaH200Config
from parslbox.system_configs.anvil_cpu import AnvilCpuConfig
from parslbox.system_configs.anvil_gpu import AnvilGpuConfig
from parslbox.system_configs.anvil_ai import AnvilAiConfig
from parslbox.system_configs.expanse_cpu import ExpanseCpuConfig
from parslbox.system_configs.expanse_gpu import ExpanseGpuConfig
from parslbox.system_configs.midway3_cpu import Midway3CpuConfig
from parslbox.system_configs.midway3_gpu import Midway3GpuConfig
from parslbox.system_configs.bridges2_cpu import Bridges2CpuConfig
from parslbox.system_configs.bridges2_v100 import Bridges2V100Config
from parslbox.system_configs.bridges2_h100 import Bridges2H100Config

# Create a dictionary that maps the config name to its configuration class
CONFIG_FACTORIES = {
    "polaris": PolarisConfig,
    "sophia": SophiaConfig,
    "crux": CruxConfig,
    "aurora-tile": AuroraTileConfig,
    "aurora-gpu": AuroraGpuConfig,
    "lcrc-swing": LcrcSwingConfig,
    "lcrc-improv": LcrcImprovConfig,
    "pinnacles-cenvalarc": PinnaclesCenvalarcConfig,
    "perlmutter-gpu": PerlmutterGpuConfig,
    "perlmutter-cpu": PerlmutterCpuConfig,
    # srun launcher variant — distributes workers across compute nodes instead of
    # concentrating them on the head node. Use for runs > ~10k workers.
    "perlmutter-gpu-srun": PerlmutterGpuSrunConfig,
    "kestrel-cpu": KestrelCpuConfig,
    "kestrel-gpu": KestrelGpuConfig,
    "delta-cpu": DeltaCpuConfig,
    "delta-gpu": DeltaGpuConfig,
    "delta-h200": DeltaH200Config,
    "anvil-cpu": AnvilCpuConfig,
    "anvil-gpu": AnvilGpuConfig,
    "anvil-ai": AnvilAiConfig,
    "expanse-cpu": ExpanseCpuConfig,
    "expanse-gpu": ExpanseGpuConfig,
    "midway3-cpu": Midway3CpuConfig,
    "midway3-gpu": Midway3GpuConfig,
    "bridges2-cpu": Bridges2CpuConfig,
    "bridges2-v100": Bridges2V100Config,
    "bridges2-h100": Bridges2H100Config,
    # To add a new system, create its module and add it here.
}

def load_config(name: str, run_dir: Path, retries: int, max_workers: int = None) -> tuple[Config, str]:
    """
    Loads a Parsl configuration by name.

    This function acts as a dispatcher, finding the correct configuration
    class based on the provided name and calling it with the given
    runtime parameters.

    Args:
        name (str): The name of the configuration to load (e.g., "polaris").
        run_dir (Path): The path for Parsl's run directory.
        retries (int): The number of retries for failed Parsl apps.
        max_workers (int, optional): Optional override for total workers across all nodes.
                                     If None, uses system default (MAX_WORKERS_PER_NODE * nodes).

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
    parsl_config = config_instance.get_config(run_dir=run_dir, retries=retries, max_workers=max_workers)
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

def get_system_specs() -> list[dict]:
    """Name and hardware specs for every configured system.

    Read off the class attributes rather than an instance: these are plain
    class constants, so constructing eleven configs to read five of them
    each buys nothing.

    Returns:
        list[dict]: One dict per system with 'name', 'scheduler',
            'cores_per_node', 'gpus_per_node', 'gpu_type' (None on CPU-only
            systems) and 'mpi_backend'.
    """
    return [
        {
            "name": name,
            "scheduler": cls.SCHEDULER,
            "cores_per_node": cls.CORES_PER_NODE,
            "gpus_per_node": cls.GPUS_PER_NODE,
            "gpu_type": cls.GPU_TYPE,
            "mpi_backend": cls.MPI_BACKEND,
        }
        for name, cls in CONFIG_FACTORIES.items()
    ]


def get_available_systems() -> list[str]:
    """
    Get list of all available system configuration names.
    
    Returns:
        list[str]: List of available system names.
    """
    return list(CONFIG_FACTORIES.keys())
