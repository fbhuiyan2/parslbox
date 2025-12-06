from datetime import datetime
from pathlib import Path
from parslbox.utils import path_utils
import yaml
from typing import Optional

def minutes_to_hms(minutes: int) -> str:
    """Convert minutes to HH:MM:SS format for PBS/SLURM."""
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours:02d}:{mins:02d}:00"


def get_default_run_dir() -> Path:
    """Generate default run directory with current time and date in hhmmss_ddmmyy format."""
    now = datetime.now()
    time_str = now.strftime("%H%M%S")  # hhmmss format (hours + minutes + seconds)
    date_str = now.strftime("%d%m%y")  # ddmmyy format
    dir_name = f"{time_str}_{date_str}"
    return Path.home() / ".parslbox" / "runs" / dir_name


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load the parslbox configuration file."""
    if config_path is None:
        config_path = path_utils.PBX_CONFIG_FILE
    
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)