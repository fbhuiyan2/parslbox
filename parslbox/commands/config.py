"""
ParslBox Configuration Command

Provides interactive config file creation with curated system/app selections.
Renamed from 'init' to 'config' for better clarity.
"""

import typer
from pathlib import Path
from typing import Optional, List
import os
from datetime import datetime

from parslbox.utils.config_generator import ConfigGenerator, get_builtin_apps
from parslbox.system_configs.loader import get_available_systems, get_system_config
from parslbox.utils import path_utils
from parslbox.database import database

app = typer.Typer()

# Mode detection - set to True when called from CLI
_is_interactive = False


def config_setup(
    path: Optional[str] = None,
    systems: Optional[List[str]] = None,
    apps: Optional[List[str]] = None,
    create_db: Optional[bool] = None,
) -> dict:
    """
    Core configuration setup logic - used by both CLI and API.
    
    Args:
        path: Where to create config:
              - None: Use PBX_CONFIG_PATH if set, else default home (CLI prompts if interactive)
              - ".": Current directory
              - "~" or "home": Default home directory (~/.parslbox)
              - Custom path: Specified directory
        systems: List of system names to include
                 None = error in API mode, prompt in CLI mode
        apps: List of app names to include (empty list = python only)
              None = error in API mode, prompt in CLI mode
        create_db: Whether to create database
                   None = error in API mode, prompt in CLI mode
    
    Returns:
        dict: {
            "config_path": str,
            "db_path": str or None,
            "env_vars_needed": dict or None,
            "systems_selected": List[str],
            "apps_selected": List[str],
        }
    
    Raises:
        FileExistsError: If config file exists (API mode only - CLI prompts)
        ValueError: If invalid systems/apps provided
        RuntimeError: If required parameter is None in API mode
    """
    # Validate API mode has all required parameters
    if not _is_interactive:
        if path is None:
            raise RuntimeError("API mode requires 'path' parameter")
        if systems is None:
            raise RuntimeError("API mode requires 'systems' parameter")
        if apps is None:
            raise RuntimeError("API mode requires 'apps' parameter")
        if create_db is None:
            raise RuntimeError("API mode requires 'create_db' parameter")
    
    # Step 1: Determine config path
    config_path = _determine_config_path(path)
    
    # Step 2: Handle existing config
    _handle_existing_config(config_path)
    
    # Step 3: Select systems
    if systems is None:
        systems = _select_systems()
    else:
        # Validate systems
        available = get_available_systems()
        invalid = [s for s in systems if s not in available]
        if invalid:
            raise ValueError(f"Invalid system(s): {invalid}. Available: {available}")
    
    # Require at least 1 system
    if not systems:
        raise ValueError("At least one system must be selected")
    
    # Step 4: Select apps
    if apps is None:
        apps = _select_apps()
    
    # If no apps selected, default to python only
    if not apps:
        apps = ["python"]
        if _is_interactive:
            typer.secho("\nNo apps selected. Defaulting to 'python' app.", fg=typer.colors.YELLOW)
    
    # Step 5: Generate config
    generator = ConfigGenerator(systems, apps)
    config_content = generator.generate()
    
    # Step 6: Write config file
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    
    # Step 7: Handle database
    db_path = _handle_database(config_path.parent, create_db)
    
    # Step 8: Handle environment variables (if not default location)
    env_vars = _handle_env_vars(config_path.parent)
    
    return {
        "config_path": str(config_path),
        "db_path": str(db_path) if db_path else None,
        "env_vars_needed": env_vars,
        "systems_selected": systems,
        "apps_selected": apps,
    }


@app.command()
def config(
    path: Optional[str] = typer.Argument(
        None,
        help="Where to create config: '.' (current), '~' (home), or custom path. Defaults to PBX_CONFIG_PATH or home."
    ),
):
    """
    Create or reconfigure ParslBox configuration.
    
    Creates a curated config.yaml based on your system and app selections.
    
    Examples:
      pbx config           # Use PBX_CONFIG_PATH or default home
      pbx config ~         # Force default home directory
      pbx config .         # Current directory
      pbx config /custom   # Custom path
    """
    global _is_interactive
    _is_interactive = True
    
    try:
        result = config_setup(
            path=path,
            systems=None,  # Prompt in CLI
            apps=None,     # Prompt in CLI
            create_db=None # Prompt in CLI
        )
        
        # Display success message
        typer.secho("\n" + "="*70, fg=typer.colors.GREEN)
        typer.secho("✓ Configuration created successfully!", fg=typer.colors.GREEN, bold=True)
        typer.secho("="*70, fg=typer.colors.GREEN)
        
        typer.secho(f"\n  Config file: {result['config_path']}", fg=typer.colors.CYAN)
        
        if result.get('db_path'):
            typer.secho(f"  Database:    {result['db_path']}", fg=typer.colors.CYAN)
        
        typer.secho(f"\n  Systems: {', '.join(result['systems_selected'])}", fg=typer.colors.YELLOW)
        typer.secho(f"  Apps:    {', '.join(result['apps_selected'])}", fg=typer.colors.YELLOW)
        
        if result.get('env_vars_needed'):
            typer.secho("\n" + "-"*70, fg=typer.colors.YELLOW)
            typer.secho("⚠  Environment variables required:", fg=typer.colors.YELLOW, bold=True)
            typer.secho("-"*70, fg=typer.colors.YELLOW)
            for var, val in result['env_vars_needed'].items():
                typer.secho(f"  export {var}={val}", fg=typer.colors.MAGENTA)
            typer.secho("\nAdd these to your shell profile (~/.bashrc, ~/.zshrc, etc.)", dim=True)
        
        typer.secho("\n" + "-"*70)
        typer.secho("Next steps:", bold=True)
        typer.secho("-"*70)
        typer.secho("  1. Edit config to add executable paths and environment setup")
        typer.secho("  2. Run 'pbx add' to add jobs to the database")
        typer.secho("  3. Run 'pbx run' to execute your workflow")
        typer.secho("")
        
    except (FileExistsError, ValueError, RuntimeError) as e:
        typer.secho(f"\n❌ Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


# ============================================================================
# Helper Functions
# ============================================================================

def _determine_config_path(path: Optional[str]) -> Path:
    """
    Determine and normalize config path.
    
    Priority:
    1. If path is provided: use it
    2. If PBX_CONFIG_PATH env var is set: use it
    3. Default to ~/.parslbox
    """
    if path is None:
        # Check for PBX_CONFIG_PATH environment variable
        env_path = os.getenv("PBX_CONFIG_PATH")
        
        if env_path:
            # Use environment variable
            config_dir = Path(env_path).expanduser().resolve()
            if _is_interactive:
                typer.secho(f"\n✓ Using PBX_CONFIG_PATH: {config_dir}", fg=typer.colors.CYAN)
        elif _is_interactive:
            # Prompt user
            typer.secho("\nWhere should the config be created?", bold=True)
            typer.echo("  1. Current directory (.)")
            typer.echo("  2. Home directory (~/.parslbox) [Default]")
            typer.echo("  3. Custom path")
            
            choice = typer.prompt("\nChoice", type=int, default=2)
            
            if choice == 1:
                path = "."
            elif choice == 2:
                path = "~"
            elif choice == 3:
                path = typer.prompt("Enter custom path")
            else:
                typer.secho("Invalid choice. Using default (home directory).", fg=typer.colors.YELLOW)
                path = "~"
            
            # Fall through to path normalization below
            config_dir = None  # Will be set below
        else:
            # API mode with no path and no env var - use default
            config_dir = Path.home() / ".parslbox"
    
    # Normalize path if it was provided or set by prompt
    if path is not None:
        if path in ["~", "home"]:
            config_dir = Path.home() / ".parslbox"
        elif path in [".", "./"]:
            config_dir = Path.cwd()
        else:
            config_dir = Path(path).expanduser().resolve()
    
    config_path = config_dir / "config.yaml"
    
    if _is_interactive and path is not None:
        typer.secho(f"\n✓ Config will be created at: {config_path}", fg=typer.colors.CYAN)
    
    return config_path


def _handle_existing_config(config_path: Path) -> None:
    """Handle existing config file - prompt to keep/delete in CLI, error in API."""
    if not config_path.exists():
        return
    
    if not _is_interactive:
        raise FileExistsError(f"Config file already exists: {config_path}")
    
    typer.secho(f"\n⚠  Config file already exists at: {config_path}", fg=typer.colors.YELLOW, bold=True)
    typer.echo("\nWhat would you like to do?")
    typer.echo("  1. Keep old config (rename it)")
    typer.echo("  2. Delete old config")
    
    choice = typer.prompt("Choice", type=int, default=1)
    
    if choice == 1:
        # Prompt for backup name
        backup_name = _prompt_for_backup_name(config_path)
        backup_path = config_path.parent / backup_name
        
        # Validate backup doesn't exist
        while backup_path.exists():
            typer.secho(f"\n❌ Error: '{backup_name}' already exists.", fg=typer.colors.RED)
            backup_name = _prompt_for_backup_name(config_path)
            backup_path = config_path.parent / backup_name
        
        config_path.rename(backup_path)
        typer.secho(f"✓ Renamed old config to: {backup_name}", fg=typer.colors.GREEN)
    
    elif choice == 2:
        config_path.unlink()
        typer.secho("✓ Deleted old config", fg=typer.colors.GREEN)
    
    else:
        typer.secho("Invalid choice. Aborting.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def _prompt_for_backup_name(original_path: Path) -> str:
    """Prompt for backup filename with suggested default."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{original_path.name}.backup.{timestamp}"
    
    backup_name = typer.prompt(
        "\nEnter new name for the old config file",
        default=default_name
    )
    
    return backup_name


def _handle_database(db_dir: Path, create_db: Optional[bool]) -> Optional[Path]:
    """Handle database creation."""
    db_path = db_dir / "job_database_pbx.db"
    
    # Check if database already exists FIRST
    if db_path.exists():
        # Database exists - just inform user and use it
        if _is_interactive:
            typer.secho(f"✓ Using existing database at: {db_path}", fg=typer.colors.GREEN)
        return db_path
    
    # Database doesn't exist - ask if user wants to create one
    if create_db is None:
        if not _is_interactive:
            raise RuntimeError("create_db cannot be None in API mode")
        
        response = typer.prompt(f"\nCreate database in {db_dir}? [Y/n]", default="Y")
        create_db = response.lower() in ["y", "yes", ""]
    
    if not create_db:
        return None
    
    # Create database
    database.initialize_database(db_path)
    
    if _is_interactive:
        typer.secho(f"✓ Created database at: {db_path}", fg=typer.colors.GREEN)
    
    return db_path


def _handle_env_vars(config_dir: Path) -> Optional[dict]:
    """Display env vars if config is not in default location."""
    default_config_dir = Path.home() / ".parslbox"
    
    # If using default location, no env vars needed
    if config_dir.resolve() == default_config_dir.resolve():
        return None
    
    # Non-default location - need env vars
    env_vars = {
        "PBX_CONFIG_PATH": str(config_dir),
        "PBX_DB_PATH": str(config_dir),
    }
    
    return env_vars


def _select_systems() -> List[str]:
    """Interactive system selection."""
    if not _is_interactive:
        raise RuntimeError("Cannot prompt for systems in API mode")
    
    available_systems = get_available_systems()
    
    typer.secho("\nAvailable systems:", bold=True)
    for i, sys_name in enumerate(available_systems, 1):
        typer.echo(f"  {i}. {sys_name}")
    
    while True:
        try:
            input_str = typer.prompt("\nSelect systems (e.g., 1 3 5 or 'all')")
            selected = _parse_selection(input_str, available_systems)
            
            if not selected:
                typer.secho("Error: At least one system must be selected.", fg=typer.colors.RED)
                continue
            
            typer.secho(f"✓ Selected: {', '.join(selected)}", fg=typer.colors.GREEN)
            return selected
            
        except ValueError as e:
            typer.secho(f"Error: {e}", fg=typer.colors.RED)
            continue


def _select_apps() -> List[str]:
    """Interactive app selection."""
    if not _is_interactive:
        raise RuntimeError("Cannot prompt for apps in API mode")
    
    available_apps = get_builtin_apps()
    
    typer.secho("\nAvailable applications:", bold=True)
    for i, app_name in enumerate(available_apps, 1):
        typer.echo(f"  {i}. {app_name}")
    
    while True:
        try:
            input_str = typer.prompt(
                "\nSelect applications (e.g., 1 2, 'all', or Enter to skip)",
                default=""
            )
            
            if not input_str.strip():
                # Empty selection - will default to python later
                return []
            
            selected = _parse_selection(input_str, available_apps)
            typer.secho(f"✓ Selected: {', '.join(selected)}", fg=typer.colors.GREEN)
            return selected
            
        except ValueError as e:
            typer.secho(f"Error: {e}", fg=typer.colors.RED)
            continue


def _parse_selection(input_str: str, options: List[str]) -> List[str]:
    """Parse space-separated or comma-separated input and return selected items."""
    try:
        # Handle "all" keyword
        if input_str.strip().lower() == "all":
            return list(options)

        # First try to split by whitespace (space, tab, etc.)
        # This handles: "1 3 5", "1  3   5", "1\t3\t5"
        parts = input_str.split()
        
        # If no spaces found, try comma separation
        # This handles: "1,3,5", "1, 3, 5"
        if len(parts) == 1 and ',' in input_str:
            parts = [x.strip() for x in input_str.split(',') if x.strip()]
        
        if not parts:
            return []
        
        indices = [int(x) - 1 for x in parts]
        
        # Validate all indices
        invalid = [i+1 for i in indices if i < 0 or i >= len(options)]
        if invalid:
            raise ValueError(f"Invalid selection(s): {invalid}. Valid range: 1-{len(options)}")
        
        # Return selected items (preserve order, remove duplicates)
        seen = set()
        result = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                result.append(options[i])
        
        return result
    
    except ValueError as e:
        if "invalid literal" in str(e):
            raise ValueError(f"Invalid input format. Expected space or comma-separated numbers (e.g., 1 3 5)")
        raise
