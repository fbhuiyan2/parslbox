import typer
from parslbox.database import database
from parslbox.utils import path_utils
from parslbox.utils import pbx_config_utils as config_utils
from parslbox.commands.config import app as config_cmd
from parslbox.commands.ls import app as list_jobs
from parslbox.commands.add import app as add_job
from parslbox.commands.rm import app as remove_jobs
from parslbox.commands.update import app as update_jobs
from parslbox.commands.run import app as run_jobs
from parslbox.commands.info import app as info_jobs
from parslbox.commands.filter import app as filter_jobs
from parslbox.commands.qsub import app as qsub_jobs
from parslbox.commands.sbatch import app as sbatch_jobs

app = typer.Typer(help="A CLI tool to manage parsl workflows and jobs.",
                  no_args_is_help=True,)

@app.callback()
def main_callback(ctx: typer.Context):
    """
    This function runs BEFORE any command.
    It ensures the database and config are ready.
    """
    # Don't check anything if running 'config' command
    if ctx.invoked_subcommand == "config":
        return
    
    # Initialize database in default location (always needed)
    database.initialize_database(path_utils.DB_FILE)
    
    # Check if config exists
    if not path_utils.PBX_CONFIG_FILE.exists():
        # Offer to create config (we're always in CLI mode here. API goes thru ParslBox.__init__())
        typer.secho(
            f"⚠  Config file not found at: {path_utils.PBX_CONFIG_FILE}",
            fg=typer.colors.YELLOW
        )
        response = typer.prompt("Create config now? [Y/n]", default="Y")
        
        if response.lower() in ["y", "yes", ""]:
            # Run pbx config
            from parslbox.commands.config import config as config_cmd_func
            import parslbox.commands.config as config_module
            config_module._is_interactive = True
            
            try:
                config_cmd_func(path=None)  # Will use PBX_CONFIG_PATH or default
            except typer.Exit:
                typer.secho("\n❌ Config creation cancelled.", fg=typer.colors.RED)
                typer.secho("Config is required to run ParslBox commands.", fg=typer.colors.YELLOW)
                raise
            
            typer.secho(
                f"\n✓ Config created! Please re-run your command: pbx {ctx.invoked_subcommand}",
                fg=typer.colors.GREEN
            )
            raise typer.Exit(code=0)
        else:
            typer.secho("Config is required. Run 'pbx config' when ready.", fg=typer.colors.CYAN)
            raise typer.Exit(code=1)

# Add commands to the main application
app.add_typer(config_cmd, name="config", help="Create or reconfigure ParslBox configuration")
app.add_typer(list_jobs)
app.add_typer(add_job)
app.add_typer(remove_jobs)
app.add_typer(update_jobs)
app.add_typer(run_jobs)
app.add_typer(info_jobs)
app.add_typer(filter_jobs)
app.add_typer(qsub_jobs)
app.add_typer(sbatch_jobs)


if __name__ == "__main__":
    app()
