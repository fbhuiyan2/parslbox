import typer
from parslbox.helpers import database, path_utils, config_utils
from parslbox.commands.ls import app as list_jobs
from parslbox.commands.add import app as add_job
from parslbox.commands.rm import app as remove_jobs
from parslbox.commands.update import app as update_jobs
from parslbox.commands.run import app as run_jobs
from parslbox.commands.info import app as info_jobs
from parslbox.commands.filter import app as filter_jobs
from parslbox.commands.qsub import app as qsub_jobs

app = typer.Typer(help="A CLI tool to manage parsl workflows and jobs.",
                  no_args_is_help=True,)

@app.callback()
def main_callback():
    """
    This function runs BEFORE any command.
    It ensures the database directory and file are ready.
    """
    # print("DEBUG: main_callback is running, initializing database...")
    database.initialize_database(path_utils.DB_FILE)
    config_utils.initialize_config_file()

# Add the 'jobs' subcommand to the main application
app.add_typer(list_jobs)
app.add_typer(add_job)
app.add_typer(remove_jobs)
app.add_typer(update_jobs)
app.add_typer(run_jobs)
app.add_typer(info_jobs)
app.add_typer(filter_jobs)
app.add_typer(qsub_jobs)


if __name__ == "__main__":
    app()
