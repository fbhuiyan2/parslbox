import logging
import os
import shlex
from pathlib import Path
from parslbox.apps.appbase import AppBase
from parslbox.apps.app_helpers import (
    split_input, read_text_safe, last_segment, job_input_file, newest_file,
    num_nodes, ranks_per_node, threads_per_rank,
)


class GromacsApp(AppBase):
    """
    GROMACS application implementation for ParslBox.

    Runs `gmx_mpi mdrun` on a prebuilt run input (.tpr); `gmx grompp` is not
    run by pbx. The job's input file is the .tpr; extra mdrun flags go in
    `--app-args`.

    Every launch passes `-cpi <name>.cpt`. GROMACS starts from step 0 when no
    checkpoint exists, so the first run and every restart use the same
    command. `-deffnm`, `-s` and `-cpi` are set by pbx, so do not pass them in
    `--app-args`; neither `-noappend` nor `-g`, which rename the log that the
    success check reads (`<name>.log`).

    `-ntomp` assumes a CPU job holds the whole node (pbx does not tell apps
    the node occupancy); pass `-ntomp` in `--app-args` for a CPU job at
    occupancy below 1.

    Restart is "re-run as is" (Scene B in `AppBase.restart`): the command
    already resumes from the checkpoint.
    """

    INPUT_REQUIRED = True
    DFLT_INPUT = "topol.tpr"

    def get_command_template(self, **kwargs) -> str:
        mpi_prefix = kwargs['mpi_prefix']
        mpi_opts_str = kwargs['mpi_opts_str']
        executable = kwargs['executable'] or 'gmx_mpi'
        total_gpus = kwargs['total_gpus']
        mpi_commands = kwargs.get('mpi_commands', {})

        tpr, rest = split_input(kwargs['in_file'])
        stem = os.path.splitext(tpr)[0]

        # A thread-MPI `gmx` ignores the launcher: under mpiexec -n N it would
        # start N independent simulations writing the same files. Fail from the
        # job script, not by raising: a raise inside the Parsl bash app leaves
        # no pbx_job_<id>.err behind, so the reason would only reach the pbx
        # run log.
        total_ranks = ranks_per_node(mpi_commands) * num_nodes(mpi_commands)
        if Path(executable).name == 'gmx' and total_ranks > 1:
            msg = (
                f"ERROR: '{executable}' looks like the thread-MPI build of GROMACS, "
                f"but this job launches {total_ranks} ranks. Point executable_path "
                f"at the MPI build (gmx_mpi), or run a single rank."
            )
            return f"echo {shlex.quote(msg)} >&2\nexit 1"

        from parslbox.system_configs.loader import get_system_config
        system_config = get_system_config(kwargs['config_name'])

        args = ["mdrun", "-deffnm", stem, "-s", tpr, "-cpi", f"{stem}.cpt"]
        rest_flags = rest.split()
        if "-ntomp" not in rest_flags:
            args += ["-ntomp", str(threads_per_rank(system_config, total_gpus, mpi_commands))]
        if total_gpus > 0 and "-nb" not in rest_flags:
            args += ["-nb", "gpu"]
        if rest:
            args.append(rest)

        cmd = " ".join(p for p in [mpi_prefix, mpi_opts_str, executable, *args] if p)
        logging.getLogger(__name__).info(f"GROMACS command: {cmd}")
        return cmd

    def restart(self, job_dict: dict) -> dict | None:
        return {}

    def check_success(self, job_id: int, job_path: Path, db_path: Path, error_message: str = None) -> str:
        """
        Done only when the last mdrun segment of `<name>.log` finished and was
        not cut short, and the run exited cleanly.

        "Finished mdrun" alone is not enough: a run stopped by `-maxh` or by a
        signal still prints it, after a "Run time exceeded" / "Received the
        ... signal, stopping" line. A non-zero exit is Failed whatever the log
        says, so a log left by an earlier completed run cannot turn a failed
        relaunch into Done; re-running a failed job costs little, since the
        command resumes from the checkpoint.
        """
        logger = logging.getLogger(__name__)

        if error_message:
            logger.warning(f"Job {job_id}: execution error: {error_message}. Marking as Failed.")
            return "Failed"

        tpr = job_input_file(job_id, db_path)
        log_file = (Path(job_path) / f"{os.path.splitext(tpr)[0]}.log" if tpr
                    else newest_file(job_path, "*.log"))
        text = read_text_safe(log_file) if log_file else None
        if text is None:
            logger.warning(f"Job {job_id}: GROMACS log not found. Marking as Failed.")
            return "Failed"

        segment = last_segment(text, "Started mdrun on rank")
        if "Finished mdrun on rank" not in segment:
            logger.warning(f"Job {job_id}: 'Finished mdrun' not found in the log. Marking as Failed.")
            return "Failed"
        if "Run time exceeded" in segment or "signal, stopping" in segment:
            logger.warning(f"Job {job_id}: mdrun was stopped early (time limit or signal). Marking as Failed.")
            return "Failed"

        logger.info(f"Job {job_id}: mdrun finished in {log_file.name}.")
        return "Done"
