import logging
import os
from pathlib import Path
from parslbox.apps.appbase import AppBase
from parslbox.apps.app_helpers import split_input, read_text_safe, job_input_file, newest_file

# pw.x prints "JOB DONE." however the run ended, including when SCF did not
# converge or a time limit stopped it, and exits 0 unless built with
# __RETURN_EXIT_STATUS. These lines say the run did not finish its work.
_STOPPED_EARLY = (
    "convergence NOT achieved",
    "Maximum CPU time exceeded",
    "Program stopped by user request",
)


class QeApp(AppBase):
    """
    Quantum ESPRESSO `pw.x` for ParslBox.

    Runs `pw.x -in <input>` with standard output written to `<input>.out`
    (the input's extension replaced). The input file is required, since QE has
    no conventional name. Parallelization flags such as `-nk` go in
    `--app-args`. `OMP_NUM_THREADS` is set to 1 after `environment_setup`
    runs, so it cannot be changed there.

    Only `pw.x` is covered. Restart is not supported (Scene C in
    `AppBase.restart`).
    """

    INPUT_REQUIRED = True
    DFLT_INPUT = None

    @staticmethod
    def _output_name(input_file: str) -> str:
        out = f"{os.path.splitext(input_file)[0]}.out"
        return out if out != input_file else f"{input_file}.out"

    def get_command_template(self, **kwargs) -> str:
        mpi_prefix = kwargs['mpi_prefix']
        mpi_opts_str = kwargs['mpi_opts_str']
        executable = kwargs['executable'] or 'pw.x'

        inp, rest = split_input(kwargs['in_file'])
        parts = [mpi_prefix, mpi_opts_str, executable, "-in", inp]
        if rest:
            parts.append(rest)
        # Fresh each run, so a restart cannot read a marker left by an earlier one.
        parts += [">", self._output_name(inp)]

        cmd = " ".join(p for p in parts if p)
        logging.getLogger(__name__).info(f"QE command: {cmd}")
        return cmd

    def get_additional_setup(self, **kwargs) -> str:
        return "export OMP_NUM_THREADS=1"

    def check_success(self, job_id: int, job_path: Path, db_path: Path, error_message: str = None) -> str:
        """
        Done when the output has "JOB DONE." and none of the stopped-early
        lines, and the run exited cleanly. Ion-relaxation non-convergence is
        not detected.
        """
        logger = logging.getLogger(__name__)

        if error_message:
            logger.warning(f"Job {job_id}: execution error: {error_message}. Marking as Failed.")
            return "Failed"

        inp = job_input_file(job_id, db_path)
        out_file = (Path(job_path) / self._output_name(inp) if inp
                    else newest_file(job_path, "*.out"))
        text = read_text_safe(out_file) if out_file else None
        if text is None:
            logger.warning(f"Job {job_id}: pw.x output file not found. Marking as Failed.")
            return "Failed"

        if "JOB DONE." not in text:
            logger.warning(f"Job {job_id}: 'JOB DONE.' not found in {out_file.name}. Marking as Failed.")
            return "Failed"
        for line in _STOPPED_EARLY:
            if line in text:
                logger.warning(f"Job {job_id}: pw.x stopped early ('{line}'). Marking as Failed.")
                return "Failed"
        return "Done"
