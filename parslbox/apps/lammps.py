import logging
from pathlib import Path
from parslbox.apps.appbase import AppBase
from parslbox.apps.app_helpers import read_text_safe, gpus_visible_per_rank


class LammpsApp(AppBase):
    """
    Plain LAMMPS (no Kokkos) for ParslBox.

    CPU jobs run `lmp -in <input>`. GPU jobs run the GPU package:
    `lmp -sf gpu -pk gpu N -in <input>`, where N is the number of GPUs each
    rank can see (one for srun `map_gpu` or a per-rank wrapper script; every
    GPU on the node when nothing isolates them). The GPU package needs a
    LAMMPS built with it; for a Kokkos build use `lammps-kk`. On a GPU system a
    job gets all GPUs unless `--nocc` is given, so add `--nocc 1.0` (or any
    occupancy) to run this app on CPUs. `OMP_NUM_THREADS` is set to 1 after
    `environment_setup` runs, so it cannot be changed there.

    Restart is not supported (Scene C in `AppBase.restart`).
    """

    INPUT_REQUIRED = True
    DFLT_INPUT = "in.lammps"

    def get_command_template(self, **kwargs) -> str:
        mpi_prefix = kwargs['mpi_prefix']
        mpi_opts_str = kwargs['mpi_opts_str']
        executable = kwargs['executable'] or 'lmp'
        in_file = kwargs['in_file']
        total_gpus = kwargs['total_gpus']

        if total_gpus > 0:
            from parslbox.system_configs.loader import get_system_config
            gpus_per_node = get_system_config(kwargs['config_name']).GPUS_PER_NODE
            n = gpus_visible_per_rank(mpi_prefix, total_gpus, gpus_per_node,
                                      kwargs.get('mpi_commands', {}))
            lammps_args = f"-sf gpu -pk gpu {n} -in {in_file}"
        else:
            lammps_args = f"-in {in_file}"

        cmd = " ".join(p for p in [mpi_prefix, mpi_opts_str, executable, lammps_args] if p)
        logging.getLogger(__name__).info(f"LAMMPS command: {cmd}")
        return cmd

    def get_additional_setup(self, **kwargs) -> str:
        return "export OMP_NUM_THREADS=1"

    def check_success(self, job_id: int, job_path: Path, db_path: Path, error_message: str = None) -> str:
        """
        Done when `log.lammps` holds "Total wall time:". Execution errors are
        ignored, as for `lammps-kk`: LAMMPS can exit ungracefully after a
        successful run.
        """
        logger = logging.getLogger(__name__)

        if error_message:
            logger.info(f"Job {job_id}: execution error ignored for LAMMPS, "
                        f"checking the log instead: {error_message}")

        text = read_text_safe(Path(job_path) / "log.lammps")
        if text is None:
            logger.warning(f"Job {job_id}: log.lammps not found. Marking as Failed.")
            return "Failed"
        if "Total wall time:" in text:
            return "Done"
        logger.warning(f"Job {job_id}: success marker not found in log.lammps. Marking as Failed.")
        return "Failed"
