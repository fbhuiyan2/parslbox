import logging
from pathlib import Path
from parslbox.apps.appbase import AppBase


class OrcaApp(AppBase):
    """
    ORCA application implementation for ParslBox.

    ORCA manages its own internal MPI parallelism via a bundled OpenMPI.
    It is NOT launched with an external mpirun/mpiexec. Instead, pbx
    sets USES_MPI = True for proper multi-node resource allocation but
    the command template ignores mpi_prefix entirely.

    ORCA requires:
    - A .nodes file listing hostnames and available slots per host
    - A --host argument passed through the command line

    MPI config note: Since ORCA ignores pbx's mpi_prefix, most mpi:
    settings (backend, use_hostlist, cpu_bind_method, add, disable)
    have no effect. The only setting that matters is use_short_hostnames,
    which controls whether hostnames in the .nodes file and --host arg
    are stripped of their domain suffix.
    """

    INPUT_REQUIRED = True
    DFLT_INPUT = "input.inp"

    def get_command_template(self, **kwargs) -> str:
        executable = kwargs.get('executable', 'orca')
        in_file = kwargs['in_file']
        mpi_commands = kwargs.get('mpi_commands', {})

        hostnames_str = mpi_commands.get('PBX_HOSTNAMES', '')
        cores_per_node = mpi_commands.get('PBX_CORES_PER_NODE', '1')

        nodes_file = in_file.rsplit('.', 1)[0] + '.nodes'

        hostnames = hostnames_str.split(',') if hostnames_str else []
        nodes_content = '\n'.join(f"{h} slots={cores_per_node}" for h in hostnames)
        host_arg = ','.join(f"{h}:{cores_per_node}" for h in hostnames)

        return f"""cat > {nodes_file} << 'ORCA_NODES_EOF'
{nodes_content}
ORCA_NODES_EOF
echo "INFO: ORCA nodes file: {nodes_file}"
echo "INFO: ORCA host arg: --host {host_arg}"
{executable} ./{in_file} "--host {host_arg}"
"""

    def get_additional_setup(self, **kwargs) -> str:
        return "export OMP_NUM_THREADS=1"

    def check_success(self, job_id: int, job_path: Path, db_path: Path, error_message: str = None) -> str:
        logger = logging.getLogger(__name__)

        if error_message:
            logger.warning(f"Job {job_id}: ORCA execution error: {error_message}")
            return "Failed"

        for f in Path(job_path).glob("*.out"):
            try:
                if "****ORCA TERMINATED NORMALLY****" in f.read_text():
                    logger.info(f"Job {job_id}: ORCA terminated normally (found in {f.name})")
                    return "Done"
            except Exception:
                continue

        logger.warning(f"Job {job_id}: ORCA normal termination string not found in any .out file")
        return "Failed"
