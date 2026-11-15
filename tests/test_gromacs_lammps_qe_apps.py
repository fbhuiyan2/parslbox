"""Command lines and success checks of the gromacs, lammps and qe apps."""
import pytest

from parslbox.apps import app_helpers
from parslbox.apps.app_registry import APP_FACTORY, get_app_config
from parslbox.apps.gromacs import GromacsApp
from parslbox.apps.lammps import LammpsApp
from parslbox.apps.lammps_kk import LammpsKokkosApp
from parslbox.apps.qe import QeApp
from parslbox.database import database
from parslbox.system_configs.loader import get_system_config

MAP_GPU = "srun --ntasks 4 --gpu-bind=map_gpu:0,1,2,3"
WRAPPER = "mpiexec -n 4 --ppn 4 ./gpu_wrapper_pbx_cuda_7.sh"
MASK_GPU = "srun --ntasks 2 --gpu-bind=mask_gpu:0x3,0xc"
PLAIN = "mpiexec -n 4 --ppn 4"


def _mpi(rpn, hosts=("n0",)):
    return {"PBX_RANKS_PER_NODE": str(rpn), "PBX_HOSTNAMES": ",".join(hosts)}


def _kwargs(config="polaris", prefix=PLAIN, in_file="md.tpr", gpus=0,
            executable="", mpi=None, opts=""):
    return dict(mpi_prefix=prefix, mpi_opts_str=opts, executable=executable,
                in_file=in_file, total_gpus=gpus, config_name=config,
                app_config={}, mpi_commands=mpi or _mpi(4))


def _gmx(**kw):
    return GromacsApp().get_command_template(**_kwargs(**kw))


class TestRegistration:
    @pytest.mark.parametrize("name,cls", [
        ("gromacs", GromacsApp), ("lammps", LammpsApp), ("qe", QeApp)])
    def test_registered(self, name, cls):
        assert APP_FACTORY[name] is cls

    def test_input_defaults(self):
        assert get_app_config("gromacs") == {"INPUT_REQUIRED": True, "DFLT_INPUT": "topol.tpr"}
        assert get_app_config("lammps") == {"INPUT_REQUIRED": True, "DFLT_INPUT": "in.lammps"}
        assert get_app_config("qe") == {"INPUT_REQUIRED": True, "DFLT_INPUT": None}


class TestHelpers:
    def test_split_input(self):
        assert app_helpers.split_input("md.tpr -nsteps 5") == ("md.tpr", "-nsteps 5")
        assert app_helpers.split_input("md.tpr") == ("md.tpr", "")
        assert app_helpers.split_input("") == ("", "")

    def test_last_segment(self):
        assert app_helpers.last_segment("a S1 b S2 c", "S") == "S2 c"
        assert app_helpers.last_segment("no marker", "S") == "no marker"

    def test_read_text_safe(self, tmp_path):
        (tmp_path / "f").write_bytes(b"ok \x80\x94 done")
        assert "done" in app_helpers.read_text_safe(tmp_path / "f")
        assert app_helpers.read_text_safe(tmp_path / "missing") is None

    @pytest.mark.parametrize("prefix,total,rpn,expected", [
        (MAP_GPU, 4, 4, 1), (WRAPPER, 4, 4, 1), (MASK_GPU, 4, 2, 2),
        (PLAIN, 4, 4, 4), (PLAIN, 2, 2, 2)])
    def test_gpus_visible_per_rank_matches_lammps_kk(self, prefix, total, rpn, expected):
        """lammps-kk is the reference for `-k on g N`; plain LAMMPS must agree."""
        mpi = _mpi(rpn)
        got = app_helpers.gpus_visible_per_rank(prefix, total, 4, mpi)
        assert got == expected
        kk = LammpsKokkosApp().get_command_template(**_kwargs(
            prefix=prefix, in_file="in.x", gpus=total, mpi=mpi))
        assert f"-k on g {got} " in kk


class TestThreadsPerRank:
    def test_cpu_job_splits_the_node(self):
        cfg = get_system_config("perlmutter-cpu")
        assert app_helpers.threads_per_rank(cfg, 0, _mpi(16)) == 8

    def test_full_node_gpu_job(self):
        cfg = get_system_config("polaris")
        assert app_helpers.threads_per_rank(cfg, 4, _mpi(4)) == 16

    def test_sub_node_gpu_job_counts_only_the_cores_of_its_gpus(self):
        cfg = get_system_config("polaris")
        assert app_helpers.threads_per_rank(cfg, 1, _mpi(1)) == 16
        assert app_helpers.threads_per_rank(cfg, 2, _mpi(2)) == 16

    def test_more_ranks_than_gpus_share_the_gpus_cores(self):
        cfg = get_system_config("polaris")
        assert app_helpers.threads_per_rank(cfg, 4, _mpi(8)) == 8

    def test_multi_node_uses_gpus_per_node(self):
        cfg = get_system_config("polaris")
        assert app_helpers.threads_per_rank(cfg, 8, _mpi(4, ("a", "b"))) == 16

    def test_excluded_cores_are_not_counted(self):
        cfg = get_system_config("aurora-gpu")  # 208 cores, 4 excluded, 6 GPUs
        assert app_helpers.threads_per_rank(cfg, 6, _mpi(6)) == 34

    def test_never_below_one(self):
        cfg = get_system_config("polaris")
        assert app_helpers.threads_per_rank(cfg, 0, _mpi(200)) == 1


class TestGromacsCommand:
    def test_cpu_job(self):
        cmd = _gmx(config="perlmutter-cpu", prefix="srun -n 16", mpi=_mpi(16))
        assert cmd == ("srun -n 16 gmx_mpi mdrun -deffnm md -s md.tpr "
                       "-cpi md.cpt -ntomp 8")

    def test_gpu_job_offloads_nonbonded(self):
        cmd = _gmx(gpus=4, mpi=_mpi(4))
        assert cmd.endswith("-cpi md.cpt -ntomp 16 -nb gpu")

    def test_the_default_input_name_gives_topol_files(self):
        cmd = _gmx(in_file="topol.tpr")
        assert "-deffnm topol -s topol.tpr -cpi topol.cpt" in cmd

    def test_app_args_are_appended_and_do_not_change_the_stem(self):
        cmd = _gmx(in_file="md.tpr -nsteps 5000")
        assert "-deffnm md -s md.tpr" in cmd
        assert cmd.endswith("-nsteps 5000")

    def test_a_user_ntomp_replaces_ours(self):
        cmd = _gmx(in_file="md.tpr -ntomp 3", gpus=4)
        assert cmd.count("-ntomp") == 1 and "-ntomp 3" in cmd

    def test_a_user_nb_replaces_ours(self):
        cmd = _gmx(in_file="md.tpr -nb cpu", gpus=4)
        assert cmd.count("-nb") == 1 and "-nb cpu" in cmd

    def test_cpu_job_gets_no_nb_flag(self):
        assert "-nb" not in _gmx(gpus=0)

    def test_explicit_executable_and_mpi_opts(self):
        cmd = _gmx(executable="/opt/gmx/bin/gmx_mpi", opts="--cpu-bind none")
        assert cmd.startswith(f"{PLAIN} --cpu-bind none /opt/gmx/bin/gmx_mpi mdrun")

    @pytest.mark.parametrize("exe", ["gmx", "/opt/gromacs/bin/gmx"])
    def test_thread_mpi_build_with_several_ranks_fails_from_the_job_script(self, exe):
        """A raise inside the Parsl bash app leaves no .err file, so the
        command itself prints the reason to stderr and exits non-zero."""
        cmd = _gmx(executable=exe, mpi=_mpi(2))
        assert "thread-MPI" in cmd and ">&2" in cmd
        assert cmd.rstrip().endswith("exit 1")
        assert "mdrun" not in cmd.replace("thread-MPI build of GROMACS", "")

    def test_thread_mpi_build_counts_ranks_across_nodes(self):
        assert "exit 1" in _gmx(executable="gmx", mpi=_mpi(1, ("a", "b")))

    def test_flags_that_merely_start_like_ours_do_not_suppress_ours(self):
        cmd = _gmx(in_file="md.tpr -ntomp_pme 2 -nbfile x", gpus=4)
        assert "-ntomp 16" in cmd and "-nb gpu" in cmd

    def test_thread_mpi_build_with_one_rank_is_allowed(self):
        cmd = _gmx(executable="gmx", mpi=_mpi(1))
        assert "gmx mdrun" in cmd and "exit 1" not in cmd

    def test_restart_reruns_as_is(self):
        assert GromacsApp().restart({"job_id": 1}) == {}


GMX_FINISHED = ("Started mdrun on rank 0 Mon\n...\nFinished mdrun on rank 0 Mon\n")


def _add_job(tmp_path, in_file):
    db = tmp_path / "jobs.db"
    database.initialize_database(db)
    jid = database.add_job(db, str(tmp_path), "gromacs", 1, 0, 1.0, None, in_file=in_file)
    return db, jid


class TestGromacsCheckSuccess:
    def _check(self, tmp_path, log_text, in_file="md.tpr", error=None, use_db=True):
        (tmp_path / f"{in_file.split()[0].rsplit('.', 1)[0]}.log").write_text(log_text)
        db, jid = _add_job(tmp_path, in_file) if use_db else (None, 1)
        return GromacsApp().check_success(jid, tmp_path, db, error)

    def test_a_finished_run_is_done(self, tmp_path):
        assert self._check(tmp_path, GMX_FINISHED) == "Done"

    def test_the_log_is_found_from_the_database_row_ignoring_app_args(self, tmp_path):
        assert self._check(tmp_path, GMX_FINISHED, in_file="md.tpr -nsteps 9") == "Done"

    def test_without_a_database_the_newest_log_is_used(self, tmp_path):
        assert self._check(tmp_path, GMX_FINISHED, use_db=False) == "Done"

    def test_a_time_limit_stop_is_not_done(self, tmp_path):
        text = ("Started mdrun on rank 0\nStep 900: Run time exceeded 0.100 hours, "
                "will terminate the run within 100 steps\nFinished mdrun on rank 0\n")
        assert self._check(tmp_path, text) == "Failed"

    def test_a_signal_stop_is_not_done(self, tmp_path):
        text = ("Started mdrun on rank 0\nReceived the TERM signal, stopping "
                "within 100 steps\nFinished mdrun on rank 0\n")
        assert self._check(tmp_path, text) == "Failed"

    def test_only_the_last_appended_segment_counts(self, tmp_path):
        text = GMX_FINISHED + "Started mdrun on rank 0 later\nstep 10\n"
        assert self._check(tmp_path, text) == "Failed"

    def test_an_earlier_stop_does_not_taint_a_later_clean_segment(self, tmp_path):
        text = ("Started mdrun on rank 0\nRun time exceeded\nFinished mdrun on rank 0\n"
                + GMX_FINISHED)
        assert self._check(tmp_path, text) == "Done"

    def test_a_log_without_the_marker_is_failed(self, tmp_path):
        assert self._check(tmp_path, "Started mdrun on rank 0\nFatal error\n") == "Failed"

    def test_a_missing_log_is_failed(self, tmp_path):
        assert GromacsApp().check_success(1, tmp_path, None) == "Failed"

    def test_an_execution_error_is_failed_even_with_a_finished_log(self, tmp_path):
        """A completed log from an earlier run must not turn a failed
        relaunch into Done."""
        assert self._check(tmp_path, GMX_FINISHED, error="exit 1") == "Failed"

    def test_the_log_named_by_the_database_row_wins_over_a_newer_file(self, tmp_path):
        import os
        (tmp_path / "md.log").write_text(GMX_FINISHED)
        other = tmp_path / "other.log"
        other.write_text("Started mdrun on rank 0\nFatal error\n")
        os.utime(tmp_path / "md.log", (1, 1))
        db, jid = _add_job(tmp_path, "md.tpr")
        assert GromacsApp().check_success(jid, tmp_path, db) == "Done"
        assert GromacsApp().check_success(jid, tmp_path, None) == "Failed"

    def test_undecodable_bytes_do_not_hide_a_finished_run(self, tmp_path):
        (tmp_path / "md.log").write_bytes(b"x \x80\x94\n" + GMX_FINISHED.encode())
        assert GromacsApp().check_success(1, tmp_path, None) == "Done"


class TestLammpsCommand:
    def _cmd(self, **kw):
        return LammpsApp().get_command_template(**_kwargs(in_file="in.x", **kw))

    def test_cpu_job(self):
        assert self._cmd() == f"{PLAIN} lmp -in in.x"

    @pytest.mark.parametrize("prefix,total,rpn,n", [
        (MAP_GPU, 4, 4, 1), (WRAPPER, 4, 4, 1), (MASK_GPU, 4, 2, 2), (PLAIN, 4, 4, 4)])
    def test_gpu_package_gpu_count_follows_the_binding(self, prefix, total, rpn, n):
        cmd = self._cmd(prefix=prefix, gpus=total, mpi=_mpi(rpn))
        assert cmd.endswith(f"lmp -sf gpu -pk gpu {n} -in in.x")

    def test_app_args_ride_along_with_the_input(self):
        cmd = LammpsApp().get_command_template(**_kwargs(in_file="in.x -var T 300"))
        assert cmd.endswith("-in in.x -var T 300")

    def test_threads_are_pinned_to_one(self):
        assert LammpsApp().get_additional_setup() == "export OMP_NUM_THREADS=1"

    def test_restart_is_not_supported(self):
        with pytest.raises(NotImplementedError):
            LammpsApp().restart({})


class TestLammpsCheckSuccess:
    def test_marker_means_done_whatever_the_exit_code(self, tmp_path):
        (tmp_path / "log.lammps").write_text("Total wall time: 0:00:10\n")
        assert LammpsApp().check_success(1, tmp_path, None, "exit 1") == "Done"

    def test_no_marker_is_failed(self, tmp_path):
        (tmp_path / "log.lammps").write_text("ERROR: bad\n")
        assert LammpsApp().check_success(1, tmp_path, None) == "Failed"

    def test_missing_log_is_failed(self, tmp_path):
        assert LammpsApp().check_success(1, tmp_path, None) == "Failed"

    def test_undecodable_bytes_do_not_hide_a_finished_run(self, tmp_path):
        (tmp_path / "log.lammps").write_bytes(b"\x80\x94\nTotal wall time: 0:01:23\n")
        assert LammpsApp().check_success(1, tmp_path, None) == "Done"


class TestQeCommand:
    def _cmd(self, in_file="pw.in", **kw):
        return QeApp().get_command_template(**_kwargs(in_file=in_file, **kw))

    def test_output_is_redirected_next_to_the_input(self):
        assert self._cmd() == f"{PLAIN} pw.x -in pw.in > pw.out"

    def test_app_args_go_before_the_redirect(self):
        assert self._cmd("pw.in -nk 4").endswith("pw.x -in pw.in -nk 4 > pw.out")

    def test_an_input_already_named_out_is_not_overwritten(self):
        assert self._cmd("scf.out").endswith("-in scf.out > scf.out.out")

    def test_input_without_extension(self):
        assert self._cmd("scf").endswith("-in scf > scf.out")

    def test_explicit_executable(self):
        assert " /opt/qe/bin/pw.x -in" in self._cmd(executable="/opt/qe/bin/pw.x")

    def test_threads_default_to_one(self):
        assert QeApp().get_additional_setup() == "export OMP_NUM_THREADS=1"

    def test_restart_is_not_supported(self):
        with pytest.raises(NotImplementedError):
            QeApp().restart({})


class TestQeCheckSuccess:
    def _check(self, tmp_path, out_text, error=None, in_file="pw.in", use_db=True):
        (tmp_path / "pw.out").write_text(out_text)
        db, jid = _add_job(tmp_path, in_file) if use_db else (None, 1)
        return QeApp().check_success(jid, tmp_path, db, error)

    def test_a_converged_run_is_done(self, tmp_path):
        assert self._check(tmp_path, "!  total energy\n   JOB DONE.\n") == "Done"

    def test_without_a_database_the_newest_output_is_used(self, tmp_path):
        assert self._check(tmp_path, "   JOB DONE.\n", use_db=False) == "Done"

    def test_job_done_after_scf_failure_is_not_done(self, tmp_path):
        """pw.x prints JOB DONE. even when SCF did not converge."""
        text = "     convergence NOT achieved after 100 iterations: stopping\n   JOB DONE.\n"
        assert self._check(tmp_path, text) == "Failed"

    def test_job_done_after_a_time_limit_stop_is_not_done(self, tmp_path):
        text = "     Maximum CPU time exceeded\n   JOB DONE.\n"
        assert self._check(tmp_path, text) == "Failed"

    def test_job_done_after_a_user_stop_is_not_done(self, tmp_path):
        text = "     Program stopped by user request\n   JOB DONE.\n"
        assert self._check(tmp_path, text) == "Failed"

    def test_no_marker_is_failed(self, tmp_path):
        assert self._check(tmp_path, "Error in routine\n") == "Failed"

    def test_an_execution_error_is_failed_even_with_the_marker(self, tmp_path):
        assert self._check(tmp_path, "   JOB DONE.\n", error="exit 1") == "Failed"

    def test_missing_output_is_failed(self, tmp_path):
        assert QeApp().check_success(1, tmp_path, None) == "Failed"

    def test_the_output_named_by_the_database_row_wins_over_a_newer_file(self, tmp_path):
        import os
        (tmp_path / "pw.out").write_text("   JOB DONE.\n")
        os.utime(tmp_path / "pw.out", (1, 1))
        (tmp_path / "other.out").write_text("Error in routine\n")
        db, jid = _add_job(tmp_path, "pw.in")
        assert QeApp().check_success(jid, tmp_path, db) == "Done"
        assert QeApp().check_success(jid, tmp_path, None) == "Failed"

    def test_the_output_is_not_confused_with_pbx_job_files(self, tmp_path):
        (tmp_path / "pbx_job_1.out").write_text("   JOB DONE.\n")
        assert QeApp().check_success(1, tmp_path, None) == "Failed"
