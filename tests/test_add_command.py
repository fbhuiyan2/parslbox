import re
import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from parslbox.commands.add import app as add_app
from parslbox.commands.helpers.job_info_validator import validate_resource_parameters
from parslbox.database import database


class TestValidateResourceParameters:
    """Unit tests for validate_resource_parameters — ranks_per_node + node_occupancy."""

    @pytest.fixture
    def sys_config(self):
        mock = MagicMock()
        mock.GPUS_PER_NODE = 4
        mock.CORES_PER_NODE = 64
        mock.EXCLUDE_CORES = None
        return mock

    @pytest.fixture
    def sys_config_excluded(self):
        mock = MagicMock()
        mock.GPUS_PER_NODE = 4
        mock.CORES_PER_NODE = 64
        mock.EXCLUDE_CORES = [0, 1, 2, 3]
        return mock

    def test_lammps_cpu_full_node(self, sys_config):
        """LAMMPS CPU job, full node → ranks = CORES_PER_NODE."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=1.0,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 64

    def test_lammps_cpu_half_node(self, sys_config):
        """LAMMPS CPU job, nocc=0.5 → ranks = 32."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.5,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 32

    def test_lammps_cpu_quarter_node(self, sys_config):
        """LAMMPS CPU job, nocc=0.25 → ranks = 16."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.25,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 16

    def test_lammps_cpu_half_node_with_excluded(self, sys_config_excluded):
        """LAMMPS CPU, nocc=0.5, 4 excluded cores → ranks = (64-4)*0.5 = 30."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.5,
            system_config=sys_config_excluded, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 30

    def test_python_cpu_half_node(self, sys_config):
        """Python CPU job, nocc=0.5 → ranks stays 1 (Python override)."""
        from parslbox.apps.python import PythonApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.5,
            system_config=sys_config, app_class=PythonApp,
        )
        assert params['final_ranks_per_node'] == 1

    def test_lammps_gpu_job_ignores_occupancy(self, sys_config):
        """GPU jobs should not scale ranks by occupancy."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=2, nnodes=1,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 2

    def test_no_app_class_cpu_half_node(self, sys_config):
        """Fallback (no app class), nocc=0.5 → ranks = 32."""
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.5,
            system_config=sys_config, app_class=None,
        )
        assert params['final_ranks_per_node'] == 32

    def test_user_override_not_scaled(self, sys_config):
        """User-specified ranks_per_node should NOT be scaled by occupancy."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.5, ranks_per_node=8,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_ranks_per_node'] == 8

    # --- Multi-node on GPU system: CPU vs GPU ---

    def test_multinode_gpu_system_no_flags(self, sys_config):
        """Multi-node on GPU system, no -g or -o → auto-assign all GPUs."""
        params, info, _ = validate_resource_parameters(
            ngpus=0, nnodes=2,
            system_config=sys_config,
        )
        assert params['final_num_nodes'] == 2
        assert params['final_ngpus'] == 8  # 2 nodes * 4 GPUs
        assert params['final_node_occupancy'] == 1.0

    def test_multinode_gpu_system_with_nocc(self, sys_config):
        """Multi-node on GPU system with -o → CPU-only, no GPUs."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=2, node_occupancy=1.0,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_num_nodes'] == 2
        assert params['final_ngpus'] == 0
        assert params['final_node_occupancy'] == 1.0
        assert params['final_ranks_per_node'] == 64

    def test_multinode_gpu_system_explicit_gpus(self, sys_config):
        """Multi-node with explicit -g → use specified GPU count."""
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=4, nnodes=2,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_num_nodes'] == 2
        assert params['final_ngpus'] == 4
        assert params['final_ranks_per_node'] == 4  # GPUS_PER_NODE for multi-node

    def test_multinode_cpu_system(self, sys_config):
        """Multi-node on CPU-only system (no GPUs) → CPU job."""
        sys_config.GPUS_PER_NODE = 0
        from parslbox.apps.lammps_kk import LammpsKokkosApp as LammpsKKApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=2,
            system_config=sys_config, app_class=LammpsKKApp,
        )
        assert params['final_num_nodes'] == 2
        assert params['final_ngpus'] == 0
        assert params['final_ranks_per_node'] == 64

    def test_multinode_python_cpu_with_nocc(self, sys_config):
        """Multi-node Python on GPU system with -o → CPU-only, 1 rank."""
        from parslbox.apps.python import PythonApp
        params, _, _ = validate_resource_parameters(
            ngpus=0, nnodes=2, node_occupancy=1.0,
            system_config=sys_config, app_class=PythonApp,
        )
        assert params['final_num_nodes'] == 2
        assert params['final_ngpus'] == 0
        assert params['final_ranks_per_node'] == 1


class TestRanksMayOutnumberResources:
    """`-rpn` above the job's GPU or core count used to be rejected for GPU
    jobs and silently mis-allocated for CPU ones. Sharing is supported now,
    so it warns instead -- LAMMPS and GROMACS both run several ranks per GPU.

    The old bound was `GPUS_PER_NODE`, which is the node's capacity rather
    than the job's: `-g 1 -rpn 6` slipped through it and produced five ranks
    with no GPU at all.
    """

    @pytest.fixture
    def sys_config(self):
        mock = MagicMock()
        mock.GPUS_PER_NODE = 4
        mock.CORES_PER_NODE = 64
        mock.EXCLUDE_CORES = None
        return mock

    def test_more_ranks_than_gpus_is_allowed(self, sys_config):
        params, _, _ = validate_resource_parameters(
            ngpus=1, nnodes=1, ranks_per_node=8, system_config=sys_config)
        assert params['final_ranks_per_node'] == 8
        assert params['final_ngpus'] == 1

    def test_more_ranks_than_gpus_warns(self, sys_config):
        _, _, warnings = validate_resource_parameters(
            ngpus=1, nnodes=1, ranks_per_node=8, system_config=sys_config)
        assert any("8 ranks per node share 1 GPU" in w for w in warnings)

    def test_the_warning_says_how_many_ranks_land_on_each_gpu(self, sys_config):
        _, _, warnings = validate_resource_parameters(
            ngpus=2, nnodes=1, ranks_per_node=7, system_config=sys_config)
        assert any("up to 4 ranks land on one GPU" in w for w in warnings)

    def test_one_rank_per_gpu_says_nothing(self, sys_config):
        _, _, warnings = validate_resource_parameters(
            ngpus=4, nnodes=1, ranks_per_node=4, system_config=sys_config)
        assert warnings == []

    def test_fewer_ranks_than_gpus_says_nothing(self, sys_config):
        _, _, warnings = validate_resource_parameters(
            ngpus=4, nnodes=1, ranks_per_node=2, system_config=sys_config)
        assert warnings == []

    def test_multinode_gpu_bound_is_the_per_node_count(self, sys_config):
        """Multi-node jobs take every GPU on each node, so `ngpus` is the
        total and GPUS_PER_NODE is what a rank count competes with."""
        _, _, warnings = validate_resource_parameters(
            ngpus=8, nnodes=2, ranks_per_node=4, system_config=sys_config)
        assert warnings == []

    def test_multinode_gpu_oversubscription_warns_against_the_per_node_count(self, sys_config):
        _, _, warnings = validate_resource_parameters(
            ngpus=8, nnodes=2, ranks_per_node=6, system_config=sys_config)
        assert any("6 ranks per node share 4 GPU(s) per node" in w for w in warnings)

    def test_multinode_cpu_does_not_warn_about_a_fraction_it_will_not_get(self, sys_config):
        """JobResourceSpec forces multi-node occupancy to 1.0 and the job
        gets whole nodes, so 40 ranks on a 64-core node are not sharing.
        Comparing against `nocc x cores` claimed they shared 32."""
        _, _, warnings = validate_resource_parameters(
            ngpus=0, nnodes=2, node_occupancy=0.5, ranks_per_node=40,
            system_config=sys_config)
        assert warnings == []

    def test_uneven_gpu_blocks_warn(self, sys_config):
        """6 GPUs across 4 ranks is [0,1] [2,3] [4] [5]. LAMMPS Kokkos takes
        one `-k on g N` for every rank and cannot express that."""
        sys_config.GPUS_PER_NODE = 6
        _, _, warnings = validate_resource_parameters(
            ngpus=6, nnodes=1, ranks_per_node=4, system_config=sys_config)
        assert any("do not divide evenly" in w for w in warnings)
        assert any("2 rank(s) get 2 GPUs and the rest get 1" in w for w in warnings)

    def test_an_even_split_does_not_warn(self, sys_config):
        sys_config.GPUS_PER_NODE = 6
        _, _, warnings = validate_resource_parameters(
            ngpus=6, nnodes=1, ranks_per_node=3, system_config=sys_config)
        assert warnings == []

    def test_more_ranks_than_owned_cores_is_allowed_and_warns(self, sys_config):
        params, _, warnings = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.25, ranks_per_node=40,
            system_config=sys_config)
        assert params['final_ranks_per_node'] == 40
        assert any("40 ranks per node share 16 core(s)" in w for w in warnings)

    def test_the_core_bound_follows_occupancy_not_the_whole_node(self, sys_config):
        """`-o 0.25 -rpn 40` owns 16 of 64 cores. The old check compared
        against 64 and passed it silently."""
        _, _, warnings = validate_resource_parameters(
            ngpus=0, nnodes=1, node_occupancy=0.25, ranks_per_node=16,
            system_config=sys_config)
        assert warnings == []

    def test_hardware_threads_are_still_a_hard_ceiling(self, sys_config):
        from parslbox.commands.helpers.job_info_validator import ValidationError
        with pytest.raises(ValidationError, match="exceeds usable cores per node"):
            validate_resource_parameters(
                ngpus=1, nnodes=1, ranks_per_node=65, system_config=sys_config)

    def test_the_ceiling_excludes_reserved_cores(self, sys_config):
        from parslbox.commands.helpers.job_info_validator import ValidationError
        sys_config.EXCLUDE_CORES = [0, 1, 2, 3]
        with pytest.raises(ValidationError, match=r"exceeds usable cores per node \(60\)"):
            validate_resource_parameters(
                ngpus=1, nnodes=1, ranks_per_node=61, system_config=sys_config)

    def test_a_gpu_job_is_no_longer_capped_at_the_gpu_count(self, sys_config):
        """Regression: this raised 'exceeds GPUs per node (4)'."""
        params, _, _ = validate_resource_parameters(
            ngpus=1, nnodes=1, ranks_per_node=64, system_config=sys_config)
        assert params['final_ranks_per_node'] == 64

    def _spec_string(self, sys_config, **kwargs):
        from parslbox.commands.helpers.job_info_validator import (
            calculate_resource_display_info,
        )
        params, _, _ = validate_resource_parameters(system_config=sys_config, **kwargs)
        return calculate_resource_display_info(params, sys_config)['resource_spec_string']

    def test_the_add_summary_counts_ranks_not_gpus(self, sys_config):
        """It derived total ranks from `ngpus`, so a shared-GPU job was
        announced as `r:1` while `pbx ls` showed the same job as `r:8`."""
        assert self._spec_string(
            sys_config, ngpus=1, nnodes=1, ranks_per_node=8) == "n:1-r:8-g:1-nocc:NA"

    def test_the_add_summary_matches_what_ls_renders(self, sys_config):
        """`pbx ls` builds the same string from num_nodes * ranks_per_node."""
        for ngpus, nnodes, ranks in [(1, 1, 8), (4, 1, 4), (4, 1, 2), (8, 2, 6), (8, 2, 9)]:
            params, _, _ = validate_resource_parameters(
                ngpus=ngpus, nnodes=nnodes, ranks_per_node=ranks,
                system_config=sys_config)
            ls_style = (f"n:{params['final_num_nodes']}"
                        f"-r:{params['final_num_nodes'] * params['final_ranks_per_node']}"
                        f"-g:{params['final_ngpus']}-nocc:NA")
            assert self._spec_string(
                sys_config, ngpus=ngpus, nnodes=nnodes,
                ranks_per_node=ranks) == ls_style

    def test_one_rank_per_gpu_renders_as_before(self, sys_config):
        assert self._spec_string(
            sys_config, ngpus=2, nnodes=1, ranks_per_node=2) == "n:1-r:2-g:2-nocc:NA"


class TestAddCommand:
    """Test suite for the add command functionality."""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            db_path = Path(tmp.name)
            database.initialize_database(db_path)
            yield db_path
            # Cleanup
            if db_path.exists():
                db_path.unlink()
    
    @pytest.fixture
    def temp_job_dirs(self):
        """Create temporary job directories for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test job directories
            job1_dir = temp_path / "job1"
            job2_dir = temp_path / "job2"
            job1_dir.mkdir()
            job2_dir.mkdir()
            
            # Create input files
            (job1_dir / "in.lammps").write_text("# LAMMPS input file")
            (job2_dir / "input.vasp").write_text("# VASP input file")
            
            yield {
                "job1": job1_dir,
                "job2": job2_dir,
                "temp_dir": temp_path
            }
    
    @pytest.fixture
    def mock_system_config(self):
        """Mock system configuration."""
        mock_config = MagicMock()
        mock_config.GPUS_PER_NODE = 4
        mock_config.CORES_PER_NODE = 64
        return mock_config
    
    def test_single_node_gpu_job(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding a single-node GPU job."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True,
                 'DFLT_INPUT': 'in.lammps'
             }):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk",
                "--config", "polaris",
                "--ngpus", "2",
                "--tag", "test-gpu"
            ])
            
            assert result.exit_code == 0
            assert "✅ Added 1 job(s), IDs:" in result.stdout
            assert "Resource specification: n:1-r:2-g:2-nocc:NA" in result.stdout
            
            # Verify database entry
            jobs = database.get_jobs(temp_db)
            assert len(jobs) == 1
            job = jobs[0]
            assert job['app'] == 'lammps-kk'
            assert job['num_nodes'] == 1
            assert job['ngpus'] == 2
            assert job['node_occupancy'] == 1.0
            assert job['tag'] == 'test-gpu'
    
    def test_single_node_cpu_job(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding a single-node CPU job with node allocation."""
        runner = CliRunner()
        
        # Create a temporary environment file for Python app
        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('typer.confirm', return_value=True):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "python",
                "--config", "polaris",
                "--nocc", "0.5",
                "--tag", "test-cpu",
                "--envfile", str(env_file)
            ])
            
            assert result.exit_code == 0
            assert "✅ Added 1 job(s), IDs:" in result.stdout
            assert "Resource specification: n:1-r:1-g:0-nocc:0.5" in result.stdout

            # Verify database entry
            jobs = database.get_jobs(temp_db)
            assert len(jobs) == 1
            job = jobs[0]
            assert job['app'] == 'python'
            assert job['num_nodes'] == 1
            assert job['ngpus'] == 0
            assert job['node_occupancy'] == 0.5
            assert job['tag'] == 'test-cpu'
    
    def test_multi_node_job(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding a multi-node job."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True,
                 'DFLT_INPUT': 'in.lammps'
             }):

            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk",
                "--config", "polaris",
                "--nnodes", "2"
            ])
            
            assert result.exit_code == 0
            assert "✅ Added 1 job(s), IDs:" in result.stdout
            assert "Multi-node job will use 8 total GPUs (4 per node)" in result.stdout
            assert "Resource specification: n:2-r:8-g:8-nocc:NA" in result.stdout
            
            # Verify database entry
            jobs = database.get_jobs(temp_db)
            assert len(jobs) == 1
            job = jobs[0]
            assert job['app'] == 'lammps-kk'
            assert job['num_nodes'] == 2
            assert job['node_occupancy'] == 1.0
    
    def test_conflicting_parameters(self, temp_job_dirs, mock_system_config):
        """Test handling of conflicting parameters."""
        runner = CliRunner()
        
        # Create a temporary environment file
        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            temp_db = Path(tmp.name)
            database.initialize_database(temp_db)
            
            with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
                 patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
                 patch('parslbox.commands.add.is_app_registered', return_value=True), \
                 patch('parslbox.commands.add.get_app_config', return_value={
                     'INPUT_REQUIRED': False,
                     'DFLT_INPUT': None
                 }), \
                 patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
                 patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
                 patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                     'INPUT_REQUIRED': False,
                     'DFLT_INPUT': None
                 }), \
                 patch('typer.confirm', return_value=False):
                
                # Test CLI interactive handling of conflicting parameters
                result = runner.invoke(add_app, [
                    str(temp_job_dirs["job1"]),
                    "--app", "python",
                    "--config", "polaris",
                    "--ngpus", "2",
                    "--nocc", "0.5",
                    "--envfile", str(env_file)
                ])
                
                assert result.exit_code == 0
                assert "⚠️  Warning: Both --ngpus and --nocc specified" in result.stdout
                assert "Setting ngpus=0 for CPU-only job" in result.stdout
                assert "Resource specification: n:1-r:1-g:0-nocc:0.5" in result.stdout

            # Cleanup
            if temp_db.exists():
                temp_db.unlink()
    
    def test_gpu_limit_validation(self, temp_db, temp_job_dirs, mock_system_config):
        """Test validation of GPU limits."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk",
                "--config", "polaris",
                "--ngpus", "8"  # More than 4 GPUs per node
            ])
            
            assert result.exit_code == 1
            assert "❌ Error: Requested 8 GPUs but only 4 available per node" in result.stdout
    
    def test_invalid_nocc(self, temp_db, temp_job_dirs, mock_system_config):
        """Test validation of node occupancy range."""
        runner = CliRunner()
        
        # Create a temporary environment file
        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "python",
                "--config", "polaris",
                "--nocc", "1.5",  # Invalid range
                "--envfile", str(env_file)
            ])
            
            # Accept either exit code 1 (application error) or 2 (typer validation error)
            assert result.exit_code in [1, 2]
            assert "❌ Error: --nocc must be between 0.0 and 1.0" in result.stdout
    
    def test_invalid_nnodes(self, temp_db, temp_job_dirs, mock_system_config):
        """Test validation of node count."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk",
                "--config", "polaris",
                "--nnodes", "0"  # Invalid node count
            ])
            
            assert result.exit_code == 1
            assert "❌ Error: --nnodes must be at least 1" in result.stdout
    
    def test_unknown_app(self, temp_db, temp_job_dirs):
        """Test handling of unknown application."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.is_app_registered', return_value=False), \
             patch('parslbox.apps.app_registry.get_registered_apps', return_value=['lammps-kk', 'vasp', 'python']):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "unknown_app",
                "--config", "polaris"
            ])
            
            assert result.exit_code == 1
            assert "❌ Error: Unknown application: 'unknown_app'." in result.stdout
            assert "Available built-in apps:" in result.stdout
    
    def test_nonexistent_path(self, temp_db, mock_system_config):
        """Test handling of nonexistent path."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True):
            
            result = runner.invoke(add_app, [
                "/nonexistent/path",
                "--app", "lammps-kk",
                "--config", "polaris",
            ])
            
            assert result.exit_code == 1
            assert "❌ Failed to add 1 job(s):" in result.stdout
            assert "[1 job(s)] Path '/nonexistent/path' does not exist:" in result.stdout
            assert "/nonexistent/path" in result.stdout
    
    def test_duplicate_job_path(self, temp_db, temp_job_dirs, mock_system_config):
        """Test handling of duplicate job paths."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }):
            
            # Add job first time
            result1 = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--config", "polaris",
                "--app", "lammps-kk"
            ])
            assert result1.exit_code == 0

            # Try to add same job again
            result2 = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--config", "polaris",
                "--app", "lammps-kk"
            ])
            assert result2.exit_code == 1
            assert "❌ Failed to add 1 job(s):" in result2.stdout
            assert "already exists in the database" in result2.stdout
    
    def test_add_all_subdirectories(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding all subdirectories with 'all' argument."""
        runner = CliRunner()
        
        # Create a temporary environment file for Python app
        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('typer.confirm', return_value=True):
            
            # Change to temp directory
            import os
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_job_dirs["temp_dir"])
                
                result = runner.invoke(add_app, [
                    "all",
                    "--config", "polaris",
                    "--app", "python",
                    "--envfile", str(env_file)
                ])
                
                assert result.exit_code == 0
                assert "Successfully added 2 job(s)" in result.stdout
                
                # Verify both jobs were added
                jobs = database.get_jobs(temp_db)
                assert len(jobs) == 2
                
            finally:
                os.chdir(original_cwd)
    
    def test_add_all_with_base_dir(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding all subdirectories of an explicit dir with 'all:<dir>'."""
        runner = CliRunner()

        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")

        base = temp_job_dirs["temp_dir"] / "batch"
        base.mkdir()
        (base / "run_a").mkdir()
        (base / "run_b").mkdir()

        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('typer.confirm', return_value=True):

            result = runner.invoke(add_app, [
                f"all:{base}",
                "--config", "polaris",
                "--app", "python",
                "--envfile", str(env_file)
            ])

            assert result.exit_code == 0
            assert "Successfully added 2 job(s)" in result.stdout

            jobs = database.get_jobs(temp_db)
            assert len(jobs) == 2

    def test_input_file_handling(self, temp_db, temp_job_dirs, mock_system_config):
        """Test input file handling for different app configurations."""
        runner = CliRunner()
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True):
            
            # Test app with required input and default
            with patch('parslbox.apps.app_registry.get_app_config', return_value={
                'INPUT_REQUIRED': True,
                'DFLT_INPUT': 'in.lammps'
            }):
                result = runner.invoke(add_app, [
                    str(temp_job_dirs["job1"]),
                    "--config", "polaris",
                    "--app", "lammps-kk"
                ])
                
                assert result.exit_code == 0
                assert "Using default input file 'in.lammps'" in result.stdout
                
                jobs = database.get_jobs(temp_db)
                assert jobs[0]['in_file'] == 'in.lammps'
    
    def test_default_parameters(self, temp_db, temp_job_dirs, mock_system_config):
        """Test default parameter values."""
        runner = CliRunner()
        
        # Create a temporary environment file for Python app
        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")
        
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': False,
                 'DFLT_INPUT': None
             }), \
             patch('typer.confirm', return_value=True):
            
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--config", "polaris",
                "--app", "python",
                "--envfile", str(env_file)
            ])
            
            assert result.exit_code == 0
            assert "Single-node job on GPU system: auto-assigned 4 GPUs" in result.stdout
            # Python is 1 rank per node owning all 4 GPUs, so r:1. This read
            # r:4 while the display derived ranks from the GPU count -- which
            # disagreed with `pbx ls` on the very same job.
            assert "Resource specification: n:1-r:1-g:4-nocc:NA" in result.stdout

            # Verify default values - GPU system auto-assigns GPUs
            jobs = database.get_jobs(temp_db)
            job = jobs[0]
            assert job['num_nodes'] == 1
            assert job['ngpus'] == 4
            assert job['ranks_per_node'] == 1
            assert job['node_occupancy'] == 1.0
            assert job['status'] == 'Ready'

    def test_add_with_app_args(self, temp_db, temp_job_dirs, mock_system_config):
        """Test adding a job with --args appends to in_file."""
        runner = CliRunner()

        env_file = temp_job_dirs["temp_dir"] / "test_env.sh"
        env_file.write_text("#!/bin/bash\necho 'test environment'")

        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.commands.add.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.add.is_app_registered', return_value=True), \
             patch('parslbox.commands.add.get_app_config', return_value={
                 'INPUT_REQUIRED': True,
                 'DFLT_INPUT': None
             }), \
             patch('parslbox.commands.helpers.job_info_validator.get_system_config', return_value=mock_system_config), \
             patch('parslbox.commands.helpers.job_info_validator.is_app_registered', return_value=True), \
             patch('parslbox.commands.helpers.job_info_validator.get_app_config', return_value={
                 'INPUT_REQUIRED': True,
                 'DFLT_INPUT': None
             }), \
             patch('typer.confirm', return_value=True):

            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--config", "polaris",
                "--app", "python",
                "--input", "script.py",
                "--args", "--file afile -o 8 bfile",
                "--envfile", str(env_file)
            ])

            assert result.exit_code == 0
            assert "✅ Added 1 job(s), IDs:" in result.stdout

            # Verify in_file contains script name + args
            jobs = database.get_jobs(temp_db)
            assert len(jobs) == 1
            assert jobs[0]['in_file'] == "script.py --file afile -o 8 bfile"

    def _seed_parents(self, temp_db, n):
        """Add `n` simple parent jobs and return their IDs."""
        ids = []
        for i in range(n):
            ids.append(database.add_job(
                db_path=temp_db, path=f"/parent_{i}", app="lammps-kk",
                num_nodes=1, ngpus=0, node_occupancy=1.0, tag="parent",
            ))
        return ids

    def test_parents_supports_range(self, temp_db, temp_job_dirs, mock_system_config):
        """--parents '1-3 5' should expand to [1,2,3,5]."""
        self._seed_parents(temp_db, 5)
        runner = CliRunner()
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True, 'DFLT_INPUT': 'in.lammps',
             }):
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk", "--config", "polaris",
                "--ngpus", "2", "--tag", "child",
                "--parents", "1-3 5",
            ])
        assert result.exit_code == 0, result.stdout
        # The new child is job_id 6 (parents seeded 1-5)
        child = [j for j in database.get_jobs(temp_db) if j['tag'] == 'child'][0]
        import json
        assert [int(x) for x in json.loads(child['parents'])] == [1, 2, 3, 5]

    def test_parents_range_past_the_last_job_is_refused(self, temp_db, temp_job_dirs, mock_system_config):
        """`--parents` was rewired to `resolve_job_ids`; without this the
        one-line change can be reverted and '1-100000000' once again builds
        a hundred million integers before anything checks the database."""
        self._seed_parents(temp_db, 5)
        runner = CliRunner()
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True, 'DFLT_INPUT': 'in.lammps',
             }):
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk", "--config", "polaris",
                "--ngpus", "2", "--tag", "child",
                "--parents", "1-100000000",
            ])
        assert result.exit_code != 0
        assert "highest job id" in result.stdout
        assert len(result.stdout) < 400
        assert not [j for j in database.get_jobs(temp_db) if j['tag'] == 'child']

    def test_parents_mixed_ranges_and_singles(self, temp_db, temp_job_dirs, mock_system_config):
        """--parents '1-2 4 6-7' should expand to [1,2,4,6,7]."""
        self._seed_parents(temp_db, 7)
        runner = CliRunner()
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True, 'DFLT_INPUT': 'in.lammps',
             }):
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk", "--config", "polaris",
                "--ngpus", "2", "--tag", "child",
                "--parents", "1-2 4 6-7",
            ])
        assert result.exit_code == 0, result.stdout
        child = [j for j in database.get_jobs(temp_db) if j['tag'] == 'child'][0]
        import json
        assert [int(x) for x in json.loads(child['parents'])] == [1, 2, 4, 6, 7]

    def test_parents_invalid_range_errors(self, temp_db, temp_job_dirs, mock_system_config):
        """Backwards range '5-3' should produce a clear error and not add anything."""
        self._seed_parents(temp_db, 5)
        runner = CliRunner()
        with patch('parslbox.commands.add.path_utils.DB_FILE', temp_db), \
             patch('parslbox.system_configs.loader.get_system_config', return_value=mock_system_config), \
             patch('parslbox.apps.app_registry.is_app_registered', return_value=True), \
             patch('parslbox.apps.app_registry.get_app_config', return_value={
                 'INPUT_REQUIRED': True, 'DFLT_INPUT': 'in.lammps',
             }):
            result = runner.invoke(add_app, [
                str(temp_job_dirs["job1"]),
                "--app", "lammps-kk", "--config", "polaris",
                "--ngpus", "2",
                "--parents", "5-3",
            ])
        assert result.exit_code != 0
        assert "Invalid parent job IDs" in result.stdout


class TestGpuRequestOnACpuOnlySystem:
    """`GPUS_PER_NODE == 0` (crux, perlmutter-cpu, lcrc-improv). The GPU
    availability check only ran for single-node requests, so a multi-node GPU
    request fell past it: it was accepted with ranks_per_node=0, which launches
    `mpiexec -n 0`, and once the sharing warning started dividing by that count
    it raised a raw ZeroDivisionError that `pbx add` does not catch."""

    @pytest.fixture
    def cpu_only(self):
        mock = MagicMock()
        mock.GPUS_PER_NODE = 0
        mock.CORES_PER_NODE = 128
        mock.EXCLUDE_CORES = None
        return mock

    @pytest.mark.parametrize("kwargs", [
        dict(ngpus=2, nnodes=1),
        dict(ngpus=2, nnodes=2),
        dict(ngpus=2, nnodes=2, ranks_per_node=4),
        dict(ngpus=1, nnodes=3, ranks_per_node=1),
    ])
    def test_it_is_rejected_for_every_node_count(self, cpu_only, kwargs):
        from parslbox.commands.helpers.job_info_validator import ValidationError
        with pytest.raises(ValidationError, match="this system has no GPUs"):
            validate_resource_parameters(system_config=cpu_only, **kwargs)

    def test_cpu_jobs_on_the_same_system_are_unaffected(self, cpu_only):
        params, _, warnings = validate_resource_parameters(
            node_occupancy=0.5, nnodes=1, ranks_per_node=40,
            system_config=cpu_only)
        assert params['final_ranks_per_node'] == 40
        assert warnings == []          # 40 ranks fit in the 64 cores it owns

    def test_cpu_sharing_still_warns_on_the_same_system(self, cpu_only):
        _, _, warnings = validate_resource_parameters(
            node_occupancy=0.5, nnodes=1, ranks_per_node=100,
            system_config=cpu_only)
        assert any("100 ranks per node share 64 core(s)" in w for w in warnings)


_CLI_FLAG = re.compile(r"--[a-zA-Z]")


class TestErrorTextReadsTheSameFromEveryLayer:
    """`validate_resource_parameters` raises into the CLI, the API and the MCP
    tool alike, so its messages must not spell parameters as CLI flags: the
    MCP caller's field is `ranks_per_node` and there is no `--ranks-per-node`
    anywhere in its schema. Nothing pinned this wording, so it could drift
    back without failing anything."""

    @pytest.fixture
    def sys_config(self):
        mock = MagicMock()
        mock.GPUS_PER_NODE = 4
        mock.CORES_PER_NODE = 64
        mock.EXCLUDE_CORES = None
        return mock

    def test_the_ceiling_error_names_the_parameter_not_the_flag(self, sys_config):
        from parslbox.commands.helpers.job_info_validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            validate_resource_parameters(ngpus=4, nnodes=1, ranks_per_node=500,
                                         system_config=sys_config)
        msg = str(exc.value)
        assert msg == "ranks_per_node (500) exceeds usable cores per node (64)"
        assert not _CLI_FLAG.search(msg)

    def test_the_positive_integer_error_names_the_parameter_not_the_flag(self, sys_config):
        from parslbox.commands.helpers.job_info_validator import ValidationError
        with pytest.raises(ValidationError) as exc:
            validate_resource_parameters(ngpus=4, nnodes=1, ranks_per_node=0,
                                         system_config=sys_config)
        assert not _CLI_FLAG.search(str(exc.value))
        assert "ranks_per_node" in str(exc.value)

    def test_mps_is_only_suggested_on_nvidia(self):
        """MPS is an NVIDIA service. Aurora is Intel, and the sharing warning
        was telling its users to enable something that does not exist there."""
        nvidia = MagicMock(GPUS_PER_NODE=4, CORES_PER_NODE=64,
                           EXCLUDE_CORES=None, GPU_TYPE="cuda")
        intel = MagicMock(GPUS_PER_NODE=4, CORES_PER_NODE=64,
                          EXCLUDE_CORES=None, GPU_TYPE="intel")
        _, _, nv = validate_resource_parameters(
            ngpus=2, nnodes=1, ranks_per_node=8, system_config=nvidia)
        _, _, it = validate_resource_parameters(
            ngpus=2, nnodes=1, ranks_per_node=8, system_config=intel)
        assert any("MPS" in w for w in nv)
        assert not any("MPS" in w for w in it)
        assert not any("CUDA" in w for w in it)
        # both still say the ranks share
        assert all(any("share 2 GPU(s)" in w for w in ws) for ws in (nv, it))

    def test_no_warning_carries_cli_flag_syntax(self, sys_config):
        """The warnings go into the API's msg_log and the MCP response string."""
        for kwargs in (dict(ngpus=1, nnodes=1, ranks_per_node=8),
                       dict(ngpus=4, nnodes=1, ranks_per_node=3),
                       dict(node_occupancy=0.25, nnodes=1, ranks_per_node=40)):
            _, _, warnings = validate_resource_parameters(
                system_config=sys_config, **kwargs)
            for w in warnings:
                # `--` also appears as prose punctuation; only a dash pair
                # glued to a word is flag syntax.
                assert not _CLI_FLAG.search(w), (kwargs, w)
