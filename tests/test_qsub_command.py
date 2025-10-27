import pytest
import tempfile
import yaml
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typer.testing import CliRunner
from datetime import datetime

from parslbox.commands.qsub import app as qsub_app, minutes_to_hms, get_default_run_dir, load_config


class TestQsubCommand:
    """Test suite for the qsub command functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration for testing."""
        return {
            'schedulers': {
                'pbs': {
                    'template': '''#!/bin/bash
#PBS -N {job_name}
#PBS -q {queue}
#PBS -l select={select}:ncpus=32:ngpus=4
#PBS -l walltime={walltime}
#PBS -l filesystems={filesystems}
#PBS -A {project}

{python_env_setup}

cd {run_dir}
parslbox run --config {config} {run_options}
'''
                }
            },
            'polaris': {
                'python_env_setup': 'module load conda\nconda activate parslbox'
            },
            'sophia': {
                'python_env_setup': 'source /opt/miniconda3/bin/activate parslbox'
            }
        }
    
    @pytest.fixture
    def temp_config_file(self, mock_config):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(mock_config, tmp)
            config_path = Path(tmp.name)
            yield config_path
            # Cleanup
            if config_path.exists():
                config_path.unlink()
    
    @pytest.fixture
    def temp_run_dir(self):
        """Create a temporary run directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_minutes_to_hms_conversion(self):
        """Test conversion of minutes to HH:MM:SS format."""
        assert minutes_to_hms(90) == "01:30:00"
        assert minutes_to_hms(60) == "01:00:00"
        assert minutes_to_hms(30) == "00:30:00"
        assert minutes_to_hms(150) == "02:30:00"
        assert minutes_to_hms(0) == "00:00:00"
        assert minutes_to_hms(1440) == "24:00:00"  # 24 hours
    
    @patch('parslbox.commands.qsub.datetime')
    def test_get_default_run_dir(self, mock_datetime):
        """Test generation of default run directory."""
        # Mock datetime to return a specific time
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%H%M%S": "143022",  # 14:30:22
            "%d%m%y": "151223"   # 15/12/23
        }[fmt]
        mock_datetime.now.return_value = mock_now
        
        result = get_default_run_dir()
        expected = Path.home() / ".parslbox" / "runs" / "143022_151223"
        assert result == expected
    
    def test_load_config_success(self, temp_config_file):
        """Test successful config loading."""
        with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file):
            config = load_config()
            assert 'schedulers' in config
            assert 'pbs' in config['schedulers']
            assert 'polaris' in config
    
    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist."""
        with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', Path('/nonexistent/config.yaml')):
            with pytest.raises(FileNotFoundError):
                load_config()
    
    def test_qsub_basic_functionality(self, temp_config_file, temp_run_dir):
        """Test basic qsub functionality with minimal parameters."""
        runner = CliRunner()
        
        with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file), \
             patch('parslbox.commands.qsub.get_default_run_dir', return_value=temp_run_dir), \
             patch('subprocess.run') as mock_subprocess:
            
            # Mock successful qsub submission
            mock_result = MagicMock()
            mock_result.stdout = "12345.polaris-pbs-01.alcf.anl.gov"
            mock_subprocess.return_value = mock_result
            
            result = runner.invoke(qsub_app, [
                "--config", "polaris",
                "--job-name", "test-job",
                "--queue", "debug",
                "--select", "1",
                "--walltime", "30",
                "--project", "test-project"
            ])
            
            assert result.exit_code == 0
            assert "📁 Created run directory:" in result.stdout
            assert "📝 Generated submit script:" in result.stdout
            assert "🚀 Job submitted successfully!" in result.stdout
            assert "Job ID: 12345.polaris-pbs-01.alcf.anl.gov" in result.stdout
            
            # Verify submit script was created
            submit_file = temp_run_dir / "submit.sh"
            assert submit_file.exists()
            
            # Verify qsub was called
            mock_subprocess.assert_called_once_with(
                ['qsub', 'submit.sh'],
                cwd=temp_run_dir,
                capture_output=True,
                text=True,
                check=True
            )
    
    def test_qsub_with_all_options(self, temp_config_file, temp_run_dir):
        """Test qsub with all optional parameters."""
        runner = CliRunner()
        
        with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file), \
             patch('parslbox.commands.qsub.get_default_run_dir', return_value=temp_run_dir), \
             patch('subprocess.run') as mock_subprocess:
            
            # Mock successful qsub submission
            mock_result = MagicMock()
            mock_result.stdout = "12345.polaris-pbs-01.alcf.anl.gov"
            mock_subprocess.return_value = mock_result
            
            result = runner.invoke(qsub_app, [
                "--config", "polaris",
                "--job-name", "test-job",
                "--queue", "prod",
                "--select", "4",
                "--walltime", "120",
                "--project", "test-project",
                "--filesystems", "home:eagle",
                "--apps", "lammps,vasp",
                "--tags", "run1,run2",
                "--retries", "2"
            ])
            
            assert result.exit_code == 0
            assert "🚀 Job submitted successfully!" in result.stdout
            
            # Verify submit script content
            submit_file = temp_run_dir / "submit.sh"
            assert submit_file.exists()
            
            content = submit_file.read_text()
            assert "#PBS -N test-job" in content
            assert "#PBS -q prod" in content
            assert "#PBS -l select=4" in content
            assert "#PBS -l walltime=02:00:00" in content
            assert "#PBS -l filesystems=home:eagle" in content
            assert "#PBS -A test-project" in content
            assert "module load conda" in content
            assert "conda activate parslbox" in content
            assert "--apps lammps,vasp" in content
            assert "--tags run1,run2" in content
            assert "--retries 2" in content
    
    def test_qsub_without_filesystems(self, temp_config_file, temp_run_dir):
        """Test qsub without filesystems parameter."""
        runner = CliRunner()
        
        with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file), \
             patch('parslbox.commands.qsub.get_default_run_dir', return_value=temp_run_dir), \
             patch('subprocess.run') as mock_subprocess:
            
            # Mock successful qsub submission
            mock_result = MagicMock()
            mock_result.stdout = "12345.polaris-pbs-01.alcf.anl.gov"
            mock_subprocess.return_value = mock_result
            
            result = runner.invoke(qsub_app, [
                "--config", "polaris",
                "--job-name", "test-job",
                "--queue", "debug",
                "--select", "1",
                "--walltime", "30",
                "--project", "test-project"
            ])
            
            assert result.exit_code == 0
            
            # Verify filesystems line is removed from submit script
            submit_file = temp_run_dir / "submit.sh"
            content = submit_file.read_text()
            assert "#PBS -l filesystems=" not in content
    
    def test_qsub_custom_run_dir(self, temp_config_file):
        """Test qsub with custom run directory."""
        runner = CliRunner()
        
        with tempfile.TemporaryDirectory() as custom_dir:
            custom_run_dir = Path(custom_dir) / "custom_run"
            
            with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file), \
                 patch('subprocess.run') as mock_subprocess:
                
                # Mock successful qsub submission
                mock_result = MagicMock()
                mock_result.stdout = "12345.polaris-pbs-01.alcf.anl.gov"
                mock_subprocess.return_value = mock_result
                
                result = runner.invoke(qsub_app, [
                    "--config", "polaris",
                    "--job-name", "test-job",
                    "--queue", "debug",
                    "--select", "1",
                    "--walltime", "30",
                    "--project", "test-project",
                    "--run-dir", str(custom_run_dir)
                ])
                
                assert result.exit_code == 0
                assert custom_run_dir.exists()
                assert (custom_run_dir / "submit.sh").exists()
    
    def test_qsub_config_not_found(self, temp_config_file):
        """Test qsub with missing configuration file."""
        runner = CliRunner()
        
        with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', Path('/nonexistent/config.yaml')):
            result = runner.invoke(qsub_app, [
                "--config", "polaris",
                "--job-name", "test-job",
                "--queue", "debug",
                "--select", "1",
                "--walltime", "30",
                "--project", "test-project"
            ])
            
            assert result.exit_code == 1
            assert "❌ Error:" in result.stdout
            assert "Configuration file not found" in result.stdout
    
    def test_qsub_missing_pbs_template(self, temp_config_file):
        """Test qsub when PBS template is missing from config."""
        runner = CliRunner()
        
        # Create config without PBS scheduler
        config_without_pbs = {
            'polaris': {
                'python_env_setup': 'module load conda'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
            yaml.dump(config_without_pbs, tmp)
            config_path = Path(tmp.name)
        
        try:
            with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', config_path):
                result = runner.invoke(qsub_app, [
                    "--config", "polaris",
                    "--job-name", "test-job",
                    "--queue", "debug",
                    "--select", "1",
                    "--walltime", "30",
                    "--project", "test-project"
                ])
                
                assert result.exit_code == 1
                assert "❌ Error: PBS scheduler template not found" in result.stdout
        finally:
            config_path.unlink()
    
    def test_qsub_invalid_system_config(self, temp_config_file):
        """Test qsub with invalid system configuration."""
        runner = CliRunner()
        
        with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file):
            result = runner.invoke(qsub_app, [
                "--config", "nonexistent-system",
                "--job-name", "test-job",
                "--queue", "debug",
                "--select", "1",
                "--walltime", "30",
                "--project", "test-project"
            ])
            
            assert result.exit_code == 1
            assert "❌ Error: System 'nonexistent-system' not found" in result.stdout
    
    def test_qsub_submission_failure(self, temp_config_file, temp_run_dir):
        """Test qsub when job submission fails."""
        runner = CliRunner()
        
        with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file), \
             patch('parslbox.commands.qsub.get_default_run_dir', return_value=temp_run_dir), \
             patch('subprocess.run') as mock_subprocess:
            
            # Mock failed qsub submission
            mock_subprocess.side_effect = subprocess.CalledProcessError(
                1, ['qsub', 'submit.sh'], stderr="qsub: Job rejected by server"
            )
            
            result = runner.invoke(qsub_app, [
                "--config", "polaris",
                "--job-name", "test-job",
                "--queue", "debug",
                "--select", "1",
                "--walltime", "30",
                "--project", "test-project"
            ])
            
            assert result.exit_code == 1
            assert "❌ Error submitting job:" in result.stdout
            assert "Job rejected by server" in result.stdout
            assert "📄 Submit script saved at:" in result.stdout
    
    def test_qsub_command_not_found(self, temp_config_file, temp_run_dir):
        """Test qsub when qsub command is not available."""
        runner = CliRunner()
        
        with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file), \
             patch('parslbox.commands.qsub.get_default_run_dir', return_value=temp_run_dir), \
             patch('subprocess.run') as mock_subprocess:
            
            # Mock qsub command not found
            mock_subprocess.side_effect = FileNotFoundError("qsub command not found")
            
            result = runner.invoke(qsub_app, [
                "--config", "polaris",
                "--job-name", "test-job",
                "--queue", "debug",
                "--select", "1",
                "--walltime", "30",
                "--project", "test-project"
            ])
            
            assert result.exit_code == 1
            assert "❌ Error: 'qsub' command not found" in result.stdout
            assert "Make sure PBS is available" in result.stdout
            assert "📄 Submit script saved at:" in result.stdout
    
    def test_qsub_different_system_configs(self, temp_config_file, temp_run_dir):
        """Test qsub with different system configurations."""
        runner = CliRunner()
        
        with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file), \
             patch('parslbox.commands.qsub.get_default_run_dir', return_value=temp_run_dir), \
             patch('subprocess.run') as mock_subprocess:
            
            # Mock successful qsub submission
            mock_result = MagicMock()
            mock_result.stdout = "12345.sophia-pbs-01"
            mock_subprocess.return_value = mock_result
            
            result = runner.invoke(qsub_app, [
                "--config", "sophia",
                "--job-name", "test-job",
                "--queue", "debug",
                "--select", "1",
                "--walltime", "30",
                "--project", "test-project"
            ])
            
            assert result.exit_code == 0
            
            # Verify sophia-specific python environment setup
            submit_file = temp_run_dir / "submit.sh"
            content = submit_file.read_text()
            assert "source /opt/miniconda3/bin/activate parslbox" in content
    
    def test_qsub_walltime_formatting(self, temp_config_file, temp_run_dir):
        """Test various walltime values and their formatting."""
        runner = CliRunner()
        
        test_cases = [
            (30, "00:30:00"),
            (60, "01:00:00"),
            (90, "01:30:00"),
            (120, "02:00:00"),
            (1440, "24:00:00")
        ]
        
        for walltime_minutes, expected_format in test_cases:
            with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file), \
                 patch('parslbox.commands.qsub.get_default_run_dir', return_value=temp_run_dir), \
                 patch('subprocess.run') as mock_subprocess:
                
                # Mock successful qsub submission
                mock_result = MagicMock()
                mock_result.stdout = "12345.polaris-pbs-01.alcf.anl.gov"
                mock_subprocess.return_value = mock_result
                
                result = runner.invoke(qsub_app, [
                    "--config", "polaris",
                    "--job-name", "test-job",
                    "--queue", "debug",
                    "--select", "1",
                    "--walltime", str(walltime_minutes),
                    "--project", "test-project"
                ])
                
                assert result.exit_code == 0
                
                # Verify walltime format in submit script
                submit_file = temp_run_dir / "submit.sh"
                content = submit_file.read_text()
                assert f"#PBS -l walltime={expected_format}" in content
    
    def test_qsub_run_options_combinations(self, temp_config_file, temp_run_dir):
        """Test different combinations of run options."""
        runner = CliRunner()
        
        test_cases = [
            # (apps, tags, retries, expected_in_script)
            ("lammps", None, 0, "--apps lammps"),
            (None, "run1", 0, "--tags run1"),
            (None, None, 2, "--retries 2"),
            ("lammps,vasp", "run1,run2", 1, "--apps lammps,vasp --tags run1,run2 --retries 1"),
            (None, None, 0, "")  # No options
        ]
        
        for apps, tags, retries, expected in test_cases:
            with patch('parslbox.commands.qsub.path_utils.PBX_CONFIG_FILE', temp_config_file), \
                 patch('parslbox.commands.qsub.get_default_run_dir', return_value=temp_run_dir), \
                 patch('subprocess.run') as mock_subprocess:
                
                # Mock successful qsub submission
                mock_result = MagicMock()
                mock_result.stdout = "12345.polaris-pbs-01.alcf.anl.gov"
                mock_subprocess.return_value = mock_result
                
                cmd_args = [
                    "--config", "polaris",
                    "--job-name", "test-job",
                    "--queue", "debug",
                    "--select", "1",
                    "--walltime", "30",
                    "--project", "test-project"
                ]
                
                if apps:
                    cmd_args.extend(["--apps", apps])
                if tags:
                    cmd_args.extend(["--tags", tags])
                if retries > 0:
                    cmd_args.extend(["--retries", str(retries)])
                
                result = runner.invoke(qsub_app, cmd_args)
                
                assert result.exit_code == 0
                
                # Verify run options in submit script
                submit_file = temp_run_dir / "submit.sh"
                content = submit_file.read_text()
                if expected:
                    assert expected in content
                else:
                    # Check that parslbox run command has no extra options
                    assert "parslbox run --config polaris" in content
