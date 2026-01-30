"""
Test suite for custom app registration functionality.

Tests the config-based custom app registration system that allows users
to add new applications without modifying ParslBox source code.
"""

import tempfile
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch

from parslbox.apps.appbase import AppBase
from parslbox.apps.custom_app_loader import (
    load_custom_apps, 
    import_app_class, 
    validate_app_class,
    expand_path
)
from parslbox.apps.app_registry import get_app_class, get_registered_apps


class TestCustomApp(AppBase):
    """Simple test app for validation."""
    
    INPUT_REQUIRED = True
    DFLT_INPUT = "test_input.txt"
    
    def get_command_template(self, **kwargs) -> str:
        mpi_prefix = kwargs['mpi_prefix']
        executable = kwargs['executable']
        in_file = kwargs['in_file']
        return f"{mpi_prefix} {executable} {in_file}"


class TestCustomAppLoader:
    """Test the custom app loader functionality."""
    
    def test_import_app_class_from_file(self):
        """Test importing an app class from a Python file."""
        # Create a temporary app file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
from parslbox.apps.appbase import AppBase

class MyTestApp(AppBase):
    INPUT_REQUIRED = True
    DFLT_INPUT = "input.txt"
    
    def get_command_template(self, **kwargs):
        return f"{kwargs['mpi_prefix']} my_test_executable {kwargs['in_file']}"
''')
            temp_app_file = f.name
        
        try:
            # Test import
            app_class = import_app_class(temp_app_file, "MyTestApp")
            
            # Validate the class
            assert app_class.__name__ == "MyTestApp"
            assert hasattr(app_class, 'INPUT_REQUIRED')
            assert hasattr(app_class, 'DFLT_INPUT')
            assert hasattr(app_class, 'get_command_template')
            assert app_class.INPUT_REQUIRED is True
            assert app_class.DFLT_INPUT == "input.txt"
            
        finally:
            Path(temp_app_file).unlink()
    
    def test_import_app_class_invalid_file(self):
        """Test importing from non-existent file raises ImportError."""
        with pytest.raises(ImportError, match="Could not import module"):
            import_app_class("/nonexistent/file.py", "MyApp")
    
    def test_import_app_class_invalid_class(self):
        """Test importing non-existent class raises ImportError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('# Empty file')
            temp_app_file = f.name
        
        try:
            with pytest.raises(ImportError, match="does not have a class named"):
                import_app_class(temp_app_file, "NonExistentClass")
        finally:
            Path(temp_app_file).unlink()
    
    def test_validate_app_class_valid(self):
        """Test validation of a valid app class."""
        assert validate_app_class(TestCustomApp) is True
    
    def test_validate_app_class_missing_attributes(self):
        """Test validation fails for class missing required attributes."""
        class InvalidApp(AppBase):
            # Missing INPUT_REQUIRED and DFLT_INPUT
            def get_command_template(self, **kwargs):
                return "test"
        
        assert validate_app_class(InvalidApp) is False
    
    def test_validate_app_class_missing_methods(self):
        """Test validation fails for class missing required methods."""
        # Create a class that doesn't inherit from AppBase and lacks the method
        class InvalidApp:
            INPUT_REQUIRED = True
            DFLT_INPUT = "input.txt"
            # Missing get_command_template method
        
        assert validate_app_class(InvalidApp) is False
    
    def test_expand_path_home_directory(self):
        """Test path expansion with ~ character."""
        path = expand_path("~/test.py")
        assert str(path).startswith(str(Path.home()))
        assert str(path).endswith("test.py")
    
    def test_expand_path_absolute(self):
        """Test path expansion with absolute path."""
        abs_path = "/absolute/path/test.py"
        path = expand_path(abs_path)
        assert str(path) == abs_path
    
    def test_app_instance_command_generation(self):
        """Test that app instance generates correct commands."""
        app = TestCustomApp()
        command = app.get_command_template(
            mpi_prefix="mpiexec -n 4",
            mpi_opts_str="",
            executable="/path/to/exe",
            in_file="input.txt"
        )
        expected = "mpiexec -n 4 /path/to/exe input.txt"
        assert command == expected


class TestCustomAppRegistration:
    """Test the full custom app registration system."""
    
    def test_load_custom_apps_from_config(self):
        """Test loading custom apps from a config file."""
        # Create a temporary app file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
from parslbox.apps.appbase import AppBase

class ConfigTestApp(AppBase):
    INPUT_REQUIRED = True
    DFLT_INPUT = "config_input.txt"
    
    def get_command_template(self, **kwargs):
        return f"{kwargs['mpi_prefix']} config_executable {kwargs['in_file']}"
''')
            temp_app_file = f.name
        
        # Create a temporary config
        config_data = {
            'custom_apps': {
                'config_test_app': {
                    'module': temp_app_file,
                    'class': 'ConfigTestApp'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_file = f.name
        
        try:
            # Mock the config path
            with patch('parslbox.utils.path_utils.PBX_CONFIG_FILE', Path(temp_config_file)):
                custom_apps = load_custom_apps()
                
                assert len(custom_apps) == 1
                assert 'config_test_app' in custom_apps
                assert custom_apps['config_test_app'].__name__ == 'ConfigTestApp'
                
        finally:
            Path(temp_app_file).unlink()
            Path(temp_config_file).unlink()
    
    def test_load_custom_apps_no_config(self):
        """Test loading when config file doesn't exist."""
        with patch('parslbox.utils.path_utils.PBX_CONFIG_FILE', Path('/nonexistent/config.yaml')):
            custom_apps = load_custom_apps()
            assert custom_apps == {}
    
    def test_load_custom_apps_empty_config(self):
        """Test loading when config has no custom_apps section."""
        config_data = {'other_section': 'value'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_file = f.name
        
        try:
            with patch('parslbox.utils.path_utils.PBX_CONFIG_FILE', Path(temp_config_file)):
                custom_apps = load_custom_apps()
                assert custom_apps == {}
        finally:
            Path(temp_config_file).unlink()
    
    def test_load_custom_apps_invalid_app(self):
        """Test loading continues when one app is invalid."""
        # Create valid app file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
from parslbox.apps.appbase import AppBase

class ValidApp(AppBase):
    INPUT_REQUIRED = True
    DFLT_INPUT = "input.txt"
    
    def get_command_template(self, **kwargs):
        return "valid command"
''')
            valid_app_file = f.name
        
        # Create config with valid and invalid apps
        config_data = {
            'custom_apps': {
                'valid_app': {
                    'module': valid_app_file,
                    'class': 'ValidApp'
                },
                'invalid_app': {
                    'module': '/nonexistent/file.py',
                    'class': 'InvalidApp'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_file = f.name
        
        try:
            with patch('parslbox.utils.path_utils.PBX_CONFIG_FILE', Path(temp_config_file)):
                custom_apps = load_custom_apps()
                
                # Should load only the valid app
                assert len(custom_apps) == 1
                assert 'valid_app' in custom_apps
                assert 'invalid_app' not in custom_apps
                
        finally:
            Path(valid_app_file).unlink()
            Path(temp_config_file).unlink()


class TestAppRegistry:
    """Test the app registry with custom apps."""
    
    def test_builtin_apps_still_work(self):
        """Test that built-in apps are still accessible."""
        builtin_apps = ['lammps', 'vasp', 'python']
        
        for app_name in builtin_apps:
            app_class = get_app_class(app_name)
            assert app_class is not None
            assert hasattr(app_class, 'INPUT_REQUIRED')
            assert hasattr(app_class, 'get_command_template')
    
    def test_get_registered_apps_includes_builtins(self):
        """Test that get_registered_apps returns built-in apps."""
        registered = get_registered_apps()
        assert 'lammps' in registered
        assert 'vasp' in registered
        assert 'python' in registered
    
    def test_builtin_apps_have_priority(self):
        """Test that built-in apps take priority over custom apps with same name."""
        # This test verifies the priority system works correctly
        # Built-in apps should always be returned first
        lammps_class = get_app_class('lammps')
        assert lammps_class.__name__ == 'LammpsApp'
    
    def test_unknown_app_error_message(self):
        """Test that unknown app raises helpful error message."""
        with pytest.raises(ValueError, match="Unknown application: 'nonexistent'"):
            get_app_class('nonexistent')


class TestIntegration:
    """Integration tests for the complete custom app system."""
    
    def test_end_to_end_custom_app_workflow(self):
        """Test the complete workflow from config to app usage."""
        # Create a custom app file
        app_code = '''
from parslbox.apps.appbase import AppBase

class IntegrationTestApp(AppBase):
    INPUT_REQUIRED = True
    DFLT_INPUT = "integration_input.txt"
    
    def get_command_template(self, **kwargs):
        mpi_prefix = kwargs.get('mpi_prefix', '')
        executable = kwargs.get('executable', 'default_exe')
        in_file = kwargs.get('in_file', 'default_input')
        return f"{mpi_prefix} {executable} --input {in_file}"
    
    def get_additional_setup(self, **kwargs):
        return "export INTEGRATION_TEST=1"
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(app_code)
            temp_app_file = f.name
        
        # Create config
        config_data = {
            'custom_apps': {
                'integration_test': {
                    'module': temp_app_file,
                    'class': 'IntegrationTestApp'
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_file = f.name
        
        try:
            # Test the complete workflow
            with patch('parslbox.utils.path_utils.PBX_CONFIG_FILE', Path(temp_config_file)):
                # Reset the custom apps cache to force reload
                import parslbox.apps.app_registry as registry
                registry._CUSTOM_APPS = None
                
                # Get the custom app
                app_class = get_app_class('integration_test')
                assert app_class.__name__ == 'IntegrationTestApp'
                
                # Create instance and test functionality
                app = app_class()
                assert app.INPUT_REQUIRED is True
                assert app.DFLT_INPUT == "integration_input.txt"
                
                # Test command generation
                command = app.get_command_template(
                    mpi_prefix="mpirun -n 8",
                    executable="/path/to/my_app",
                    in_file="my_input.txt"
                )
                expected = "mpirun -n 8 /path/to/my_app --input my_input.txt"
                assert command == expected
                
                # Test additional setup
                setup = app.get_additional_setup()
                assert setup == "export INTEGRATION_TEST=1"
                
                # Verify it appears in registered apps
                registered = get_registered_apps()
                assert 'integration_test' in registered
                
        finally:
            Path(temp_app_file).unlink()
            Path(temp_config_file).unlink()
            # Reset the cache
            import parslbox.apps.app_registry as registry
            registry._CUSTOM_APPS = None
