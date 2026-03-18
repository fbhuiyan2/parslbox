#!/usr/bin/env python3
"""
ParslBox Test Job Creation Script

This script automates the creation of test jobs for lammps-kk, python, and vasp applications.
It creates directory structures, copies necessary files, and adds jobs to pbx with varied
resource configurations and parent dependencies.

Usage:
    python create_test_jobs.py <config_name> [--lammps N] [--python script_name N] [--vasp N]

Examples:
    python create_test_jobs.py polaris --lammps 10 --python hello_affinity.py 5 --vasp 8
    python create_test_jobs.py crux --lammps 5
    python create_test_jobs.py polaris --python test_script.py 3 --vasp 4
"""

import argparse
import shutil
import random
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Import parslbox modules
sys.path.append(str(Path(__file__).parent.parent))
from parslbox.api import ParslBox
from parslbox.utils.pbx_config_utils import load_app_config
from parslbox.system_configs.loader import get_system_config


class TestJobCreator:
    """Main class for creating test jobs."""
    
    def __init__(self, config_name: str, tag: str = None, lmp_exm_dir: str = None):
        """
        Initialize the test job creator.
        
        Args:
            config_name: Name of the system configuration (e.g., 'polaris', 'crux')
            tag: Optional tag to apply to all created jobs
            lmp_exm_dir: Optional custom LAMMPS examples directory path
        """
        self.gpu_enabled = True
        self.cpu_only = False
        self.config_name = config_name
        self.tag = tag or "test"  # Default tag if none provided
        self.lmp_exm_dir = Path(lmp_exm_dir) if lmp_exm_dir else None
        self.system_config = get_system_config(config_name)
        self.tests_dir = Path("tests")
        self.created_jobs = []  # Track created job IDs for parent assignment
        self.PARENTS_MAX = 1
        
        # Initialize ParslBox API
        self.pbx = ParslBox()
        
        # Resource configuration options
        self.gpu_options = [1, 2, 8]
        self.fullnode_options = [1, 2]
        self.cpu_occupancy_options = [1]
        
    def setup_directories(self):
        """Create the main test directory structure."""
        print("🏗️  Setting up directory structure...")
        
        # Create main tests directory
        self.tests_dir.mkdir(exist_ok=True)
        
        # Create app-specific subdirectories
        (self.tests_dir / "lammps").mkdir(exist_ok=True)
        (self.tests_dir / "python").mkdir(exist_ok=True)
        (self.tests_dir / "vasp").mkdir(exist_ok=True)
        
        print(f"✅ Created directory structure in {self.tests_dir.absolute()}")
    
    def get_lammps_examples_path(self) -> Path:
        """
        Get the LAMMPS examples path from custom directory or derive from executable path.
        
        Returns:
            Path to LAMMPS examples directory
            
        Raises:
            ValueError: If lammps is not configured or path cannot be derived
        """
        # If custom LAMMPS examples directory is provided, use it
        if self.lmp_exm_dir:
            if not self.lmp_exm_dir.exists():
                raise ValueError(
                    f"Custom LAMMPS examples directory not found: {self.lmp_exm_dir}"
                )
            return self.lmp_exm_dir
        
        # Otherwise, derive from executable path in config
        app_config = load_app_config("lammps-kk", self.config_name)
        
        if not app_config or "executable_path" not in app_config:
            raise ValueError(
                f"LAMMPS not configured for system '{self.config_name}'. "
                f"Please configure the executable_path in ~/.parslbox/config.yaml"
            )
        
        executable_path = Path(app_config["executable_path"])
        
        # Derive examples path: replace /build/lmp with /examples
        if executable_path.name == "lmp" and "build" in executable_path.parent.name:
            examples_path = executable_path.parent.parent / "examples"
        else:
            # Fallback: assume examples is a sibling of the executable's parent
            examples_path = executable_path.parent / "examples"
        
        if not examples_path.exists():
            raise ValueError(
                f"LAMMPS examples directory not found at {examples_path}. "
                f"Please verify your LAMMPS installation."
            )
        
        return examples_path
    
    def create_lammps_jobs(self, num_jobs: int):
        """
        Create LAMMPS test jobs.
        
        Args:
            num_jobs: Number of LAMMPS jobs to create
        """
        print(f"🧪 Creating {num_jobs} LAMMPS jobs...")
        
        # Get examples path and friction source
        examples_path = self.get_lammps_examples_path()
        friction_src = examples_path / "friction"
        
        if not friction_src.exists():
            raise ValueError(f"Friction example not found at {friction_src}")
        
        lammps_dir = self.tests_dir / "lammps"
        
        # Copy friction example N times
        for i in range(1, num_jobs + 1):
            dest_dir = lammps_dir / f"friction_{i}"
            
            # Copy the friction directory
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(friction_src, dest_dir)
            
            # Remove log files
            for log_file in dest_dir.glob("log*"):
                log_file.unlink()
            
            print(f"  📁 Created {dest_dir}")
        
        # Change to lammps directory and add jobs
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(lammps_dir)
            
            # Add all jobs with varied resources
            job_dirs = [f"friction_{i}" for i in range(1, num_jobs + 1)]
            self._add_lammps_jobs_to_pbx(job_dirs)
            
        finally:
            os.chdir(original_cwd)
    
    def create_python_jobs(self, script_name: str, num_jobs: int):
        """
        Create Python test jobs.
        
        Args:
            script_name: Name of the Python script
            num_jobs: Number of Python jobs to create
        """
        print(f"🐍 Creating {num_jobs} Python jobs for {script_name}...")
        
        script_path = Path(script_name)
        if not script_path.exists():
            raise ValueError(f"Python script not found: {script_name}")
        
        # Derive environment file name
        env_file_name = f"{script_path.stem}_env.sh"
        env_file_path = Path(env_file_name)
        
        if not env_file_path.exists():
            print(f"⚠️  Warning: Environment file {env_file_name} not found in current directory")
            env_file_path = None
        
        python_dir = self.tests_dir / "python"
        
        # Create job directories
        job_dirs = []
        for i in range(1, num_jobs + 1):
            dest_dir = python_dir / f"{script_path.stem}_{i}"
            dest_dir.mkdir(exist_ok=True)
            
            # Copy python script
            shutil.copy2(script_path, dest_dir)
            
            # Copy environment file if it exists
            if env_file_path and env_file_path.exists():
                shutil.copy2(env_file_path, dest_dir)
            
            job_dirs.append(dest_dir.name)
            print(f"  📁 Created {dest_dir}")
        
        # Change to python directory and add jobs
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(python_dir)
            
            # Use absolute path for environment file if it exists
            self._add_python_jobs_to_pbx(job_dirs, script_path.name, env_file_name)
            
        finally:
            os.chdir(original_cwd)
    
    def create_vasp_jobs(self, num_jobs: int):
        """
        Create VASP test jobs.
        
        Args:
            num_jobs: Number of VASP jobs to create
        """
        print(f"⚛️  Creating {num_jobs} VASP jobs...")
        
        # Check for original VASP files
        vasp_original = Path("vasp_original_files")
        if not vasp_original.exists():
            raise ValueError(
                f"VASP original files directory not found: {vasp_original}. "
                f"Please create this directory with VASP input files."
            )
        
        vasp_dir = self.tests_dir / "vasp"
        
        # Create numbered directories and copy files
        job_dirs = []
        for i in range(1, num_jobs + 1):
            dest_dir = vasp_dir / str(i)
            
            # Copy original files
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(vasp_original, dest_dir)
            
            job_dirs.append(str(i))
            print(f"  📁 Created {dest_dir}")
        
        # Change to vasp directory and add jobs
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(vasp_dir)
            
            self._add_vasp_jobs_to_pbx(job_dirs)
            
        finally:
            os.chdir(original_cwd)
    
    def _generate_resource_config(self, job_index: int, total_jobs: int) -> Tuple[dict, str]:
        """
        Generate varied resource configurations for jobs.
        
        Args:
            job_index: Current job index (0-based)
            total_jobs: Total number of jobs
            
        Returns:
            Tuple of (api_kwargs, description)
        """
        # If CPU-only mode is enabled, only create CPU jobs
        if self.cpu_only:
            config_type = job_index % 3  # 3 CPU configuration types
            
            if config_type == 0:
                # Partial node CPU job
                occupancy = random.choice(self.cpu_occupancy_options)
                return {"node_occupancy": occupancy}, f"Partial node CPU, {occupancy} occupancy"
            elif config_type == 1:
                # Full node CPU job
                return {"node_occupancy": 1.0}, "Full node CPU"
            else:
                # Multi-node CPU job
                nnodes = random.choice(self.multinode_options)
                return {"nnodes": nnodes, "node_occupancy": 1.0}, f"Multi-node CPU, {nnodes} nodes"
        
        # If GPU mode is enabled, only create GPU jobs
        elif self.gpu_enabled:
            config_type = job_index % 3  # 3 GPU configuration types
            
            if config_type == 0:
                # Single GPU job
                ngpus = 1
                return {"ngpus": ngpus}, f"Single GPU job, {ngpus} GPU"
            elif config_type == 1:
                # Multi-GPU single node job
                ngpus = random.choice(self.gpu_options)
                if ngpus > self.system_config.GPUS_PER_NODE:
                    ngpus = self.system_config.GPUS_PER_NODE
                return {"ngpus": ngpus}, f"Multi-GPU job, {ngpus} GPUs"
            else:
                # Multi-node GPU job (All GPUs per node)
                nnodes = random.choice(self.fullnode_options)
                return {"nnodes": nnodes}, f"Multi-node GPU, {nnodes} nodes"
        
        # Fallback: if neither GPU nor CPU-only is set properly, default to single GPU
        else:
            return {"ngpus": 1}, "Default single GPU job"
    
    def _generate_parent_dependencies(self, current_job_count: int) -> List[int]:
        """
        Generate random parent dependencies for a job.
        
        Args:
            current_job_count: Number of jobs created so far
            
        Returns:
            List of parent job IDs as integers
        """
        if current_job_count == 0:
            return []
        
        # Randomly decide number of parents (0 to min(PARENTS_MAX, current_job_count))
        max_parents = min(self.PARENTS_MAX, current_job_count)
        num_parents = random.randint(0, max_parents)
        
        if num_parents == 0:
            return []
        
        # Select random parent job IDs from existing jobs
        if not self.created_jobs:
            return []
        
        available_parents = self.created_jobs[-current_job_count:] if current_job_count <= len(self.created_jobs) else self.created_jobs
        if not available_parents:
            return []
        
        selected_parents = random.sample(available_parents, min(num_parents, len(available_parents)))
        return selected_parents
    
    def _add_lammps_jobs_to_pbx(self, job_dirs: List[str]):
        """Add LAMMPS jobs to pbx with varied configurations."""
        print("  🔧 Adding LAMMPS jobs to pbx...")
        
        for i, job_dir in enumerate(job_dirs):
            resource_kwargs, resource_desc = self._generate_resource_config(i, len(job_dirs))
            parent_ids = self._generate_parent_dependencies(len(self.created_jobs))
            
            try:
                # Use ParslBox API to add job
                job_ids, failed_jobs, msg_log = self.pbx.add_jobs(
                    paths=[job_dir],
                    app="lammps-kk",
                    config=self.config_name,
                    input_file="in.friction",
                    tag=self.tag,
                    parents=parent_ids if parent_ids else None,
                    **resource_kwargs
                )
                
                # Display any warnings or info messages
                for warning in msg_log["warnings"]:
                    print(f"    ⚠️  Warning: {warning}")
                for info in msg_log["info"]:
                    print(f"    ℹ️  Info: {info}")
                
                # Check for failures
                if failed_jobs:
                    for path, error in failed_jobs:
                        print(f"    ❌ Failed to add {path}: {error}")
                else:
                    # Success - track the job ID
                    if job_ids:
                        job_id = job_ids[0]  # Should only be one job
                        self.created_jobs.append(job_id)
                        parent_info = f" (parents: {parent_ids})" if parent_ids else ""
                        print(f"    ✅ Added {job_dir} - {resource_desc}, Job ID: {job_id}{parent_info}")
                    else:
                        print(f"    ❌ No job ID returned for {job_dir}")
                        
            except Exception as e:
                print(f"    ❌ Failed to add {job_dir}: {e}")
    
    def _add_python_jobs_to_pbx(self, job_dirs: List[str], script_name: str, env_file: Optional[str]):
        """Add Python jobs to pbx with varied configurations."""
        print("  🔧 Adding Python jobs to pbx...")
        
        for i, job_dir in enumerate(job_dirs):
            resource_kwargs, resource_desc = self._generate_resource_config(i, len(job_dirs))
            parent_ids = self._generate_parent_dependencies(len(self.created_jobs))
            
            # Determine environment file path
            env_file_path = None
            if env_file:
                env_file_path = str((Path(job_dir) / env_file).resolve())
            
            try:
                # Use ParslBox API to add job
                job_ids, failed_jobs, msg_log = self.pbx.add_jobs(
                    paths=[job_dir],
                    app="python",
                    config=self.config_name,
                    input_file=script_name,
                    tag=self.tag,
                    env_file=env_file_path,
                    parents=parent_ids if parent_ids else None,
                    **resource_kwargs
                )
                
                # Display any warnings or info messages
                for warning in msg_log["warnings"]:
                    print(f"    ⚠️  Warning: {warning}")
                for info in msg_log["info"]:
                    print(f"    ℹ️  Info: {info}")
                
                # Check for failures
                if failed_jobs:
                    for path, error in failed_jobs:
                        print(f"    ❌ Failed to add {path}: {error}")
                else:
                    # Success - track the job ID
                    if job_ids:
                        job_id = job_ids[0]  # Should only be one job
                        self.created_jobs.append(job_id)
                        parent_info = f" (parents: {parent_ids})" if parent_ids else ""
                        print(f"    ✅ Added {job_dir} - {resource_desc}, Job ID: {job_id}{parent_info}")
                    else:
                        print(f"    ❌ No job ID returned for {job_dir}")
                        
            except Exception as e:
                print(f"    ❌ Failed to add {job_dir}: {e}")
    
    def _add_vasp_jobs_to_pbx(self, job_dirs: List[str]):
        """Add VASP jobs to pbx with varied configurations."""
        print("  🔧 Adding VASP jobs to pbx...")
        
        for i, job_dir in enumerate(job_dirs):
            resource_kwargs, resource_desc = self._generate_resource_config(i, len(job_dirs))
            parent_ids = self._generate_parent_dependencies(len(self.created_jobs))
            
            try:
                # Use ParslBox API to add job
                job_ids, failed_jobs, msg_log = self.pbx.add_jobs(
                    paths=[job_dir],
                    app="vasp",
                    config=self.config_name,
                    tag=self.tag,
                    parents=parent_ids if parent_ids else None,
                    **resource_kwargs
                )
                
                # Display any warnings or info messages
                for warning in msg_log["warnings"]:
                    print(f"    ⚠️  Warning: {warning}")
                for info in msg_log["info"]:
                    print(f"    ℹ️  Info: {info}")
                
                # Check for failures
                if failed_jobs:
                    for path, error in failed_jobs:
                        print(f"    ❌ Failed to add {path}: {error}")
                else:
                    # Success - track the job ID
                    if job_ids:
                        job_id = job_ids[0]  # Should only be one job
                        self.created_jobs.append(job_id)
                        parent_info = f" (parents: {parent_ids})" if parent_ids else ""
                        print(f"    ✅ Added {job_dir} - {resource_desc}, Job ID: {job_id}{parent_info}")
                    else:
                        print(f"    ❌ No job ID returned for {job_dir}")
                        
            except Exception as e:
                print(f"    ❌ Failed to add {job_dir}: {e}")


def main():
    """Main function to parse arguments and create test jobs."""
    parser = argparse.ArgumentParser(
        description="Create test jobs for ParslBox applications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s polaris --lammps 10 --python hello_affinity.py 5 --vasp 8
  %(prog)s crux --lammps 5
  %(prog)s polaris --python test_script.py 3 --vasp 4
        """
    )
    
    parser.add_argument(
        "config_name",
        help="System configuration name (e.g., 'polaris', 'crux', 'sophia')"
    )
    
    parser.add_argument(
        "--lammps",
        type=int,
        metavar="N",
        help="Create N LAMMPS friction jobs"
    )
    
    parser.add_argument(
        "--python",
        nargs=2,
        metavar=("SCRIPT", "N"),
        help="Create N Python jobs using SCRIPT"
    )
    
    parser.add_argument(
        "--vasp",
        type=int,
        metavar="N",
        help="Create N VASP jobs"
    )
    
    parser.add_argument(
        "--tag",
        type=str,
        metavar="TAG",
        help="Tag to apply to all created jobs (default: 'test_jobs')"
    )
    
    parser.add_argument(
        "--lmp-exm-dir",
        type=str,
        metavar="DIR",
        help="Custom LAMMPS examples directory path (overrides auto-detection)"
    )
    
    args = parser.parse_args()
    
    # Validate that at least one job type is specified
    if not any([args.lammps, args.python, args.vasp]):
        parser.error("At least one job type must be specified (--lammps, --python, or --vasp)")
    
    try:
        # Create the test job creator
        creator = TestJobCreator(args.config_name, args.tag, args.lmp_exm_dir)
        
        # Setup directory structure
        creator.setup_directories()
        
        # Create jobs based on arguments
        if args.lammps:
            creator.create_lammps_jobs(args.lammps)
        
        if args.python:
            script_name, num_jobs_str = args.python
            num_jobs = int(num_jobs_str)
            creator.create_python_jobs(script_name, num_jobs)
        
        if args.vasp:
            creator.create_vasp_jobs(args.vasp)
        
        print(f"\n🎉 Successfully created test jobs! Total jobs created: {len(creator.created_jobs)}")
        print(f"📁 Test directory: {creator.tests_dir.absolute()}")
        print("\n💡 Next steps:")
        print("   - Review the created jobs with: pbx ls")
        print("   - Submit jobs with: pbx qsub")
        print("   - Monitor progress with: pbx ls --status Running")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
