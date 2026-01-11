#!/usr/bin/env python3
"""
LAMMPS Strong Scaling Orchestrator for ParslBox

This script automates the creation of LAMMPS strong scaling test jobs.
It creates directory structures for different GPU counts, copies necessary files,
and adds jobs to pbx with proper dependencies for strong scaling analysis.

Usage:
    python lammps_strong_scale_orchestrator.py <config_name> --gpus 1 2 4 --fstruct model.lmp --ff forcefield.dat [--in in.lammps]

Examples:
    python lammps_strong_scale_orchestrator.py polaris --gpus 1 2 4 --fstruct data.lmp --ff pair_coeff.dat
    python lammps_strong_scale_orchestrator.py crux --gpus 1 2 --fstruct model.lmp --ff forcefield.dat --in in.friction
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# Import parslbox modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from parslbox.api import ParslBox
from parslbox.utils.pbx_config_utils import load_app_config
from parslbox.system_configs.loader import get_system_config


class LammpsStrongScaleOrchestrator:
    """Main class for creating LAMMPS strong scaling test jobs."""
    
    def __init__(self, config_name: str, tag: str = "strong_scale"):
        """
        Initialize the strong scaling orchestrator.
        
        Args:
            config_name: Name of the system configuration (e.g., 'polaris', 'crux')
            tag: Tag to apply to all created jobs
        """
        self.config_name = config_name
        self.tag = tag
        self.system_config = get_system_config(config_name)
        self.created_jobs = []  # Track created job IDs for parent assignment
        self.base_dir = Path.cwd()
        self.pbx = ParslBox()  # Initialize ParslBox API
        
        # Validate LAMMPS is configured
        self._validate_lammps_config()
    
    def _validate_lammps_config(self):
        """Validate that LAMMPS is properly configured for this system."""
        app_config = load_app_config("lammps", self.config_name)
        
        if not app_config or "executable_path" not in app_config:
            raise ValueError(
                f"LAMMPS not configured for system '{self.config_name}'. "
                f"Please configure the executable_path in ~/.parslbox/config.yaml"
            )
    
    def setup_directories(self, gpu_counts: List[int]):
        """
        Create GPU-specific directories for strong scaling tests.
        
        Args:
            gpu_counts: List of GPU counts to create directories for
        """
        print("🏗️  Setting up directory structure...")
        
        for gpu_count in gpu_counts:
            gpu_dir = self.base_dir / f"{gpu_count}gpu"
            gpu_dir.mkdir(exist_ok=True)
            print(f"  📁 Created {gpu_dir}")
    
    def copy_input_files(self, gpu_counts: List[int], input_file: str, 
                        struct_file: str, ff_file: str):
        """
        Copy input files to each GPU directory.
        
        Args:
            gpu_counts: List of GPU counts
            input_file: LAMMPS input file name
            struct_file: Structure/data file name
            ff_file: Force field file name
        """
        print("📋 Copying input files to GPU directories...")
        
        # Validate input files exist
        files_to_copy = [input_file, struct_file, ff_file]
        for file_name in files_to_copy:
            file_path = self.base_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_name}")
        
        # Validate analysis scripts exist in current directory
        analysis_script = self.base_dir / "plot_strong-scale-results.py"
        env_script = self.base_dir / "python_env-setup.sh"
        
        if not analysis_script.exists():
            raise FileNotFoundError(
                "Analysis script 'plot_strong-scale-results.py' not found in current directory. "
                "Please ensure both analysis scripts are present."
            )
        
        if not env_script.exists():
            raise FileNotFoundError(
                "Environment script 'python_env-setup.sh' not found in current directory. "
                "Please ensure both analysis scripts are present."
            )
        
        # Copy files to each GPU directory
        for gpu_count in gpu_counts:
            gpu_dir = self.base_dir / f"{gpu_count}gpu"
            
            for file_name in files_to_copy:
                src_file = self.base_dir / file_name
                dst_file = gpu_dir / file_name
                shutil.copy2(src_file, dst_file)
            
            print(f"  ✅ Copied files to {gpu_dir}")
    
    def create_lammps_jobs(self, gpu_counts: List[int], input_file: str):
        """
        Create LAMMPS jobs with different GPU counts and proper dependencies.
        
        Args:
            gpu_counts: List of GPU counts for scaling test
            input_file: LAMMPS input file name
        """
        print("🧪 Creating LAMMPS strong scaling jobs...")
        
        parent_job_ids = None
        
        for i, gpu_count in enumerate(gpu_counts):
            job_dir = f"{gpu_count}gpu"
            
            try:
                # Use ParslBox API to add job
                job_ids, failed_jobs, msg_log = self.pbx.add_jobs(
                    paths=[job_dir],
                    app="lammps",
                    config=self.config_name,
                    input_file=input_file,
                    ngpus=gpu_count,
                    tag=self.tag,
                    parents=parent_job_ids
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
                    raise RuntimeError(f"Failed to add job for {job_dir}")
                
                # Success
                if job_ids:
                    job_id = job_ids[0]  # Should only be one job
                    self.created_jobs.append(job_id)
                    parent_job_ids = [job_id]  # This job becomes parent for next job
                    parent_info = f" (parent: {parent_job_ids[0]})" if i > 0 else ""
                    print(f"    ✅ Added {job_dir} - {gpu_count} GPUs, Job ID: {job_id}{parent_info}")
                else:
                    raise RuntimeError(f"No job ID returned for {job_dir}")
                    
            except Exception as e:
                print(f"    ❌ Failed to add {job_dir}: {e}")
                raise
        
        return parent_job_ids[0] if parent_job_ids else None  # Return the last job ID for analysis job dependency
    
    def create_analysis_job(self, last_job_id: Optional[int]):
        """
        Create Python analysis job to process results.
        
        Args:
            last_job_id: Job ID of the last LAMMPS job to use as parent
        """
        print("📊 Creating analysis job...")
        
        try:
            # Use ParslBox API to add analysis job
            job_ids, failed_jobs, msg_log = self.pbx.add_jobs(
                paths=["."],  # Run in current directory
                app="python",
                config=self.config_name,
                input_file="plot_strong-scale-results.py",
                env_file=str(self.base_dir / "python_env-setup.sh"),
                tag=self.tag,
                node_occupancy=0.1,  # Light CPU usage for analysis
                parents=[last_job_id] if last_job_id is not None else None
            )
            
            # Display any warnings or info messages
            for warning in msg_log["warnings"]:
                print(f"    ⚠️  Warning: {warning}")
            for info in msg_log["info"]:
                print(f"    ℹ️  Info: {info}")
            
            # Check for failures
            if failed_jobs:
                for path, error in failed_jobs:
                    print(f"    ❌ Failed to add analysis job at {path}: {error}")
                raise RuntimeError("Failed to add analysis job")
            
            # Success
            if job_ids:
                job_id = job_ids[0]  # Should only be one job
                parent_info = f" (parent: {last_job_id})" if last_job_id else ""
                print(f"    ✅ Added analysis job, Job ID: {job_id}{parent_info}")
            else:
                raise RuntimeError("No job ID returned for analysis job")
                
        except Exception as e:
            print(f"    ❌ Failed to add analysis job: {e}")
            raise
    
    def orchestrate(self, gpu_counts: List[int], input_file: str, 
                   struct_file: str, ff_file: str):
        """
        Main orchestration method to set up and create all jobs.
        
        Args:
            gpu_counts: List of GPU counts for scaling test
            input_file: LAMMPS input file name
            struct_file: Structure/data file name
            ff_file: Force field file name
        """
        print(f"🚀 Starting LAMMPS strong scaling orchestration for {self.config_name}")
        print(f"   GPU counts: {gpu_counts}")
        print(f"   Input file: {input_file}")
        print(f"   Structure file: {struct_file}")
        print(f"   Force field file: {ff_file}")
        print()
        
        # Setup directories
        self.setup_directories(gpu_counts)
        
        # Copy input files
        self.copy_input_files(gpu_counts, input_file, struct_file, ff_file)
        
        # Create LAMMPS jobs with dependencies
        last_job_id = self.create_lammps_jobs(gpu_counts, input_file)
        
        # Create analysis job
        self.create_analysis_job(last_job_id)
        
        print(f"\n🎉 Successfully created strong scaling test!")
        print(f"📁 Working directory: {self.base_dir.absolute()}")
        print(f"🏷️  Job tag: {self.tag}")
        print("\n💡 Next steps:")
        print("   - Review the created jobs with: pbx ls")
        print("   - Submit jobs with: pbx qsub")
        print("   - Monitor progress with: pbx ls --status Running")
        print("   - Analysis results will be generated automatically after all jobs complete")


def main():
    """Main function to parse arguments and orchestrate strong scaling tests."""
    parser = argparse.ArgumentParser(
        description="Create LAMMPS strong scaling test jobs for ParslBox",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s polaris --gpus 1 2 4 --fstruct data.lmp --ff pair_coeff.dat
  %(prog)s crux --gpus 1 2 --fstruct model.lmp --ff forcefield.dat --in in.friction
  %(prog)s polaris --gpus 1 2 4 8 --fstruct data.lmp --ff pair_coeff.dat --in in.lammps
        """
    )
    
    parser.add_argument(
        "config_name",
        help="System configuration name (e.g., 'polaris', 'crux', 'sophia')"
    )
    
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        required=True,
        metavar="N",
        help="GPU counts for strong scaling test (e.g., --gpus 1 2 4)"
    )
    
    parser.add_argument(
        "--fstruct",
        required=True,
        metavar="FILE",
        help="Structure/data file name (e.g., data.lmp, model.lmp)"
    )
    
    parser.add_argument(
        "--ff",
        required=True,
        metavar="FILE",
        help="Force field file name (e.g., pair_coeff.dat, forcefield.dat)"
    )
    
    parser.add_argument(
        "--in",
        dest="input_file",
        default="in.lammps",
        metavar="FILE",
        help="LAMMPS input file name (default: in.lammps)"
    )
    
    parser.add_argument(
        "--tag",
        default="strong_scale",
        metavar="TAG",
        help="Tag to apply to all created jobs (default: 'strong_scale')"
    )
    
    args = parser.parse_args()
    
    # Validate GPU counts
    if len(args.gpus) < 2:
        parser.error("At least 2 GPU counts are required for strong scaling analysis")
    
    if any(gpu_count <= 0 for gpu_count in args.gpus):
        parser.error("All GPU counts must be positive integers")
    
    # Sort GPU counts to ensure proper dependency order
    gpu_counts = sorted(args.gpus)
    
    try:
        # Create the orchestrator
        orchestrator = LammpsStrongScaleOrchestrator(args.config_name, args.tag)
        
        # Run the orchestration
        orchestrator.orchestrate(
            gpu_counts=gpu_counts,
            input_file=args.input_file,
            struct_file=args.fstruct,
            ff_file=args.ff
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
