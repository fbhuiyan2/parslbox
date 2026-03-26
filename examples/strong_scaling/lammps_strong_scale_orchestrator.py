#!/usr/bin/env python3
"""
LAMMPS Strong Scaling Orchestrator for ParslBox

This script automates the creation of LAMMPS strong scaling test jobs.
It creates directory structures for different GPU/core counts, copies
necessary files, and adds jobs to pbx with proper dependencies for
strong scaling analysis.

Usage:
    python lammps_strong_scale_orchestrator.py <config_name> (--gpus | --cores) N [N ...] --fstruct FILE --ff FILE [options]

Arguments:
    config_name              System configuration name (e.g., polaris, crux, sophia)

Required (mutually exclusive):
    --gpus N [N ...]         GPU counts for scaling test (e.g., --gpus 1 2 4)
    --cores N [N ...]        Core counts for scaling test (e.g., --cores 1 8 64)

Required:
    --fstruct FILE           Structure/data file name (e.g., data.lmp, model.lmp)
    --ff FILE                Force field file name (e.g., pair_coeff.dat, forcefield.dat)

Optional:
    --in FILE                LAMMPS input file name (default: in.lammps)
    --rankspercore N         Ranks per core for CPU scaling (default: 1)
    --tag TAG                Tag to apply to all created jobs (default: strong_scale)

Examples:
    # GPU scaling on Polaris
    python lammps_strong_scale_orchestrator.py polaris --gpus 1 2 4 --fstruct data.lmp --ff pair_coeff.dat

    # GPU scaling with custom input file
    python lammps_strong_scale_orchestrator.py polaris --gpus 1 2 4 8 --fstruct data.lmp --ff pair_coeff.dat --in in.lammps

    # CPU scaling on Sophia
    python lammps_strong_scale_orchestrator.py sophia --cores 1 8 64 128 --fstruct data.lmp --ff pair_coeff.dat

    # CPU scaling with oversubscription
    python lammps_strong_scale_orchestrator.py sophia --cores 1 8 64 --fstruct data.lmp --ff forcefield.dat --rankspercore 2

    # Custom job tag
    python lammps_strong_scale_orchestrator.py crux --gpus 1 2 --fstruct model.lmp --ff forcefield.dat --tag my_test
"""

import argparse
import math
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
    
    def __init__(self, config_name: str, scaling_mode: str, tag: str = "strong_scale"):
        """
        Initialize the strong scaling orchestrator.
        
        Args:
            config_name: Name of the system configuration (e.g., 'polaris', 'crux')
            scaling_mode: Either 'gpu' or 'core' to indicate scaling dimension
            tag: Tag to apply to all created jobs
        """
        self.config_name = config_name
        self.scaling_mode = scaling_mode
        self.tag = tag
        self.system_config = get_system_config(config_name)
        self.created_jobs = []  # Track created job IDs for parent assignment
        self.base_dir = Path.cwd()
        self.pbx = ParslBox()  # Initialize ParslBox API
        
        # Validate LAMMPS is configured
        self._validate_lammps_config()
    
    def _validate_lammps_config(self):
        """Validate that LAMMPS is properly configured for this system."""
        app_config = load_app_config("lammps-kk", self.config_name)
        
        if not app_config or "executable_path" not in app_config:
            raise ValueError(
                f"LAMMPS not configured for system '{self.config_name}'. "
                f"Please configure the executable_path in ~/.parslbox/config.yaml"
            )
    
    def _get_units_per_node(self) -> int:
        """Get the number of GPUs or cores per node for this system."""
        if self.scaling_mode == 'gpu':
            return self.system_config.GPUS_PER_NODE
        else:  # core mode
            return self.system_config.CORES_PER_NODE
    
    def _validate_scale_counts(self, scale_counts: List[int]) -> None:
        """
        Validate that scale counts are compatible with multi-node requirements.
        
        For multi-node jobs, ParslBox automatically uses all GPUs/cores on allocated nodes.
        Therefore, counts must be multiples of units_per_node for multi-node cases.
        
        Args:
            scale_counts: List of GPU/core counts to validate
            
        Raises:
            ValueError: If any count is incompatible with multi-node behavior
        """
        units_per_node = self._get_units_per_node()
        unit_name = "GPU" if self.scaling_mode == 'gpu' else "core"
        
        for count in scale_counts:
            if count > units_per_node:
                # Multi-node case: must be multiple of units_per_node
                if count % units_per_node != 0:
                    raise ValueError(
                        f"{unit_name} count {count} is not compatible with multi-node jobs. "
                        f"For {self.config_name} with {units_per_node} {unit_name}s per node, "
                        f"multi-node jobs must use multiples of {units_per_node} {unit_name}s "
                        f"(e.g., {units_per_node}, {units_per_node*2}, {units_per_node*3}, etc.)."
                    )
    
    def setup_directories(self, scale_counts: List[int]):
        """
        Create scale-specific directories for strong scaling tests.
        
        Args:
            scale_counts: List of GPU/core counts to create directories for
        """
        print("🏗️  Setting up directory structure...")
        
        unit_suffix = "gpu" if self.scaling_mode == 'gpu' else "core"
        
        for count in scale_counts:
            dir_name = self.base_dir / f"{count}{unit_suffix}"
            dir_name.mkdir(exist_ok=True)
            print(f"  📁 Created {dir_name}")
    
    def copy_input_files(self, scale_counts: List[int], input_file: str, 
                        struct_file: str, ff_file: str):
        """
        Copy input files to each scale directory.
        
        Args:
            scale_counts: List of GPU/core counts
            input_file: LAMMPS input file name
            struct_file: Structure/data file name
            ff_file: Force field file name
        """
        print("📋 Copying input files to directories...")
        
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
        
        # Copy files to each directory
        unit_suffix = "gpu" if self.scaling_mode == 'gpu' else "core"
        
        for count in scale_counts:
            dir_name = self.base_dir / f"{count}{unit_suffix}"
            
            for file_name in files_to_copy:
                src_file = self.base_dir / file_name
                dst_file = dir_name / file_name
                shutil.copy2(src_file, dst_file)
            
            print(f"  ✅ Copied files to {dir_name}")
    
    def create_lammps_jobs(self, scale_counts: List[int], input_file: str, rankspercore: int = 1):
        """
        Create LAMMPS jobs with different scale counts and proper dependencies.
        
        Args:
            scale_counts: List of GPU/core counts for scaling test
            input_file: LAMMPS input file name
            rankspercore: Ranks per core (for CPU scaling)
        """
        print("🧪 Creating LAMMPS strong scaling jobs...")
        
        parent_job_ids = None
        units_per_node = self._get_units_per_node()
        unit_suffix = "gpu" if self.scaling_mode == 'gpu' else "core"
        
        for i, count in enumerate(scale_counts):
            job_dir = f"{count}{unit_suffix}"
            
            # Calculate required nodes for this count
            required_nodes = math.ceil(count / units_per_node)
            
            # Determine job type for logging
            if required_nodes > 1:
                job_type = f"multi-node ({required_nodes} nodes)"
            else:
                job_type = "single-node"
            
            try:
                # Prepare job parameters based on scaling mode
                if self.scaling_mode == 'gpu':
                    job_params = {
                        'paths': [job_dir],
                        'app': 'lammps-kk',
                        'config': self.config_name,
                        'input_file': input_file,
                        'ngpus': count,
                        'nnodes': required_nodes,
                        'tag': self.tag
                    }
                else:  # core mode
                    job_params = {
                        'paths': [job_dir],
                        'app': 'lammps-kk',
                        'config': self.config_name,
                        'input_file': input_file,
                        'ncores': count,
                        'nnodes': required_nodes,
                        'rankspercore': rankspercore,
                        'tag': self.tag
                    }
                
                # Use ParslBox API to add job
                job_ids, failed_jobs, msg_log = self.pbx.add_jobs(**job_params)
                
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
                    print(f"    ✅ Added {job_dir} - {count} {unit_suffix}s, {job_type}, Job ID: {job_id}{parent_info}")
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
    
    def orchestrate(self, scale_counts: List[int], input_file: str, 
                   struct_file: str, ff_file: str, rankspercore: int = 1):
        """
        Main orchestration method to set up and create all jobs.
        
        Args:
            scale_counts: List of GPU/core counts for scaling test
            input_file: LAMMPS input file name
            struct_file: Structure/data file name
            ff_file: Force field file name
            rankspercore: Ranks per core (for CPU scaling)
        """
        unit_name = "GPU" if self.scaling_mode == 'gpu' else "core"
        
        print(f"🚀 Starting LAMMPS strong scaling orchestration for {self.config_name}")
        print(f"   Scaling mode: {unit_name}")
        print(f"   {unit_name} counts: {scale_counts}")
        print(f"   Input file: {input_file}")
        print(f"   Structure file: {struct_file}")
        print(f"   Force field file: {ff_file}")
        if self.scaling_mode == 'core':
            print(f"   Ranks per core: {rankspercore}")
        print()
        
        # Validate scale counts are compatible with multi-node requirements
        self._validate_scale_counts(scale_counts)
        
        # Setup directories
        self.setup_directories(scale_counts)
        
        # Copy input files
        self.copy_input_files(scale_counts, input_file, struct_file, ff_file)
        
        # Create LAMMPS jobs with dependencies
        last_job_id = self.create_lammps_jobs(scale_counts, input_file, rankspercore)
        
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
    )
    
    parser.add_argument(
        "config_name",
        help="System configuration name (e.g., 'polaris', 'crux', 'sophia')"
    )
    
    # Mutually exclusive group for GPU or core scaling
    scaling_group = parser.add_mutually_exclusive_group(required=True)
    scaling_group.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        metavar="N",
        help="GPU counts for strong scaling test (e.g., --gpus 1 2 4)"
    )
    scaling_group.add_argument(
        "--cores",
        nargs="+",
        type=int,
        metavar="N",
        help="Core counts for strong scaling test (e.g., --cores 1 8 64)"
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
        "--rankspercore",
        type=int,
        default=1,
        metavar="N",
        help="Ranks per core for CPU scaling (default: 1)"
    )
    
    parser.add_argument(
        "--tag",
        default="strong_scale",
        metavar="TAG",
        help="Tag to apply to all created jobs (default: 'strong_scale')"
    )
    
    args = parser.parse_args()
    
    # Determine scaling mode and get scale counts
    if args.gpus:
        scaling_mode = 'gpu'
        scale_counts = sorted(args.gpus)
    else:
        scaling_mode = 'core'
        scale_counts = sorted(args.cores)
    
    # Validate scale counts
    if len(scale_counts) < 2:
        parser.error("At least 2 scale points are required for strong scaling analysis")
    
    if any(count <= 0 for count in scale_counts):
        parser.error("All scale counts must be positive integers")
    
    try:
        # Create the orchestrator
        orchestrator = LammpsStrongScaleOrchestrator(args.config_name, scaling_mode, args.tag)
        
        # Run the orchestration
        orchestrator.orchestrate(
            scale_counts=scale_counts,
            input_file=args.input_file,
            struct_file=args.fstruct,
            ff_file=args.ff,
            rankspercore=args.rankspercore
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
