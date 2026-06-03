#!/usr/bin/env python3
"""
LAMMPS Weak Scaling Orchestrator for ParslBox

This script automates the creation of LAMMPS weak scaling test jobs.
It creates directory structures for different GPU/core counts, replicates the
atomic structure to maintain constant atoms per compute unit, and adds jobs
to pbx with proper dependencies for weak scaling analysis.

Usage:
    python lammps_weak_scale_orchestrator.py <config_name> (--gpus | --cores) N [N ...] --fstruct FILE --ff FILE [options]

Arguments:
    config_name              System configuration name (e.g., polaris, crux, sophia)

Required (mutually exclusive):
    --gpus N [N ...]         GPU counts for scaling test (e.g., --gpus 1 2 4)
    --cores N [N ...]        Core counts for scaling test (e.g., --cores 1 8 64)

Required:
    --fstruct FILE           Small structure/data file (<100 atoms, <100 A^3)
    --ff FILE                Force field file name (e.g., pair_coeff.dat, forcefield.dat)

Optional:
    --in FILE                LAMMPS input file name (default: in.lammps)
    --natoms-per-unit N      Target atoms per GPU/core (default: 250)
    --rankspercore N         Ranks per core for CPU scaling (default: 1)
    --tag TAG                Tag to apply to all created jobs (default: weak_scale)

Examples:
    # GPU weak scaling on Polaris
    python lammps_weak_scale_orchestrator.py polaris --gpus 1 2 4 --fstruct small.lmp --ff pair_coeff.dat

    # GPU scaling with custom atoms per GPU
    python lammps_weak_scale_orchestrator.py polaris --gpus 1 2 4 8 --fstruct data.lmp --ff forcefield.dat --natoms-per-unit 500

    # CPU weak scaling on Sophia
    python lammps_weak_scale_orchestrator.py sophia --cores 1 8 64 128 --fstruct small.lmp --ff pair_coeff.dat

    # CPU scaling with oversubscription
    python lammps_weak_scale_orchestrator.py sophia --cores 1 8 64 --fstruct data.lmp --ff forcefield.dat --rankspercore 2

    # Custom job tag
    python lammps_weak_scale_orchestrator.py polaris --gpus 1 2 4 --fstruct small.lmp --ff pair_coeff.dat --tag my_weak_test
"""

import argparse
import math
import re
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Dict

# Import parslbox modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from parslbox.api import ParslBox
from parslbox.utils.pbx_config_utils import load_app_config
from parslbox.system_configs.loader import get_system_config


class LammpsWeakScaleOrchestrator:
    """Main class for creating LAMMPS weak scaling test jobs."""
    
    def __init__(self, config_name: str, scaling_mode: str, tag: str = "weak_scale"):
        """
        Initialize the weak scaling orchestrator.
        
        Args:
            config_name: Name of the system configuration (e.g., 'polaris', 'crux')
            scaling_mode: Either 'gpu' or 'core' to indicate scaling dimension
            tag: Tag to apply to all created jobs
        """
        self.config_name = config_name
        self.scaling_mode = scaling_mode
        self.tag = tag
        self.system_config = get_system_config(config_name)
        self.created_jobs = []
        self.base_dir = Path.cwd()
        self.pbx = ParslBox()
        
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
    
    def parse_lammps_data_file(self, file_path: Path) -> Dict[str, any]:
        """
        Parse LAMMPS data file to extract atom count and box information.
        
        Args:
            file_path: Path to LAMMPS data file
            
        Returns:
            Dictionary with 'natoms' and 'box_bounds' keys
            
        Raises:
            ValueError: If file cannot be parsed or is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"LAMMPS data file not found: {file_path}")
        
        natoms = None
        box_bounds = {}
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                
                # Extract atom count
                if 'atoms' in line and natoms is None:
                    parts = line.split()
                    if len(parts) >= 2:
                        natoms = int(parts[0])
                
                # Extract box bounds
                if 'xlo xhi' in line:
                    parts = line.split()
                    box_bounds['x'] = (float(parts[0]), float(parts[1]))
                elif 'ylo yhi' in line:
                    parts = line.split()
                    box_bounds['y'] = (float(parts[0]), float(parts[1]))
                elif 'zlo zhi' in line:
                    parts = line.split()
                    box_bounds['z'] = (float(parts[0]), float(parts[1]))
            
            if natoms is None:
                raise ValueError("Could not find atom count in LAMMPS data file")
            
            if len(box_bounds) != 3:
                raise ValueError("Could not find complete box bounds in LAMMPS data file")
            
            # Calculate box volume
            lx = box_bounds['x'][1] - box_bounds['x'][0]
            ly = box_bounds['y'][1] - box_bounds['y'][0]
            lz = box_bounds['z'][1] - box_bounds['z'][0]
            volume = lx * ly * lz
            
            return {
                'natoms': natoms,
                'box_bounds': box_bounds,
                'box_volume': volume,
                'box_dimensions': (lx, ly, lz)
            }
            
        except Exception as e:
            raise ValueError(f"Error parsing LAMMPS data file: {e}")
    
    def calculate_replication_factors(self, initial_atoms: int, target_atoms: int) -> Tuple[int, int, int]:
        """
        Calculate optimal replication factors (nx, ny, nz) to get closest to target atoms.
        
        Uses non-cubic replication to minimize difference from target atom count.
        
        Args:
            initial_atoms: Number of atoms in original structure
            target_atoms: Target number of atoms
            
        Returns:
            Tuple of (nx, ny, nz) replication factors
        """
        if target_atoms <= initial_atoms:
            return (1, 1, 1)
        
        scale_factor = target_atoms / initial_atoms
        
        # Start with cubic replication as baseline
        base = round(scale_factor ** (1/3))
        
        # Try different combinations to find closest match
        best_factors = (base, base, base)
        best_diff = abs(initial_atoms * base**3 - target_atoms)
        
        # Search range around base value
        search_range = max(1, base // 2)
        for nx in range(max(1, base - search_range), base + search_range + 1):
            for ny in range(max(1, base - search_range), base + search_range + 1):
                for nz in range(max(1, base - search_range), base + search_range + 1):
                    actual_atoms = initial_atoms * nx * ny * nz
                    diff = abs(actual_atoms - target_atoms)
                    
                    if diff < best_diff:
                        best_diff = diff
                        best_factors = (nx, ny, nz)
        
        return best_factors
    
    def create_modified_input_file(self, original_input: Path, replicate_factors: Tuple[int, int, int],
                                   output_path: Path) -> None:
        """
        Create modified LAMMPS input file with replicate command inserted.
        
        Args:
            original_input: Path to original input file
            replicate_factors: Tuple of (nx, ny, nz) for replication
            output_path: Path where modified input file should be written
        """
        with open(original_input, 'r') as f:
            lines = f.readlines()
        
        # Find the line with read_data command
        modified_lines = []
        replicate_inserted = False
        
        for line in lines:
            modified_lines.append(line)
            
            # Insert replicate command after read_data
            if not replicate_inserted and line.strip().startswith('read_data'):
                nx, ny, nz = replicate_factors
                if nx > 1 or ny > 1 or nz > 1:
                    modified_lines.append(f"replicate {nx} {ny} {nz}\n")
                    replicate_inserted = True
        
        # Write modified input file
        with open(output_path, 'w') as f:
            f.writelines(modified_lines)
    
    def setup_directories(self, scale_counts: List[int]):
        """
        Create scale-specific directories for weak scaling tests.
        
        Args:
            scale_counts: List of GPU/core counts to create directories for
        """
        print("🏗️  Setting up directory structure...")
        
        unit_suffix = "gpu" if self.scaling_mode == 'gpu' else "core"
        
        for count in scale_counts:
            dir_name = self.base_dir / f"{count}{unit_suffix}"
            dir_name.mkdir(exist_ok=True)
            print(f"  📁 Created {dir_name}")
    
    def copy_and_modify_files(self, scale_counts: List[int], input_file: str,
                             struct_file: str, ff_file: str, natoms_per_unit: int,
                             initial_atoms: int):
        """
        Copy input files and create modified versions for each scale point.
        
        Args:
            scale_counts: List of GPU/core counts
            input_file: LAMMPS input file name
            struct_file: Structure/data file name
            ff_file: Force field file name
            natoms_per_unit: Target atoms per GPU/core
            initial_atoms: Number of atoms in original structure
        """
        print("📋 Copying and modifying input files...")
        
        # Validate input files exist
        files_to_copy = [input_file, struct_file, ff_file]
        for file_name in files_to_copy:
            file_path = self.base_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_name}")
        
        # Validate analysis scripts exist
        analysis_script = self.base_dir / "plot_weak-scale-results.py"
        env_script = self.base_dir / "python_env-setup.sh"
        
        if not analysis_script.exists():
            raise FileNotFoundError(
                "Analysis script 'plot_weak-scale-results.py' not found in current directory."
            )
        
        if not env_script.exists():
            raise FileNotFoundError(
                "Environment script 'python_env-setup.sh' not found in current directory."
            )
        
        unit_suffix = "gpu" if self.scaling_mode == 'gpu' else "core"
        
        # Process each scale point
        for count in scale_counts:
            dir_name = self.base_dir / f"{count}{unit_suffix}"
            
            # Calculate target atoms and replication factors
            target_atoms = natoms_per_unit * count
            replicate_factors = self.calculate_replication_factors(initial_atoms, target_atoms)
            actual_atoms = initial_atoms * replicate_factors[0] * replicate_factors[1] * replicate_factors[2]
            
            # Copy structure file (original, will be replicated by LAMMPS)
            shutil.copy2(self.base_dir / struct_file, dir_name / struct_file)
            
            # Copy force field file
            shutil.copy2(self.base_dir / ff_file, dir_name / ff_file)
            
            # Create modified input file with replicate command
            self.create_modified_input_file(
                self.base_dir / input_file,
                replicate_factors,
                dir_name / input_file
            )
            
            print(f"  ✅ {count}{unit_suffix}: replicate {replicate_factors[0]}x{replicate_factors[1]}x{replicate_factors[2]} "
                  f"({actual_atoms} atoms, target: {target_atoms})")
    
    def create_lammps_jobs(self, scale_counts: List[int], input_file: str, rankspercore: int = 1):
        """
        Create LAMMPS jobs with different scale counts and proper dependencies.
        
        Args:
            scale_counts: List of GPU/core counts for scaling test
            input_file: LAMMPS input file name
            rankspercore: Ranks per core (for CPU scaling)
        """
        print("🧪 Creating LAMMPS weak scaling jobs...")
        
        parent_job_ids = None
        units_per_node = self._get_units_per_node()
        unit_suffix = "gpu" if self.scaling_mode == 'gpu' else "core"
        
        for i, count in enumerate(scale_counts):
            job_dir = f"{count}{unit_suffix}"
            
            # Calculate required nodes
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
                    cores_per_node = units_per_node
                    if required_nodes == 1 and count < cores_per_node:
                        job_params = {
                            'paths': [job_dir],
                            'app': 'lammps-kk',
                            'config': self.config_name,
                            'input_file': input_file,
                            'nnodes': 1,
                            'node_occupancy': count / cores_per_node,
                            'ranks_per_node': rankspercore * count,
                            'tag': self.tag,
                        }
                    else:
                        job_params = {
                            'paths': [job_dir],
                            'app': 'lammps-kk',
                            'config': self.config_name,
                            'input_file': input_file,
                            'nnodes': required_nodes,
                            'ranks_per_node': rankspercore * cores_per_node,
                            'tag': self.tag,
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
                    job_id = job_ids[0]
                    self.created_jobs.append(job_id)
                    parent_job_ids = [job_id]
                    parent_info = f" (parent: {parent_job_ids[0]})" if i > 0 else ""
                    print(f"    ✅ Added {job_dir} - {count} {unit_suffix}s, {job_type}, Job ID: {job_id}{parent_info}")
                else:
                    raise RuntimeError(f"No job ID returned for {job_dir}")
                    
            except Exception as e:
                print(f"    ❌ Failed to add {job_dir}: {e}")
                raise
        
        return parent_job_ids[0] if parent_job_ids else None
    
    def create_analysis_job(self, last_job_id: Optional[int]):
        """
        Create Python analysis job to process results.
        
        Args:
            last_job_id: Job ID of the last LAMMPS job to use as parent
        """
        print("📊 Creating analysis job...")
        
        try:
            job_ids, failed_jobs, msg_log = self.pbx.add_jobs(
                paths=["."],
                app="python",
                config=self.config_name,
                input_file="plot_weak-scale-results.py",
                env_file=str(self.base_dir / "python_env-setup.sh"),
                tag=self.tag,
                node_occupancy=0.1,
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
                job_id = job_ids[0]
                parent_info = f" (parent: {last_job_id})" if last_job_id else ""
                print(f"    ✅ Added analysis job, Job ID: {job_id}{parent_info}")
            else:
                raise RuntimeError("No job ID returned for analysis job")
                
        except Exception as e:
            print(f"    ❌ Failed to add analysis job: {e}")
            raise
    
    def orchestrate(self, scale_counts: List[int], input_file: str,
                   struct_file: str, ff_file: str, natoms_per_unit: int,
                   rankspercore: int = 1):
        """
        Main orchestration method to set up and create all jobs.
        
        Args:
            scale_counts: List of GPU/core counts for scaling test
            input_file: LAMMPS input file name
            struct_file: Structure/data file name
            ff_file: Force field file name
            natoms_per_unit: Target atoms per GPU/core
            rankspercore: Ranks per core (for CPU scaling)
        """
        unit_name = "GPU" if self.scaling_mode == 'gpu' else "core"
        
        print(f"🚀 Starting LAMMPS weak scaling orchestration for {self.config_name}")
        print(f"   Scaling mode: {unit_name}")
        print(f"   {unit_name} counts: {scale_counts}")
        print(f"   Atoms per {unit_name}: {natoms_per_unit}")
        print(f"   Input file: {input_file}")
        print(f"   Structure file: {struct_file}")
        print(f"   Force field file: {ff_file}")
        if self.scaling_mode == 'core':
            print(f"   Ranks per core: {rankspercore}")
        print()
        
        # Validate scale counts
        self._validate_scale_counts(scale_counts)
        
        # Parse LAMMPS data file
        print("📖 Parsing LAMMPS data file...")
        data_info = self.parse_lammps_data_file(self.base_dir / struct_file)
        initial_atoms = data_info['natoms']
        box_volume = data_info['box_volume']
        print(f"  ✅ Found {initial_atoms} atoms, box volume: {box_volume:.2f} Ų")
        
        # Validate initial structure size
        if initial_atoms >= 100:
            print(f"  ⚠️  Warning: Initial structure has {initial_atoms} atoms (recommended <100)")
        if box_volume >= 1000000:  # 100³
            print(f"  ⚠️  Warning: Box volume is {box_volume:.2f} Ų (recommended <100³)")
        print()
        
        # Setup directories
        self.setup_directories(scale_counts)
        
        # Copy and modify input files
        self.copy_and_modify_files(scale_counts, input_file, struct_file, ff_file,
                                   natoms_per_unit, initial_atoms)
        
        # Create LAMMPS jobs with dependencies
        last_job_id = self.create_lammps_jobs(scale_counts, input_file, rankspercore)
        
        # Create analysis job
        self.create_analysis_job(last_job_id)
        
        print(f"\n🎉 Successfully created weak scaling test!")
        print(f"📁 Working directory: {self.base_dir.absolute()}")
        print(f"🏷️  Job tag: {self.tag}")
        print("\n💡 Next steps:")
        print("   - Review the created jobs with: pbx ls")
        print("   - Submit jobs with: pbx qsub")
        print("   - Monitor progress with: pbx ls --status Running")
        print("   - Analysis results will be generated automatically after all jobs complete")


def main():
    """Main function to parse arguments and orchestrate weak scaling tests."""
    parser = argparse.ArgumentParser(
        description="Create LAMMPS weak scaling test jobs for ParslBox",
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
        help="GPU counts for weak scaling test (e.g., --gpus 1 2 4)"
    )
    scaling_group.add_argument(
        "--cores",
        nargs="+",
        type=int,
        metavar="N",
        help="Core counts for weak scaling test (e.g., --cores 1 8 64)"
    )
    
    parser.add_argument(
        "--fstruct",
        required=True,
        metavar="FILE",
        help="Structure/data file name (small structure <100 atoms, <100 Ų)"
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
        "--natoms-per-unit",
        type=int,
        default=250,
        metavar="N",
        help="Target atoms per GPU/core (default: 250)"
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
        default="weak_scale",
        metavar="TAG",
        help="Tag to apply to all created jobs (default: 'weak_scale')"
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
        parser.error("At least 2 scale points are required for weak scaling analysis")
    
    if any(count <= 0 for count in scale_counts):
        parser.error("All scale counts must be positive integers")
    
    # Validate natoms_per_unit
    if args.natoms_per_unit <= 0:
        parser.error("--natoms-per-unit must be a positive integer")
    
    try:
        # Create the orchestrator
        orchestrator = LammpsWeakScaleOrchestrator(args.config_name, scaling_mode, args.tag)
        
        # Run the orchestration
        orchestrator.orchestrate(
            scale_counts=scale_counts,
            input_file=args.input_file,
            struct_file=args.fstruct,
            ff_file=args.ff,
            natoms_per_unit=args.natoms_per_unit,
            rankspercore=args.rankspercore
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
