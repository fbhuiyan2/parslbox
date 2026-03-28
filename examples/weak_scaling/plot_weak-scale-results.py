#!/usr/bin/env python3
"""
LAMMPS Weak Scaling Results Analysis Script

This script analyzes LAMMPS log files from weak scaling tests and generates
publication-ready plots showing performance consistency and efficiency.

For weak scaling, the problem size scales with compute resources, so ideal
performance should remain constant. The script extracts performance metrics
and calculates weak scaling efficiency.

Usage:
    python plot_weak-scale-results.py [--base-dir DIR] [--output-prefix PREFIX] [--mode MODE]

Examples:
    python plot_weak-scale-results.py
    python plot_weak-scale-results.py --mode gpu
    python plot_weak-scale-results.py --mode core --output-prefix my_weak_scaling
"""

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from parslbox.apps.utils import report_status


class LammpsWeakScaleAnalyzer:
    """Analyzer for LAMMPS weak scaling performance data."""
    
    def __init__(self, base_dir: Path = None, output_prefix: str = "weak_scaling", mode: str = "auto"):
        """
        Initialize the analyzer.
        
        Args:
            base_dir: Base directory containing GPU/core subdirectories
            output_prefix: Prefix for output files
            mode: Scaling mode - 'gpu', 'core', or 'auto' (auto-detect)
        """
        self.base_dir = base_dir or Path.cwd()
        self.output_prefix = output_prefix
        self.mode = mode
        
        # Set up matplotlib for publication-ready plots
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Configure matplotlib for publication-ready plots."""
        plt.style.use('default')
        
        plt.rcParams.update({
            'figure.figsize': (10, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.size': 20,
            'axes.titlesize': 20,
            'axes.labelsize': 20,
            'xtick.labelsize': 18,
            'ytick.labelsize': 18,
            'legend.fontsize': 18,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'grid.alpha': 0.3,
            'axes.grid': False,
            'axes.axisbelow': True
        })
    
    def _calculate_x_axis_params(self, max_count: int) -> Tuple[float, float, float]:
        """Calculate x-axis limits and tick intervals."""
        if max_count <= 100:
            xlim_max = 100
            major_tick_interval = 10
            minor_tick_interval = 2  # 4 minor ticks between majors (10/5 = 2)
        elif max_count <= 1000:
            xlim_max = math.ceil(max_count / 100) * 100
            major_tick_interval = 100
            minor_tick_interval = 20  # 4 minor ticks between majors (100/5 = 20)
        else:
            xlim_max = math.ceil(max_count / 500) * 500
            major_tick_interval = 500
            minor_tick_interval = 100  # 4 minor ticks between majors (500/5 = 100)
        
        return xlim_max, major_tick_interval, minor_tick_interval
    
    def _calculate_y_axis_params_performance(self, max_value: float, is_ns_per_day: bool = False) -> Tuple[float, float, float]:
        """
        Calculate y-axis limits and tick intervals for performance plots.

        Args:
            max_value: Maximum value in the data
            is_ns_per_day: True for ns/day plot, False for timesteps/s plot

        Returns:
            Tuple of (ylim_max, major_tick_interval, minor_tick_interval)
        """
        if is_ns_per_day:
            # Round to nearest 0.5
            ylim_max = math.ceil(max_value * 2) / 2
            if max_value < 0.5:
                major_tick_interval = 0.1
                minor_tick_interval = 0.05  # 1 minor tick between majors (0.1/2 = 0.05)
            elif max_value < 5:
                ylim_max = math.ceil(max_value)  # Round to ceiling integer 
                major_tick_interval = 1
                minor_tick_interval = 0.5  # 1 minor tick between majors (1/2 = 0.5)
            elif max_value < 10:
                ylim_max = math.ceil(max_value / 10) * 10  # Round to nearest 10
                major_tick_interval = 2
                minor_tick_interval = 0.5  # 1 minor tick between majors (1/2 = 0.5)
            else:
                ylim_max = math.ceil(max_value / 10) * 10  # Round to nearest 10
                major_tick_interval = 5
                minor_tick_interval = 2.5  # 1 minor tick between majors (5/2 = 2.5)
        else:
            # Round to nearest 10
            ylim_max = math.ceil(max_value / 10) * 10
            if max_value < 50:
                major_tick_interval = 5
                minor_tick_interval = 1  # 4 minor ticks between majors (5/5 = 1)
            else:
                major_tick_interval = 10
                minor_tick_interval = 2  # 4 minor ticks between majors (10/5 = 2)

        return ylim_max, major_tick_interval, minor_tick_interval

    def _setup_axis_ticks(self, ax, xlim_max: float, ylim_max: float,
                         x_major: float, x_minor: float,
                         y_major: float, y_minor: float):
        """Set up axis limits and tick formatting."""
        ax.set_xlim(0, xlim_max)
        ax.set_ylim(0, ylim_max)
        
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_minor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_major))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_minor))
        
        ax.tick_params(which='both', top=True, right=True, bottom=True, left=True)
        ax.tick_params(which='major', length=6, width=1.5)
        ax.tick_params(which='minor', length=3, width=1)
        
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(False, which='minor')
    
    def detect_scaling_mode(self) -> str:
        """
        Auto-detect scaling mode by checking directory names.
        
        Returns:
            'gpu' or 'core' based on directory naming
        """
        for item in self.base_dir.iterdir():
            if item.is_dir():
                if item.name.endswith('gpu'):
                    return 'gpu'
                elif item.name.endswith('core'):
                    return 'core'
        
        raise ValueError("Could not auto-detect scaling mode. No directories ending in 'gpu' or 'core' found.")
    
    def find_scale_directories(self) -> List[int]:
        """
        Find all GPU/core directories in the base directory.
        
        Returns:
            List of GPU/core counts found (sorted)
        """
        if self.mode == 'auto':
            self.mode = self.detect_scaling_mode()
        
        suffix = 'gpu' if self.mode == 'gpu' else 'core'
        scale_dirs = []
        
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.endswith(suffix):
                try:
                    count = int(item.name[:-len(suffix)])
                    scale_dirs.append(count)
                except ValueError:
                    continue
        
        return sorted(scale_dirs)
    
    def extract_performance_data(self, log_file: Path) -> Optional[Dict[str, float]]:
        """Extract performance data from a LAMMPS log file."""
        if not log_file.exists():
            print(f"⚠️  Log file not found: {log_file}")
            return None
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # LAMMPS uses different unit prefixes: katom-step/s, Matom-step/s, Gatom-step/s, etc.
            performance_pattern = r'Performance:\s+([\d.]+)\s+ns/day,\s+([\d.]+)\s+hours/ns,\s+([\d.]+)\s+timesteps/s,\s+([\d.]+)\s+(\w?)atom-step/s'

            # Multipliers to normalize to katom-step/s
            unit_to_katom = {'k': 1.0, 'M': 1e3, 'G': 1e6, '': 1e-3}

            for line in reversed(lines):
                match = re.search(performance_pattern, line)
                if match:
                    prefix = match.group(5)
                    multiplier = unit_to_katom.get(prefix, 1.0)
                    return {
                        'ns_per_day': float(match.group(1)),
                        'hours_per_ns': float(match.group(2)),
                        'timesteps_per_s': float(match.group(3)),
                        'katom_step_per_s': float(match.group(4)) * multiplier
                    }
            
            print(f"⚠️  No performance data found in: {log_file}")
            return None
            
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")
            return None
    
    def extract_atom_count(self, log_file: Path) -> Optional[int]:
        """
        Extract total atom count from LAMMPS log file.
        
        Looks for lines like "  created 512 atoms" or similar.
        """
        if not log_file.exists():
            return None
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Look for atom count after replication
            for line in lines:
                if 'atoms' in line.lower():
                    # Try to find patterns like "created X atoms" or "X atoms in group"
                    match = re.search(r'(\d+)\s+atoms', line)
                    if match:
                        return int(match.group(1))
            
            return None
            
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")
            return None
    
    def parse_log_files(self) -> Dict[int, Dict[str, float]]:
        """Parse all log files and extract performance data."""
        print("📊 Parsing LAMMPS log files...")
        
        scale_counts = self.find_scale_directories()
        if not scale_counts:
            raise ValueError(f"No {self.mode} directories found. Expected directories like '1{self.mode}', '2{self.mode}', etc.")
        
        performance_data = {}
        suffix = 'gpu' if self.mode == 'gpu' else 'core'
        
        for count in scale_counts:
            dir_name = self.base_dir / f"{count}{suffix}"
            log_file = dir_name / "log.lammps"
            
            print(f"  📄 Processing {log_file}")
            
            perf_data = self.extract_performance_data(log_file)
            atom_count = self.extract_atom_count(log_file)
            
            if perf_data:
                perf_data['atom_count'] = atom_count
                performance_data[count] = perf_data
                atom_info = f", {atom_count} atoms" if atom_count else ""
                print(f"    ✅ Found performance data: {perf_data['timesteps_per_s']:.2f} timesteps/s{atom_info}")
            else:
                print(f"    ❌ No performance data found")
        
        if not performance_data:
            raise ValueError("No performance data found in any log files")
        
        return performance_data
    
    def calculate_weak_scaling_efficiency(self, performance_data: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
        """
        Calculate weak scaling efficiency metrics.
        
        For weak scaling, efficiency = (Performance_N / Performance_1) × 100%
        Ideal weak scaling maintains constant performance.
        """
        print("📈 Calculating weak scaling efficiency...")
        
        scale_counts = sorted(performance_data.keys())
        baseline_count = scale_counts[0]
        baseline_perf = performance_data[baseline_count]
        
        efficiency_data = {}
        
        for count in scale_counts:
            perf = performance_data[count]
            
            # Weak scaling efficiency (performance should stay constant)
            efficiency_timesteps = (perf['timesteps_per_s'] / baseline_perf['timesteps_per_s']) * 100
            efficiency_katom = (perf['katom_step_per_s'] / baseline_perf['katom_step_per_s']) * 100
            
            # Performance per atom (normalized metric)
            if perf.get('atom_count') and perf['atom_count'] > 0:
                perf_per_atom = perf['timesteps_per_s'] / perf['atom_count']
            else:
                perf_per_atom = None
            
            efficiency_data[count] = {
                'efficiency_timesteps': efficiency_timesteps,
                'efficiency_katom': efficiency_katom,
                'perf_per_atom': perf_per_atom,
                'scale_ratio': count / baseline_count
            }
            
            print(f"  🔢 {count} {self.mode}s: {efficiency_timesteps:.1f}% efficiency")
        
        return efficiency_data
    
    def create_performance_plot(self, performance_data: Dict[int, Dict[str, float]]) -> Path:
        """Create performance vs scale count plot."""
        scale_counts = sorted(performance_data.keys())
        timesteps_per_s = [performance_data[c]['timesteps_per_s'] for c in scale_counts]
        ns_per_day = [performance_data[c]['ns_per_day'] for c in scale_counts]
        
        max_count = max(scale_counts)
        max_timesteps = max(timesteps_per_s)
        max_ns_per_day = max(ns_per_day)
        
        x_lim_max, x_major, x_minor = self._calculate_x_axis_params(max_count)
        y1_lim_max, y1_major, y1_minor = self._calculate_y_axis_params_performance(max_timesteps, is_ns_per_day=False)
        y2_lim_max, y2_major, y2_minor = self._calculate_y_axis_params_performance(max_ns_per_day, is_ns_per_day=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        unit_label = "GPUs" if self.mode == 'gpu' else "Cores"
        
        # Timesteps/s plot
        ax1.plot(scale_counts, timesteps_per_s, 'o-', color='blue', label='Timesteps/s')
        ax1.axhline(y=timesteps_per_s[0], color='gray', linestyle='--', linewidth=2, label='Baseline (ideal)')
        ax1.set_xlabel(f'Number of {unit_label}')
        ax1.set_ylabel('Timesteps per Second')
        ax1.legend()
        self._setup_axis_ticks(ax1, x_lim_max, y1_lim_max, x_major, x_minor, y1_major, y1_minor)
        
        # ns/day plot
        ax2.plot(scale_counts, ns_per_day, 'o-', color='green', label='ns/day')
        ax2.axhline(y=ns_per_day[0], color='gray', linestyle='--', linewidth=2, label='Baseline (ideal)')
        ax2.set_xlabel(f'Number of {unit_label}')
        ax2.set_ylabel('Nanoseconds per Day')
        ax2.legend()
        self._setup_axis_ticks(ax2, x_lim_max, y2_lim_max, x_major, x_minor, y2_major, y2_minor)
        
        plt.tight_layout()
        
        output_file = self.base_dir / f"{self.output_prefix}_performance.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_efficiency_plot(self, efficiency_data: Dict[int, Dict[str, float]]) -> Path:
        """Create weak scaling efficiency plot."""
        scale_counts = sorted(efficiency_data.keys())
        efficiency_timesteps = [efficiency_data[c]['efficiency_timesteps'] for c in scale_counts]
        
        max_count = max(scale_counts)
        max_efficiency = max(efficiency_timesteps)
        
        x_lim_max, x_major, x_minor = self._calculate_x_axis_params(max_count)
        
        # Y-axis for efficiency (0-110% or slightly above max)
        y_lim_max = max(110, math.ceil(max_efficiency / 10) * 10)
        y_major = 10 if y_lim_max <= 100 else 20
        y_minor = 2.5 if y_lim_max <= 100 else 5
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unit_label = "GPUs" if self.mode == 'gpu' else "Cores"
        
        ax.plot(scale_counts, efficiency_timesteps, 'o-', color='blue', linewidth=2, label='Weak Scaling Efficiency')
        ax.axhline(y=100, color='gray', linestyle='--', linewidth=2, label='Ideal Efficiency (100%)')
        ax.set_xlabel(f'Number of {unit_label}')
        ax.set_ylabel('Weak Scaling Efficiency (%)')
        ax.legend()
        
        self._setup_axis_ticks(ax, x_lim_max, y_lim_max, x_major, x_minor, y_major, y_minor)
        
        plt.tight_layout()
        
        output_file = self.base_dir / f"{self.output_prefix}_efficiency.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_system_size_plot(self, performance_data: Dict[int, Dict[str, float]]) -> Path:
        """Create system size (total atoms) vs scale count plot."""
        scale_counts = sorted(performance_data.keys())
        atom_counts = [performance_data[c].get('atom_count', 0) for c in scale_counts]
        
        # Skip if no atom count data
        if all(count == 0 or count is None for count in atom_counts):
            print("  ⚠️  Skipping system size plot (no atom count data)")
            return None
        
        max_count = max(scale_counts)
        max_atoms = max(atom_counts)
        
        x_lim_max, x_major, x_minor = self._calculate_x_axis_params(max_count)
        
        # Y-axis for atom count
        y_lim_max = math.ceil(max_atoms / 1000) * 1000
        y_major = 1000 if max_atoms < 10000 else 5000
        y_minor = 200 if max_atoms < 10000 else 1000
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        unit_label = "GPUs" if self.mode == 'gpu' else "Cores"
        
        ax.plot(scale_counts, atom_counts, 'o-', color='purple', linewidth=2, label='Total Atoms')
        
        # Add ideal linear scaling line
        atoms_per_unit = atom_counts[0] / scale_counts[0] if scale_counts[0] > 0 else 0
        ideal_atoms = [atoms_per_unit * c for c in scale_counts]
        ax.plot(scale_counts, ideal_atoms, '--', color='gray', linewidth=2, label='Ideal Linear Scaling')
        
        ax.set_xlabel(f'Number of {unit_label}')
        ax.set_ylabel('Total Atoms')
        ax.legend()
        
        self._setup_axis_ticks(ax, x_lim_max, y_lim_max, x_major, x_minor, y_major, y_minor)
        
        plt.tight_layout()
        
        output_file = self.base_dir / f"{self.output_prefix}_system_size.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def save_data_csv(self, performance_data: Dict[int, Dict[str, float]],
                     efficiency_data: Dict[int, Dict[str, float]]) -> Path:
        """Save raw data to CSV file."""
        scale_counts = sorted(performance_data.keys())
        
        data_rows = []
        for count in scale_counts:
            perf = performance_data[count]
            eff = efficiency_data[count]
            
            row = {
                f'{self.mode}_count': count,
                'atom_count': perf.get('atom_count', None),
                'ns_per_day': perf['ns_per_day'],
                'hours_per_ns': perf['hours_per_ns'],
                'timesteps_per_s': perf['timesteps_per_s'],
                'katom_step_per_s': perf['katom_step_per_s'],
                'efficiency_timesteps': eff['efficiency_timesteps'],
                'efficiency_katom': eff['efficiency_katom'],
                'perf_per_atom': eff.get('perf_per_atom', None)
            }
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        output_file = self.base_dir / f"{self.output_prefix}_data.csv"
        df.to_csv(output_file, index=False, float_format='%.4f')
        
        return output_file
    
    def save_summary_report(self, performance_data: Dict[int, Dict[str, float]],
                           efficiency_data: Dict[int, Dict[str, float]]) -> Path:
        """Save a summary report of the weak scaling analysis."""
        scale_counts = sorted(performance_data.keys())
        baseline_count = scale_counts[0]
        
        output_file = self.base_dir / f"{self.output_prefix}_summary.txt"
        
        unit_label = "GPU" if self.mode == 'gpu' else "core"
        
        with open(output_file, 'w') as f:
            f.write("LAMMPS Weak Scaling Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Scaling mode: {unit_label.upper()}\n")
            f.write(f"Baseline: {baseline_count} {unit_label}(s)\n")
            f.write(f"{unit_label.upper()} counts tested: {scale_counts}\n\n")
            
            f.write("System Size:\n")
            f.write("-" * 20 + "\n")
            for count in scale_counts:
                perf = performance_data[count]
                atom_count = perf.get('atom_count', 'N/A')
                f.write(f"{count:3d} {unit_label}s: {atom_count} atoms\n")
            
            f.write("\nPerformance Data:\n")
            f.write("-" * 20 + "\n")
            for count in scale_counts:
                perf = performance_data[count]
                f.write(f"{count:3d} {unit_label}s: {perf['timesteps_per_s']:8.2f} timesteps/s, "
                       f"{perf['ns_per_day']:8.3f} ns/day\n")
            
            f.write("\nWeak Scaling Efficiency:\n")
            f.write("-" * 20 + "\n")
            for count in scale_counts:
                eff = efficiency_data[count]
                f.write(f"{count:3d} {unit_label}s: {eff['efficiency_timesteps']:5.1f}% efficiency\n")
            
            # Calculate average efficiency (excluding baseline)
            if len(scale_counts) > 1:
                avg_efficiency = np.mean([efficiency_data[c]['efficiency_timesteps']
                                        for c in scale_counts[1:]])
                f.write(f"\nAverage weak scaling efficiency: {avg_efficiency:.1f}%\n")
            
            # Find best and worst efficiency
            best_count = max(scale_counts, key=lambda x: efficiency_data[x]['efficiency_timesteps'])
            worst_count = min(scale_counts, key=lambda x: efficiency_data[x]['efficiency_timesteps'])
            
            f.write(f"Best efficiency: {efficiency_data[best_count]['efficiency_timesteps']:.1f}% "
                   f"at {best_count} {unit_label}s\n")
            f.write(f"Worst efficiency: {efficiency_data[worst_count]['efficiency_timesteps']:.1f}% "
                   f"at {worst_count} {unit_label}s\n")
        
        return output_file
    
    def analyze(self) -> Tuple[Path, ...]:
        """Run complete analysis and generate all outputs."""
        print("🚀 Starting LAMMPS weak scaling analysis...")
        print(f"📁 Base directory: {self.base_dir.absolute()}")
        
        # Parse log files
        performance_data = self.parse_log_files()
        
        unit_label = "GPU" if self.mode == 'gpu' else "core"
        print(f"📊 Detected scaling mode: {unit_label.upper()}")
        
        # Calculate weak scaling efficiency
        efficiency_data = self.calculate_weak_scaling_efficiency(performance_data)
        
        # Create plots
        print("📊 Creating plots...")
        performance_plot = self.create_performance_plot(performance_data)
        print(f"  ✅ Saved: {performance_plot}")
        
        efficiency_plot = self.create_efficiency_plot(efficiency_data)
        print(f"  ✅ Saved: {efficiency_plot}")
        
        system_size_plot = self.create_system_size_plot(performance_data)
        if system_size_plot:
            print(f"  ✅ Saved: {system_size_plot}")
        
        # Save data
        print("💾 Saving data files...")
        data_csv = self.save_data_csv(performance_data, efficiency_data)
        print(f"  ✅ Saved: {data_csv}")
        
        summary_report = self.save_summary_report(performance_data, efficiency_data)
        print(f"  ✅ Saved: {summary_report}")
        
        print("\n🎉 Analysis complete!")
        
        return (performance_plot, efficiency_plot, system_size_plot, data_csv, summary_report)


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze LAMMPS weak scaling results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --mode gpu
  %(prog)s --mode core --output-prefix my_weak_scaling
        """
    )
    
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory containing GPU/core subdirectories (default: current directory)"
    )
    
    parser.add_argument(
        "--output-prefix",
        default="weak_scaling",
        help="Prefix for output files (default: 'weak_scaling')"
    )
    
    parser.add_argument(
        "--mode",
        choices=['gpu', 'core', 'auto'],
        default='auto',
        help="Scaling mode: 'gpu', 'core', or 'auto' to detect (default: 'auto')"
    )
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = LammpsWeakScaleAnalyzer(
            base_dir=args.base_dir,
            output_prefix=args.output_prefix,
            mode=args.mode
        )
        
        # Run analysis
        results = analyzer.analyze()
        
        print("\n📋 Generated files:")
        for result in results:
            if result:
                print(f"  📊 {result.name}")

        # Report status to PBX based on whether expected .png files exist
        png_files = [r for r in results if r and r.suffix == '.png' and r.is_file()]
        if png_files:
            report_status("done")
        else:
            report_status("failed")

    except Exception as e:
        print(f"❌ Error: {e}")
        report_status("failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
