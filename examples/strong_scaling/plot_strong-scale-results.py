#!/usr/bin/env python3
"""
LAMMPS Strong Scaling Results Analysis Script

This script analyzes LAMMPS log files from strong scaling tests and generates
publication-ready plots showing performance scaling and efficiency.

The script looks for performance data in log.lammps files from GPU directories
and extracts metrics like timesteps/s and katom-step/s to calculate strong
scaling efficiency.

Usage:
    python plot_strong-scale-results.py [--base-dir DIR] [--output-prefix PREFIX]

Examples:
    python plot_strong-scale-results.py
    python plot_strong-scale-results.py --base-dir /path/to/results --output-prefix my_scaling
"""

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


class LammpsStrongScaleAnalyzer:
    """Analyzer for LAMMPS strong scaling performance data."""
    
    def __init__(self, base_dir: Path = None, output_prefix: str = "strong_scaling"):
        """
        Initialize the analyzer.
        
        Args:
            base_dir: Base directory containing GPU subdirectories
            output_prefix: Prefix for output files
        """
        self.base_dir = base_dir or Path.cwd()
        self.output_prefix = output_prefix
        
        # Set up matplotlib for publication-ready plots
        self._setup_matplotlib()
    
    def _setup_matplotlib(self):
        """Configure matplotlib for publication-ready plots."""
        # Use a clean style
        plt.style.use('default')
        
        # Set publication-ready parameters
        plt.rcParams.update({
            'figure.figsize': (10, 8),
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.size': 20,
            'axes.titlesize': 20,
            'axes.labelsize': 20,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 18,
            'lines.linewidth': 2,
            'lines.markersize': 8,
            'grid.alpha': 0.3,
            'axes.grid': False,
            'axes.axisbelow': True
        })
    
    def _calculate_x_axis_params(self, max_gpu: int) -> Tuple[float, float, float]:
        """
        Calculate x-axis limits and tick intervals based on max GPU count.
        
        Args:
            max_gpu: Maximum GPU count in the data
            
        Returns:
            Tuple of (xlim_max, major_tick_interval, minor_tick_interval)
        """
        if max_gpu <= 100:
            xlim_max = 100
            major_tick_interval = 10
            minor_tick_interval = 2  # 4 minor ticks between majors (10/5 = 2)
        elif max_gpu <= 1000:
            xlim_max = math.ceil(max_gpu / 100) * 100
            major_tick_interval = 50
            minor_tick_interval = 10  # 4 minor ticks between majors (50/5 = 10)
        else:
            xlim_max = math.ceil(max_gpu / 500) * 500
            major_tick_interval = 100
            minor_tick_interval = 20  # 4 minor ticks between majors (100/5 = 20)
        
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
            else:
                major_tick_interval = 0.25
                minor_tick_interval = 0.125  # 1 minor tick between majors (0.25/2 = 0.125)
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
    
    def _calculate_y_axis_params_speedup(self, max_speedup: float) -> Tuple[float, float, float]:
        """
        Calculate y-axis limits and tick intervals for speedup plot.
        
        Args:
            max_speedup: Maximum actual speedup value in the data
            
        Returns:
            Tuple of (ylim_max, major_tick_interval, minor_tick_interval)
        """
        # Round to nearest ceiling 50s
        ylim_max = math.ceil(max_speedup / 50) * 50
        
        # Determine tick intervals based on range
        if ylim_max <= 100:
            major_tick_interval = 10
            minor_tick_interval = 2  # 4 minor ticks between majors (10/5 = 2)
        else:
            major_tick_interval = 25
            minor_tick_interval = 5  # 4 minor ticks between majors (25/5 = 5)
        
        return ylim_max, major_tick_interval, minor_tick_interval
    
    def _setup_axis_ticks(self, ax, xlim_max: float, ylim_max: float, 
                         x_major: float, x_minor: float, 
                         y_major: float, y_minor: float):
        """
        Set up axis limits and tick formatting for a plot.
        
        Args:
            ax: Matplotlib axis object
            xlim_max: Maximum x-axis limit
            ylim_max: Maximum y-axis limit
            x_major: Major tick interval for x-axis
            x_minor: Minor tick interval for x-axis
            y_major: Major tick interval for y-axis
            y_minor: Minor tick interval for y-axis
        """
        # Set axis limits
        ax.set_xlim(0, xlim_max)
        ax.set_ylim(0, ylim_max)
        
        # Set major and minor ticks
        ax.xaxis.set_major_locator(ticker.MultipleLocator(x_major))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_minor))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_major))
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_minor))
        
        # Enable ticks on all sides
        ax.tick_params(which='both', top=True, right=True, bottom=True, left=True)
        ax.tick_params(which='major', length=6, width=1.5)
        ax.tick_params(which='minor', length=3, width=1)
        
        # Enable grid for major ticks only
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(False, which='minor')
    
    def find_gpu_directories(self) -> List[int]:
        """
        Find all GPU directories in the base directory.
        
        Returns:
            List of GPU counts found (sorted)
        """
        gpu_dirs = []
        
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.endswith('gpu'):
                try:
                    # Extract GPU count from directory name (e.g., "4gpu" -> 4)
                    gpu_count = int(item.name[:-3])
                    gpu_dirs.append(gpu_count)
                except ValueError:
                    continue
        
        return sorted(gpu_dirs)
    
    def extract_performance_data(self, log_file: Path) -> Optional[Dict[str, float]]:
        """
        Extract performance data from a LAMMPS log file.
        
        Looks for the performance line from the bottom of the file:
        "Performance: X.XXX ns/day, X.XXX hours/ns, X.XXX timesteps/s, X.XXX katom-step/s"
        
        Args:
            log_file: Path to log.lammps file
            
        Returns:
            Dictionary with performance metrics or None if not found
        """
        if not log_file.exists():
            print(f"⚠️  Log file not found: {log_file}")
            return None
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Search from bottom of file for performance line
            performance_pattern = r'Performance:\s+([\d.]+)\s+ns/day,\s+([\d.]+)\s+hours/ns,\s+([\d.]+)\s+timesteps/s,\s+([\d.]+)\s+katom-step/s'
            
            for line in reversed(lines):
                match = re.search(performance_pattern, line)
                if match:
                    return {
                        'ns_per_day': float(match.group(1)),
                        'hours_per_ns': float(match.group(2)),
                        'timesteps_per_s': float(match.group(3)),
                        'katom_step_per_s': float(match.group(4))
                    }
            
            print(f"⚠️  No performance data found in: {log_file}")
            return None
            
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")
            return None
    
    def extract_timestep_data(self, log_file: Path) -> Optional[float]:
        """
        Extract timestep value from a LAMMPS log file.
        
        Looks for lines starting with "timestep" followed by a float value,
        searching from the bottom of the file to get the most recent value.
        
        Args:
            log_file: Path to log.lammps file
            
        Returns:
            Timestep value as float or None if not found
        """
        if not log_file.exists():
            print(f"⚠️  Log file not found: {log_file}")
            return None
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            # Search from bottom of file for timestep line
            timestep_pattern = r'^timestep\s+([\d.]+)'
            
            for line in reversed(lines):
                line = line.strip()
                match = re.search(timestep_pattern, line)
                if match:
                    return float(match.group(1))
            
            print(f"⚠️  No timestep data found in: {log_file}")
            return None
            
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")
            return None
    
    def parse_log_files(self) -> Dict[int, Dict[str, float]]:
        """
        Parse all log files and extract performance and timestep data.
        
        Returns:
            Dictionary mapping GPU count to performance metrics including timestep
        """
        print("📊 Parsing LAMMPS log files...")
        
        gpu_counts = self.find_gpu_directories()
        if not gpu_counts:
            raise ValueError("No GPU directories found. Expected directories like '1gpu', '2gpu', etc.")
        
        performance_data = {}
        
        for gpu_count in gpu_counts:
            gpu_dir = self.base_dir / f"{gpu_count}gpu"
            log_file = gpu_dir / "log.lammps"
            
            print(f"  📄 Processing {log_file}")
            
            perf_data = self.extract_performance_data(log_file)
            timestep_data = self.extract_timestep_data(log_file)
            
            if perf_data:
                # Add timestep data to performance data
                perf_data['timestep'] = timestep_data
                performance_data[gpu_count] = perf_data
                timestep_info = f", timestep: {timestep_data}" if timestep_data else ", timestep: N/A"
                print(f"    ✅ Found performance data: {perf_data['timesteps_per_s']:.2f} timesteps/s, {perf_data['ns_per_day']:.3f} ns/day{timestep_info}")
            else:
                print(f"    ❌ No performance data found")
        
        if not performance_data:
            raise ValueError("No performance data found in any log files")
        
        return performance_data
    
    def calculate_scaling_efficiency(self, performance_data: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
        """
        Calculate strong scaling efficiency metrics.
        
        Args:
            performance_data: Performance data from parse_log_files
            
        Returns:
            Dictionary with scaling efficiency data
        """
        print("📈 Calculating scaling efficiency...")
        
        gpu_counts = sorted(performance_data.keys())
        baseline_gpu = gpu_counts[0]
        baseline_perf = performance_data[baseline_gpu]
        
        efficiency_data = {}
        
        for gpu_count in gpu_counts:
            perf = performance_data[gpu_count]
            
            # Calculate speedup (performance ratio)
            speedup_timesteps = perf['timesteps_per_s'] / baseline_perf['timesteps_per_s']
            speedup_katom = perf['katom_step_per_s'] / baseline_perf['katom_step_per_s']
            
            # Calculate efficiency (speedup / GPU ratio)
            gpu_ratio = gpu_count / baseline_gpu
            efficiency_timesteps = (speedup_timesteps / gpu_ratio) * 100  # Percentage
            efficiency_katom = (speedup_katom / gpu_ratio) * 100  # Percentage
            
            efficiency_data[gpu_count] = {
                'speedup_timesteps': speedup_timesteps,
                'speedup_katom': speedup_katom,
                'efficiency_timesteps': efficiency_timesteps,
                'efficiency_katom': efficiency_katom,
                'gpu_ratio': gpu_ratio
            }
            
            print(f"  🔢 {gpu_count} GPUs: {speedup_timesteps:.2f}x speedup, {efficiency_timesteps:.1f}% efficiency")
        
        return efficiency_data
    
    def create_performance_plot(self, performance_data: Dict[int, Dict[str, float]]) -> Path:
        """
        Create performance vs GPU count plot.
        
        Args:
            performance_data: Performance data from parse_log_files
            
        Returns:
            Path to saved plot file
        """
        gpu_counts = sorted(performance_data.keys())
        timesteps_per_s = [performance_data[gpu]['timesteps_per_s'] for gpu in gpu_counts]
        ns_per_day = [performance_data[gpu]['ns_per_day'] for gpu in gpu_counts]
        
        # Calculate axis parameters
        max_gpu = max(gpu_counts)
        max_timesteps = max(timesteps_per_s)
        max_ns_per_day = max(ns_per_day)
        
        x_lim_max, x_major, x_minor = self._calculate_x_axis_params(max_gpu)
        y1_lim_max, y1_major, y1_minor = self._calculate_y_axis_params_performance(max_timesteps, is_ns_per_day=False)
        y2_lim_max, y2_major, y2_minor = self._calculate_y_axis_params_performance(max_ns_per_day, is_ns_per_day=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Timesteps/s plot
        ax1.plot(gpu_counts, timesteps_per_s, 'o-', color='blue', label='Timesteps/s')
        ax1.set_xlabel('Number of GPUs')
        ax1.set_ylabel('Timesteps per Second')
        #ax1.set_title('LAMMPS Performance: Timesteps per Second vs GPU Count')
        ax1.legend()
        
        # Set up axis formatting for timesteps/s plot
        self._setup_axis_ticks(ax1, x_lim_max, y1_lim_max, x_major, x_minor, y1_major, y1_minor)
        
        # ns/day plot
        ax2.plot(gpu_counts, ns_per_day, 'o-', color='green', label='ns/day')
        ax2.set_xlabel('Number of GPUs')
        ax2.set_ylabel('Nanoseconds per Day')
        #ax2.set_title('LAMMPS Performance: ns/day vs GPU Count')
        ax2.legend()
        
        # Set up axis formatting for ns/day plot
        self._setup_axis_ticks(ax2, x_lim_max, y2_lim_max, x_major, x_minor, y2_major, y2_minor)
        
        plt.tight_layout()
        
        output_file = self.base_dir / f"{self.output_prefix}_performance.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def create_scaling_plot(self, efficiency_data: Dict[int, Dict[str, float]]) -> Path:
        """
        Create strong scaling efficiency plot.
        
        Args:
            efficiency_data: Efficiency data from calculate_scaling_efficiency
            
        Returns:
            Path to saved plot file
        """
        gpu_counts = sorted(efficiency_data.keys())
        speedup_timesteps = [efficiency_data[gpu]['speedup_timesteps'] for gpu in gpu_counts]
        efficiency_timesteps = [efficiency_data[gpu]['efficiency_timesteps'] for gpu in gpu_counts]
        
        # Calculate axis parameters
        max_gpu = max(gpu_counts)
        max_actual_speedup = max(speedup_timesteps)  # Use actual speedup, not ideal
        max_efficiency = max(efficiency_timesteps)
        
        x_lim_max, x_major, x_minor = self._calculate_x_axis_params(max_gpu)
        y1_lim_max, y1_major, y1_minor = self._calculate_y_axis_params_speedup(max_actual_speedup)
        
        # For efficiency plot, use 0-110% or slightly above max efficiency
        efficiency_ylim_max = max(110, math.ceil(max_efficiency / 10) * 10)
        efficiency_major = 10 if efficiency_ylim_max <= 100 else 20
        efficiency_minor = 2.5 if efficiency_ylim_max <= 100 else 5
        
        # Ideal scaling line
        baseline_gpu = gpu_counts[0]
        ideal_speedup = [gpu / baseline_gpu for gpu in gpu_counts]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Speedup plot
        ax1.plot(gpu_counts, speedup_timesteps, 'o-', color='blue', linewidth=2, label='Actual Speedup')
        ax1.plot(gpu_counts, ideal_speedup, '--', color='gray', linewidth=2, label='Ideal Speedup')
        ax1.set_xlabel('Number of GPUs')
        ax1.set_ylabel('Speedup Factor')
        #ax1.set_title('LAMMPS Strong Scaling: Speedup vs GPU Count')
        ax1.legend()
        
        # Set up axis formatting for speedup plot
        self._setup_axis_ticks(ax1, x_lim_max, y1_lim_max, x_major, x_minor, y1_major, y1_minor)
        
        # Efficiency plot
        ax2.plot(gpu_counts, efficiency_timesteps, 'o-', color='green', linewidth=2, label='Scaling Efficiency')
        ax2.axhline(y=100, color='gray', linestyle='--', linewidth=2, label='Ideal Efficiency (100%)')
        ax2.set_xlabel('Number of GPUs')
        ax2.set_ylabel('Scaling Efficiency (%)')
        #ax2.set_title('LAMMPS Strong Scaling: Efficiency vs GPU Count')
        ax2.legend()
        
        # Set up axis formatting for efficiency plot
        self._setup_axis_ticks(ax2, x_lim_max, efficiency_ylim_max, x_major, x_minor, efficiency_major, efficiency_minor)
        
        plt.tight_layout()
        
        output_file = self.base_dir / f"{self.output_prefix}_efficiency.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_file
    
    def save_data_csv(self, performance_data: Dict[int, Dict[str, float]], 
                     efficiency_data: Dict[int, Dict[str, float]]) -> Path:
        """
        Save raw data to CSV file.
        
        Args:
            performance_data: Performance data
            efficiency_data: Efficiency data
            
        Returns:
            Path to saved CSV file
        """
        gpu_counts = sorted(performance_data.keys())
        
        data_rows = []
        for gpu_count in gpu_counts:
            perf = performance_data[gpu_count]
            eff = efficiency_data[gpu_count]
            
            row = {
                'gpu_count': gpu_count,
                'timestep': perf.get('timestep', None),
                'ns_per_day': perf['ns_per_day'],
                'hours_per_ns': perf['hours_per_ns'],
                'timesteps_per_s': perf['timesteps_per_s'],
                'katom_step_per_s': perf['katom_step_per_s'],
                'speedup_timesteps': eff['speedup_timesteps'],
                'speedup_katom': eff['speedup_katom'],
                'efficiency_timesteps': eff['efficiency_timesteps'],
                'efficiency_katom': eff['efficiency_katom']
            }
            data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        output_file = self.base_dir / f"{self.output_prefix}_data.csv"
        df.to_csv(output_file, index=False, float_format='%.4f')
        
        return output_file
    
    def save_summary_report(self, performance_data: Dict[int, Dict[str, float]], 
                           efficiency_data: Dict[int, Dict[str, float]]) -> Path:
        """
        Save a summary report of the scaling analysis.
        
        Args:
            performance_data: Performance data
            efficiency_data: Efficiency data
            
        Returns:
            Path to saved summary file
        """
        gpu_counts = sorted(performance_data.keys())
        baseline_gpu = gpu_counts[0]
        
        output_file = self.base_dir / f"{self.output_prefix}_summary.txt"
        
        with open(output_file, 'w') as f:
            f.write("LAMMPS Strong Scaling Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Baseline: {baseline_gpu} GPU(s)\n")
            f.write(f"GPU counts tested: {gpu_counts}\n")
            
            # Add timestep information
            baseline_timestep = performance_data[baseline_gpu].get('timestep')
            if baseline_timestep is not None:
                f.write(f"Timestep: {baseline_timestep}\n")
            else:
                f.write("Timestep: N/A\n")
            f.write("\n")
            
            f.write("Performance Data:\n")
            f.write("-" * 20 + "\n")
            for gpu_count in gpu_counts:
                perf = performance_data[gpu_count]
                f.write(f"{gpu_count:2d} GPUs: {perf['timesteps_per_s']:8.2f} timesteps/s, "
                       f"{perf['ns_per_day']:8.3f} ns/day\n")
            
            f.write("\nScaling Efficiency:\n")
            f.write("-" * 20 + "\n")
            for gpu_count in gpu_counts:
                eff = efficiency_data[gpu_count]
                f.write(f"{gpu_count:2d} GPUs: {eff['speedup_timesteps']:5.2f}x speedup, "
                       f"{eff['efficiency_timesteps']:5.1f}% efficiency\n")
            
            # Calculate average efficiency (excluding baseline)
            if len(gpu_counts) > 1:
                avg_efficiency = np.mean([efficiency_data[gpu]['efficiency_timesteps'] 
                                        for gpu in gpu_counts[1:]])
                f.write(f"\nAverage scaling efficiency: {avg_efficiency:.1f}%\n")
            
            # Find best efficiency
            best_gpu = max(gpu_counts, key=lambda x: efficiency_data[x]['efficiency_timesteps'])
            best_eff = efficiency_data[best_gpu]['efficiency_timesteps']
            f.write(f"Best efficiency: {best_eff:.1f}% at {best_gpu} GPUs\n")
            
            # Add ns/day performance summary
            f.write(f"\nSimulation Speed Summary:\n")
            f.write("-" * 25 + "\n")
            for gpu_count in gpu_counts:
                perf = performance_data[gpu_count]
                f.write(f"{gpu_count:2d} GPUs: {perf['ns_per_day']:8.3f} ns/day\n")
        
        return output_file
    
    def analyze(self) -> Tuple[Path, Path, Path, Path]:
        """
        Run complete analysis and generate all outputs.
        
        Returns:
            Tuple of (performance_plot, efficiency_plot, data_csv, summary_report) paths
        """
        print("🚀 Starting LAMMPS strong scaling analysis...")
        print(f"📁 Base directory: {self.base_dir.absolute()}")
        
        # Parse log files
        performance_data = self.parse_log_files()
        
        # Calculate scaling efficiency
        efficiency_data = self.calculate_scaling_efficiency(performance_data)
        
        # Create plots
        print("📊 Creating performance plots...")
        performance_plot = self.create_performance_plot(performance_data)
        print(f"  ✅ Saved: {performance_plot}")
        
        efficiency_plot = self.create_scaling_plot(efficiency_data)
        print(f"  ✅ Saved: {efficiency_plot}")
        
        # Save data
        print("💾 Saving data files...")
        data_csv = self.save_data_csv(performance_data, efficiency_data)
        print(f"  ✅ Saved: {data_csv}")
        
        summary_report = self.save_summary_report(performance_data, efficiency_data)
        print(f"  ✅ Saved: {summary_report}")
        
        print("\n🎉 Analysis complete!")
        
        return performance_plot, efficiency_plot, data_csv, summary_report


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze LAMMPS strong scaling results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --base-dir /path/to/results
  %(prog)s --output-prefix my_scaling_test
        """
    )
    
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base directory containing GPU subdirectories (default: current directory)"
    )
    
    parser.add_argument(
        "--output-prefix",
        default="strong_scaling",
        help="Prefix for output files (default: 'strong_scaling')"
    )
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = LammpsStrongScaleAnalyzer(
            base_dir=args.base_dir,
            output_prefix=args.output_prefix
        )
        
        # Run analysis
        performance_plot, efficiency_plot, data_csv, summary_report = analyzer.analyze()
        
        print("\n📋 Generated files:")
        print(f"  📊 Performance plot: {performance_plot.name}")
        print(f"  📈 Efficiency plot: {efficiency_plot.name}")
        print(f"  📄 Data CSV: {data_csv.name}")
        print(f"  📝 Summary report: {summary_report.name}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
