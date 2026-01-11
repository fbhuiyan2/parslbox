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
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
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
    
    def parse_log_files(self) -> Dict[int, Dict[str, float]]:
        """
        Parse all log files and extract performance data.
        
        Returns:
            Dictionary mapping GPU count to performance metrics
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
            if perf_data:
                performance_data[gpu_count] = perf_data
                print(f"    ✅ Found performance data: {perf_data['timesteps_per_s']:.2f} timesteps/s")
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
        katom_step_per_s = [performance_data[gpu]['katom_step_per_s'] for gpu in gpu_counts]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Timesteps/s plot
        ax1.plot(gpu_counts, timesteps_per_s, 'o-', color='blue', label='Timesteps/s')
        ax1.set_xlabel('Number of GPUs')
        ax1.set_ylabel('Timesteps per Second')
        ax1.set_title('LAMMPS Performance: Timesteps per Second vs GPU Count')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Set integer ticks for GPU count
        ax1.set_xticks(gpu_counts)
        
        # katom-step/s plot
        ax2.plot(gpu_counts, katom_step_per_s, 'o-', color='red', label='katom-step/s')
        ax2.set_xlabel('Number of GPUs')
        ax2.set_ylabel('katom-step per Second')
        ax2.set_title('LAMMPS Performance: katom-step per Second vs GPU Count')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Set integer ticks for GPU count
        ax2.set_xticks(gpu_counts)
        
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
        
        # Ideal scaling line
        baseline_gpu = gpu_counts[0]
        ideal_speedup = [gpu / baseline_gpu for gpu in gpu_counts]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Speedup plot
        ax1.plot(gpu_counts, speedup_timesteps, 'o-', color='blue', linewidth=2, label='Actual Speedup')
        ax1.plot(gpu_counts, ideal_speedup, '--', color='gray', linewidth=2, label='Ideal Speedup')
        ax1.set_xlabel('Number of GPUs')
        ax1.set_ylabel('Speedup Factor')
        ax1.set_title('LAMMPS Strong Scaling: Speedup vs GPU Count')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(gpu_counts)
        
        # Efficiency plot
        ax2.plot(gpu_counts, efficiency_timesteps, 'o-', color='green', linewidth=2, label='Scaling Efficiency')
        ax2.axhline(y=100, color='gray', linestyle='--', linewidth=2, label='Ideal Efficiency (100%)')
        ax2.set_xlabel('Number of GPUs')
        ax2.set_ylabel('Scaling Efficiency (%)')
        ax2.set_title('LAMMPS Strong Scaling: Efficiency vs GPU Count')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xticks(gpu_counts)
        ax2.set_ylim(0, max(110, max(efficiency_timesteps) * 1.1))
        
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
            f.write(f"GPU counts tested: {gpu_counts}\n\n")
            
            f.write("Performance Data:\n")
            f.write("-" * 20 + "\n")
            for gpu_count in gpu_counts:
                perf = performance_data[gpu_count]
                f.write(f"{gpu_count:2d} GPUs: {perf['timesteps_per_s']:8.2f} timesteps/s, "
                       f"{perf['katom_step_per_s']:8.2f} katom-step/s\n")
            
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
