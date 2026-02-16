#!/bin/bash
# Python Environment Setup for LAMMPS Strong Scaling Analysis
# This script sets up the required Python environment for running the analysis script

echo "🐍 Setting up Python environment for LAMMPS strong scaling analysis..."

# Load Python module (adjust based on your system)
# Uncomment and modify the appropriate lines for your system:

# For systems with module environment (like Polaris, Crux, etc.)
# module load python
# module load python/3.9
# module load conda
# module load miniconda3

# For Polaris specifically:
# module load conda/2023-10-04

# For systems using conda/mamba
# conda activate base
# conda activate myenv

# Install required packages if not already available
# Uncomment if packages need to be installed:
# pip install matplotlib numpy pandas
# conda install matplotlib numpy pandas

# Set any required environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# For matplotlib backend (useful for headless systems)
export MPLBACKEND=Agg

# Increase matplotlib font cache if needed
export MPLCONFIGDIR=/tmp/matplotlib-$USER

echo "✅ Python environment setup complete"
echo "📦 Required packages: matplotlib, numpy, pandas"
echo "🔧 Backend set to: $MPLBACKEND"

