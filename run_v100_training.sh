#!/bin/bash
# V100 Training Execution Script

# Exit on any error
set -e

# Ensure we're in the correct directory
cd "$(dirname "$0")"

# Create virtual environment if not exists
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Set CUDA and GPU environment variables
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="7.0"  # V100 architecture

# Run training script
python3 train_v100_gpu_fixed.py

# Deactivate virtual environment
deactivate 