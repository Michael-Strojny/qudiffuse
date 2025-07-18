#!/bin/bash
set -e

# Ensure we're using the correct Python version
PYTHON_CMD=$(which python3.10)
if [ -z "$PYTHON_CMD" ]; then
    echo "❌ Python 3.10 not found. Please install."
    exit 1
fi

# Create virtual environment
$PYTHON_CMD -m venv qudiffuse_venv
source qudiffuse_venv/bin/activate

# Upgrade pip and setuptools
pip install --upgrade pip setuptools wheel

# Install CUDA and GPU dependencies first
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install additional monitoring and progress tracking libraries
pip install \
    pynvml==11.5.0 \
    GPUtil==1.4.0 \
    nvidia-ml-py3==7.352.0 \
    psutil==7.0.0 \
    tqdm==4.67.1

# Install project requirements
pip install -r requirements.txt

# Verify GPU availability and environment
python3 -c "
import sys
import torch
import torchvision
import pynvml
import GPUtil
import psutil

print('✅ Python Version:', sys.version)
print('✅ CUDA Available:', torch.cuda.is_available())
print('✅ GPU Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')
print('✅ PyTorch Version:', torch.__version__)
print('✅ TorchVision Version:', torchvision.__version__)
print('✅ PYNVML Version:', pynvml.__version__)
print('✅ GPUtil Version:', GPUtil.__version__)
print('✅ psutil Version:', psutil.__version__)

# Verify GPU information
try:
    import torch.cuda
    print('✅ CUDA Devices:', torch.cuda.device_count())
    print('✅ Current Device:', torch.cuda.current_device())
    print('✅ Device Name:', torch.cuda.get_device_name(0))
except Exception as e:
    print('❌ CUDA Device Check Failed:', str(e))
"

# Run training script
python3 train_v100_gpu_fixed.py 