#!/bin/bash

# Strict error handling
set -euo pipefail

# Color codes for logging
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Maximum number of connection attempts
MAX_RETRIES=3
RETRY_DELAY=10  # seconds

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $*" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

# Configuration
REMOTE_HOST="66.115.179.150"
REMOTE_PORT="51733"
REMOTE_USER="root"
REMOTE_PYTHON="/usr/bin/python3"
REMOTE_REPO_PATH="/root/qudiffuse"
REMOTE_SCRIPT_PATH="$REMOTE_REPO_PATH/scripts/train_timestep_dbns.py"
CHECKPOINT="/root/checkpoint_epoch_50.pth"

# SSH Connection with comprehensive retry mechanism
ssh_with_retry() {
    local command="$1"
    local retry_count=0

    while [ $retry_count -lt $MAX_RETRIES ]; do
        log_info "Attempting SSH connection (Attempt $((retry_count + 1))/$MAX_RETRIES)..."
        
        # Attempt SSH connection with timeout
        if ssh -o ConnectTimeout=15 -p "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST" "$command"; then
            return 0
        else
            log_warning "SSH connection failed. Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
            ((retry_count++))
        fi
    done

    log_error "❌ SSH Connection Failed after $MAX_RETRIES attempts"
    return 1
}

# Dependency installation
install_dependencies() {
    log_info "🔍 Installing Python dependencies..."
    
    # Ensure pip is up to date
    $REMOTE_PYTHON -m pip install --upgrade pip
    
    # Install core dependencies with comprehensive error handling
    $REMOTE_PYTHON -m pip install \
        torch \
        torchvision \
        numpy \
        pynvml \
        tqdm \
        pydantic \
        || { log_error "❌ Dependency installation failed"; return 1; }
}

# Repository synchronization
sync_repository() {
    log_info "🔄 Synchronizing QuDiffuse repository..."
    
    # Ensure remote repository directory exists
    ssh_with_retry "mkdir -p $REMOTE_REPO_PATH"
    
    # Synchronize local repository to remote with comprehensive error handling
    rsync -avz \
        -e "ssh -p $REMOTE_PORT" \
        --exclude='.git' \
        --exclude='*.pyc' \
        --exclude='__pycache__' \
        "$PWD/" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_REPO_PATH/" \
        || { log_error "❌ Repository synchronization failed"; return 1; }
}

# SSH Connection with comprehensive error handling
ssh_train() {
    log_info "🚀 Initiating Authentic DBN Training via SSH"
    
    # Validate checkpoint exists
    if ! ssh_with_retry "test -f $CHECKPOINT"; then
        log_error "❌ Checkpoint not found at $CHECKPOINT"
        return 1
    fi
    
    # Execute remote training with comprehensive logging
    ssh_with_retry << EOSSH
        set -euo pipefail
        
        # Set Python path to include repository
        export PYTHONPATH="$REMOTE_REPO_PATH:$PYTHONPATH"
        
        # Install dependencies
        $REMOTE_PYTHON -m pip install torch torchvision numpy pynvml tqdm pydantic
        
        # Verify GPU availability
        $REMOTE_PYTHON -c "
import torch
import sys

try:
    print(f'✅ CUDA Available: {torch.cuda.is_available()}')
    print(f'✅ CUDA Device Count: {torch.cuda.device_count()}')
    print(f'✅ Current CUDA Device: {torch.cuda.current_device()}')
    print(f'✅ CUDA Device Name: {torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'❌ CUDA Check Failed: {e}')
    sys.exit(1)
"
        
        # Verify script exists
        if [[ ! -f "$REMOTE_SCRIPT_PATH" ]]; then
            echo "❌ Training script not found at $REMOTE_SCRIPT_PATH"
            exit 1
        fi
        
        # Execute training script
        $REMOTE_PYTHON "$REMOTE_SCRIPT_PATH" \
            --autoencoder "$CHECKPOINT" \
            --timesteps 50 \
            --batch-size 32 \
            --epochs-per-timestep 100 \
            --log-level DEBUG \
            --authentic-mode true
EOSSH
}

# Main execution
main() {
    log_info "🔬 Starting QuDiffuse Authentic DBN Training Pipeline"
    
    # Synchronize repository
    sync_repository || exit 1
    
    # Install dependencies
    install_dependencies || exit 1
    
    # Execute training
    if ssh_train; then
        log_info "🎉 Training Completed Successfully!"
    else
        log_error "❌ Training Failed"
        exit 1
    fi
}

# Execute main function
main 