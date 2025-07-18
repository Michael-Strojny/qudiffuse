#!/bin/bash

# Strict error handling
set -euo pipefail

# Color codes for logging
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $*" >&2
}

# Configuration
REMOTE_HOST="66.115.179.150"
REMOTE_PORT="51733"
REMOTE_USER="root"
SSH_KEY_PATH="${HOME}/.ssh/vast_ai_key"
MAX_RETRIES=3
RETRY_DELAY=10

# Comprehensive connectivity check
check_connectivity() {
    log_info "🌐 Checking network connectivity..."
    
    # Check DNS resolution
    if ! nslookup "$REMOTE_HOST" &>/dev/null; then
        log_error "❌ DNS resolution failed for $REMOTE_HOST"
        return 1
    fi
    
    # Check port accessibility
    if ! nc -zv "$REMOTE_HOST" "$REMOTE_PORT" &>/dev/null; then
        log_error "❌ Port $REMOTE_PORT is not accessible on $REMOTE_HOST"
        return 1
    fi
    
    log_info "✅ Basic network connectivity verified"
    return 0
}

# SSH Connection diagnostic
diagnose_ssh() {
    log_info "🔍 Performing SSH diagnostics..."
    
    # Verbose SSH connection test
    ssh -vv \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -i "$SSH_KEY_PATH" \
        -p "$REMOTE_PORT" \
        "$REMOTE_USER@$REMOTE_HOST" \
        'echo "SSH connection successful"' || true
}

# Attempt SSH connection with retry
ssh_with_retry() {
    local command="$1"
    local retry_count=0

    while [ $retry_count -lt $MAX_RETRIES ]; do
        log_info "Attempting SSH connection (Attempt $((retry_count + 1))/$MAX_RETRIES)..."
        
        if ssh \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            -i "$SSH_KEY_PATH" \
            -p "$REMOTE_PORT" \
            "$REMOTE_USER@$REMOTE_HOST" \
            "$command"; then
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

# System information gathering
gather_system_info() {
    log_info "📋 Gathering system information..."
    
    ssh_with_retry '
        echo "=== System Information ==="
        uname -a
        echo -e "\n=== CPU Info ==="
        lscpu | grep "Model name\\|Socket(s)\\|Core(s) per socket\\|Thread(s) per core"
        echo -e "\n=== Memory Info ==="
        free -h
        echo -e "\n=== Disk Space ==="
        df -h
        echo -e "\n=== GPU Information ==="
        nvidia-smi
    '
}

# Main recovery workflow
main() {
    log_info "🚀 Starting Remote Instance Recovery and Diagnostic Workflow"
    
    # Check basic connectivity
    if ! check_connectivity; then
        log_error "❌ Network connectivity check failed"
        exit 1
    fi
    
    # Perform SSH diagnostics
    diagnose_ssh
    
    # Attempt to gather system information
    if gather_system_info; then
        log_info "✅ Remote instance appears to be operational"
    else
        log_error "❌ Unable to gather system information"
        exit 1
    fi
}

# Execute main function
main 