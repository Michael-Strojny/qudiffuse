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
VAST_AI_INSTANCE_ID=""  # Replace with your actual instance ID
VAST_AI_API_KEY=""      # Replace with your vast.ai API key

# Validate configuration
validate_config() {
    if [[ -z "$VAST_AI_INSTANCE_ID" ]]; then
        log_error "❌ VAST_AI_INSTANCE_ID is not set"
        return 1
    fi
    
    if [[ -z "$VAST_AI_API_KEY" ]]; then
        log_error "❌ VAST_AI_API_KEY is not set"
        return 1
    fi
}

# Check instance status
check_instance_status() {
    log_info "🔍 Checking instance status..."
    
    # Use vast.ai API to get instance details
    local status=$(curl -s \
        -H "Authorization: $VAST_AI_API_KEY" \
        "https://vast.ai/api/v0/instances/$VAST_AI_INSTANCE_ID/" | jq -r '.status')
    
    log_info "Instance Status: $status"
    
    case "$status" in
        "running")
            log_info "✅ Instance is running"
            return 0
            ;;
        "stopped")
            log_warning "⏸️ Instance is stopped"
            return 1
            ;;
        *)
            log_error "❌ Unknown instance status: $status"
            return 2
            ;;
    esac
}

# Start instance
start_instance() {
    log_info "🚀 Starting instance..."
    
    local response=$(curl -s \
        -X POST \
        -H "Authorization: $VAST_AI_API_KEY" \
        "https://vast.ai/api/v0/instances/$VAST_AI_INSTANCE_ID/start/")
    
    log_debug "Start Response: $response"
    
    # Wait for instance to start
    local max_attempts=10
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        log_info "Waiting for instance to start... (Attempt $((attempt + 1))/$max_attempts)"
        sleep 30
        
        if check_instance_status; then
            log_info "✅ Instance started successfully"
            return 0
        fi
        
        ((attempt++))
    done
    
    log_error "❌ Failed to start instance after $max_attempts attempts"
    return 1
}

# Restart instance
restart_instance() {
    log_info "🔄 Restarting instance..."
    
    # Stop instance if running
    if check_instance_status; then
        log_info "Stopping instance..."
        curl -s \
            -X POST \
            -H "Authorization: $VAST_AI_API_KEY" \
            "https://vast.ai/api/v0/instances/$VAST_AI_INSTANCE_ID/stop/"
        
        sleep 30  # Wait for instance to stop
    fi
    
    # Start instance
    start_instance
}

# Regenerate SSH keys
regenerate_ssh_keys() {
    log_info "🔑 Regenerating SSH keys..."
    
    # Remove existing keys
    rm -f ~/.ssh/vast_ai_key ~/.ssh/vast_ai_key.pub
    
    # Generate new ED25519 key
    ssh-keygen -t ed25519 \
               -f ~/.ssh/vast_ai_key \
               -N "" \
               -C "vast_ai_training_key"
    
    # Update vast.ai instance with new public key
    local public_key=$(cat ~/.ssh/vast_ai_key.pub)
    
    curl -s \
        -X POST \
        -H "Authorization: $VAST_AI_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"public_key\": \"$public_key\"}" \
        "https://vast.ai/api/v0/instances/$VAST_AI_INSTANCE_ID/update/"
}

# Main execution
main() {
    log_info "🚀 Starting Vast.ai Instance Management"
    
    # Validate configuration
    validate_config || exit 1
    
    # Check current status
    if ! check_instance_status; then
        log_warning "Instance not running. Attempting to start..."
        start_instance || exit 1
    fi
    
    # Regenerate SSH keys
    regenerate_ssh_keys
    
    log_info "🎉 Instance Management Complete!"
}

# Execute main function
main 