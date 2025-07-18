#!/bin/bash

# Strict error handling
set -euo pipefail

# Color codes for logging
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
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

# Configuration
SSH_KEY_PATH="${HOME}/.ssh/vast_ai_key"
REMOTE_HOST="66.115.179.150"
REMOTE_PORT="51733"
REMOTE_USER="root"

# Ensure .ssh directory exists with correct permissions
prepare_ssh_directory() {
    log_info "🔐 Preparing SSH directory..."
    mkdir -p "${HOME}/.ssh"
    chmod 700 "${HOME}/.ssh"
}

# Generate SSH key pair
generate_ssh_key() {
    log_info "🔑 Generating new SSH key pair..."
    
    # Remove existing key if it exists
    if [[ -f "${SSH_KEY_PATH}" ]]; then
        log_warning "Removing existing SSH key..."
        rm "${SSH_KEY_PATH}" "${SSH_KEY_PATH}.pub"
    fi
    
    # Generate new SSH key with strong encryption
    ssh-keygen -t ed25519 \
               -f "${SSH_KEY_PATH}" \
               -N "" \
               -C "vast_ai_training_key"
    
    # Set correct permissions
    chmod 600 "${SSH_KEY_PATH}"
    chmod 644 "${SSH_KEY_PATH}.pub"
}

# Create SSH config for vast.ai
create_ssh_config() {
    log_info "📝 Creating SSH configuration..."
    
    SSH_CONFIG_PATH="${HOME}/.ssh/config"
    
    # Backup existing config if it exists
    if [[ -f "${SSH_CONFIG_PATH}" ]]; then
        cp "${SSH_CONFIG_PATH}" "${SSH_CONFIG_PATH}.bak"
    fi
    
    # Write new configuration
    cat > "${SSH_CONFIG_PATH}" << EOL
# Vast.ai Training Instance
Host vast_ai_training
    HostName ${REMOTE_HOST}
    Port ${REMOTE_PORT}
    User ${REMOTE_USER}
    IdentityFile ${SSH_KEY_PATH}
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOL

    # Set correct permissions
    chmod 600 "${SSH_CONFIG_PATH}"
}

# Test SSH connection
test_ssh_connection() {
    log_info "🌐 Testing SSH connection..."
    
    # Attempt connection with verbose output
    if ssh -vv \
           -o StrictHostKeyChecking=no \
           -o UserKnownHostsFile=/dev/null \
           -i "${SSH_KEY_PATH}" \
           -p "${REMOTE_PORT}" \
           "${REMOTE_USER}@${REMOTE_HOST}" \
           'echo "SSH connection successful"'; then
        log_info "✅ SSH Connection Successful!"
        return 0
    else
        log_error "❌ SSH Connection Failed"
        return 1
    fi
}

# Main execution
main() {
    log_info "🚀 Starting Vast.ai SSH Configuration"
    
    # Prepare SSH directory
    prepare_ssh_directory
    
    # Generate SSH key pair
    generate_ssh_key
    
    # Create SSH configuration
    create_ssh_config
    
    # Test SSH connection
    if ! test_ssh_connection; then
        log_error "❌ SSH Configuration Failed"
        exit 1
    fi
    
    log_info "🎉 Vast.ai SSH Configuration Complete!"
}

# Execute main function
main 