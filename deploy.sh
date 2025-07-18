#!/bin/bash

# QuDiffusive Web Platform Deployment Script
# Deploys to remote server: ssh -p 17782 root@89.25.97.3 -L 8080:localhost:8080

set -e

# Configuration
REMOTE_HOST="89.25.97.3"
REMOTE_PORT="17782"
REMOTE_USER="root"
REMOTE_PATH="/opt/qudiffusive"
LOCAL_PATH="."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if SSH key exists
check_ssh_key() {
    if [ ! -f ~/.ssh/id_rsa ] && [ ! -f ~/.ssh/id_ed25519 ]; then
        error "No SSH key found. Please set up SSH key authentication."
        exit 1
    fi
}

# Test SSH connection
test_ssh_connection() {
    log "Testing SSH connection to $REMOTE_USER@$REMOTE_HOST:$REMOTE_PORT..."
    
    if ssh -p $REMOTE_PORT -o ConnectTimeout=10 -o BatchMode=yes $REMOTE_USER@$REMOTE_HOST "echo 'SSH connection successful'" > /dev/null 2>&1; then
        success "SSH connection established"
    else
        error "Failed to connect to remote server"
        echo "Please ensure:"
        echo "1. SSH key is properly configured"
        echo "2. Server is accessible at $REMOTE_HOST:$REMOTE_PORT"
        echo "3. User $REMOTE_USER has proper permissions"
        exit 1
    fi
}

# Build frontend
build_frontend() {
    log "Building React frontend..."
    
    if [ -d "web_platform/frontend" ]; then
        cd web_platform/frontend
        
        if [ ! -f "package.json" ]; then
            error "package.json not found in frontend directory"
            exit 1
        fi
        
        log "Installing frontend dependencies..."
        npm ci
        
        log "Building frontend..."
        npm run build
        
        if [ ! -d "build" ]; then
            error "Frontend build failed - build directory not found"
            exit 1
        fi
        
        cd ../..
        success "Frontend built successfully"
    else
        warning "Frontend directory not found, skipping frontend build"
    fi
}

# Create deployment package
create_deployment_package() {
    log "Creating deployment package..."
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    PACKAGE_NAME="qudiffusive-$(date +%Y%m%d-%H%M%S).tar.gz"
    
    # Copy files to temporary directory
    cp -r src/ $TEMP_DIR/
    cp -r web_platform/ $TEMP_DIR/
    cp -r scripts/ $TEMP_DIR/
    cp requirements.txt $TEMP_DIR/
    cp Dockerfile $TEMP_DIR/
    cp docker-compose.yml $TEMP_DIR/
    cp nginx.conf $TEMP_DIR/
    cp deploy.sh $TEMP_DIR/
    
    # Create .env file for production
    cat > $TEMP_DIR/.env << EOF
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8080
SECRET_KEY=$(openssl rand -hex 32)
MODEL_DEVICE=cpu
LOG_LEVEL=INFO
REDIS_URL=redis://redis:6379
EOF
    
    # Create archive
    cd $TEMP_DIR
    tar -czf $PACKAGE_NAME .
    mv $PACKAGE_NAME /tmp/
    cd - > /dev/null
    
    # Clean up
    rm -rf $TEMP_DIR
    
    echo "/tmp/$PACKAGE_NAME"
}

# Deploy to remote server
deploy_to_remote() {
    local package_path=$1
    
    log "Deploying to remote server..."
    
    # Create remote directory
    ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST "mkdir -p $REMOTE_PATH"
    
    # Upload package
    log "Uploading deployment package..."
    scp -P $REMOTE_PORT $package_path $REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH/
    
    # Extract and deploy
    local package_name=$(basename $package_path)
    
    ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST << EOF
        cd $REMOTE_PATH
        
        # Stop existing services
        if [ -f docker-compose.yml ]; then
            echo "Stopping existing services..."
            docker-compose down || true
        fi
        
        # Backup current deployment
        if [ -d current ]; then
            echo "Backing up current deployment..."
            mv current backup-\$(date +%Y%m%d-%H%M%S) || true
        fi
        
        # Extract new deployment
        echo "Extracting new deployment..."
        mkdir -p current
        cd current
        tar -xzf ../$package_name
        
        # Install system dependencies if needed
        if ! command -v docker &> /dev/null; then
            echo "Installing Docker..."
            curl -fsSL https://get.docker.com -o get-docker.sh
            sh get-docker.sh
            systemctl enable docker
            systemctl start docker
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            echo "Installing Docker Compose..."
            curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
            chmod +x /usr/local/bin/docker-compose
        fi
        
        # Create necessary directories
        mkdir -p logs model_cache data ssl
        
        # Set permissions
        chown -R 1000:1000 logs model_cache data
        
        # Start services
        echo "Starting services..."
        docker-compose up -d --build
        
        # Wait for services to be ready
        echo "Waiting for services to start..."
        sleep 30
        
        # Check health
        if curl -f http://localhost:8080/health > /dev/null 2>&1; then
            echo "Deployment successful! Services are healthy."
        else
            echo "Warning: Health check failed. Check logs with: docker-compose logs"
        fi
        
        # Clean up
        rm ../$package_name
EOF
    
    success "Deployment completed"
}

# Show deployment status
show_status() {
    log "Checking deployment status..."
    
    ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST << 'EOF'
        cd /opt/qudiffusive/current
        
        echo "=== Docker Services ==="
        docker-compose ps
        
        echo ""
        echo "=== Health Check ==="
        if curl -f http://localhost:8080/health 2>/dev/null; then
            echo "✅ Application is healthy"
        else
            echo "❌ Application health check failed"
        fi
        
        echo ""
        echo "=== Recent Logs ==="
        docker-compose logs --tail=20 qudiffusive-web
EOF
}

# Main deployment function
main() {
    log "Starting QuDiffusive Web Platform deployment..."
    
    # Pre-deployment checks
    check_ssh_key
    test_ssh_connection
    
    # Build and package
    build_frontend
    package_path=$(create_deployment_package)
    
    # Deploy
    deploy_to_remote $package_path
    
    # Clean up local package
    rm $package_path
    
    # Show status
    show_status
    
    success "Deployment completed successfully!"
    echo ""
    echo "Access the application at:"
    echo "  Local tunnel: http://localhost:8080 (via SSH tunnel)"
    echo "  Remote: http://$REMOTE_HOST:8080"
    echo ""
    echo "To create SSH tunnel:"
    echo "  ssh -p $REMOTE_PORT root@$REMOTE_HOST -L 8080:localhost:8080"
    echo ""
    echo "To check logs:"
    echo "  ssh -p $REMOTE_PORT root@$REMOTE_HOST 'cd /opt/qudiffusive/current && docker-compose logs -f'"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "status")
        test_ssh_connection
        show_status
        ;;
    "logs")
        test_ssh_connection
        ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST 'cd /opt/qudiffusive/current && docker-compose logs -f'
        ;;
    "restart")
        test_ssh_connection
        log "Restarting services..."
        ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST 'cd /opt/qudiffusive/current && docker-compose restart'
        show_status
        ;;
    "stop")
        test_ssh_connection
        log "Stopping services..."
        ssh -p $REMOTE_PORT $REMOTE_USER@$REMOTE_HOST 'cd /opt/qudiffusive/current && docker-compose down'
        ;;
    *)
        echo "Usage: $0 [deploy|status|logs|restart|stop]"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy application to remote server (default)"
        echo "  status  - Check deployment status"
        echo "  logs    - Show application logs"
        echo "  restart - Restart services"
        echo "  stop    - Stop services"
        exit 1
        ;;
esac
