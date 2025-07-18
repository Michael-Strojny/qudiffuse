#!/bin/bash

# QuDiffusive Web Platform Local Deployment Script
# Deploys the complete platform locally using Docker Compose

set -e

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

# Check if Docker is installed and running
check_docker() {
    log "🐳 Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    success "Docker and Docker Compose are available"
}

# Build React frontend
build_frontend() {
    log "🏗️ Building React frontend..."
    
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

# Create necessary directories
create_directories() {
    log "📁 Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p model_cache
    mkdir -p data
    
    # Set proper permissions
    chmod 755 logs model_cache data
    
    success "Directories created"
}

# Generate environment file if it doesn't exist
setup_environment() {
    log "⚙️ Setting up environment..."
    
    if [ ! -f ".env.local" ]; then
        warning ".env.local not found, creating default configuration"
        
        cat > .env.local << EOF
# QuDiffusive Web Platform - Local Environment Configuration
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8080
SECRET_KEY=$(openssl rand -hex 32)
MODEL_DEVICE=cpu
REDIS_URL=redis://redis:6379
LOG_LEVEL=INFO
DEBUG=false
EOF
    fi
    
    success "Environment configured"
}

# Stop existing services
stop_services() {
    log "🛑 Stopping existing services..."
    
    if [ -f "docker-compose.yml" ]; then
        docker-compose down --remove-orphans || true
    fi
    
    # Clean up any orphaned containers
    docker container prune -f || true
    
    success "Existing services stopped"
}

# Build and start services
start_services() {
    log "🚀 Building and starting services..."
    
    # Build the application
    log "Building QuDiffusive web application..."
    docker-compose build --no-cache
    
    # Start services
    log "Starting services..."
    docker-compose up -d
    
    success "Services started"
}

# Wait for services to be ready
wait_for_services() {
    log "⏳ Waiting for services to be ready..."
    
    # Wait for Redis
    log "Waiting for Redis..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker-compose exec -T redis redis-cli ping &> /dev/null; then
            success "Redis is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        error "Redis failed to start within timeout"
        exit 1
    fi
    
    # Wait for web application
    log "Waiting for web application..."
    timeout=120
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8080/health &> /dev/null; then
            success "Web application is ready"
            break
        fi
        sleep 3
        timeout=$((timeout - 3))
    done
    
    if [ $timeout -le 0 ]; then
        error "Web application failed to start within timeout"
        echo "Check logs with: docker-compose logs qudiffusive-web"
        exit 1
    fi
    
    # Wait for Nginx
    log "Waiting for Nginx..."
    timeout=30
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost/ &> /dev/null; then
            success "Nginx is ready"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        warning "Nginx may not be ready, but continuing..."
    fi
}

# Show deployment status
show_status() {
    log "📊 Deployment Status"
    echo "===================="
    
    echo ""
    echo "🐳 Docker Services:"
    docker-compose ps
    
    echo ""
    echo "🌐 Application URLs:"
    echo "  Main Application:    http://localhost/"
    echo "  Direct Backend:      http://localhost:8080/"
    echo "  API Documentation:   http://localhost:8080/docs"
    echo "  Health Check:        http://localhost:8080/health"
    echo "  Redis:              localhost:6379"
    
    echo ""
    echo "🔍 Health Checks:"
    
    # Test main application
    if curl -f http://localhost/ &> /dev/null; then
        echo "  ✅ Main Application (Nginx): OK"
    else
        echo "  ❌ Main Application (Nginx): FAILED"
    fi
    
    # Test backend directly
    if curl -f http://localhost:8080/health &> /dev/null; then
        echo "  ✅ Backend API: OK"
    else
        echo "  ❌ Backend API: FAILED"
    fi
    
    # Test Redis
    if docker-compose exec -T redis redis-cli ping &> /dev/null; then
        echo "  ✅ Redis: OK"
    else
        echo "  ❌ Redis: FAILED"
    fi
    
    echo ""
    echo "📝 Useful Commands:"
    echo "  View logs:           docker-compose logs -f"
    echo "  Stop services:       docker-compose down"
    echo "  Restart services:    docker-compose restart"
    echo "  Rebuild:             docker-compose build --no-cache"
    echo "  Shell access:        docker-compose exec qudiffusive-web bash"
}

# Run tests
run_tests() {
    log "🧪 Running integration tests..."
    
    # Wait a bit more for services to stabilize
    sleep 5
    
    if python test_web_platform.py --url http://localhost:8080; then
        success "Integration tests passed"
    else
        warning "Some integration tests failed, but deployment may still be functional"
    fi
}

# Main deployment function
main() {
    log "🚀 Starting QuDiffusive Web Platform Local Deployment"
    echo "======================================================"
    
    # Pre-deployment checks
    check_docker
    
    # Build and setup
    build_frontend
    create_directories
    setup_environment
    
    # Deploy
    stop_services
    start_services
    wait_for_services
    
    # Verify deployment
    show_status
    
    # Run tests if requested
    if [ "${1:-}" = "--test" ]; then
        run_tests
    fi
    
    success "🎉 Local deployment completed successfully!"
    echo ""
    echo "🌟 QuDiffusive Web Platform is now running locally!"
    echo "   Access it at: http://localhost/"
    echo ""
    echo "💡 Tips:"
    echo "   - Use 'docker-compose logs -f' to view real-time logs"
    echo "   - Use 'docker-compose down' to stop all services"
    echo "   - Use './deploy_local.sh --test' to run integration tests"
}

# Handle command line arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "test")
        main --test
        ;;
    "status")
        show_status
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "stop")
        log "🛑 Stopping all services..."
        docker-compose down
        success "All services stopped"
        ;;
    "restart")
        log "🔄 Restarting services..."
        docker-compose restart
        show_status
        ;;
    "rebuild")
        log "🔨 Rebuilding and restarting..."
        docker-compose down
        docker-compose build --no-cache
        docker-compose up -d
        wait_for_services
        show_status
        ;;
    "clean")
        log "🧹 Cleaning up Docker resources..."
        docker-compose down --volumes --remove-orphans
        docker system prune -f
        success "Cleanup completed"
        ;;
    *)
        echo "Usage: $0 [deploy|test|status|logs|stop|restart|rebuild|clean]"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy application locally (default)"
        echo "  test     - Deploy and run integration tests"
        echo "  status   - Show deployment status"
        echo "  logs     - Show application logs"
        echo "  stop     - Stop all services"
        echo "  restart  - Restart services"
        echo "  rebuild  - Rebuild and restart services"
        echo "  clean    - Clean up all Docker resources"
        exit 1
        ;;
esac
