#!/bin/bash

# QuDiffusive Web Platform - Simple Local Runner
# Runs the platform locally without Docker for development

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

# Check Python dependencies
check_dependencies() {
    log "🐍 Checking Python dependencies..."
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
        error "Python is not installed. Please install Python 3.10+ first."
        exit 1
    fi
    
    # Try to import required packages
    python_cmd="python3"
    if ! command -v python3 &> /dev/null; then
        python_cmd="python"
    fi
    
    missing_packages=()
    
    packages=("fastapi" "uvicorn" "websockets" "pydantic" "structlog" "numpy" "PIL")
    
    for package in "${packages[@]}"; do
        if ! $python_cmd -c "import $package" &> /dev/null; then
            missing_packages+=("$package")
        fi
    done
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        warning "Missing packages: ${missing_packages[*]}"
        log "Installing missing packages..."
        
        pip_packages="fastapi uvicorn websockets pydantic pydantic-settings structlog numpy pillow python-multipart psutil"
        
        if command -v pip3 &> /dev/null; then
            pip3 install $pip_packages
        elif command -v pip &> /dev/null; then
            pip install $pip_packages
        else
            error "pip is not available. Please install pip first."
            exit 1
        fi
    fi
    
    success "Python dependencies are available"
}

# Build frontend if needed
build_frontend() {
    log "🏗️ Checking frontend build..."
    
    if [ -d "web_platform/frontend" ]; then
        cd web_platform/frontend
        
        if [ ! -f "package.json" ]; then
            warning "package.json not found, skipping frontend build"
            cd ../..
            return
        fi
        
        # Check if build exists and is newer than package.json
        if [ -d "build" ] && [ "build" -nt "package.json" ]; then
            success "Frontend build is up to date"
            cd ../..
            return
        fi
        
        if command -v npm &> /dev/null; then
            log "Installing frontend dependencies..."
            npm ci
            
            log "Building frontend..."
            npm run build
            
            if [ ! -d "build" ]; then
                error "Frontend build failed"
                exit 1
            fi
            
            success "Frontend built successfully"
        else
            warning "npm not found, skipping frontend build"
        fi
        
        cd ../..
    else
        warning "Frontend directory not found"
    fi
}

# Create necessary directories
setup_directories() {
    log "📁 Setting up directories..."
    
    mkdir -p logs
    mkdir -p model_cache
    mkdir -p data
    
    success "Directories created"
}

# Set environment variables
setup_environment() {
    log "⚙️ Setting up environment..."
    
    export ENVIRONMENT=development
    export HOST=0.0.0.0
    export PORT=8080
    export SECRET_KEY=local-development-secret-key
    export MODEL_DEVICE=cpu
    export LOG_LEVEL=INFO
    export DEBUG=true
    
    success "Environment configured"
}

# Start Redis if available
start_redis() {
    log "🔴 Checking Redis..."
    
    if command -v redis-server &> /dev/null; then
        # Check if Redis is already running
        if redis-cli ping &> /dev/null; then
            success "Redis is already running"
            export REDIS_URL=redis://localhost:6379
        else
            log "Starting Redis server..."
            redis-server --daemonize yes --port 6379
            sleep 2
            
            if redis-cli ping &> /dev/null; then
                success "Redis started successfully"
                export REDIS_URL=redis://localhost:6379
            else
                warning "Failed to start Redis, using in-memory storage"
                export REDIS_URL=""
            fi
        fi
    else
        warning "Redis not found, using in-memory storage"
        export REDIS_URL=""
    fi
}

# Start the web platform
start_platform() {
    log "🚀 Starting QuDiffusive Web Platform..."
    
    # Use the startup script with debug mode
    python start_platform.py --debug --host 0.0.0.0 --port 8080 &
    PLATFORM_PID=$!
    
    # Wait for the platform to start
    log "⏳ Waiting for platform to be ready..."
    
    timeout=60
    while [ $timeout -gt 0 ]; do
        if curl -f http://localhost:8080/health &> /dev/null; then
            success "Platform is ready!"
            break
        fi
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        error "Platform failed to start within timeout"
        kill $PLATFORM_PID 2>/dev/null || true
        exit 1
    fi
    
    echo $PLATFORM_PID > .platform.pid
}

# Show status
show_status() {
    log "📊 Platform Status"
    echo "=================="
    
    echo ""
    echo "🌐 Application URLs:"
    echo "  Main Application:    http://localhost:8080/"
    echo "  API Documentation:   http://localhost:8080/docs"
    echo "  Health Check:        http://localhost:8080/health"
    
    echo ""
    echo "🔍 Health Checks:"
    
    if curl -f http://localhost:8080/health &> /dev/null; then
        echo "  ✅ Web Application: OK"
    else
        echo "  ❌ Web Application: FAILED"
    fi
    
    if [ -n "$REDIS_URL" ] && redis-cli ping &> /dev/null; then
        echo "  ✅ Redis: OK"
    else
        echo "  ⚠️  Redis: Not available (using in-memory storage)"
    fi
    
    echo ""
    echo "💡 Tips:"
    echo "  - Press Ctrl+C to stop the platform"
    echo "  - Use './run_local.sh stop' to stop background services"
    echo "  - Use './run_local.sh test' to run integration tests"
}

# Stop services
stop_services() {
    log "🛑 Stopping services..."
    
    if [ -f ".platform.pid" ]; then
        PID=$(cat .platform.pid)
        if kill -0 $PID 2>/dev/null; then
            kill $PID
            success "Platform stopped"
        fi
        rm -f .platform.pid
    fi
    
    # Stop Redis if we started it
    if command -v redis-cli &> /dev/null; then
        redis-cli shutdown 2>/dev/null || true
    fi
    
    success "All services stopped"
}

# Run tests
run_tests() {
    log "🧪 Running integration tests..."
    
    if python test_web_platform.py --url http://localhost:8080; then
        success "Integration tests passed"
    else
        warning "Some integration tests failed"
    fi
}

# Main function
main() {
    log "🚀 Starting QuDiffusive Web Platform (Local Mode)"
    echo "================================================="
    
    # Setup
    check_dependencies
    build_frontend
    setup_directories
    setup_environment
    start_redis
    
    # Start platform
    start_platform
    
    # Show status
    show_status
    
    success "🎉 Platform is running locally!"
    echo ""
    echo "🌟 Access QuDiffusive Web Platform at: http://localhost:8080"
    echo ""
    echo "Press Ctrl+C to stop the platform..."
    
    # Wait for interrupt
    trap 'stop_services; exit 0' INT TERM
    
    # Keep the script running
    if [ -f ".platform.pid" ]; then
        PID=$(cat .platform.pid)
        wait $PID 2>/dev/null || true
    fi
}

# Handle command line arguments
case "${1:-run}" in
    "run")
        main
        ;;
    "test")
        check_dependencies
        setup_environment
        start_redis
        start_platform
        run_tests
        stop_services
        ;;
    "stop")
        stop_services
        ;;
    "status")
        show_status
        ;;
    *)
        echo "Usage: $0 [run|test|stop|status]"
        echo ""
        echo "Commands:"
        echo "  run     - Start the platform locally (default)"
        echo "  test    - Start platform and run tests"
        echo "  stop    - Stop all services"
        echo "  status  - Show platform status"
        exit 1
        ;;
esac
