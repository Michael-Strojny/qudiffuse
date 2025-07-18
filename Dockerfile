# QuDiffusive Web Platform Dockerfile
# Multi-stage build for production deployment

# Stage 1: Build React frontend
FROM node:18-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY web_platform/frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY web_platform/frontend/ ./

# Build the React app
RUN npm run build

# Stage 2: Python backend with QuDiffusive
FROM python:3.10-slim AS backend

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Install Python dependencies for local development
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    websockets \
    pydantic \
    pydantic-settings \
    structlog \
    redis \
    pillow \
    python-multipart \
    python-jose \
    passlib \
    python-dotenv \
    aiofiles \
    jinja2 \
    httpx \
    prometheus-client \
    psutil \
    numpy

# Copy source code
COPY web_platform/ ./web_platform/
COPY torch_placeholder.py ./
COPY start_platform.py ./

# Copy built frontend
COPY --from=frontend-builder /app/frontend/build ./web_platform/frontend/build

# Create necessary directories
RUN mkdir -p logs model_cache data

# Set ownership
RUN chown -R app:app /app

# Switch to app user
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command
CMD ["python", "-m", "web_platform.main"]
