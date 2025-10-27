#!/bin/bash
# Deployment script for PDF Layout Extraction API
# Usage: ./deploy.sh [tag] [remote_host]

set -e

# Configuration
IMAGE_NAME="pdf-layout-extraction"
CONTAINER_NAME="pdf-layout-api"
DEFAULT_TAG="latest"
DEFAULT_REMOTE_HOST="localhost:5532"
DEFAULT_PORT="8887"

# Get parameters
TAG=${1:-$DEFAULT_TAG}
REMOTE_HOST=${2:-$DEFAULT_REMOTE_HOST}
PORT=${3:-$DEFAULT_PORT}

echo "🚀 Deploying PDF Layout Extraction API"
echo "======================================"
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "Container: ${CONTAINER_NAME}"
echo "Remote Host: ${REMOTE_HOST}"
echo "Port: ${PORT}"
echo ""

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Check if remote Docker daemon is accessible
echo "🔍 Checking remote Docker daemon..."
if ! docker -H ${REMOTE_HOST} version &> /dev/null; then
    echo "❌ Cannot connect to remote Docker daemon at ${REMOTE_HOST}"
    echo "   Make sure the remote machine is running and accessible"
    echo "   Check your SSH tunnel: ssh -L 5532:localhost:2376 user@remote-host"
    exit 1
fi
echo "✅ Remote Docker daemon is accessible"

# Check if image exists
echo "🔍 Checking if image exists..."
if ! docker -H ${REMOTE_HOST} image inspect ${IMAGE_NAME}:${TAG} &> /dev/null; then
    echo "❌ Image ${IMAGE_NAME}:${TAG} not found"
    echo "   Please build the image first: ./build.sh ${TAG} ${REMOTE_HOST}"
    exit 1
fi
echo "✅ Image found"

# Stop and remove existing container if it exists
echo "🛑 Stopping existing container (if any)..."
if docker -H ${REMOTE_HOST} ps -q -f name=${CONTAINER_NAME} | grep -q .; then
    docker -H ${REMOTE_HOST} stop ${CONTAINER_NAME}
    echo "✅ Existing container stopped"
fi

if docker -H ${REMOTE_HOST} ps -aq -f name=${CONTAINER_NAME} | grep -q .; then
    docker -H ${REMOTE_HOST} rm ${CONTAINER_NAME}
    echo "✅ Existing container removed"
fi

# Create logs directory on remote host
echo "📁 Creating logs directory..."
docker -H ${REMOTE_HOST} run --rm -v /tmp:/host-tmp alpine mkdir -p /host-tmp/pdf-api-logs || true

# Run the container
echo "🚀 Starting new container..."
docker -H ${REMOTE_HOST} run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    -p ${PORT}:8887 \
    -v /tmp/pdf-api-logs:/app/logs \
    --restart unless-stopped \
    ${IMAGE_NAME}:${TAG}

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Container started successfully!"
    echo ""
    echo "📋 Container Information:"
    echo "   Name: ${CONTAINER_NAME}"
    echo "   Image: ${IMAGE_NAME}:${TAG}"
    echo "   Port: ${PORT}"
    echo "   Status: Running"
    echo ""
    echo "🔍 Monitoring Commands:"
    echo "   View logs: docker -H ${REMOTE_HOST} logs -f ${CONTAINER_NAME}"
    echo "   Check status: docker -H ${REMOTE_HOST} ps -f name=${CONTAINER_NAME}"
    echo "   Health check: curl http://localhost:${PORT}/health"
    echo ""
    echo "🌐 API Endpoints:"
    echo "   Health: http://localhost:${PORT}/health"
    echo "   API Docs: http://localhost:${PORT}/docs"
    echo "   Extract Layout: http://localhost:${PORT}/extract-layout"
    echo ""
    echo "⏳ Waiting for service to be ready..."
    sleep 10
    
    # Health check
    echo "🔍 Performing health check..."
    if curl -f http://localhost:${PORT}/health &> /dev/null; then
        echo "✅ Service is healthy and ready!"
    else
        echo "⚠️  Service may still be starting up. Check logs:"
        echo "   docker -H ${REMOTE_HOST} logs ${CONTAINER_NAME}"
    fi
else
    echo ""
    echo "❌ Failed to start container!"
    exit 1
fi
