#!/bin/bash
# Build script for PDF Layout Extraction API Docker image
# Usage: ./build.sh [tag] [remote_host]

set -e

# Configuration
IMAGE_NAME="pdf-layout-extraction"
DEFAULT_TAG="latest"
DEFAULT_REMOTE_HOST="localhost:5532"

# Get parameters
TAG=${1:-$DEFAULT_TAG}
REMOTE_HOST=${2:-$DEFAULT_REMOTE_HOST}

echo "🐳 Building PDF Layout Extraction API Docker Image"
echo "=================================================="
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "Remote Host: ${REMOTE_HOST}"
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

# Build the image
echo ""
echo "🔨 Building Docker image..."
echo "This may take several minutes for the first build..."

docker -H ${REMOTE_HOST} build \
    -f docker/Dockerfile \
    -t ${IMAGE_NAME}:${TAG} \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Docker image built successfully!"
    echo "   Image: ${IMAGE_NAME}:${TAG}"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Deploy with: ./deploy.sh ${TAG} ${REMOTE_HOST}"
    echo "   2. Or use docker-compose: docker-compose -H ${REMOTE_HOST} up -d"
    echo "   3. Check logs: docker -H ${REMOTE_HOST} logs pdf-layout-api"
else
    echo ""
    echo "❌ Docker build failed!"
    exit 1
fi
