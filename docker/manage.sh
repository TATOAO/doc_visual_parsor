#!/bin/bash
# Management script for PDF Layout Extraction API
# Usage: ./manage.sh [command] [options]

set -e

# Configuration
IMAGE_NAME="pdf-layout-extraction"
CONTAINER_NAME="pdf-layout-api"
DEFAULT_REMOTE_HOST="localhost:5532"
DEFAULT_PORT="8887"

REMOTE_HOST=${REMOTE_HOST:-$DEFAULT_REMOTE_HOST}
PORT=${PORT:-$DEFAULT_PORT}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Docker connection
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! docker -H ${REMOTE_HOST} version &> /dev/null; then
        log_error "Cannot connect to remote Docker daemon at ${REMOTE_HOST}"
        log_info "Make sure the remote machine is running and accessible"
        log_info "Check your SSH tunnel: ssh -L 5532:localhost:2376 user@remote-host"
        exit 1
    fi
}

# Show status
show_status() {
    log_info "Container Status:"
    docker -H ${REMOTE_HOST} ps -f name=${CONTAINER_NAME} --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo ""
    log_info "Image Information:"
    docker -H ${REMOTE_HOST} images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
}

# Show logs
show_logs() {
    local lines=${1:-50}
    log_info "Showing last ${lines} lines of logs:"
    docker -H ${REMOTE_HOST} logs --tail ${lines} ${CONTAINER_NAME}
}

# Follow logs
follow_logs() {
    log_info "Following logs (Ctrl+C to stop):"
    docker -H ${REMOTE_HOST} logs -f ${CONTAINER_NAME}
}

# Health check
health_check() {
    log_info "Performing health check..."
    if curl -f http://localhost:${PORT}/health &> /dev/null; then
        log_success "Service is healthy!"
        curl -s http://localhost:${PORT}/health | jq . 2>/dev/null || curl -s http://localhost:${PORT}/health
    else
        log_error "Service is not responding"
        log_info "Check if the container is running: docker -H ${REMOTE_HOST} ps -f name=${CONTAINER_NAME}"
    fi
}

# Stop container
stop_container() {
    log_info "Stopping container..."
    if docker -H ${REMOTE_HOST} ps -q -f name=${CONTAINER_NAME} | grep -q .; then
        docker -H ${REMOTE_HOST} stop ${CONTAINER_NAME}
        log_success "Container stopped"
    else
        log_warning "Container is not running"
    fi
}

# Start container
start_container() {
    log_info "Starting container..."
    if docker -H ${REMOTE_HOST} ps -aq -f name=${CONTAINER_NAME} | grep -q .; then
        docker -H ${REMOTE_HOST} start ${CONTAINER_NAME}
        log_success "Container started"
    else
        log_error "Container does not exist. Please deploy first."
        exit 1
    fi
}

# Restart container
restart_container() {
    log_info "Restarting container..."
    docker -H ${REMOTE_HOST} restart ${CONTAINER_NAME}
    log_success "Container restarted"
}

# Remove container
remove_container() {
    log_info "Removing container..."
    stop_container
    if docker -H ${REMOTE_HOST} ps -aq -f name=${CONTAINER_NAME} | grep -q .; then
        docker -H ${REMOTE_HOST} rm ${CONTAINER_NAME}
        log_success "Container removed"
    else
        log_warning "Container does not exist"
    fi
}

# Test API
test_api() {
    local pdf_file=${1:-"3800.pdf"}
    
    if [ ! -f "${pdf_file}" ]; then
        log_error "PDF file not found: ${pdf_file}"
        log_info "Usage: $0 test [pdf_file]"
        exit 1
    fi
    
    log_info "Testing API with file: ${pdf_file}"
    
    # Health check first
    if ! curl -f http://localhost:${PORT}/health &> /dev/null; then
        log_error "Service is not healthy. Please check the container status."
        exit 1
    fi
    
    # Test extraction
    log_info "Sending PDF for layout extraction..."
    response=$(curl -s -X POST "http://localhost:${PORT}/extract-layout" \
        -F "file=@${pdf_file}" \
        -F "confidence_threshold=0.1" \
        -F "max_pages=2")
    
    if echo "${response}" | jq -e '.success' &> /dev/null; then
        log_success "API test successful!"
        echo "${response}" | jq '.message, .metadata'
    else
        log_error "API test failed!"
        echo "${response}"
    fi
}

# Show help
show_help() {
    echo "PDF Layout Extraction API Management Script"
    echo "=========================================="
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  status          Show container and image status"
    echo "  logs [lines]    Show container logs (default: 50 lines)"
    echo "  follow          Follow container logs in real-time"
    echo "  health          Perform health check"
    echo "  start           Start the container"
    echo "  stop            Stop the container"
    echo "  restart         Restart the container"
    echo "  remove          Stop and remove the container"
    echo "  test [pdf]      Test API with a PDF file"
    echo "  help            Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  REMOTE_HOST     Docker remote host (default: localhost:5532)"
    echo "  PORT            API port (default: 8887)"
    echo ""
    echo "Examples:"
    echo "  $0 status"
    echo "  $0 logs 100"
    echo "  $0 test sample.pdf"
    echo "  REMOTE_HOST=localhost:5532 $0 health"
}

# Main script
main() {
    local command=${1:-"help"}
    
    case $command in
        "status")
            check_docker
            show_status
            ;;
        "logs")
            check_docker
            show_logs $2
            ;;
        "follow")
            check_docker
            follow_logs
            ;;
        "health")
            health_check
            ;;
        "start")
            check_docker
            start_container
            ;;
        "stop")
            check_docker
            stop_container
            ;;
        "restart")
            check_docker
            restart_container
            ;;
        "remove")
            check_docker
            remove_container
            ;;
        "test")
            test_api $2
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
