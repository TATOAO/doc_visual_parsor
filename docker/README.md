# Docker Deployment Guide for PDF Layout Extraction API

This guide provides comprehensive instructions for deploying the PDF Layout Extraction API using Docker on a remote GPU machine.

## 🏗️ Architecture

The Docker setup includes:
- **Multi-stage build** for optimized image size
- **CUDA support** for GPU acceleration
- **Security hardening** with non-root user
- **Health checks** for monitoring
- **Volume mounts** for persistent data

## 📋 Prerequisites

### Local Machine
- Docker installed
- SSH access to remote GPU machine
- PDF files for testing

### Remote GPU Machine
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- SSH server running
- Sufficient disk space (~5GB for image)

## 🔧 Setup Instructions

### 1. SSH Tunnel Setup

Create an SSH tunnel to access the remote Docker daemon:

```bash
# Replace with your actual remote host details
ssh -L 5532:localhost:2376 user@your-gpu-machine.com
```

Keep this SSH session open while working with Docker.

### 2. Verify Remote Docker Access

```bash
# Test connection to remote Docker daemon
docker -H localhost:5532 version
```

### 3. Build the Docker Image

```bash
# Build with default settings
./docker/build.sh

# Build with custom tag
./docker/build.sh v1.0.0

# Build for specific remote host
./docker/build.sh latest localhost:5532
```

### 4. Deploy the Service

```bash
# Deploy with default settings
./docker/deploy.sh

# Deploy with custom port
./docker/deploy.sh latest localhost:5532 9000
```

## 🚀 Quick Start Commands

### Build and Deploy in One Go

```bash
# Build the image
./docker/build.sh

# Deploy the service
./docker/deploy.sh

# Check status
./docker/manage.sh status

# Test the API
./docker/manage.sh test 3800.pdf
```

## 🛠️ Management Commands

Use the management script for common operations:

```bash
# Show container status
./docker/manage.sh status

# View logs
./docker/manage.sh logs

# Follow logs in real-time
./docker/manage.sh follow

# Health check
./docker/manage.sh health

# Start/stop/restart
./docker/manage.sh start
./docker/manage.sh stop
./docker/manage.sh restart

# Remove container
./docker/manage.sh remove

# Test API
./docker/manage.sh test sample.pdf
```

## 🐳 Docker Compose Alternative

For easier management, use Docker Compose:

```bash
# Start service
docker-compose -H localhost:5532 up -d

# View logs
docker-compose -H localhost:5532 logs -f

# Stop service
docker-compose -H localhost:5532 down

# Restart service
docker-compose -H localhost:5532 restart
```

## 📊 Monitoring and Troubleshooting

### Health Checks

The container includes built-in health checks:

```bash
# Manual health check
curl http://localhost:8887/health

# Using management script
./docker/manage.sh health
```

### Log Monitoring

```bash
# View recent logs
./docker/manage.sh logs 100

# Follow logs in real-time
./docker/manage.sh follow

# Direct Docker command
docker -H localhost:5532 logs -f pdf-layout-api
```

### Common Issues

#### 1. Connection Refused
```bash
# Check SSH tunnel
ssh -L 5532:localhost:2376 user@remote-host

# Verify Docker daemon
docker -H localhost:5532 version
```

#### 2. GPU Not Available
```bash
# Check GPU availability
docker -H localhost:5532 run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

#### 3. Model Loading Issues
```bash
# Check container logs
./docker/manage.sh logs

# Verify model files
docker -H localhost:5532 exec pdf-layout-api ls -la /app/model_parameters/
```

#### 4. Memory Issues
```bash
# Monitor memory usage
docker -H localhost:5532 stats pdf-layout-api

# Check available memory
docker -H localhost:5532 exec pdf-layout-api free -h
```

## 🔒 Security Considerations

### Container Security
- Runs as non-root user (`appuser`)
- Minimal base image with only necessary packages
- No unnecessary network exposure
- Health checks for monitoring

### Network Security
- Use SSH tunnels for secure remote access
- Consider VPN for production deployments
- Implement proper firewall rules
- Use HTTPS in production (with reverse proxy)

## 📈 Performance Optimization

### GPU Utilization
- Ensure CUDA drivers are properly installed
- Monitor GPU usage: `nvidia-smi`
- Use appropriate batch sizes
- Consider model quantization for faster inference

### Memory Management
- Monitor container memory usage
- Adjust `max_pages` parameter for large PDFs
- Use SSD storage for better I/O performance
- Consider increasing container memory limits

### Scaling
- Use load balancer for multiple instances
- Implement request queuing for high load
- Consider horizontal scaling with multiple containers

## 🔄 Updates and Maintenance

### Updating the Image
```bash
# Build new version
./docker/build.sh v1.1.0

# Deploy new version
./docker/deploy.sh v1.1.0

# Clean up old images
docker -H localhost:5532 image prune
```

### Model Updates
```bash
# Update model files on remote host
scp new_model.onnx user@remote-host:/path/to/model_parameters/

# Restart container to load new model
./docker/manage.sh restart
```

## 📝 Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- `PYTHONUNBUFFERED`: Python output buffering
- `PYTHONDONTWRITEBYTECODE`: Python bytecode generation

### Volume Mounts
- `/app/logs`: Application logs
- `/app/model_parameters`: Model files (read-only)

### Port Configuration
- Default: `8887`
- Configurable via deploy script
- Health check endpoint: `/health`

## 🧪 Testing

### API Testing
```bash
# Test with sample PDF
./docker/manage.sh test sample.pdf

# Manual API test
curl -X POST "http://localhost:8887/extract-layout" \
     -F "file=@sample.pdf" \
     -F "confidence_threshold=0.1" \
     -F "max_pages=3"
```

### Load Testing
```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test with multiple requests
ab -n 10 -c 2 -T "multipart/form-data" \
   -p sample.pdf http://localhost:8887/extract-layout
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [CUDA Documentation](https://docs.nvidia.com/cuda/)

## 🆘 Support

For issues and questions:
1. Check container logs: `./docker/manage.sh logs`
2. Verify health status: `./docker/manage.sh health`
3. Test API functionality: `./docker/manage.sh test`
4. Review this documentation
5. Check GitHub issues for known problems
