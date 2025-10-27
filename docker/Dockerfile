# Multi-stage Dockerfile for PDF Layout Extraction API
# Optimized for GPU deployment with CUDA support

# Stage 1: Base image with CUDA support
# Alternative base images (uncomment one if needed):
# FROM nvidia/cuda:11.8-devel-ubuntu22.04 AS base
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/anujkosambi/python-onnxruntime-gpu-cuda:2.0.0 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0


# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libfreetype6-dev \
    libharfbuzz-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libtiff-dev \
    libjbig2dec0-dev \
    libz-dev \
    tesseract-ocr \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip
RUN python -m pip install --upgrade pip

# Stage 2: Python dependencies
FROM base AS python-deps

RUN apt-get update && apt-get install -y build-essential

RUN pip install --upgrade pip setuptools wheel 
# Install uv for faster package management
RUN pip install uv

# Copy project files

RUN uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --system numpy==1.26.4 fastapi==0.104.0 uvicorn==0.24.0 python-multipart==0.0.6 paddleocr
RUN uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --system onnxruntime>=1.15.0
RUN uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --system onnxruntime-gpu>=1.15.0
RUN uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --system PyMuPDF==1.25.1



WORKDIR /app
COPY pyproject.toml ./
COPY requirements.txt ./

RUN uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --system -r requirements.txt

# Install Python dependencies
RUN uv pip install --system .[gpu]

# Stage 3: Final image
FROM base AS final

# Copy Python dependencies from previous stage
COPY --from=python-deps /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=python-deps /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy application code
COPY doc_chunking/ ./doc_chunking/
COPY model_parameters/ ./model_parameters/

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8887

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8887/health || exit 1

# Default command
CMD ["python", "-m", "doc_chunking.app.app"]
