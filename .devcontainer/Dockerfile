FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Set UTF-8 locale
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Python, pip, and uv (minimal footprint)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv curl ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip uv -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /workspace

COPY requirements.txt .
COPY doc_chunking/ ./doc_chunking/
COPY model_parameters/ ./model_parameters/

