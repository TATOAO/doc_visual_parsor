FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime AS base

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Set UTF-8 locale
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install Python, pip, and uv (minimal footprint)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git curl ca-certificates tar && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip uv -i https://pypi.tuna.tsinghua.edu.cn/simple

# ------------------------------
# Pre-bake Cursor server binary
# ------------------------------
# 1️⃣ Create the directory Cursor expects
RUN mkdir -p /root/.cursor-server

# 2️⃣ Download and extract the Cursor server binary during build
# This prevents Cursor from trying to download it when the container starts
RUN CURSOR_VERSION="5c17eb2968a37f66bc6662f48d6356a100b67be8" && \
    CURSOR_URL="https://downloads.cursor.com/production/${CURSOR_VERSION}/linux/x64/cursor-reh-linux-x64.tar.gz" && \
    curl -L "${CURSOR_URL}" -o /tmp/cursor-reh-linux-x64.tar.gz && \
    tar -xzf /tmp/cursor-reh-linux-x64.tar.gz -C /root/.cursor-server/ && \
    rm /tmp/cursor-reh-linux-x64.tar.gz && \
    chmod +x /root/.cursor-server/cursor-reh-linux-x64/bin/cursor-reh || \
    chmod +x /root/.cursor-server/cursor-reh-linux-x64/cursor-reh || true

# Alternative: If you have the tar.gz file locally, uncomment the following and comment the above RUN command:
# COPY cursor-reh-linux-x64.tar.gz /tmp/
# RUN tar -xzf /tmp/cursor-reh-linux-x64.tar.gz -C /root/.cursor-server/ && \
#     rm /tmp/cursor-reh-linux-x64.tar.gz && \
#     chmod +x /root/.cursor-server/cursor-reh-linux-x64/bin/cursor-reh || \
#     chmod +x /root/.cursor-server/cursor-reh-linux-x64/cursor-reh || true

WORKDIR /workspace

# Copy requirements.txt first before using it
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN uv venv && \
    uv pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# Copy the rest of the project files
COPY . .

