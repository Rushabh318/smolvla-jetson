# Edge SmolVLA Inference - Dockerfile for Jetson Orin Nano
# Base image: NVIDIA L4T PyTorch container for JetPack 6.x

# ============================================
# Base Image
# ============================================
# Use the official NVIDIA L4T PyTorch image
# This includes PyTorch with CUDA support pre-installed
ARG L4T_VERSION=r36.4.0
ARG PYTORCH_VERSION=2.5
FROM nvcr.io/nvidia/l4t-pytorch:${L4T_VERSION}-pth${PYTORCH_VERSION}-py3

# ============================================
# Environment Variables
# ============================================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# MuJoCo rendering backend (EGL for headless GPU rendering)
ENV MUJOCO_GL=egl

# HuggingFace cache location
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# ============================================
# System Dependencies
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # MuJoCo dependencies
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglfw3-dev \
    libglew-dev \
    # General utilities
    git \
    wget \
    curl \
    vim \
    # Build tools (for some pip packages)
    build-essential \
    cmake \
    # Video encoding (for recording)
    ffmpeg \
    # Clean up
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# Working Directory
# ============================================
WORKDIR /app

# ============================================
# Python Dependencies
# ============================================
# Copy requirements first for better caching
COPY requirements.txt .

# Install Python packages
# Note: PyTorch is already installed in base image
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# MuJoCo Menagerie Setup
# ============================================
RUN mkdir -p third_party && \
    cd third_party && \
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git

# ============================================
# Application Code
# ============================================
# Copy application code
COPY configs/ ./configs/
COPY simulation/ ./simulation/
COPY models/ ./models/
COPY control/ ./control/
COPY inference/ ./inference/
COPY benchmarking/ ./benchmarking/
COPY visualization/ ./visualization/
COPY utils/ ./utils/
COPY main.py .
COPY CLAUDE.md .

# ============================================
# Create Runtime Directories
# ============================================
RUN mkdir -p logs outputs outputs/videos .cache/huggingface

# ============================================
# Pre-download Model (Optional)
# ============================================
# Uncomment to include model in image (adds ~2GB)
# RUN python -c "from transformers import AutoModel; AutoModel.from_pretrained('lerobot/smolvla_base')"

# ============================================
# Health Check
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; import mujoco; print('OK')" || exit 1

# ============================================
# Entrypoint
# ============================================
# Default command runs the main inference script
CMD ["python", "main.py"]

# ============================================
# Labels
# ============================================
LABEL maintainer="your-email@example.com"
LABEL description="SmolVLA inference optimized for Jetson Orin Nano"
LABEL version="1.0.0"

# ============================================
# Build Instructions
# ============================================
# Build:
#   docker build -t smolvla-jetson:latest .
#
# Run (with GPU):
#   docker run --runtime nvidia --gpus all \
#       -v $(pwd)/outputs:/app/outputs \
#       -v $(pwd)/logs:/app/logs \
#       smolvla-jetson:latest
#
# Run with visualization (requires X11):
#   docker run --runtime nvidia --gpus all \
#       -e DISPLAY=$DISPLAY \
#       -v /tmp/.X11-unix:/tmp/.X11-unix \
#       -v $(pwd)/outputs:/app/outputs \
#       smolvla-jetson:latest python main.py --visualize
#
# Interactive shell:
#   docker run --runtime nvidia --gpus all -it smolvla-jetson:latest bash