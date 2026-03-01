#!/bin/bash

# Stop on error
set -e

echo "Starting RunPod Setup..."

# 1. System Dependencies
echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    cmake \
    build-essential \
    libsndfile1 \
    ffmpeg \
    unzip \
    htop \
    tmux \
    vim

# 2. Python Environment (assuming Base RunPod image has Python/Conda)
# If not, we might need to install miniconda. Most RunPod images like PyTorch ones have it.
echo "Upgrading pip..."
pip install --upgrade pip

# 3. Install Python Requirements
echo "Installing Python libraries..."
pip install \
    torch torchaudio torchvision \
    tensorflow \
    transformers \
    peft \
    accelerate \
    bitsandbytes \
    hydra-core \
    omegaconf \
    jupyter \
    matplotlib \
    scipy \
    pandas \
    scikit-learn \
    soundfile \
    wandb

# 4. Clone/Setup LanguageModelDecoder (if needed for compilation)
# Note: The original repo instructions say LanguageModelDecoder needs compilation.
# We might need to handle this carefully if using the C++ decoder.
# For now, we focus on the Python dependencies.

echo "Setup Complete! You can now run 'jupyter lab' or python scripts."
