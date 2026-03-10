#!/bin/bash
# =============================================================================
# RunPod Setup Script for Transformer Experiments
# =============================================================================
# Run this on a fresh RunPod instance after SSH-ing in.
# Tested with: RTX 4090, PyTorch 2.x template (we use TF but it works)
#
# Usage:
#   bash setup_runpod.sh
# =============================================================================

set -e

# Resolve the repo root from wherever this script lives
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

echo "============================================="
echo "  Setting up Transformer Experiment Environment"
echo "  Repo root: $REPO_ROOT"
echo "============================================="

# 0. Git
git config --global user.name "Neo Koda"
git config --global user.email "mneocicerok@gmail.com"

# 1. Install system dependencies
echo "[1/5] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git unzip wget > /dev/null 2>&1

# 2. Install Python dependencies globally
echo "[2/5] Installing Python packages..."
pip install -q --ignore-installed blinker
pip install -q tensorflow==2.15.0.post1 \
    nvidia-cudnn-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
    omegaconf hydra-core wandb matplotlib tensorboard

# Add NVIDIA pip libraries to LD_LIBRARY_PATH so TF finds the GPU
NV_LIB_PATH="/usr/local/lib/python3.11/dist-packages/nvidia"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NV_LIB_PATH/cudnn/lib:$NV_LIB_PATH/cublas/lib:$NV_LIB_PATH/cuda_nvrtc/lib:$NV_LIB_PATH/cuda_runtime/lib
if ! grep -q "nvidia/cudnn/lib" ~/.bashrc; then
    echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$NV_LIB_PATH/cudnn/lib:$NV_LIB_PATH/cublas/lib:$NV_LIB_PATH/cuda_nvrtc/lib:$NV_LIB_PATH/cuda_runtime/lib" >> ~/.bashrc
fi

# 3. Check repo exists
if [ ! -d "$REPO_ROOT/NeuralDecoder" ]; then
    echo "  ERROR: NeuralDecoder directory not found at $REPO_ROOT/NeuralDecoder"
    echo "  Please ensure setup_runpod.sh is in the speechBCI repo root."
    exit 1
fi

# 4. Install NeuralDecoder package (--no-deps to avoid pulling conflicting transitive deps)
echo "[4/5] Installing NeuralDecoder package..."
cd "$REPO_ROOT/NeuralDecoder"
pip install -e . --no-deps
cd "$REPO_ROOT"

# 5. Verify setup
echo "[5/5] Verifying setup..."
python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {len(gpus)}')
for gpu in gpus:
    print(f'  {gpu}')

# Test TransformerEncoder import
from neuralDecoder.models import TransformerEncoder
model = TransformerEncoder(d_model=256, nhead=4, num_layers=2, d_ff=512, nClasses=41, dropout=0.1,
                           stack_kwargs={'kernel_size': 32, 'strides': 4})
x = tf.random.normal([2, 200, 256])
y = model(x, training=False)
print(f'TransformerEncoder smoke test: input {x.shape} -> output {y.shape}')
print('Model params:', model.count_params())
print()
print('Setup complete! Ready to run experiments.')
"

echo ""
echo "============================================="
echo "  Setup complete!"
echo "============================================="
