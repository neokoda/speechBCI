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

echo "============================================="
echo "  Setting up Transformer Experiment Environment"
echo "============================================="

# 0. Git
git config --global user.name "Neo Koda"
git config --global user.email "mneocicerok@gmail.com"

# 1. Install system dependencies
echo "[1/5] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq git unzip wget > /dev/null 2>&1

# 2. Install Python dependencies globally
echo "[2/5] Installing Python packages..."
pip install -q tensorflow==2.12.0 \
    numpy scipy omegaconf hydra-core wandb matplotlib tensorboard

WORKSPACE=/workspace

# If the repo isn't already here, you need to upload it or clone it
if [ ! -d "speechBCI" ]; then
    echo "  NOTE: Please upload/clone your speechBCI repo to /workspace/speechBCI"
    echo "  You can use: scp -r /local/path/to/speechBCI root@<runpod-ip>:/workspace/"
    echo "  Or: git clone <your-repo-url> speechBCI"
fi

# 4. Install NeuralDecoder package
echo "[4/5] Installing NeuralDecoder package..."
if [ -d "speechBCI/NeuralDecoder" ]; then
    cd speechBCI/NeuralDecoder
    pip install -e . 2>/dev/null || pip install -e . --no-deps
    cd $WORKSPACE
fi

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
echo ""
echo "To run Round 1 experiments:"
echo "  cd /workspace/speechBCI"
echo "  python AnalysisExamples/run_round1_experiments.py \\"
echo "    --data-dir /workspace/speechBCI/data/derived/tfRecords \\"
echo "    --output-dir /workspace/speechBCI/experiments/round1 \\"
echo "    --gpu 0"
echo ""
