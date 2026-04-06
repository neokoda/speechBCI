#!/bin/bash
# =============================================================================
# Setup Script for SpeechBCI Experiments (vast.ai / RunPod)
# =============================================================================
# Installs Python 3.11 + TensorFlow 2.15 in a venv at /workspace/venv311.
# TF 2.15 requires Python 3.8-3.11 (system Python 3.12 is incompatible).
#
# Usage:
#   bash setup_runpod.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
VENV="/workspace/venv311"

echo "============================================="
echo "  SpeechBCI Setup — Python 3.11 + TF 2.15"
echo "  Repo root : $REPO_ROOT"
echo "  Venv path : $VENV"
echo "============================================="

# 0. Git identity
git config --global user.name  "Neo Koda"
git config --global user.email "mneocicerok@gmail.com"

# 1. System deps + Python 3.11
echo "[1/6] Installing system deps and Python 3.11..."
apt-get update -qq
apt-get install -y -qq git unzip wget software-properties-common > /dev/null 2>&1
add-apt-repository -y ppa:deadsnakes/ppa > /dev/null 2>&1
apt-get update -qq
apt-get install -y -qq python3.11 python3.11-venv python3.11-dev > /dev/null 2>&1
echo "  Python 3.11: $(python3.11 --version)"

# 2. Create venv
echo "[2/6] Creating venv at $VENV..."
python3.11 -m venv "$VENV"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
$PIP install -q --upgrade pip setuptools wheel

# 3. Install TensorFlow 2.15 + CUDA libs
echo "[3/6] Installing TensorFlow 2.15..."
$PIP install -q --no-cache-dir \
    "tensorflow[and-cuda]==2.15.*" \
    "numpy<2" \
    omegaconf \
    hydra-core \
    wandb \
    matplotlib \
    tensorboard \
    scipy

# 4. Install PyTorch + HuggingFace (for LM pipeline)
echo "[4/6] Installing PyTorch + HuggingFace transformers..."
$PIP install -q --no-cache-dir \
    "torch>=2.1" --index-url https://download.pytorch.org/whl/cu121
$PIP install -q --no-cache-dir \
    transformers \
    accelerate \
    sentencepiece \
    cmudict

# Downgrade cuDNN from 9 → 8.9 AFTER PyTorch (torch pulls in cuDNN 9, overwriting 8.9).
# TF 2.15 requires libcudnn.so.8; PyTorch uses its own bundled cuDNN so --no-deps is safe.
echo "  Pinning cuDNN to 8.9 for TF 2.15 (after PyTorch install)..."
$PIP install -q --no-cache-dir --force-reinstall --no-deps "nvidia-cudnn-cu12==8.9.7.29"

# 5. Install NeuralDecoder package
echo "[5/6] Installing NeuralDecoder package..."
cd "$REPO_ROOT/NeuralDecoder"
$PIP install -q -e . --no-deps
cd "$REPO_ROOT"

# 6. Persist activation + LD_LIBRARY_PATH in ~/.bashrc
echo "[6/6] Configuring shell..."
VENV_NV="$VENV/lib/python3.11/site-packages/nvidia"
VENV_LD="$VENV_NV/cudnn/lib:$VENV_NV/cublas/lib:$VENV_NV/cuda_nvrtc/lib:$VENV_NV/cuda_runtime/lib:$VENV_NV/cufft/lib:$VENV_NV/cusolver/lib:$VENV_NV/cusparse/lib:$VENV_NV/nvjitlink/lib"
export LD_LIBRARY_PATH="$VENV_LD:${LD_LIBRARY_PATH:-}"
if ! grep -q "venv311/bin/activate" ~/.bashrc; then
    {
        echo ""
        echo "# SpeechBCI venv (Python 3.11 + TF 2.15)"
        echo "source $VENV/bin/activate"
        echo "export LD_LIBRARY_PATH=$VENV_LD:\$LD_LIBRARY_PATH"
    } >> ~/.bashrc
fi
source "$VENV/bin/activate"

# Verify
echo ""
echo "============================================="
echo "  Verifying installation..."
echo "============================================="
$PY -c "
import tensorflow as tf
print(f'TensorFlow : {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs       : {len(gpus)}')
for g in gpus:
    print(f'  {g}')

import torch
print(f'PyTorch    : {torch.__version__}')
print(f'CUDA avail : {torch.cuda.is_available()}')

from neuralDecoder.models import ConformerEncoder
print('ConformerEncoder import: OK')
print()
print('Setup complete! Activate with:')
print('  source $VENV/bin/activate')
"

echo ""
echo "============================================="
echo "  Done! Venv: $VENV"
echo "  Run eval with:"
echo "  source $VENV/bin/activate"
echo "  cd $REPO_ROOT"
echo "  python AnalysisExamples/eval_lm_pipeline.py --lm both --hf-token <TOKEN>"
echo "============================================="
