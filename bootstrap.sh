#!/bin/bash
set -e

echo "=========================================="
echo "SkyReels Video Generation - Bootstrap"
echo "=========================================="

# Configuration with defaults
REPO_URL="${REPO_URL:-https://github.com/deeshank/SkyReels-V1.git}"
REPO_BRANCH="${REPO_BRANCH:-dev}"
REPO_DIR="SkyReels-V1"
WORKSPACE_DIR="/workspace"
PORT="${PORT:-8000}"
GPU_NUM="${GPU_NUM:-1}"
TASK_TYPES="${TASK_TYPES:-t2v}"
USE_QUANT="${USE_QUANT:-true}"
USE_OFFLOAD="${USE_OFFLOAD:-true}"
HIGH_CPU_MEMORY="${HIGH_CPU_MEMORY:-true}"
PARAMETERS_LEVEL="${PARAMETERS_LEVEL:-true}"

echo "Configuration:"
echo "  Repository: $REPO_URL"
echo "  Branch: $REPO_BRANCH"
echo "  Workspace: $WORKSPACE_DIR"
echo "  Port: $PORT"
echo "  GPU Count: $GPU_NUM"
echo "  Task Types: $TASK_TYPES"
echo "  Use Quantization: $USE_QUANT"
echo "  Use Offload: $USE_OFFLOAD"
echo ""

# Check if running in RunPod
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "✓ Running in RunPod (Pod ID: $RUNPOD_POD_ID)"
fi

# Update system packages
echo "Installing system dependencies..."
apt-get update -qq 2>/dev/null || true
apt-get install -y -qq git wget curl ffmpeg libsm6 libxext6 2>/dev/null || echo "Some packages already installed"

# Check Python
echo "Checking Python..."
PYTHON_CMD=$(command -v python3 || command -v python)
if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python not found"
    exit 1
fi
echo "✓ Python: $($PYTHON_CMD --version)"

# Early export of GPU runtime environment
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${GPU_NUM:-1}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# Ensure NVIDIA modules are ready
echo "Preparing NVIDIA devices..."
apt-get install -y -qq nvidia-modprobe || true
nvidia-modprobe -u -c=0 || true
ls -l /dev/nvidia* || true
chmod a+rw /dev/nvidia* || true

# Check CUDA
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ GPUs detected:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | nl -v 0
else
    echo "⚠ WARNING: No GPU detected"
fi

# Setup workspace
echo "Setting up workspace at $WORKSPACE_DIR..."
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

# Clone or update repository
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning SkyReels repository (branch: $REPO_BRANCH)..."
    git clone --depth 1 --branch $REPO_BRANCH $REPO_URL $REPO_DIR
else
    echo "✓ Repository exists, updating..."
    cd $REPO_DIR
    git fetch origin $REPO_BRANCH
    git checkout $REPO_BRANCH
    git pull || true
    cd "$WORKSPACE_DIR"
fi

cd "$WORKSPACE_DIR/$REPO_DIR"

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel -q

# Check/ensure CUDA PyTorch exists and is CUDA-enabled
echo "Verifying PyTorch CUDA (cu124)..."
if $PYTHON_CMD - <<'PY'
import sys, torch
ok = (torch.version.cuda == '12.4') and torch.cuda.is_available() and torch.cuda.device_count() > 0
sys.exit(0 if ok else 1)
PY
then
    echo "✓ CUDA PyTorch OK (cu124)"
else
    echo "Installing PyTorch with CUDA 12.4..."
    pip uninstall -y torch torchvision torchaudio xformers || true
    pip cache purge || true
    pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
    pip install --index-url https://download.pytorch.org/whl/cu124 xformers==0.0.29.post1
fi

echo "Installing project dependencies..."
pip install --no-cache-dir --upgrade --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt

# Verify CUDA after dependency install
if ! $PYTHON_CMD - <<'PY'
import sys, torch
sys.exit(0 if torch.cuda.is_available() and torch.version.cuda == '12.4' else 1)
PY
then
    echo "WARNING: CUDA got disabled by dependency install; restoring CUDA wheels..."
    pip uninstall -y torch torchvision torchaudio xformers || true
    pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
    pip install --index-url https://download.pytorch.org/whl/cu124 xformers==0.0.29.post1
fi

# Export GPU runtime environment
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all}
export NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${GPU_NUM:-1}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# Uncomment for multi-GPU NCCL debug if needed
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=0
# export NCCL_SHM_DISABLE=0

# Final CUDA sanity check
$PYTHON_CMD - <<'PY' || { echo "ERROR: CUDA not available to PyTorch. Aborting."; exit 1; }
import torch
assert torch.cuda.is_available(), "torch.cuda.is_available() False"
print("CUDA OK - devices:", torch.cuda.device_count())
PY

echo "Installing FastAPI..."
pip install fastapi uvicorn[standard] python-multipart

# Create directories
echo "Creating directories..."
mkdir -p api_outputs logs

# Check if api_server.py exists (should be in the repo)
if [ ! -f "api_server.py" ]; then
    echo "ERROR: api_server.py not found in repository!"
    echo "This file should be included in the repo at: $REPO_URL"
    exit 1
fi

echo "✓ api_server.py found"

# Set HF token if provided
if [ -n "$HF_TOKEN" ]; then
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Export environment variables
export PORT=$PORT
export GPU_NUM=$GPU_NUM
export TASK_TYPES=$TASK_TYPES
export USE_QUANT=$USE_QUANT
export USE_OFFLOAD=$USE_OFFLOAD
export HIGH_CPU_MEMORY=$HIGH_CPU_MEMORY
export PARAMETERS_LEVEL=$PARAMETERS_LEVEL

# Get public IP
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "localhost")

# Start server
echo ""
echo "=========================================="
echo "🚀 Starting SkyReels API Server"
echo "=========================================="
echo ""
echo "Server: http://$PUBLIC_IP:$PORT"
echo "API Docs: http://$PUBLIC_IP:$PORT/docs"
echo "Health: http://$PUBLIC_IP:$PORT/health"
echo ""
echo "GPUs: $GPU_NUM | Tasks: $TASK_TYPES"
echo ""
echo "⏳ First request may take 5-10 min (downloading models)"
echo "=========================================="
echo ""

# Run server
exec $PYTHON_CMD api_server.py
