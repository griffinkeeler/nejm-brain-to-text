#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_VENV_DIR="/scratch/mcetin_lab/gkeeler/venvs/b2txt25"
VENV_DIR="${1:-$DEFAULT_VENV_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"

echo "Repo root: $REPO_ROOT"
echo "Virtual environment: $VENV_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN is not available in PATH."
  echo "Set PYTHON_BIN to a Python 3.10+ executable before running this script."
  exit 1
fi

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$(printf '%s\n' "3.10" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.10" ]]; then
  echo "Error: $PYTHON_BIN is Python $PYTHON_VERSION, but this project requires Python 3.10+."
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "Using existing virtual environment."
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

echo "Installing project dependencies..."
pip install \
  redis==5.2.1 \
  jupyter==1.1.1 \
  numpy==2.1.2 \
  pandas==2.3.0 \
  matplotlib==3.10.1 \
  scipy==1.15.2 \
  scikit-learn==1.6.1 \
  tqdm==4.67.1 \
  g2p_en==2.1.0 \
  h5py==3.13.0 \
  omegaconf==2.3.0 \
  editdistance==0.8.1 \
  huggingface-hub==0.33.1 \
  transformers==4.53.0 \
  tokenizers==0.21.2 \
  accelerate==1.8.1 \
  bitsandbytes==0.46.0 \
  -e "$REPO_ROOT"

echo
echo "Setup complete."
echo "Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "Verify torch with:"
echo "  python -c \"import torch; print(torch.__version__); print(torch.cuda.is_available())\""
echo
echo "Train with:"
echo "  cd $REPO_ROOT/model_training"
echo "  python train_model.py"
