#!/bin/bash
#SBATCH --job-name=setup
#SBATCH --output=logs/setup-%j.out
#SBATCH --error=logs/setup-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$SLURM_SUBMIT_DIR"
else
  REPO_ROOT="$(pwd)"
fi

if [[ ! -f "$REPO_ROOT/setup_bluehive.sh" ]]; then
  echo "Error: Could not locate the repository root."
  echo "Expected to find setup_bluehive.sh under: $REPO_ROOT"
  echo "Run this script from the repository directory, or submit with:"
  echo "  sbatch --chdir=/path/to/repo setup_bluehive.sh"
  exit 1
fi

mkdir -p "$REPO_ROOT/logs"

DEFAULT_VENV_DIR="/scratch/mcetin_lab/gkeeler/venvs/b2txt25"
VENV_DIR="${1:-$DEFAULT_VENV_DIR}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_CANDIDATES=("$PYTHON_BIN")
else
  PYTHON_CANDIDATES=(python3.10 python3 python)
fi

echo "Repo root: $REPO_ROOT"
echo "Virtual environment: $VENV_DIR"

PYTHON_BIN=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  echo "Error: Could not find a usable Python executable in PATH."
  echo "Tried: ${PYTHON_CANDIDATES[*]}"
  echo "Load a Python module first, for example:"
  echo "  module avail python"
  echo "  module load python/3.10"
  echo "Then rerun, or set PYTHON_BIN explicitly."
  exit 1
fi

echo "Python executable: $(command -v "$PYTHON_BIN")"

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$(printf '%s\n' "3.10" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.10" ]]; then
  echo "Error: $PYTHON_BIN is Python $PYTHON_VERSION, but this project requires Python 3.10+."
  echo "Load a newer Python module or set PYTHON_BIN to a Python 3.10+ executable."
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
