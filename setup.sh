#!/bin/bash
set -e

echo "Setting up GRPO training environment..."

# Copy environment template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file from template"
else
    echo "✓ .env file already exists"
fi

# Install uv for faster package management
echo "Installing uv package manager..."
pip install uv

# Set up Python environment
echo "Setting up Python environment..."
uv venv
echo "✓ Created Python virtual environment"

# Detect correct path (bin or Scripts based on OS)
if [ -d ".venv/bin" ]; then
    ACTIVATE_PATH=".venv/bin/activate"
else
    ACTIVATE_PATH=".venv/Scripts/activate"
fi
. $ACTIVATE_PATH
echo "Using activation script: $ACTIVATE_PATH"

uv pip install --upgrade setuptools wheel

# Install requirements
echo "Installing Python dependencies..."
uv pip install -r requirements.txt

# Install flash-attention if CUDA is available
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    echo "✓ CUDA detected. Installing flash-attention..."
    uv pip install flash-attn --no-build-isolation
    echo "✓ flash-attention installed successfully"
else
    echo "⚠ CUDA not detected. Skipping flash-attention installation."
fi

echo "✓ Setup complete! Please configure your .env file with API keys before running training."
