#!/bin/bash
set -e

echo "Setting up training environment..."

# Install tmux if not already installed
if command -v tmux > /dev/null 2>&1; then
    echo "tmux is already installed"
else
    echo "tmux not found. Attempting to install..."
    if command -v apt-get > /dev/null 2>&1; then
        apt-get update && apt-get install -y tmux
        echo "tmux installed successfully"
    else
        echo "Warning: Could not install tmux (apt-get not available)"
    fi
fi

# Copy environment template
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file from template"
else
    echo ".env file already exists"
fi

# Install uv for faster package management
echo "Installing uv package manager..."
pip install uv

# Set up Python environment
echo "Setting up Python environment..."
if [ ! -d ".venv" ]; then
    uv venv
    echo "Created Python virtual environment"
else
    echo "Virtual environment already exists"
fi

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
uv sync

# Install flash-attention if CUDA is available
if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    echo "CUDA detected. Installing flash-attention..."
    uv pip install flash-attn --no-build-isolation
    echo "flash-attention installed successfully"
else
    echo "CUDA not detected. Skipping flash-attention installation."
fi

echo "Setup complete! Please configure your .env file with API keys before running training."
