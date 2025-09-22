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
. .venv/bin/activate
uv pip install --upgrade setuptools wheel

# Install requirements
echo "Installing Python dependencies..."
uv pip install -r requirements.txt

# Install flash-attention separately (requires special compilation flags)
echo "Installing flash-attention..."
uv pip install flash-attn --no-build-isolation

echo "✓ Setup complete! Please configure your .env file with API keys before running training."