#!/bin/bash
# LLM Watermark Setup Script
# Automates the setup process for the LLM watermark fine-tuning pipeline

set -e  # Exit on any error

echo "LLM Watermark Fine-tuning Setup"
echo "=================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "uv installed successfully"
    
    # Source the shell configuration to make uv available
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "uv is already installed"
fi

# Display uv version
echo "uv version: $(uv --version)"

# Install dependencies
echo "Installing project dependencies..."
uv sync

# Install optional dependencies based on system
echo "Detecting system and installing optimizations..."

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected - installing performance optimizations..."
    uv sync --extra performance
elif [[ $(uname) == "Darwin" ]] && [[ $(uname -m) == "arm64" ]]; then
    echo "Apple Silicon detected - installing MPS-compatible packages..."
    uv sync --extra performance
else
    echo "CPU-only setup detected"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p models logs output

# Display device configuration examples
echo ""
echo "Configuration:"
echo "   Main config:  config.yaml (edit this file)"
echo "   Examples:     examples/ directory"
echo "   - CUDA GPU:   examples/config_cuda.yaml"
echo "   - Apple MPS:  examples/config_mps.yaml"  
echo "   - CPU only:   examples/config_cpu.yaml"

# Display usage examples
echo ""
echo "Quick Start Commands:"
echo "   # Basic training (uses config.yaml):"
echo "   uv run python -m llm_watermark.main"
echo ""
echo "   # With custom output directory:"
echo "   uv run python -m llm_watermark.main --output-dir ./experiment"
echo ""
echo "   # Verbose logging:"
echo "   uv run python -m llm_watermark.main --verbose"
echo ""
echo "   # Run examples:"
echo "   uv run python example_usage.py"
echo ""
echo "To customize settings, edit config.yaml"
echo "   Reference examples are in examples/ directory"

echo ""
echo "Setup complete! Ready to start watermark fine-tuning."
echo ""
echo "For more information, see README.md"
