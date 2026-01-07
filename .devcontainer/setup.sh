#!/bin/bash
set -e

echo "🚀 Setting up DragonWizard development environment..."

# Install uv package manager
echo "📦 Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Verify installations
echo "✅ Verifying installations..."
python --version
node --version
npm --version
uv --version

# Initialize git submodules
echo "📥 Initializing git submodules..."
git submodule update --init --recursive

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
uv sync

# Build MCP dice server
echo "🎲 Building MCP dice server..."
cd external/dice-rolling-mcp
npm install
npm run build:mcp
cd ../..

# Set up environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
fi

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "Quick start:"
echo "  make test          - Run tests"
echo "  make help          - Show all available commands"
echo "  uv run dragonwizard --version"
echo ""
