.PHONY: help setup install-python install-dice-server test clean

# Default target
help:
	@echo "DragonWizard Development Commands"
	@echo "=================================="
	@echo ""
	@echo "Setup:"
	@echo "  make setup              - Complete project setup (Python + Dice Server)"
	@echo "  make install-python     - Install Python dependencies via uv"
	@echo "  make install-dice-server - Build the MCP dice server"
	@echo ""
	@echo "Development:"
	@echo "  make test               - Run all tests"
	@echo "  make test-unit          - Run unit tests only"
	@echo "  make clean              - Clean build artifacts"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Python 3.13+"
	@echo "  - uv package manager"
	@echo "  - Node.js 18+ and npm"
	@echo "  - git"

# Complete setup
setup: install-python install-dice-server
	@echo ""
	@echo "✅ Setup complete!"
	@echo ""
	@echo "To run the project:"
	@echo "  uv run dragonwizard --help"
	@echo ""
	@echo "To run tests:"
	@echo "  make test"

# Install Python dependencies
install-python:
	@echo "📦 Installing Python dependencies..."
	@command -v uv >/dev/null 2>&1 || { echo "❌ Error: uv is not installed. Install from https://github.com/astral-sh/uv"; exit 1; }
	uv sync
	@echo "✅ Python dependencies installed"

# Initialize and build dice server
install-dice-server:
	@echo "🎲 Setting up MCP dice server..."
	@command -v node >/dev/null 2>&1 || { echo "❌ Error: Node.js is not installed. Install from https://nodejs.org"; exit 1; }
	@command -v npm >/dev/null 2>&1 || { echo "❌ Error: npm is not installed. Install Node.js from https://nodejs.org"; exit 1; }
	@if [ ! -d "external/dice-rolling-mcp/.git" ]; then \
		echo "Initializing git submodule..."; \
		git submodule update --init --recursive; \
	fi
	@echo "Installing dice server npm dependencies..."
	cd external/dice-rolling-mcp && npm install
	@echo "Building dice server..."
	cd external/dice-rolling-mcp && npm run build
	@echo "✅ Dice server built at external/dice-rolling-mcp/dist/index.js"

# Run all tests
test:
	@echo "🧪 Running all tests..."
	uv run pytest -v

# Run only unit tests
test-unit:
	@echo "🧪 Running unit tests..."
	uv run pytest tests/unit/ -v

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf .pytest_cache
	rm -rf dragonwizard/__pycache__
	rm -rf tests/**/__pycache__
	rm -rf dist/
	rm -rf *.egg-info
	@if [ -d "external/dice-rolling-mcp/node_modules" ]; then \
		echo "Cleaning dice server node_modules..."; \
		rm -rf external/dice-rolling-mcp/node_modules; \
	fi
	@if [ -d "external/dice-rolling-mcp/dist" ]; then \
		echo "Cleaning dice server dist..."; \
		rm -rf external/dice-rolling-mcp/dist; \
	fi
	@echo "✅ Clean complete"
