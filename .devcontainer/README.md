# DragonWizard Dev Container

This devcontainer provides a fully configured development environment for DragonWizard with all required dependencies pre-installed.

## What's Included

### Languages & Runtimes
- **Python 3.13** - Latest Python with all project dependencies
- **Node.js 20** - For MCP dice server
- **npm** - Node package manager

### Tools
- **uv** - Fast Python package manager
- **git** - Version control
- **gh** - GitHub CLI

### VS Code Extensions
- Python language support (Pylance)
- Black formatter
- Ruff linter
- TOML support
- GitHub Copilot (if you have access)

### Project Setup
- Python virtual environment created with `uv`
- All Python dependencies installed
- MCP dice server built and ready
- Git submodules initialized
- Environment file created from `.env.example`

## Getting Started

### Prerequisites
- [Docker](https://www.docker.com/products/docker-desktop)
- [VS Code](https://code.visualstudio.com/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

### Opening in Dev Container

1. **Open the project in VS Code**
   ```bash
   code /path/to/dragonwizard
   ```

2. **Reopen in Container**
   - Press `F1` or `Ctrl+Shift+P`
   - Select "Dev Containers: Reopen in Container"
   - Wait for the container to build (first time takes ~5 minutes)

3. **Automatic Setup**
   - The `postCreateCommand` runs `setup.sh` automatically
   - Installs uv, syncs dependencies, builds dice server
   - You'll see progress in the terminal

4. **Start Developing!**
   ```bash
   make test          # Run tests
   make help          # Show all commands
   uv run dragonwizard --version
   ```

### Manual Container Build

If you need to rebuild the container:

```bash
# From VS Code Command Palette (F1):
Dev Containers: Rebuild Container
```

## Configuration

### Environment Variables

After the container starts, configure your `.env` file:

```bash
# Required for LLM functionality (Phase 2+)
LLM__API_KEY=your-anthropic-key-here

# Required for Discord bot (Phase 5+)
DISCORD_TOKEN=your-discord-bot-token-here

# Dice server path (auto-configured)
DICE_SERVER_PATH=./external/dice-rolling-mcp/dist/index.js
```

### Git Configuration

The devcontainer mounts your local `~/.gitconfig` so your git identity and preferences are preserved.

## Folder Structure

```
.devcontainer/
├── devcontainer.json   # Container configuration
├── setup.sh           # Post-create setup script
└── README.md          # This file
```

## Customization

### Adding VS Code Extensions

Edit `devcontainer.json`:

```json
"customizations": {
  "vscode": {
    "extensions": [
      "your.extension.id"
    ]
  }
}
```

### Installing Additional Tools

Add commands to `setup.sh` or use the `features` section in `devcontainer.json`.

### Available Features

See [devcontainers/features](https://github.com/devcontainers/features) for pre-built features like:
- Docker-in-Docker
- PostgreSQL
- Redis
- And many more...

## Troubleshooting

### Container Won't Start

1. Check Docker is running: `docker ps`
2. Check disk space: `df -h`
3. Rebuild container: `Dev Containers: Rebuild Container`

### Setup Script Fails

1. View full logs in the terminal
2. Run setup manually:
   ```bash
   bash .devcontainer/setup.sh
   ```

### Git Submodule Issues

```bash
git submodule update --init --recursive
```

### Port Already in Use

The container name is `dragonwizard-dev`. If you get conflicts:

```bash
docker stop dragonwizard-dev
docker rm dragonwizard-dev
```

## Performance Tips

### Slow Container Build
- Use `--no-cache` flag when rebuilding if needed
- Ensure Docker has adequate resources (4GB+ RAM recommended)

### Slow Python Package Installation
- `uv` is much faster than pip
- First sync is slower (downloads packages)
- Subsequent syncs are cached and fast

## Development Workflow

```bash
# Run tests
make test

# Run tests with coverage
uv run pytest --cov=dragonwizard

# Test dice server manually
uv run python tests/manual/tools/test_dice_server.py

# Format code
uv run black dragonwizard/

# Lint code
uv run ruff check dragonwizard/

# Run CLI
uv run dragonwizard --help
```

## Resources

- [Dev Containers Documentation](https://containers.dev/)
- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [DragonWizard README](../README.md)
- [Implementation Plan](../implementation.md)
