#!/bin/bash
# Initialize or reset workspace for Copilot loop demo

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

WORKSPACE_DIR="workspace"
WORKSPACE_TEMPLATE_DIR="workspace_template"
DATA_DIR="${ACE_DEMO_DATA_DIR:-.data}"

echo "🔄 Setting up Copilot Loop workspace..."

# Archive existing workspace if it exists
if [ -d "$WORKSPACE_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ARCHIVE_DIR="$DATA_DIR/archive/$TIMESTAMP"
    mkdir -p "$ARCHIVE_DIR"
    
    # Move workspace
    echo "📦 Archiving existing workspace to $ARCHIVE_DIR/workspace"
    mv "$WORKSPACE_DIR" "$ARCHIVE_DIR/workspace"
    
    # Archive skillbook if exists
    if [ -f "$DATA_DIR/skillbooks/skillbook.json" ]; then
        echo "📦 Archiving skillbook to $ARCHIVE_DIR/skillbook.json"
        cp "$DATA_DIR/skillbooks/skillbook.json" "$ARCHIVE_DIR/skillbook.json"
        rm "$DATA_DIR/skillbooks/skillbook.json"
    fi
fi

# Copy template to workspace
echo "📂 Creating fresh workspace from template..."
cp -r "$WORKSPACE_TEMPLATE_DIR" "$WORKSPACE_DIR"

# Initialize git repo
cd "$WORKSPACE_DIR"
if [ ! -d ".git" ]; then
    echo "🔧 Initializing git repository..."
    git init
    git config user.email "copilot-loop@example.com"
    git config user.name "Copilot Loop"
fi

# Create .agent directory for scratchpad
mkdir -p .agent

# Initial commit
if [ -z "$(git status --porcelain)" ]; then
    echo "📝 Creating initial commit..."
    echo "# Workspace initialized" > .agent/README.md
    git add .
    git commit -m "Initial workspace setup"
fi

cd "$SCRIPT_DIR"

# Create data directories
mkdir -p "$DATA_DIR/skillbooks"
mkdir -p "$DATA_DIR/logs"

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit prompt.md with your task"
echo "2. Edit .env.copilot with your API keys"
echo "3. Run: uv run python copilot_loop.py"
