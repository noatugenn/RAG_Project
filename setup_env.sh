#!/bin/bash
# setup_env.sh - Automated virtual environment setup script

set -e  # Exit on any error

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="jeen_project"
VENV_PATH="$PROJECT_DIR/$VENV_NAME"

echo "========================================="
echo "  Jeen Project Environment Setup"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ -d "$VENV_PATH" ]; then
    echo "✓ Virtual environment '$VENV_NAME' already exists"
else
    echo "→ Creating virtual environment '$VENV_NAME'..."
    python3 -m venv "$VENV_PATH"
    echo "✓ Virtual environment created"
fi

echo ""
echo "→ Activating virtual environment..."
source "$VENV_PATH/bin/activate"
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "→ Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install requirements
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "→ Installing dependencies from requirements.txt..."
    pip install -r "$PROJECT_DIR/requirements.txt"
    echo "✓ All dependencies installed"
else
    echo "⚠  requirements.txt not found!"
    exit 1
fi

echo ""
echo "→ Downloading NLTK punkt tokenizer..."
python -c "import nltk; nltk.download('punkt', quiet=True)"
echo "✓ NLTK data downloaded"

echo ""
echo "========================================="
echo "  ✓ Setup Complete!"
echo "========================================="
echo ""
echo "Virtual environment: $VENV_NAME"
echo "Python location: $(which python)"
echo ""
echo "Installed packages:"
pip list --format=columns | grep -E "(pypdf2|python-docx|nltk|google-generativeai|psycopg2|pgvector|python-dotenv|tqdm|tenacity)" || echo "  (checking...)"
echo ""
echo "To activate manually in future:"
echo "  source $VENV_NAME/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
