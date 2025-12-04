#!/usr/bin/env bash
set -euo pipefail

echo "=== Kiwix RAG Setup Script ==="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Error: Don't run this script as root. It will use sudo when needed."
    exit 1
fi

# Check sudo access
if ! sudo -n true 2>/dev/null; then
    echo "This script needs sudo access to install packages."
    echo "You may be prompted for your password."
    echo ""
fi

# Step 1: Update package list
echo "[1/6] Updating package list..."
sudo apt update -qq

# Step 2: Install Python and basic tools
echo "[2/6] Installing Python and basic tools..."
sudo apt install -y python3 python3-pip python3-tk curl wget > /dev/null 2>&1

# Verify Python
if ! python3 --version > /dev/null 2>&1; then
    echo "Error: Python3 installation failed"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python $PYTHON_VERSION installed"

# Step 3: Install Python dependencies
echo "[3/6] Installing Python dependencies..."
echo "  This may take a few minutes (downloading ~200MB of packages)..."
echo "  Installing: requests, sentence-transformers, chromadb"
# Try --user first, fall back to --break-system-packages if needed
INSTALL_SUCCESS=false
if python3 -m pip install --user requests sentence-transformers chromadb 2>&1; then
    INSTALL_SUCCESS=true
    echo "  ✓ Installed with --user flag"
else
    echo "  Note: --user install failed, trying --break-system-packages..."
    if python3 -m pip install --break-system-packages requests sentence-transformers chromadb 2>&1; then
        INSTALL_SUCCESS=true
        echo "  ✓ Installed with --break-system-packages flag"
    fi
fi

if [ "$INSTALL_SUCCESS" = false ]; then
    echo ""
    echo "  ❌ ERROR: Failed to install dependencies!"
    echo "  Please install manually:"
    echo "    pip3 install --break-system-packages sentence-transformers chromadb"
    echo ""
    echo "  Or if that fails, try:"
    echo "    pip3 install --user sentence-transformers chromadb"
    exit 1
fi

# Verify installation
if python3 -c "import chromadb; import sentence_transformers" 2>/dev/null; then
    echo "✓ Python dependencies installed and verified (sentence-transformers, chromadb)"
else
    echo "⚠️  Warning: Dependencies installed but verification failed. They may still work."
fi

# Step 4: Install Ollama
echo "[4/6] Installing Ollama..."
if ! command -v ollama > /dev/null 2>&1; then
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama already installed"
fi

# Step 5: Install Kiwix
echo "[5/6] Installing Kiwix tools..."
if ! command -v kiwix-serve > /dev/null 2>&1; then
    sudo apt install -y kiwix-tools > /dev/null 2>&1
    echo "✓ Kiwix tools installed"
else
    echo "✓ Kiwix tools already installed"
fi

# Step 6: Make scripts executable
echo "[6/6] Setting up scripts..."
chmod +x run_kiwix_chat.sh 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true
echo "✓ Scripts made executable"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Everything is ready! Just run:"
echo ""
echo "  ./run_kiwix_chat.sh"
echo ""
echo "The launcher will automatically:"
echo "  • Start Ollama server"
echo "  • Download the AI model (if needed)"
echo "  • Start Kiwix server (if ZIM file found)"
echo "  • Launch the chat interface"
echo ""
echo "RAG System Setup:"
echo "  • Embedding models (BGE) are ready to use"
echo "  • To enable RAG: Download a ZIM file from https://library.kiwix.org/"
echo "  • Place the .zim file in this directory"
echo "  • Build the index: python3 kiwix_chat.py --build-index"
echo "  • This creates embeddings for semantic search (one-time, may take time)"
echo ""
echo "Optional: Download Wikipedia ZIM file from https://library.kiwix.org/"
echo "          Place it in this directory to enable Wikipedia features."

