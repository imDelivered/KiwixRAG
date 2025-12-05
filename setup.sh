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
echo "  Installing: requests, sentence-transformers, chromadb, tiktoken"

# Function to verify installation
verify_install() {
    python3 -c "import chromadb; import sentence_transformers" 2>/dev/null
}

# Try --break-system-packages first (most reliable on modern systems)
INSTALL_SUCCESS=false
VERIFIED=false

echo "  Attempting installation with --break-system-packages..."
if python3 -m pip install --break-system-packages requests sentence-transformers chromadb tiktoken 2>&1; then
    if verify_install; then
        INSTALL_SUCCESS=true
        VERIFIED=true
        echo "  ✓ Installed and verified with --break-system-packages flag"
    else
        echo "  ⚠️  Installed but verification failed, trying --user method..."
    fi
fi

# If --break-system-packages didn't work, try --user
if [ "$VERIFIED" = false ]; then
    echo "  Attempting installation with --user flag..."
    if python3 -m pip install --user requests sentence-transformers chromadb tiktoken 2>&1; then
        if verify_install; then
            INSTALL_SUCCESS=true
            VERIFIED=true
            echo "  ✓ Installed and verified with --user flag"
        else
            # Sometimes --user installs but Python can't find them
            # Try adding user site-packages to path (find actual Python version)
            PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            USER_SITE="${HOME}/.local/lib/python${PYTHON_VERSION}/site-packages"
            if [ -d "$USER_SITE" ]; then
                export PYTHONPATH="${USER_SITE}:${PYTHONPATH}"
                if verify_install; then
                    INSTALL_SUCCESS=true
                    VERIFIED=true
                    echo "  ✓ Installed with --user flag (found in user site-packages)"
                fi
            fi
        fi
    fi
fi

if [ "$VERIFIED" = false ]; then
    echo ""
    echo "  ❌ ERROR: Failed to install or verify dependencies!"
    echo ""
    echo "  Please try installing manually:"
    echo "    pip3 install --break-system-packages sentence-transformers chromadb"
    echo ""
    echo "  If that fails, try:"
    echo "    pip3 install --user sentence-transformers chromadb"
    echo "    export PYTHONPATH=\${HOME}/.local/lib/python3.*/site-packages:\$PYTHONPATH"
    echo ""
    exit 1
fi

echo "✓ Python dependencies installed and verified (sentence-transformers, chromadb)"

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
echo "[6/7] Setting up scripts..."
chmod +x run_kiwix_chat.sh 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true
echo "✓ Scripts made executable"

# Step 7: Install krag command
echo "[7/7] Installing krag command..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KRAG_WRAPPER="/usr/local/bin/krag"

# Create wrapper script that finds and runs run_kiwix_chat.sh
# Use double quotes for heredoc to allow variable expansion
sudo tee "$KRAG_WRAPPER" > /dev/null << KRAG_EOF
#!/usr/bin/env bash
# krag command wrapper - finds project directory and runs run_kiwix_chat.sh

# Installation directory (set during setup)
INSTALL_DIR="$SCRIPT_DIR"

# Check if installation directory still exists and is valid
if [ -f "\$INSTALL_DIR/run_kiwix_chat.sh" ] && [ -f "\$INSTALL_DIR/kiwix_chat.py" ]; then
    PROJECT_DIR="\$INSTALL_DIR"
else
    # Search for project directory in common locations
    PROJECT_DIR=""
    for dir in "\$HOME" "\$HOME/OWRs-main" "\$HOME/OWRs" "/opt/kiwix-rag"; do
        if [ -f "\$dir/run_kiwix_chat.sh" ] && [ -f "\$dir/kiwix_chat.py" ]; then
            PROJECT_DIR="\$dir"
            break
        fi
    done
    
    # If not found, try searching from current directory up
    if [ -z "\$PROJECT_DIR" ]; then
        CURRENT_DIR="\$(pwd)"
        while [ "\$CURRENT_DIR" != "/" ]; do
            if [ -f "\$CURRENT_DIR/run_kiwix_chat.sh" ] && [ -f "\$CURRENT_DIR/kiwix_chat.py" ]; then
                PROJECT_DIR="\$CURRENT_DIR"
                break
            fi
            CURRENT_DIR="\$(dirname "\$CURRENT_DIR")"
        done
    fi
fi

if [ -z "\$PROJECT_DIR" ]; then
    echo "Error: Could not find Kiwix RAG project directory."
    echo "Please navigate to the project directory or run: ./run_kiwix_chat.sh"
    exit 1
fi

# Execute the launcher script
exec "\$PROJECT_DIR/run_kiwix_chat.sh" "\$@"
KRAG_EOF

sudo chmod +x "$KRAG_WRAPPER"
echo "✓ krag command installed to /usr/local/bin/krag"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Everything is ready! You can now run:"
echo ""
echo "  krag"
echo ""
echo "Or from the project directory:"
echo "  ./run_kiwix_chat.sh"
echo ""
echo "The launcher will automatically:"
echo "  • Start Ollama server"
echo "  • Download the AI model (if needed)"
echo "  • Start Kiwix server (if ZIM file found)"
echo "  • Launch the chat interface"
echo ""
echo "The 'krag' command is now available system-wide from any directory!"
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

