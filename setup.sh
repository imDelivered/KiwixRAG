#!/usr/bin/env bash
set -euo pipefail

echo "=== Chatbot Setup Script ==="
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
echo "[2/6] Installing Python, venv, and basic tools..."
sudo apt install -y python3 python3-venv python3-full python3-tk curl > /dev/null 2>&1

# Verify Python
if ! python3 --version > /dev/null 2>&1; then
    echo "Error: Python3 installation failed"
    exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python $PYTHON_VERSION installed"

# Step 3: Set up Virtual Environment and Dependencies
echo "[3/6] Setting up Python Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

echo "Installing dependencies (this may take a while)..."
./venv/bin/pip install -r requirements.txt > /dev/null 2>&1
echo "✓ Dependencies installed"

# Step 4: Check for ZIM file
echo "[4/6] Checking for ZIM file..."
ZIM_FILE=$(find . -maxdepth 1 -name "*.zim" | head -n 1)

if [ -n "$ZIM_FILE" ]; then
    echo "✓ Found ZIM file: $ZIM_FILE"
    echo "  Using Lazy RAG mode (Just-In-Time indexing)."
    echo "  No full index build required."
else
    echo "⚠️  No .zim file found in current directory."
    echo "   Please place a .zim file here (e.g. wikipedia_en_all_maxi.zim) to enable the chatbot."
fi

# Step 5: Install Ollama
echo "[5/6] Installing Ollama..."
if ! command -v ollama > /dev/null 2>&1; then
    curl -fsSL https://ollama.ai/install.sh | sh
    echo "✓ Ollama installed"
else
    echo "✓ Ollama already installed"
fi

# Step 5b: Pull the Models
echo "[5.5/6] Checking for AI Models..."

# Ensure ollama service is reachable
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama service not responding. Attempting to start..."
    ollama serve > /dev/null 2>&1 &
    OLLAMA_PID=$!
    sleep 5
fi

# Pull final model (llama3.1:1b)
if ollama list | grep -q "llama3.1:1b"; then
    echo "✓ Final model llama3.1:1b already present"
else
    echo "Pulling llama3.1:1b (final response model)..."
    if ollama pull llama3.1:1b; then
        echo "✓ llama3.1:1b downloaded successfully"
    else
        echo "⚠️  Failed to download llama3.1:1b"
        echo "   Please run 'ollama pull llama3.1:1b' manually."
    fi
fi

# Pull joint models (llama3.2:1b)
if ollama list | grep -q "llama3.2:1b"; then
    echo "✓ Joint model llama3.2:1b already present"
else
    echo "Pulling llama3.2:1b (entity & chunk filtering)..."
    if ollama pull llama3.2:1b; then
        echo "✓ llama3.2:1b downloaded successfully"
    else
        echo "⚠️  Failed to download llama3.2:1b"
        echo "   Please run 'ollama pull llama3.2:1b' manually."
    fi
fi

# Pull scorer model (qwen2.5:0.5b)
if ollama list | grep -q "qwen2.5:0.5b"; then
    echo "✓ Scorer model qwen2.5:0.5b already present"
else
    echo "Pulling qwen2.5:0.5b (article scoring)..."
    if ollama pull qwen2.5:0.5b; then
        echo "✓ qwen2.5:0.5b downloaded successfully"
    else
        echo "⚠️  Failed to download qwen2.5:0.5b"
        echo "   Please run 'ollama pull qwen2.5:0.5b' manually."
    fi
fi

# Explicitly pull requests and ollama if missing from venv (fallback)
./venv/bin/pip install requests ollama > /dev/null 2>&1 || true

echo "✓ All models checked/downloaded"

# Step 5c: Enable Ollama Service
if command -v systemctl > /dev/null 2>&1; then
    echo "Enabling Ollama service..."
    sudo systemctl enable --now ollama > /dev/null 2>&1 || true
fi

# Step 6: Make scripts executable and install krag command
echo "[6/6] Setting up scripts..."
chmod +x chatbot.py 2>/dev/null || true
chmod +x run_chatbot.sh 2>/dev/null || true
chmod +x setup.sh 2>/dev/null || true
chmod +x build_index.py 2>/dev/null || true
echo "✓ Scripts made executable"

# Install krag command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KRAG_WRAPPER="/usr/local/bin/krag"

sudo tee "$KRAG_WRAPPER" > /dev/null << KRAG_EOF
#!/usr/bin/env bash
# krag command wrapper

INSTALL_DIR="$SCRIPT_DIR"

if [ -f "\$INSTALL_DIR/run_chatbot.sh" ]; then
    exec "\$INSTALL_DIR/run_chatbot.sh" "\$@"
else
    echo "Error: Could not find run_chatbot.sh at \$INSTALL_DIR"
    exit 1
fi
KRAG_EOF

sudo chmod +x "$KRAG_WRAPPER"
echo "✓ 'krag' command installed to /usr/local/bin/krag"

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Everything is ready! You can now launch the app by running:"
echo ""
echo "  krag"
echo ""
echo "IMPORTANT: To enable offline AI with Wikipedia:"
echo "1. Ensure you have a ZIM file (e.g., wikipedia_en_all_maxi_2025-08.zim)"
echo "2. Run the chatbot:"
echo "   krag"
echo ""
echo "Make sure Ollama is running:"
echo "  ollama serve"
echo ""


