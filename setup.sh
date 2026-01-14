#!/usr/bin/env bash

# Hermit - Offline AI Chatbot for Wikipedia & ZIM Files
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

echo "=== Hermit Setup Script ==="
echo "Sets up the local AI environment with GPU support."
echo ""

# Ensure we are not running as root (prevents venv permission issues)
if [[ $EUID -eq 0 ]]; then
   echo "❌ Error: Please do NOT run this script with sudo."
   echo "The script will ask for your password only when necessary (app installation)."
   exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Cleanup artifacts
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# 1. System Packages
echo "[1/5] Checking System Prerequisites..."
sudo apt update -qq || echo "⚠️ Warning: apt update failed."
sudo apt install -y python3 python3-venv python3-full python3-tk python3-libzim curl cmake build-essential > /dev/null 2>&1
echo "✓ System packages verified"

# 2. Virtual Environment
echo "[2/5] Setting up Virtual Environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Upgrade pip
./venv/bin/pip install --upgrade pip

# 3. GPU support (PyTorch CUDA 12.1 - pinned for stability)
echo "[3/5] Installing PyTorch with CUDA support..."
./venv/bin/pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 >> setup.log 2>&1
echo "✓ PyTorch (CUDA) installed"

# 4. Core Dependencies
echo "[4/5] Installing Project Dependencies..."
./venv/bin/pip install -r requirements.txt >> setup.log 2>&1

echo "[4.5/5] Compiling llama-cpp-python..."
if command -v nvidia-smi &> /dev/null; then
    echo "  -> NVIDIA GPU detected. Building with CUDA support..."
    CMAKE_ARGS="-DGGML_CUDA=on" ./venv/bin/pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir >> setup.log 2>&1
    echo "✓ llama-cpp-python (CUDA) installed"
else
    echo "  -> No NVIDIA GPU detected. Installing CPU-only version..."
    ./venv/bin/pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir >> setup.log 2>&1
    echo "✓ llama-cpp-python (CPU) installed"
fi


# 5. Download Models
echo "[5/6] Downloading AI Models..."
echo "This may take a while depending on your internet connection."
./venv/bin/python download_models.py
echo "✓ Models downloaded"

# 6. Check Resources
echo "[6/6] Checking Resources..."
ZIM_FILE=$(find . -maxdepth 1 -name "*.zim" | head -n 1)
if [ -z "$ZIM_FILE" ]; then
    echo "⚠️  No .zim file found. Place your Wikipedia ZIM file here."
fi

SHARED_MODELS="shared_models"
mkdir -p "$SHARED_MODELS"
echo "✓ Model directory verified: $SHARED_MODELS"

# Global Commands
echo ""
echo "Enabling global commands..."
HERMIT_WRAPPER="/usr/local/bin/hermit"
sudo tee "$HERMIT_WRAPPER" > /dev/null << HERMIT_EOF
#!/usr/bin/env bash
INSTALL_DIR="$SCRIPT_DIR"
exec "\$INSTALL_DIR/run_chatbot.sh" "\$@"
HERMIT_EOF
sudo chmod +x "$HERMIT_WRAPPER"

FORGE_WRAPPER="/usr/local/bin/forge"
sudo tee "$FORGE_WRAPPER" > /dev/null << FORGE_EOF
#!/usr/bin/env bash
INSTALL_DIR="$SCRIPT_DIR"
exec "\$INSTALL_DIR/venv/bin/python" "\$INSTALL_DIR/forge.py" "\$@"
FORGE_EOF
sudo chmod +x "$FORGE_WRAPPER"

# Desktop Integration
echo "Configuring Desktop Integration..."
if [ -f "assets/icon.png" ]; then
    sudo cp "assets/icon.png" "/usr/share/pixmaps/hermit.png" || true
fi

HERMIT_DESKTOP="/usr/share/applications/hermit.desktop"
sudo tee "$HERMIT_DESKTOP" > /dev/null << HERMIT_ENTRY
[Desktop Entry]
Name=Hermit AI
Comment=Offline AI Chatbot
Exec=hermit
Icon=hermit
Type=Application
Terminal=false
Categories=Education;Science;Utility;AI;
HERMIT_ENTRY

if command -v update-desktop-database &> /dev/null; then
    sudo update-desktop-database > /dev/null 2>&1 || true
fi

echo "=== Setup Complete! ==="
echo "Run with: hermit"