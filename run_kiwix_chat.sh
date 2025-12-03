#!/usr/bin/env bash
set -euo pipefail

# Get script directory (works whether script is run directly or via symlink)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configure Ollama to use GPU 0
export CUDA_VISIBLE_DEVICES=0

# Ensure models are accessible - use user's own directory but link blobs from system
if [ ! -d ~/.ollama/models ]; then
    mkdir -p ~/.ollama/models
fi

# Link blobs directory to share model files (read-only access)
if [ ! -L ~/.ollama/models/blobs ] && [ -d /usr/share/ollama/.ollama/models/blobs ]; then
    ln -s /usr/share/ollama/.ollama/models/blobs ~/.ollama/models/blobs 2>/dev/null || true
fi

# Copy manifests if they exist (user needs writable manifests)
if [ -d /usr/share/ollama/.ollama/models/manifests ] && [ ! -d ~/.ollama/models/manifests ]; then
    cp -r /usr/share/ollama/.ollama/models/manifests ~/.ollama/models/ 2>/dev/null || true
fi

# Check if Ollama is running with correct GPU settings
OLLAMA_RUNNING=false
if curl -sSf http://localhost:11434 > /dev/null 2>&1; then
    # Check if running process has correct CUDA_VISIBLE_DEVICES
    if ps e -p $(pgrep -f "ollama serve" | head -1) 2>/dev/null | grep -q "CUDA_VISIBLE_DEVICES=0"; then
        OLLAMA_RUNNING=true
    fi
fi

# Restart Ollama only if not running or using wrong GPU
if [ "$OLLAMA_RUNNING" = false ]; then
    echo "Starting Ollama with GPU 0..."
    pkill -f "ollama serve" 2>/dev/null || true
    sleep 2
    CUDA_VISIBLE_DEVICES=0 nohup ollama serve > /tmp/ollama.log 2>&1 &
    for i in {1..20}; do
        if curl -sSf http://localhost:11434 > /dev/null 2>&1; then
            echo "✓ Ollama started with GPU 0"
            break
        fi
        sleep 0.3
    done
else
    echo "✓ Ollama already running with GPU 0"
fi

# Start Kiwix server on 8081 if not running and ZIM exists
# Search for any .zim file in common directories (no hardcoded language)
ZIM_FILE=""

# Search for any .zim file in common directories
for dir in "$SCRIPT_DIR" "$HOME" "/usr/share/kiwix"; do
  if [ -d "$dir" ]; then
    # Use find to handle globs properly
    found=$(find "$dir" -maxdepth 1 -name "*.zim" -type f 2>/dev/null | head -1)
    if [ -n "$found" ] && [ -f "$found" ]; then
      ZIM_FILE="$found"
      break
    fi
  fi
done

# If not found, try one more time with a broader search
if [ -z "$ZIM_FILE" ]; then
  for dir in "$SCRIPT_DIR" "$HOME"; do
    if [ -d "$dir" ]; then
      # Search recursively up to 2 levels deep
      found=$(find "$dir" -maxdepth 2 -name "*.zim" -type f 2>/dev/null | head -1)
      if [ -n "$found" ] && [ -f "$found" ]; then
        ZIM_FILE="$found"
        break
      fi
    fi
  done
fi

if [ -n "$ZIM_FILE" ] && [ -f "$ZIM_FILE" ]; then
  if ! curl -sSf http://localhost:8081 > /dev/null 2>&1; then
    # Check if kiwix-serve is available
    if ! command -v kiwix-serve > /dev/null 2>&1; then
      echo "[warning] kiwix-serve not found in PATH. Install with: sudo apt install kiwix-tools"
    else
      echo "Starting Kiwix server on port 8081..."
      # Kill any existing kiwix-serve processes on port 8081
      pkill -f "kiwix-serve.*8081" 2>/dev/null || true
      sleep 1
      
      # Start kiwix-serve with proper logging
      nohup kiwix-serve --port=8081 "$ZIM_FILE" > /tmp/kiwix.log 2>&1 &
      KIWIX_PID=$!
      
      # Wait for Kiwix to start (up to 30 seconds)
      KIWIX_STARTED=false
      for i in {1..30}; do
        if curl -sSf http://localhost:8081 > /dev/null 2>&1; then
          KIWIX_STARTED=true
          echo "✓ Kiwix server started on port 8081"
          break
        fi
        # Check if process is still running
        if ! kill -0 $KIWIX_PID 2>/dev/null; then
          echo "[error] Kiwix server process died. Check /tmp/kiwix.log for errors."
          break
        fi
        sleep 1
      done
      
      if [ "$KIWIX_STARTED" = false ]; then
        echo "[warning] Kiwix server did not start in time. Check /tmp/kiwix.log for errors."
        echo "[warning] You may need to start it manually: kiwix-serve --port=8081 \"$ZIM_FILE\""
      fi
    fi
  else
    echo "✓ Kiwix server already running on port 8081"
  fi
else
  echo "[info] ZIM file not found - skipping Kiwix server startup"
  echo "[info] Searched in: $SCRIPT_DIR, $HOME, /usr/share/kiwix"
  echo "[info] To use Kiwix, place a .zim file in one of these locations"
fi

# Pull model if missing
if ! ollama list | awk '{print $1}' | grep -qx "llama3.2:1b"; then
  ollama pull llama3.2:1b
fi

# Pass all arguments (including --zim-file if provided) to Python script
exec python3 "$SCRIPT_DIR/kiwix_chat.py" --model llama3.2:1b "$@"


