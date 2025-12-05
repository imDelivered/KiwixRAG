#!/usr/bin/env bash
# Simple launcher for chatbot GUI

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama server..."
    # Start Ollama in background if not running
    if ! pgrep -x ollama > /dev/null; then
        ollama serve > /dev/null 2>&1 &
        OLLAMA_PID=$!
        echo "Waiting for Ollama to start..."
        sleep 3
        
        # Check if it started successfully
        if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
            echo "Warning: Ollama may not have started properly."
            echo "Try running 'ollama serve' manually in another terminal."
        fi
    fi
fi

# Launch chatbot GUI
# Check for virtual environment
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    PYTHON_CMD="$SCRIPT_DIR/venv/bin/python3"
else
    PYTHON_CMD="python3"
fi

# Launch chatbot GUI
"$PYTHON_CMD" "$SCRIPT_DIR/run_chatbot.py" "$@"

