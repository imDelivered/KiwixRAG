#!/bin/bash

# Try to run GUI version first
if command -v python3 &> /dev/null; then
    # Use system python (usually has tk) or venv
    if [ -f "venv/bin/python3" ]; then
        venv/bin/python3 uninstall_gui.py
    else
        python3 uninstall_gui.py
    fi
    # If GUI ran successfully (even if cancelled), exit
    if [ $? -eq 0 ]; then
        exit 0
    fi
fi

# Fallback CLI if GUI fails
echo "⚠️  GUI failed or not available. Starting CLI Uninstaller..."
echo "🗑️  KiwixRAG Uninstaller (CLI)"
echo "   [1] Remove everything (venv, indices, cache)"
echo "   [2] Remove only indices (keep env)"
echo "   [3] Cancel"
read -p "Select option: " opt

case $opt in
    1)
        echo "Removing venv..."
        rm -rf venv
        echo "Removing indices..."
        rm -rf data
        echo "Cleaning cache..."
        find . -type d -name "__pycache__" -exec rm -rf {} +
        echo "✅ Complete."
        ;;
    2)
        echo "Removing indices..."
        rm -rf data
        echo "✅ Indices removed."
        ;;
    *)
        echo "Cancelled."
        exit 0
        ;;
esac
