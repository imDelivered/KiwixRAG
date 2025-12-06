#!/usr/bin/env python3
"""Simple launcher script for chatbot."""

import sys
import os
import argparse

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add to path and import
sys.path.insert(0, script_dir)

from chatbot import ChatbotGUI
from chatbot.config import DEFAULT_MODEL, DEBUG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KiwixRAG Chatbot")
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL, help="Ollama model to use")
    
    args = parser.parse_args()
    
    # Set DEBUG flag in config
    from chatbot import config
    config.DEBUG = args.debug
    
    if args.debug:
        print("[DEBUG] Debug mode enabled", file=sys.stderr)
        print(f"[DEBUG] Using model: {args.model}", file=sys.stderr)
        print(f"[DEBUG] Script directory: {script_dir}", file=sys.stderr)
    
    try:
        app = ChatbotGUI(args.model)
        app.run()
    except KeyboardInterrupt:
        if args.debug:
            print("\n[DEBUG] Interrupted by user", file=sys.stderr)
        pass
    except RuntimeError as e:
        if args.debug:
            print(f"[DEBUG] RuntimeError: {e}", file=sys.stderr)
        # Error already handled by GUI, no need to print to terminal
        sys.exit(1)


