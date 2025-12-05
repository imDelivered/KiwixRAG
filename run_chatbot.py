#!/usr/bin/env python3
"""Simple launcher script for chatbot."""

import sys
import os

# Ensure we're in the right directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Add to path and import
sys.path.insert(0, script_dir)

from chatbot import ChatbotGUI
from chatbot.config import DEFAULT_MODEL

if __name__ == "__main__":
    model = DEFAULT_MODEL
    if len(sys.argv) > 1:
        model = sys.argv[1]
    
    try:
        app = ChatbotGUI(model)
        app.run()
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


