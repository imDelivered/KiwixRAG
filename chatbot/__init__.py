"""Chatbot GUI application."""

import sys
from chatbot.gui import ChatbotGUI
from chatbot.models import Message, ModelPlatform
from chatbot.config import DEFAULT_MODEL

__all__ = ['ChatbotGUI', 'Message', 'ModelPlatform']


def main():
    """Main entry point."""
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


if __name__ == "__main__":
    main()

