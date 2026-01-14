
# Hermit - Offline AI Chatbot for Wikipedia & ZIM Files
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Debug Utilities.
Provides centralized logging and debug printing functionality.
"""

import sys
from chatbot import config

def debug_print(msg: str, label: str = None):
    """
    Print a debug message if config.DEBUG is True.
    """
    if config.DEBUG:
        prefix = f"[{label}] " if label else "[DEBUG] "
        print(f"{prefix}{msg}", file=sys.stderr)
