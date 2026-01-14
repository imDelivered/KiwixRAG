
# Hermit - Offline AI Chatbot for Wikipedia & ZIM Files
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Text Processing Utilities.
Handles text chunking, cleaning, and normalization.
"""

from typing import List
import re

class TextProcessor:
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks of approximately 'chunk_size' characters.
        """
        if not text:
            return []
            
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + chunk_size
            if end < text_len:
                segment = text[start:end+50]
                last_break = -1
                for punct in ["\n", ". ", "! ", "? "]:
                    idx = segment.rfind(punct)
                    if idx != -1:
                        last_break = max(last_break, idx)
                if last_break != -1:
                    end = start + last_break + 1
                else:
                    last_space = segment.rfind(" ")
                    if last_space != -1:
                        end = start + last_space
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
            if start >= end:
                start = end
        return chunks

    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
