"""Data models for chatbot."""

from dataclasses import dataclass
from enum import Enum


class ModelPlatform(Enum):
    """Platform types for model execution."""
    OLLAMA = "ollama"
    AUTO = "auto"


@dataclass
class Message:
    """Chat message with role and content."""
    role: str
    content: str


