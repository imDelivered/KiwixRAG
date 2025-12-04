"""Configuration constants and loading."""

import json
import os
from typing import Dict, Optional

from kiwix_chat.models import ModelPlatform


# Default configuration
DEFAULT_MODEL = "dolphin-llama3"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
KIWIX_BASE_URL = "http://localhost:8081"
MODEL_CONFIG_FILE = "model_config.json"

# Model-to-platform mapping (can be overridden by config file)
MODEL_PLATFORM_CONFIG: Dict[str, ModelPlatform] = {
    # Add explicit mappings here if needed
}


def load_model_config() -> Dict[str, str]:
    """Load model configuration from JSON file if it exists."""
    config = {}
    if os.path.exists(MODEL_CONFIG_FILE):
        try:
            with open(MODEL_CONFIG_FILE, 'r') as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return config


def detect_platform(model_name: str, explicit_platform: Optional[ModelPlatform] = None) -> ModelPlatform:
    """
    Detect which platform to use for a given model.
    
    Args:
        model_name: Name of the model
        explicit_platform: Explicitly specified platform (from --platform arg)
    
    Returns:
        ModelPlatform enum value
    """
    # If platform is explicitly specified, use it
    if explicit_platform and explicit_platform != ModelPlatform.AUTO:
        return explicit_platform
    
    # Check explicit config mapping first
    if model_name in MODEL_PLATFORM_CONFIG:
        return MODEL_PLATFORM_CONFIG[model_name]
    
    # Check config file
    config = load_model_config()
    if model_name in config:
        platform_str = config[model_name].lower()
        if platform_str == "ollama":
            return ModelPlatform.OLLAMA
    
    # Default to Ollama
    return ModelPlatform.OLLAMA


