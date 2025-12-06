"""Configuration constants."""

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "llama3.2:1b"  # Stock default model
STRICT_RAG_MODE = True
DEBUG = False

# Multi-Joint RAG System Configuration
USE_JOINTS = True  # Enable/disable joint system

# Joint Models
ENTITY_JOINT_MODEL = "llama3.2:1b"     # Entity extraction
SCORER_JOINT_MODEL = "qwen2.5:0.5b"    # Article scoring (fastest)
FILTER_JOINT_MODEL = "llama3.2:1b"     # Chunk filtering

# Joint Temperatures (lower = more deterministic)
ENTITY_JOINT_TEMP = 0.1
SCORER_JOINT_TEMP = 0.0  # Most deterministic for scoring
FILTER_JOINT_TEMP = 0.1

# Joint Timeout (seconds per joint call)
JOINT_TIMEOUT = 5



