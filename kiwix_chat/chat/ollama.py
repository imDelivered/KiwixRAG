"""Ollama API integration."""

import json
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from kiwix_chat.config import OLLAMA_CHAT_URL


def ollama_stream_chat(model: str, messages: List[dict]) -> Iterable[str]:
    """Stream chat with Ollama model."""
    payload = json.dumps({"model": model, "messages": messages, "stream": True}).encode("utf-8")
    req = Request(OLLAMA_CHAT_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=60) as resp:
            for raw_line in resp:
                if not raw_line:
                    continue
                try:
                    obj = json.loads(raw_line.decode("utf-8").strip())
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    raise RuntimeError(str(obj["error"]))
                message = obj.get("message", {})
                content_piece = message.get("content", "")
                if content_piece:
                    yield content_piece
                if obj.get("done"):
                    break
    except HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code}: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_CHAT_URL}: {e.reason}") from e


def ollama_full_chat(model: str, messages: List[dict]) -> str:
    """Full chat with Ollama model."""
    payload = json.dumps({"model": model, "messages": messages, "stream": False}).encode("utf-8")
    req = Request(OLLAMA_CHAT_URL, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
            obj = json.loads(data.decode("utf-8"))
            if obj.get("error"):
                raise RuntimeError(str(obj["error"]))
            message = obj.get("message", {})
            return message.get("content", "")
    except HTTPError as e:
        raise RuntimeError(f"HTTP error {e.code}: {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_CHAT_URL}: {e.reason}") from e


def stream_chat(model: str, messages: List[dict]) -> Iterable[str]:
    """
    Stream chat using Ollama.
    
    Args:
        model: Model name/identifier
        messages: List of message dicts
    
    Yields:
        Text chunks as they're generated
    """
    yield from ollama_stream_chat(model, messages)


def full_chat(model: str, messages: List[dict]) -> str:
    """
    Full chat using Ollama.
    
    Args:
        model: Model name/identifier
        messages: List of message dicts
    
    Returns:
        Generated response text
    """
    return ollama_full_chat(model, messages)


