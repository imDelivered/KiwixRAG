"""
API Client Wrapper for OpenAI-Compatible Endpoints.
Allows Hermit to use external servers (LM Studio, Ollama, etc.) instead of embedded llama-cpp-python.
"""

import json
import requests
import sys
from typing import List, Dict, Generator, Any, Union

from chatbot import config

class OpenAIClientWrapper:
    """
    Polymorphic wrapper that mimics llama_cpp.Llama but calls an API.
    """
    
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model_name = model_name
        print(f"Initialized API Client: {self.base_url} (Model: {self.model_name})")

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = False,
        **kwargs
    ) -> Union[Dict[str, Any], Generator[Dict[str, Any], None, None]]:
        """
        Mimics llama_cpp.Llama.create_chat_completion
        """
        
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # approximate mapping for repeat_penalty if present in kwargs
        presence_penalty = 0.0
        if "repeat_penalty" in kwargs:
             # loose mapping: 1.1 -> 0.1, 1.2 -> 0.2
             presence_penalty = max(0.0, kwargs["repeat_penalty"] - 1.0)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            "presence_penalty": presence_penalty
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
            
        # Add grammar if present (Ollama supports this, others might)
        if "grammar" in kwargs:
            # Check if this specific backend assumes a 'grammar' field or 'response_format'
            # For now, we'll try to pass it if the user is using a backend that supports it
            pass 

        try:
            print(f"DEBUG: Requesting URL: {url}")
            if stream:
                return self._stream_request(url, headers, payload)
            else:
                return self._blocking_request(url, headers, payload)
        except Exception as e:
            print(f"API Request Failed: {e}", file=sys.stderr)
            raise RuntimeError(f"API Connection Error: {e}")

    def _blocking_request(self, url, headers, payload):
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()

    def _stream_request(self, url, headers, payload):
        response = requests.post(url, headers=headers, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue
