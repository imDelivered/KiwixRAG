"""Ollama chat API integration."""

import json
from typing import List, Iterable
from urllib.request import Request, urlopen
from urllib.error import URLError

from chatbot.config import OLLAMA_CHAT_URL, STRICT_RAG_MODE, DEBUG
from chatbot.models import Message


def debug_print(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")


def stream_chat(model: str, messages: List[dict]) -> Iterable[str]:
    """Stream chat with Ollama model."""
    payload = json.dumps({
        "model": model, 
        "messages": messages, 
        "stream": True,
        "options": {"temperature": 0}
    }).encode("utf-8")
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
    except URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_CHAT_URL}: {e.reason}") from e


def full_chat(model: str, messages: List[dict]) -> str:
    """Full chat with Ollama model."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0}
    }).encode("utf-8")
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
    except URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {OLLAMA_CHAT_URL}: {e.reason}") from e


from chatbot.rag import RAGSystem

# Global RAG instance
_rag_system = None

def get_rag_system():
    global _rag_system
    if _rag_system is None:
        # Initialize only if index exists
        import os
        if os.path.exists("data/index/faiss.index") or os.path.exists("wikipedia_en_all_maxi_2025-08.zim") or any(f.endswith(".zim") for f in os.listdir('.')):
            try:
                print("Initializing RAG system (Hybrid/Fast)...")
                _rag_system = RAGSystem()
                _rag_system.load_resources()
            except Exception as e:
                print(f"Failed to load RAG: {e}")
                _rag_system = None
    return _rag_system

def build_messages(system_prompt: str, history: List[Message], user_query: str = None) -> List[dict]:
    """Build message list for Ollama API with RAG augmentation."""
    
    # 1. Retrieve context if we have a user query
    context_text = ""
    rag = get_rag_system()
    
    # 1. Detect Intent
    from chatbot.intent import detect_intent
    # Identify the actual query. If user_query is provided, use it.
    # Otherwise check the last message in history if it's from user.
    query_text = user_query
    if not query_text and history and history[-1].role == 'user':
        query_text = history[-1].content
        
    intent = detect_intent(query_text or "")
    debug_print(f"build_messages: Detected Intent='{intent.mode_name}', Should Retrieve={intent.should_retrieve}")
    
    # 2. Retrieve context (If Intent allows)
    context_text = ""
    rag = get_rag_system()
        
    if rag and query_text and intent.should_retrieve:
        try:
            results = rag.retrieve(query_text, top_k=5)
            debug_print(f"build_messages: RAG returned {len(results)} results")
            
            if results:
                context_text = "\n\nRelevant Context via RAG:\n"
                for i, r in enumerate(results, 1):
                    meta = r['metadata']
                    text = r['text']
                    title = meta.get('title', 'Unknown')
                    score = r.get('score', 0.0)
                    debug_print(f"build_messages: result_{i} title='{title}', score={score:.4f}, text_length={len(text)}")
                    context_text += f"\n--- Source {i}: {title} ---\n{text}\n"
                
                context_text += "\nInstructions: Answer the user's question using ONLY the provided context above. \n" \
                                "Verify your answer with the context.\n" \
                                "CITE SOURCES STRICTLY using the headers provided (e.g., 'Source 1: Article Title').\n" \
                                "DO NOT generate or hallucinate bibliography entries.\n" \
                                "If the provided text contains a list but not the specific answer (e.g. 'largest' but only small items listed), state that the information is missing."
            else:
                if STRICT_RAG_MODE:
                    context_text = "\n[SYSTEM NOTICE]: No relevant documents found in the local index.\n" \
                                   "Instructions: You MUST refuse to answer the user's question because no relevant context was found.\n" \
                                   "Reply EXACTLY with: 'I do not have enough information in my knowledge base to answer this question.'"
                else:
                    context_text = "\n[SYSTEM NOTICE]: No relevant documents found in the local index. Answering based on general knowledge.\n"
        except Exception as e:
            print(f"RAG retrieval error: {e}")

    # 3. Augment system prompt with Context AND Intent Instructions
    final_system_prompt = system_prompt + intent.system_instruction
    if context_text:
        final_system_prompt += context_text

    messages = [{"role": "system", "content": final_system_prompt}]
    for msg in history:
        if msg.role in ["user", "assistant", "system"]:
            messages.append({"role": msg.role, "content": msg.content})
            
    print(f"\n🧠 Generating response...")
    return messages
