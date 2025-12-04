"""Simplified streaming RAG with working fact replacement."""

import sys
import re
from typing import Iterable, List
from kiwix_chat.models import Chunk
from kiwix_chat.rag.embeddings import generate_embedding, generate_embeddings_batch
from kiwix_chat.rag.chunker import chunk_article
from kiwix_chat.rag.zim_reader import list_zim_articles, read_zim_article
from kiwix_chat.rag.token_buffer import TokenBuffer
from kiwix_chat.chat.ollama import ollama_stream_chat
import numpy as np


def stream_hybrid_response_simple(
    model: str,
    query: str,
    messages: List[dict],
    zim_file_path: str,
    max_tokens: int = 2048
) -> Iterable[str]:
    """Simplified streaming with working fact replacement."""
    
    # Retrieve chunks on-the-fly
    print(f"[rag] Using embedding model to find relevant content...", file=sys.stderr)
    query_embedding = generate_embedding(query, is_query=True)
    
    all_chunks = []
    article_count = 0
    max_articles = 500
    
    for title, href in list_zim_articles(zim_file_path):
        if article_count >= max_articles:
            break
        
        article_text = read_zim_article(zim_file_path, href)
        if not article_text:
            continue
        
        chunks = chunk_article(article_text, title, href, max_chunk_size=500, overlap=50)
        all_chunks.extend(chunks)
        article_count += 1
        
        if article_count % 50 == 0:
            print(f"[rag] Processed {article_count} articles, {len(all_chunks)} chunks...", file=sys.stderr)
    
    if not all_chunks:
        yield from ollama_stream_chat(model, messages)
        return
    
    # Generate embeddings and find top chunks
    print(f"[rag] Generating embeddings for {len(all_chunks)} chunks...", file=sys.stderr)
    chunk_texts = [chunk.text for chunk in all_chunks]
    chunk_embeddings = generate_embeddings_batch(chunk_texts, batch_size=64, show_progress=True, is_query=False)
    
    query_vec = np.array(query_embedding)
    scored_chunks = []
    for chunk, embedding in zip(all_chunks, chunk_embeddings):
        chunk_vec = np.array(embedding)
        similarity = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
        scored_chunks.append((similarity, chunk))
    
    scored_chunks.sort(key=lambda x: -x[0])
    top_chunks = [chunk for _, chunk in scored_chunks[:5]]
    
    # Tokenize
    token_buffer = TokenBuffer(max_tokens=max_tokens, pruning_strategy="relevance")
    relevance_scores = [1.0 - (i * 0.15) for i in range(len(top_chunks))]
    token_buffer.add_chunks(top_chunks, relevance_scores)
    
    # Get source texts
    source_texts = [chunk.text for chunk in top_chunks]
    combined_source = "\n\n".join([f"=== {chunk.article_title} ===\n{chunk.text}" for chunk in top_chunks])
    
    # Add to messages
    enhanced_messages = messages.copy()
    system_msg = {
        "role": "system",
        "content": f"""Use [FACT:keyword] for facts.

Source material:
{combined_source[:4000]}"""
    }
    enhanced_messages = [msg for msg in enhanced_messages if msg.get("role") != "system"]
    enhanced_messages.insert(0, system_msg)
    
    # Stream with fact replacement
    fact_pattern = re.compile(r'\[FACT:([^\]]+)\]')
    buffer = ""
    
    for chunk in ollama_stream_chat(model, enhanced_messages):
        buffer += chunk
        
        # Check for complete markers
        matches = list(fact_pattern.finditer(buffer))
        if matches:
            last_match = matches[-1]
            keyword = last_match.group(1).strip().lower()
            
            # Find in source
            exact_text = None
            for source in source_texts:
                if keyword in source.lower():
                    # Find sentence with keyword
                    sentences = re.split(r'[.!?]+', source)
                    for sentence in sentences:
                        if keyword in sentence.lower():
                            exact_text = sentence.strip()
                            break
                    if exact_text:
                        break
            
            if exact_text:
                # Yield before marker + replacement
                before = buffer[:last_match.start()]
                if before:
                    yield before
                yield exact_text
                buffer = buffer[last_match.end():]
                continue
        
        # If buffer getting long, yield what we can
        if len(buffer) > 200:
            # Check for markers
            markers = list(fact_pattern.finditer(buffer))
            if markers:
                # Yield up to first marker
                first_start = markers[0].start()
                if first_start > 0:
                    yield buffer[:first_start]
                    buffer = buffer[first_start:]
            else:
                # No markers, yield all
                yield buffer
                buffer = ""
    
    # Yield remaining buffer
    if buffer:
        yield buffer

