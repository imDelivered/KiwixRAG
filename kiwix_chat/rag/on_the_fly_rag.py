"""On-the-fly RAG - no pre-built index, generates embeddings from ZIM in real-time."""

import sys
from typing import Iterable, List
from kiwix_chat.models import Chunk
from kiwix_chat.rag.embeddings import generate_embedding, generate_embeddings_batch
from kiwix_chat.rag.chunker import chunk_article
from kiwix_chat.rag.zim_reader import list_zim_articles, read_zim_article
from kiwix_chat.rag.token_buffer import TokenBuffer
from kiwix_chat.chat.ollama import ollama_stream_chat


def on_the_fly_retrieve(query: str, zim_file_path: str, top_k: int = 5, max_articles: int = 50) -> List[Chunk]:
    """Retrieve relevant chunks on-the-fly without pre-built index.
    
    Process:
    1. Generate query embedding
    2. Read articles from ZIM (limited to max_articles for speed)
    3. Chunk articles
    4. Generate embeddings for chunks on-the-fly
    5. Find most similar chunks
    6. Return top_k
    
    Args:
        query: User query
        zim_file_path: Path to ZIM file
        top_k: Number of chunks to return
        max_articles: Maximum articles to process (for speed)
        
    Returns:
        List of most relevant Chunk objects
    """
    # Generate query embedding
    query_embedding = generate_embedding(query, is_query=True)
    
    # Get candidate articles (first max_articles for speed)
    # In a real system, you'd use keyword search or other heuristics to narrow down
    all_chunks = []
    article_count = 0
    
    print(f"[on-the-fly] Processing articles from ZIM (max {max_articles})...", file=sys.stderr)
    
    for title, href in list_zim_articles(zim_file_path):
        if article_count >= max_articles:
            break
        
        # Read article
        article_text = read_zim_article(zim_file_path, href)
        if not article_text:
            continue
        
        # Chunk article
        chunks = chunk_article(
            text=article_text,
            title=title,
            href=href,
            max_chunk_size=500,
            overlap=50,
            content_type="Wikipedia articles"
        )
        
        all_chunks.extend(chunks)
        article_count += 1
        
        if article_count % 10 == 0:
            print(f"[on-the-fly] Processed {article_count} articles, {len(all_chunks)} chunks...", file=sys.stderr)
    
    if not all_chunks:
        return []
    
    print(f"[on-the-fly] Generating embeddings for {len(all_chunks)} chunks...", file=sys.stderr)
    
    # Generate embeddings for all chunks on-the-fly
    chunk_texts = [chunk.text for chunk in all_chunks]
    chunk_embeddings = generate_embeddings_batch(chunk_texts, batch_size=32, show_progress=True, is_query=False)
    
    # Calculate similarity (cosine similarity)
    import numpy as np
    query_vec = np.array(query_embedding)
    
    scored_chunks = []
    for chunk, embedding in zip(all_chunks, chunk_embeddings):
        chunk_vec = np.array(embedding)
        # Cosine similarity
        similarity = np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec))
        scored_chunks.append((similarity, chunk))
    
    # Sort by similarity and return top_k
    scored_chunks.sort(key=lambda x: -x[0])  # Descending
    
    print(f"[on-the-fly] Found {len(scored_chunks)} chunks, returning top {top_k}", file=sys.stderr)
    
    return [chunk for _, chunk in scored_chunks[:top_k]]


def stream_on_the_fly_rag(
    model: str,
    query: str,
    messages: List[dict],
    zim_file_path: str,
    max_tokens: int = 2048
) -> Iterable[str]:
    """Stream response using on-the-fly RAG (no pre-built index).
    
    Args:
        model: LLM model name
        query: User query
        messages: Chat messages
        zim_file_path: Path to ZIM file
        max_tokens: Max tokens in buffer
        
    Yields:
        Text chunks as they're generated
    """
    # Retrieve chunks on-the-fly (embeddings generated in real-time)
    try:
        chunks = on_the_fly_retrieve(query, zim_file_path, top_k=5, max_articles=100)
        if not chunks:
            # No chunks found, just stream normal generation
            yield from ollama_stream_chat(model, messages)
            return
    except Exception as e:
        import sys
        print(f"[on-the-fly] Retrieval failed: {e}, using normal generation", file=sys.stderr)
        try:
            yield from ollama_stream_chat(model, messages)
        except Exception as stream_err:
            yield f"[Error: {stream_err}]"
        return
    
    # Tokenize chunks in real-time
    token_buffer = TokenBuffer(max_tokens=max_tokens, pruning_strategy="relevance")
    relevance_scores = [1.0 - (i * 0.15) for i in range(len(chunks))]
    token_buffer.add_chunks(chunks, relevance_scores)
    
    # Get source texts
    source_texts = [chunk.text for chunk in chunks]
    combined_source = "\n\n".join([f"=== {chunk.article_title} ===\n{chunk.text}" for chunk in chunks])
    
    # Add source context to messages
    enhanced_messages = messages.copy()
    system_msg = {
        "role": "system",
        "content": f"""Use [FACT:keyword] for facts.

Source material:
{combined_source[:3000]}"""
    }
    
    if enhanced_messages and enhanced_messages[0].get("role") == "system":
        enhanced_messages[0]["content"] += "\n\n" + system_msg["content"]
    else:
        enhanced_messages.insert(0, system_msg)
    
    # Stream generation
    yield from ollama_stream_chat(model, enhanced_messages)

