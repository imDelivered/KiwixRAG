"""Context builder for formatting retrieved chunks into context."""

from typing import List, Tuple

from kiwix_chat.models import Chunk


def build_context(
    chunks: List[Chunk],
    query: str,
    max_chars: int
) -> Tuple[str, List[dict]]:
    """Build formatted context from retrieved chunks.
    
    Args:
        chunks: List of retrieved Chunk objects
        query: Original user query (for relevance)
        max_chars: Maximum characters in context
        
    Returns:
        Tuple of (formatted_context_string, sources_list)
        sources_list contains dicts with 'title', 'url', 'excerpt'
    """
    if not chunks:
        return "", []
    
    # Build sources list (one per unique article)
    sources = []
    seen_articles = set()
    
    for chunk in chunks:
        article_key = (chunk.article_title, chunk.href)
        if article_key not in seen_articles:
            seen_articles.add(article_key)
            # Create excerpt (first 200 chars of chunk)
            excerpt = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            sources.append({
                'title': chunk.article_title,
                'url': chunk.href,
                'excerpt': excerpt,
                'content_type': chunk.content_type
            })
    
    # Build context string
    context_parts = []
    total_chars = 0
    
    # Group chunks by article
    chunks_by_article = {}
    for chunk in chunks:
        article_key = (chunk.article_title, chunk.href)
        if article_key not in chunks_by_article:
            chunks_by_article[article_key] = []
        chunks_by_article[article_key].append(chunk)
    
    # Add chunks, prioritizing articles with more relevant chunks
    for article_key, article_chunks in chunks_by_article.items():
        if total_chars >= max_chars:
            break
        
        # Sort chunks by index (to maintain order)
        article_chunks.sort(key=lambda c: c.chunk_idx)
        
        article_title = article_chunks[0].article_title
        article_href = article_chunks[0].href
        
        # Combine chunks from same article
        article_text_parts = []
        for chunk in article_chunks:
            if total_chars >= max_chars:
                break
            
            chunk_text = chunk.text
            remaining = max_chars - total_chars
            
            if len(chunk_text) > remaining:
                chunk_text = chunk_text[:remaining] + "\n[truncated]"
            
            article_text_parts.append(chunk_text)
            total_chars += len(chunk_text) + 1  # +1 for newline
        
        if article_text_parts:
            combined_text = "\n".join(article_text_parts)
            context_parts.append(f"=== {article_title} ===\n{combined_text}\n")
    
    # If we have space, add more chunks
    if total_chars < max_chars:
        remaining_chunks = [c for c in chunks if (c.article_title, c.href) not in chunks_by_article]
        for chunk in remaining_chunks:
            if total_chars >= max_chars:
                break
            
            remaining = max_chars - total_chars
            chunk_text = chunk.text
            if len(chunk_text) > remaining:
                chunk_text = chunk_text[:remaining] + "\n[truncated]"
            
            context_parts.append(f"=== {chunk.article_title} ===\n{chunk_text}\n")
            total_chars += len(chunk_text) + len(chunk.article_title) + 10
    
    context = "\n".join(context_parts)
    
    # Truncate if still over limit
    if len(context) > max_chars:
        context = context[:max_chars] + "\n[truncated]"
    
    return context, sources



