"""Index builder for creating vector indexes from ZIM files."""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

from kiwix_chat.kiwix.client import get_zim_content_description
from kiwix_chat.rag.chunker import chunk_article
from kiwix_chat.rag.vector_store import get_vector_store, get_index_path, is_indexed
from kiwix_chat.rag.zim_reader import list_zim_articles, read_zim_article


def _get_zim_file_hash(zim_file_path: str) -> str:
    """Calculate hash of ZIM file for change detection.
    
    Args:
        zim_file_path: Path to ZIM file
        
    Returns:
        MD5 hash of file (first 16 chars)
    """
    # For large files, hash just the first MB and file size
    try:
        with open(zim_file_path, 'rb') as f:
            first_mb = f.read(1024 * 1024)
            file_size = os.path.getsize(zim_file_path)
            combined = first_mb + str(file_size).encode()
            return hashlib.md5(combined).hexdigest()[:16]
    except Exception:
        return hashlib.md5(zim_file_path.encode()).hexdigest()[:16]


def _save_index_metadata(zim_file_path: str, metadata: dict) -> None:
    """Save index metadata to file.
    
    Args:
        zim_file_path: Path to ZIM file
        metadata: Metadata dict to save
    """
    index_path = get_index_path(zim_file_path)
    metadata_file = Path(index_path) / "metadata.json"
    
    os.makedirs(index_path, exist_ok=True)
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def _load_index_metadata(zim_file_path: str) -> Optional[dict]:
    """Load index metadata from file.
    
    Args:
        zim_file_path: Path to ZIM file
        
    Returns:
        Metadata dict or None if not found
    """
    index_path = get_index_path(zim_file_path)
    metadata_file = Path(index_path) / "metadata.json"
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def build_index(
    zim_file_path: str,
    max_articles: Optional[int] = None,
    batch_size: int = 100,
    show_progress: bool = True
) -> None:
    """Build vector index from ZIM file articles.
    
    Args:
        zim_file_path: Path to ZIM file
        max_articles: Maximum number of articles to index (None = all)
        batch_size: Number of articles to process before saving
        show_progress: Whether to show progress messages
    """
    if not os.path.isfile(zim_file_path):
        raise FileNotFoundError(f"ZIM file not found: {zim_file_path}")
    
    # Check if already indexed
    if is_indexed(zim_file_path):
        metadata = _load_index_metadata(zim_file_path)
        if metadata:
            current_hash = _get_zim_file_hash(zim_file_path)
            if metadata.get('zim_hash') == current_hash:
                if show_progress:
                    print(f"[rag] Index already exists for {os.path.basename(zim_file_path)}", file=sys.stderr)
                return
    
    if show_progress:
        content_type = get_zim_content_description()
        print(f"[rag] Building index for {os.path.basename(zim_file_path)} ({content_type})...", file=sys.stderr)
    
    # Initialize vector store
    vector_store = get_vector_store(zim_file_path)
    
    # Clear existing index if rebuilding
    vector_store.clear()
    
    # Get content type
    content_type = get_zim_content_description()
    
    # Process articles
    article_count = 0
    chunk_count = 0
    batch_chunks = []
    
    try:
        for title, href in list_zim_articles(zim_file_path):
            if max_articles and article_count >= max_articles:
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
                content_type=content_type
            )
            
            if chunks:
                batch_chunks.extend(chunks)
                chunk_count += len(chunks)
            
            article_count += 1
            
            # Save batch
            if len(batch_chunks) >= batch_size:
                vector_store.add_chunks(batch_chunks)
                if show_progress:
                    print(f"[rag] Indexed {article_count} articles, {chunk_count} chunks...", file=sys.stderr)
                batch_chunks = []
        
        # Save remaining chunks
        if batch_chunks:
            vector_store.add_chunks(batch_chunks)
        
        # Save metadata
        metadata = {
            "zim_file": zim_file_path,
            "zim_hash": _get_zim_file_hash(zim_file_path),
            "article_count": article_count,
            "chunk_count": chunk_count,
            "content_type": content_type
        }
        _save_index_metadata(zim_file_path, metadata)
        
        if show_progress:
            print(f"[rag] Index complete: {article_count} articles, {chunk_count} chunks", file=sys.stderr)
    
    except KeyboardInterrupt:
        print("\n[rag] Indexing interrupted by user", file=sys.stderr)
        # Save what we have so far
        if batch_chunks:
            vector_store.add_chunks(batch_chunks)
        raise
    except Exception as e:
        print(f"[rag] ERROR: Indexing failed: {e}", file=sys.stderr)
        raise


def index_article(
    article_text: str,
    title: str,
    href: str,
    zim_file_path: str,
    content_type: Optional[str] = None
) -> None:
    """Index a single article (for incremental updates).
    
    Args:
        article_text: Article text content
        title: Article title
        href: Article href/path
        zim_file_path: Path to ZIM file
        content_type: Type of content (auto-detected if None)
    """
    if not content_type:
        content_type = get_zim_content_description()
    
    # Chunk article
    chunks = chunk_article(
        text=article_text,
        title=title,
        href=href,
        max_chunk_size=500,
        overlap=50,
        content_type=content_type
    )
    
    if chunks:
        vector_store = get_vector_store(zim_file_path)
        vector_store.add_chunks(chunks)



