"""Vector database integration using ChromaDB for RAG system."""

import hashlib
import os
import sys
from pathlib import Path
from typing import List, Optional

from kiwix_chat.models import Chunk

# Global cache of vector stores (one per ZIM file)
_vector_stores: dict[str, 'VectorStore'] = {}


def _get_index_path(zim_file_path: str) -> str:
    """Get the path where the vector index should be stored.
    
    Args:
        zim_file_path: Path to ZIM file
        
    Returns:
        Path to index directory (stored alongside ZIM file)
    """
    zim_path = Path(zim_file_path)
    # Create index name based on ZIM file name and hash
    zim_hash = hashlib.md5(zim_file_path.encode()).hexdigest()[:8]
    index_name = f"{zim_path.stem}_{zim_hash}_rag_index"
    index_dir = zim_path.parent / index_name
    return str(index_dir)


class VectorStore:
    """Vector store for storing and searching article chunks."""
    
    def __init__(self, zim_file_path: str):
        """Initialize vector store for a ZIM file.
        
        Args:
            zim_file_path: Path to ZIM file
        """
        self.zim_file_path = zim_file_path
        self.index_path = _get_index_path(zim_file_path)
        
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            print("[rag] ERROR: chromadb not installed. Install with: pip install chromadb", file=sys.stderr)
            raise
        
        # Initialize ChromaDB client (persistent, local)
        self.client = chromadb.PersistentClient(
            path=self.index_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        collection_name = "kiwix_chunks"
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            # Collection doesn't exist, create it
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"zim_file": zim_file_path}
            )
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the vector store.
        
        Args:
            chunks: List of Chunk objects to add
        """
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []
        embeddings = []
        
        # Generate embeddings for chunks (is_query=False for passages)
        from kiwix_chat.rag.embeddings import generate_embeddings_batch
        
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_embeddings = generate_embeddings_batch(chunk_texts, show_progress=False, is_query=False)
        
        for chunk, embedding in zip(chunks, chunk_embeddings):
            # Create unique ID: include href to ensure uniqueness across articles
            # Use hash of href to keep ID short but unique
            href_hash = hashlib.md5(chunk.href.encode()).hexdigest()[:8]
            chunk_id = f"{href_hash}_{chunk.chunk_idx}"
            ids.append(chunk_id)
            texts.append(chunk.text)
            embeddings.append(embedding)
            metadatas.append({
                "article_title": chunk.article_title,
                "href": chunk.href,
                "chunk_idx": chunk.chunk_idx,
                "total_chunks": chunk.total_chunks,
                "content_type": chunk.content_type
            })
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[Chunk]:
        """Semantic search using query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_metadata: Optional metadata filters (e.g., {"content_type": "Wikipedia articles"})
            
        Returns:
            List of Chunk objects, sorted by relevance
        """
        # Query ChromaDB
        where = filter_metadata if filter_metadata else None
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where
        )
        
        # Convert results to Chunk objects
        chunks = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                metadata = results['metadatas'][0][i]
                chunks.append(Chunk(
                    text=results['documents'][0][i],
                    article_title=metadata.get('article_title', 'Unknown'),
                    href=metadata.get('href', ''),
                    chunk_idx=metadata.get('chunk_idx', 0),
                    total_chunks=metadata.get('total_chunks', 1),
                    content_type=metadata.get('content_type', 'Kiwix content')
                ))
        
        return chunks
    
    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        top_k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[Chunk]:
        """Hybrid search combining semantic and keyword matching.
        
        Args:
            query: Original query text for keyword matching
            query_embedding: Query embedding vector for semantic search
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of Chunk objects, sorted by combined relevance
        """
        # Get more results from semantic search
        semantic_k = min(top_k * 3, 50)  # Get 3x more for reranking
        semantic_results = self.search(query_embedding, top_k=semantic_k, filter_metadata=filter_metadata)
        
        if not semantic_results:
            return []
        
        # Score by keyword matching
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scored_chunks = []
        for chunk in semantic_results:
            # Semantic score (from ChromaDB distance, inverted)
            # ChromaDB returns results sorted by distance, so earlier = better
            semantic_score = 1.0 / (semantic_results.index(chunk) + 1)
            
            # Keyword score
            chunk_lower = chunk.text.lower()
            keyword_matches = sum(1 for word in query_words if word in chunk_lower)
            keyword_score = keyword_matches / max(len(query_words), 1)
            
            # Combined score (weighted: 70% semantic, 30% keyword)
            combined_score = 0.7 * semantic_score + 0.3 * keyword_score
            
            scored_chunks.append((combined_score, chunk))
        
        # Sort by combined score
        scored_chunks.sort(key=lambda x: -x[0])
        
        # Return top_k
        return [chunk for _, chunk in scored_chunks[:top_k]]
    
    def get_collection_size(self) -> int:
        """Get the number of chunks in the collection."""
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all chunks from the collection."""
        # Delete and recreate collection
        collection_name = self.collection.name
        self.client.delete_collection(name=collection_name)
        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"zim_file": self.zim_file_path}
        )


def get_vector_store(zim_file_path: str) -> VectorStore:
    """Get or create vector store for a ZIM file (singleton pattern).
    
    Args:
        zim_file_path: Path to ZIM file
        
    Returns:
        VectorStore instance
    """
    if zim_file_path not in _vector_stores:
        _vector_stores[zim_file_path] = VectorStore(zim_file_path)
    return _vector_stores[zim_file_path]


def is_indexed(zim_file_path: str) -> bool:
    """Check if a ZIM file has been indexed.
    
    Args:
        zim_file_path: Path to ZIM file
        
    Returns:
        True if index exists and has content, False otherwise
    """
    index_path = _get_index_path(zim_file_path)
    if not os.path.exists(index_path):
        return False
    
    try:
        store = get_vector_store(zim_file_path)
        return store.get_collection_size() > 0
    except Exception:
        return False


def get_index_path(zim_file_path: str) -> str:
    """Get the path where the index is stored.
    
    Args:
        zim_file_path: Path to ZIM file
        
    Returns:
        Path to index directory
    """
    return _get_index_path(zim_file_path)

