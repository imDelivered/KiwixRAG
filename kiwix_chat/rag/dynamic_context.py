"""Dynamic context management for iterative RAG system."""

import sys
from typing import List, Optional, Set, Tuple
from dataclasses import dataclass

from kiwix_chat.models import Chunk
from kiwix_chat.rag.token_buffer import TokenBuffer, TokenizedChunk
from kiwix_chat.rag.retriever import RAGRetriever


@dataclass
class ContextState:
    """State of the dynamic context."""
    buffer_utilization: float  # 0.0 to 1.0
    unused_chunks: int
    used_chunks: int
    total_chunks: int
    needs_retrieval: bool


class DynamicContextManager:
    """Manages active context window with dynamic pruning and retrieval.
    
    This class:
    - Manages active context window (token buffer)
    - Tracks generated text for semantic similarity computation
    - Triggers retrieval when chunks are pruned
    - Maintains query expansion state (original query + generated context)
    """
    
    def __init__(
        self,
        token_buffer: TokenBuffer,
        retriever: RAGRetriever,
        original_query: str,
        min_utilization: float = 0.8,
        usage_threshold: float = 0.7
    ):
        """Initialize dynamic context manager.
        
        Args:
            token_buffer: TokenBuffer instance for managing chunks
            retriever: RAGRetriever instance for fetching new chunks
            original_query: Original user query
            min_utilization: Minimum buffer utilization before triggering retrieval (0.0 to 1.0)
            usage_threshold: Usage score threshold for pruning (0.0 to 1.0)
        """
        self.token_buffer = token_buffer
        self.retriever = retriever
        self.original_query = original_query
        self.min_utilization = min_utilization
        self.usage_threshold = usage_threshold
        
        # Track generated text
        self.generated_text = ""
        self.generated_sentences = []  # Last N sentences for query expansion
        
        # Track seen chunks to avoid duplicates
        self.seen_chunk_ids: Set[Tuple[str, int]] = set()  # (article_title, chunk_idx)
        
        # Initial retrieval flag
        self._initial_retrieval_done = False
    
    def initialize(self, initial_chunks: List[Chunk], relevance_scores: Optional[List[float]] = None):
        """Initialize context with initial chunks.
        
        Args:
            initial_chunks: Initial chunks from first retrieval
            relevance_scores: Optional relevance scores for chunks
        """
        if relevance_scores is None:
            relevance_scores = [1.0 - (i * 0.15) for i in range(len(initial_chunks))]
        
        self.token_buffer.add_chunks(initial_chunks, relevance_scores)
        
        # Mark chunks as seen
        for chunk in initial_chunks:
            self.seen_chunk_ids.add((chunk.article_title, chunk.chunk_idx))
        
        self._initial_retrieval_done = True
    
    def add_generated_text(self, text: str) -> ContextState:
        """Add generated text, update usage scores, and prune if needed.
        
        Args:
            text: Newly generated text to add
            
        Returns:
            ContextState with current context state
        """
        if not text:
            return self.get_state()
        
        # Append to generated text (limit size to prevent unbounded growth)
        self.generated_text += text
        # Keep only last 20KB to prevent memory bloat (enough for semantic similarity)
        MAX_GENERATED_TEXT_SIZE = 20000
        if len(self.generated_text) > MAX_GENERATED_TEXT_SIZE:
            # Keep last portion, preserving sentence boundaries
            truncated = self.generated_text[-MAX_GENERATED_TEXT_SIZE:]
            # Try to start at sentence boundary
            first_period = truncated.find('.')
            if first_period > 0 and first_period < 1000:  # Don't lose too much
                self.generated_text = truncated[first_period+1:].strip()
            else:
                self.generated_text = truncated
        
        # Update sentence tracking (keep last 3 sentences for query expansion)
        sentences = self._extract_sentences(text)
        self.generated_sentences.extend(sentences)
        if len(self.generated_sentences) > 3:
            self.generated_sentences = self.generated_sentences[-3:]
        
        # Update usage scores based on semantic similarity
        used_count = self.token_buffer.update_usage_from_generation(self.generated_text)
        
        # Prune chunks that have been used
        pruned_chunks = self.token_buffer.prune_by_usage(threshold=self.usage_threshold)
        
        # Get current state
        state = self.get_state()
        
        # Trigger retrieval if needed
        if state.needs_retrieval:
            self._retrieve_new_chunks()
        
        return state
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract complete sentences from text.
        
        Args:
            text: Text to extract sentences from
            
        Returns:
            List of sentence strings
        """
        import re
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def should_retrieve_new_chunks(self) -> bool:
        """Check if buffer needs more chunks.
        
        Returns:
            True if retrieval should be triggered
        """
        state = self.get_state()
        return state.needs_retrieval
    
    def get_expanded_query(self) -> str:
        """Generate expanded query from original + generated context.
        
        Combines original query with recent generated sentences to create
        a more specific query for retrieving relevant chunks.
        
        Returns:
            Expanded query string
        """
        if not self.generated_sentences:
            return self.original_query
        
        # Combine original query with recent context
        recent_context = " ".join(self.generated_sentences[-2:])  # Last 2 sentences
        expanded = f"{self.original_query} {recent_context}"
        return expanded.strip()
    
    def get_active_context(self) -> str:
        """Return current context for LLM.
        
        Returns:
            Combined text from all active chunks
        """
        return self.token_buffer.get_active_text()
    
    def get_state(self) -> ContextState:
        """Get current context state.
        
        Returns:
            ContextState with current metrics
        """
        stats = self.token_buffer.get_buffer_stats()
        utilization = stats['utilization']
        
        unused_chunks = len(self.token_buffer.get_unused_chunks(threshold=0.3))
        used_chunks = len(self.token_buffer.get_used_chunks(threshold=self.usage_threshold))
        total_chunks = stats['chunk_count']
        
        # Need retrieval if utilization is below minimum
        needs_retrieval = utilization < self.min_utilization
        
        return ContextState(
            buffer_utilization=utilization,
            unused_chunks=unused_chunks,
            used_chunks=used_chunks,
            total_chunks=total_chunks,
            needs_retrieval=needs_retrieval
        )
    
    def _retrieve_new_chunks(self, top_k: int = 3) -> List[Chunk]:
        """Retrieve new chunks based on expanded query.
        
        Args:
            top_k: Number of chunks to retrieve
            
        Returns:
            List of newly retrieved Chunk objects
        """
        expanded_query = self.get_expanded_query()
        
        try:
            # Retrieve chunks
            new_chunks = self.retriever.retrieve(
                query=expanded_query,
                top_k=top_k * 2,  # Get more to filter out seen ones
                use_hybrid=True
            )
            
            # Filter out chunks we've already seen and low-quality chunks (search URLs, etc.)
            unseen_chunks = []
            for chunk in new_chunks:
                chunk_id = (chunk.article_title, chunk.chunk_idx)
                # Skip if already seen
                if chunk_id in self.seen_chunk_ids:
                    continue
                # Skip low-quality chunks (search URLs, empty titles, etc.)
                if (chunk.article_title.startswith("search?") or 
                    "pattern=" in chunk.article_title or 
                    not chunk.text or 
                    len(chunk.text.strip()) < 50):
                    continue
                unseen_chunks.append(chunk)
                self.seen_chunk_ids.add(chunk_id)
            
            # Take top_k unseen chunks
            new_chunks = unseen_chunks[:top_k]
            
            if new_chunks:
                # Add to buffer with relevance scores
                # Assign scores based on order (first = most relevant)
                relevance_scores = [1.0 - (i * 0.1) for i in range(len(new_chunks))]
                self.token_buffer.add_chunks(new_chunks, relevance_scores)
                
                print(f"[dynamic-context] Retrieved {len(new_chunks)} new chunks", file=sys.stderr)
            
            return new_chunks
        except Exception as e:
            print(f"[dynamic-context] Error retrieving new chunks: {e}", file=sys.stderr)
            return []
    
    def get_source_chunks(self) -> List[Chunk]:
        """Get all current source chunks from buffer.
        
        Returns:
            List of Chunk objects currently in buffer
        """
        return [tokenized.chunk for tokenized in self.token_buffer.buffer]

