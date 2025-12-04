"""Token buffer system for managing tokenized chunks in RAM with pruning."""

import sys
from typing import List, Optional, Tuple, Dict
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from kiwix_chat.models import Chunk


@dataclass
class TokenizedChunk:
    """A chunk that has been tokenized."""
    chunk: Chunk
    tokens: List[int]
    token_count: int
    added_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    relevance_score: float = 0.0  # Similarity score from retrieval
    usage_score: float = 0.0  # Semantic similarity to generated text (0.0 = unused, 1.0 = fully used)


class TokenBuffer:
    """Manages tokenized chunks in RAM with automatic pruning."""
    
    def __init__(self, max_tokens: int = 4096, pruning_strategy: str = "fifo"):
        """Initialize token buffer.
        
        Args:
            max_tokens: Maximum number of tokens to keep in buffer
            pruning_strategy: "fifo" (oldest first), "lru" (least recently used), 
                            "relevance" (lowest relevance score first), or
                            "usage"/"semantic" (highest usage_score first)
        """
        self.max_tokens = max_tokens
        self.pruning_strategy = pruning_strategy
        self.buffer: List[TokenizedChunk] = []
        self.current_token_count = 0
        self.tokenizer = None
        self._initialize_tokenizer()
        self._embedding_model = None  # Lazy-loaded for semantic similarity
    
    def _initialize_tokenizer(self):
        """Initialize tokenizer for the model.
        Uses tiktoken for GPT-style models, sentencepiece for LLaMA models."""
        try:
            # Try tiktoken first (works for GPT-style models)
            import tiktoken
            # Use cl100k_base (GPT-4 tokenizer) as default
            # This works with any Unicode text (supports all languages)
            # It's a reasonable approximation for most models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            print("[token-buffer] Using tiktoken (cl100k_base) tokenizer", file=sys.stderr)
        except ImportError:
            try:
                # Fallback to sentencepiece (for LLaMA models)
                import sentencepiece as spm
                # Try to load a common LLaMA tokenizer
                # Note: This requires the tokenizer file, which may not be available
                # For now, we'll use a character-based approximation
                print("[token-buffer] Warning: tiktoken not available, using character-based token estimation", file=sys.stderr)
                self.tokenizer = None  # Will use character-based estimation
            except ImportError:
                print("[token-buffer] Warning: No tokenizer library available, using character-based estimation", file=sys.stderr)
                self.tokenizer = None
    
    def _tokenize_text(self, text: str) -> Tuple[List[int], int]:
        """Tokenize text and return tokens + count.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Tuple of (token_ids, token_count)
        """
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text)
                return tokens, len(tokens)
            except Exception as e:
                print(f"[token-buffer] Tokenization error: {e}, using character estimation", file=sys.stderr)
        
        # Fallback: character-based estimation (rough approximation: 1 token ≈ 4 chars)
        # This is a very rough estimate, but better than nothing
        estimated_tokens = len(text) // 4
        return list(range(estimated_tokens)), estimated_tokens
    
    def _detokenize(self, tokens: List[int]) -> str:
        """Convert tokens back to text.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded text
        """
        if self.tokenizer:
            try:
                return self.tokenizer.decode(tokens)
            except Exception:
                pass
        
        # Fallback: can't decode without proper tokenizer
        return "[tokenized content]"
    
    def add_chunks(self, chunks: List[Chunk], relevance_scores: Optional[List[float]] = None):
        """Add chunks to buffer after tokenizing them.
        
        Args:
            chunks: List of Chunk objects to add
            relevance_scores: Optional list of relevance scores (one per chunk)
        """
        if relevance_scores is None:
            relevance_scores = [0.0] * len(chunks)
        
        for chunk, score in zip(chunks, relevance_scores):
            tokens, token_count = self._tokenize_text(chunk.text)
            
            tokenized = TokenizedChunk(
                chunk=chunk,
                tokens=tokens,
                token_count=token_count,
                relevance_score=score
            )
            
            self.buffer.append(tokenized)
            self.current_token_count += token_count
            
            # Prune if buffer is full
            self._prune_if_needed()
    
    def _prune_if_needed(self):
        """Prune buffer if it exceeds max_tokens."""
        while self.current_token_count > self.max_tokens and self.buffer:
            if self.pruning_strategy == "fifo":
                # Remove oldest (first added)
                removed = self.buffer.pop(0)
            elif self.pruning_strategy == "lru":
                # Remove least recently accessed
                self.buffer.sort(key=lambda x: x.last_accessed)
                removed = self.buffer.pop(0)
            elif self.pruning_strategy == "relevance":
                # Remove lowest relevance score
                self.buffer.sort(key=lambda x: x.relevance_score)
                removed = self.buffer.pop(0)
            elif self.pruning_strategy in ("usage", "semantic"):
                # Remove highest usage score (most used chunks)
                self.buffer.sort(key=lambda x: -x.usage_score)
                removed = self.buffer.pop(0)
            else:
                # Default to FIFO
                removed = self.buffer.pop(0)
            
            self.current_token_count -= removed.token_count
    
    def get_active_tokens(self) -> List[int]:
        """Get all active tokens from buffer (in order).
        
        Returns:
            List of token IDs representing all active chunks
        """
        all_tokens = []
        for tokenized in self.buffer:
            all_tokens.extend(tokenized.tokens)
            # Update last accessed time
            tokenized.last_accessed = datetime.now()
        return all_tokens
    
    def get_active_text(self) -> str:
        """Get text representation of active buffer.
        
        Returns:
            Combined text from all active chunks
        """
        texts = []
        for tokenized in self.buffer:
            texts.append(f"=== {tokenized.chunk.article_title} ===\n{tokenized.chunk.text}\n")
            tokenized.last_accessed = datetime.now()
        return "\n".join(texts)
    
    def get_buffer_stats(self) -> Dict:
        """Get statistics about the buffer.
        
        Returns:
            Dict with buffer statistics
        """
        return {
            "current_tokens": self.current_token_count,
            "max_tokens": self.max_tokens,
            "chunk_count": len(self.buffer),
            "utilization": self.current_token_count / self.max_tokens if self.max_tokens > 0 else 0.0
        }
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.clear()
        self.current_token_count = 0
    
    def _get_embedding_model(self):
        """Lazy load embedding model for semantic similarity."""
        if self._embedding_model is None:
            try:
                from kiwix_chat.rag.embeddings import initialize_embedding_model
                self._embedding_model = initialize_embedding_model()
            except Exception as e:
                print(f"[token-buffer] Warning: Could not load embedding model: {e}", file=sys.stderr)
                return None
        return self._embedding_model
    
    def update_usage_from_generation(self, generated_text: str) -> int:
        """Update usage scores based on semantic similarity to generated text.
        
        Computes cosine similarity between generated text and each chunk.
        Higher similarity means the chunk has been "used" more.
        
        Args:
            generated_text: Accumulated generated text from LLM
            
        Returns:
            Number of chunks that were marked as used (usage_score > 0.7)
        """
        if not generated_text or not generated_text.strip():
            return 0
        
        model = self._get_embedding_model()
        if model is None:
            return 0
        
        try:
            import numpy as np
            
            # Generate embedding for generated text
            gen_embedding = model.encode(generated_text, convert_to_numpy=True, show_progress_bar=False)
            gen_vec = np.array(gen_embedding)
            gen_norm = np.linalg.norm(gen_vec)
            
            if gen_norm == 0:
                return 0
            
            used_count = 0
            for tokenized in self.buffer:
                # Generate embedding for chunk text
                chunk_embedding = model.encode(tokenized.chunk.text, convert_to_numpy=True, show_progress_bar=False)
                chunk_vec = np.array(chunk_embedding)
                chunk_norm = np.linalg.norm(chunk_vec)
                
                if chunk_norm == 0:
                    continue
                
                # Compute cosine similarity
                similarity = np.dot(gen_vec, chunk_vec) / (gen_norm * chunk_norm)
                
                # Update usage score (clamp to [0, 1])
                tokenized.usage_score = max(0.0, min(1.0, float(similarity)))
                
                if tokenized.usage_score > 0.7:
                    used_count += 1
            
            return used_count
        except Exception as e:
            print(f"[token-buffer] Error computing usage scores: {e}", file=sys.stderr)
            return 0
    
    def prune_by_usage(self, threshold: float = 0.7) -> List[TokenizedChunk]:
        """Prune chunks where usage_score > threshold.
        
        Args:
            threshold: Usage score threshold (chunks with score > threshold are pruned)
            
        Returns:
            List of pruned TokenizedChunk objects
        """
        pruned = []
        remaining = []
        
        for tokenized in self.buffer:
            if tokenized.usage_score > threshold:
                pruned.append(tokenized)
                self.current_token_count -= tokenized.token_count
            else:
                remaining.append(tokenized)
        
        self.buffer = remaining
        return pruned
    
    def get_unused_chunks(self, threshold: float = 0.3) -> List[TokenizedChunk]:
        """Get chunks that haven't been used yet (low usage_score).
        
        Args:
            threshold: Maximum usage_score to consider a chunk "unused"
            
        Returns:
            List of TokenizedChunk objects with usage_score <= threshold
        """
        return [tokenized for tokenized in self.buffer if tokenized.usage_score <= threshold]
    
    def get_used_chunks(self, threshold: float = 0.7) -> List[TokenizedChunk]:
        """Get chunks that have been used (high usage_score).
        
        Args:
            threshold: Minimum usage_score to consider a chunk "used"
            
        Returns:
            List of TokenizedChunk objects with usage_score >= threshold
        """
        return [tokenized for tokenized in self.buffer if tokenized.usage_score >= threshold]

