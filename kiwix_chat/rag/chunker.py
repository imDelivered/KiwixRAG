"""Article chunking strategy for RAG system."""

import re
import sys
from typing import List, Optional, Callable

from kiwix_chat.models import Chunk


def chunk_article(
    text: str,
    title: str,
    href: str,
    max_chunk_size: int = 500,
    overlap: int = 50,
    content_type: str = "Kiwix content",
    tokenizer: Optional[Callable[[str], List[int]]] = None
) -> List[Chunk]:
    """Split article text into chunks with overlap.
    
    Strategy:
    - Split by sentences, respect paragraph boundaries
    - Preserve article metadata in each chunk
    - Use overlap to maintain context between chunks
    - If tokenizer provided, respects token boundaries
    
    Args:
        text: Full article text
        title: Article title
        href: Article href/path
        max_chunk_size: Maximum characters per chunk (or tokens if tokenizer provided)
        overlap: Number of characters (or tokens) to overlap between chunks
        content_type: Type of content (e.g., "Wikipedia articles")
        tokenizer: Optional tokenizer function (text -> List[int]) for token-aware chunking
        
    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []
    
    # Clean text: normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # If tokenizer provided, use token-aware chunking
    if tokenizer is not None:
        return chunk_by_tokens(text, title, href, max_chunk_size, overlap, content_type, tokenizer)
    
    # If text is small enough, return single chunk
    if len(text) <= max_chunk_size:
        return [Chunk(
            text=text,
            article_title=title,
            href=href,
            chunk_idx=0,
            total_chunks=1,
            content_type=content_type
        )]
    
    # Split into sentences (preserve sentence boundaries)
    # Pattern: sentence ending (works for most languages: . ! ? and Unicode equivalents)
    # Also handles: 。 (Chinese/Japanese), ؟ (Arabic), ։ (Armenian), etc.
    sentences = re.split(r'([.!?。？！؟։\u0964\u0965]+\s*)', text)
    
    # Recombine sentences with their punctuation
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])
    if len(sentences) % 2 == 1:
        combined_sentences.append(sentences[-1])
    
    # Build chunks from sentences
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in combined_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_size = len(sentence)
        
        # If single sentence exceeds max_chunk_size, split it by words
        if sentence_size > max_chunk_size:
            # First, save current chunk if it has content
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    article_title=title,
                    href=href,
                    chunk_idx=len(chunks),
                    total_chunks=0,  # Will update later
                    content_type=content_type
                ))
                current_chunk = []
                current_size = 0
            
            # Split long sentence by words
            words = sentence.split()
            word_chunk = []
            word_size = 0
            
            for word in words:
                word_len = len(word) + 1  # +1 for space
                if word_size + word_len > max_chunk_size and word_chunk:
                    chunk_text = ' '.join(word_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        article_title=title,
                        href=href,
                        chunk_idx=len(chunks),
                        total_chunks=0,
                        content_type=content_type
                    ))
                    # Start new chunk with overlap
                    overlap_words = word_chunk[-overlap//10:] if len(word_chunk) > overlap//10 else word_chunk
                    word_chunk = overlap_words + [word]
                    word_size = sum(len(w) + 1 for w in word_chunk)
                else:
                    word_chunk.append(word)
                    word_size += word_len
            
            # Add remaining words
            if word_chunk:
                chunk_text = ' '.join(word_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    article_title=title,
                    href=href,
                    chunk_idx=len(chunks),
                    total_chunks=0,
                    content_type=content_type
                ))
            
            continue
        
        # Check if adding this sentence would exceed max_chunk_size
        if current_size + sentence_size > max_chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                article_title=title,
                href=href,
                chunk_idx=len(chunks),
                total_chunks=0,  # Will update later
                content_type=content_type
            ))
            
            # Start new chunk with overlap (last N characters from previous chunk)
            if overlap > 0 and len(chunk_text) > overlap:
                overlap_text = chunk_text[-overlap:]
                # Try to start at word boundary
                overlap_start = overlap_text.find(' ')
                if overlap_start > 0:
                    overlap_text = overlap_text[overlap_start + 1:]
                current_chunk = [overlap_text, sentence]
                current_size = len(overlap_text) + sentence_size + 1
            else:
                current_chunk = [sentence]
                current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunks.append(Chunk(
            text=chunk_text,
            article_title=title,
            href=href,
            chunk_idx=len(chunks),
            total_chunks=0,
            content_type=content_type
        ))
    
    # Update total_chunks for all chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total
    
    return chunks


def chunk_by_tokens(
    text: str,
    title: str,
    href: str,
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    content_type: str = "Kiwix content",
    tokenizer: Callable[[str], List[int]] = None
) -> List[Chunk]:
    """Split article text into chunks based on token count.
    
    This method respects token boundaries and creates chunks optimized
    for actual token counts rather than character counts. This is more
    accurate for models that have specific token limits.
    
    Strategy:
    - Split by sentences first (preserve sentence boundaries)
    - Tokenize each sentence
    - Build chunks that respect token limits
    - Use overlap to maintain context
    
    Args:
        text: Full article text
        title: Article title
        href: Article href/path
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Number of tokens to overlap between chunks
        content_type: Type of content (e.g., "Wikipedia articles")
        tokenizer: Tokenizer function (text -> List[int])
        
    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        return []
    
    if tokenizer is None:
        # Fallback to character-based chunking
        return chunk_article(text, title, href, max_chunk_size=max_tokens * 4, overlap=overlap_tokens * 4, content_type=content_type)
    
    # Clean text: normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split into sentences (preserve sentence boundaries)
    # Pattern: sentence ending (works for most languages: . ! ? and Unicode equivalents)
    sentences = re.split(r'([.!?。？！؟։\u0964\u0965]+\s*)', text)
    
    # Recombine sentences with their punctuation
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])
    if len(sentences) % 2 == 1:
        combined_sentences.append(sentences[-1])
    
    # Tokenize all sentences
    sentence_tokens = []
    for sentence in combined_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        try:
            tokens = tokenizer(sentence)
            sentence_tokens.append((sentence, tokens, len(tokens)))
        except Exception as e:
            print(f"[chunker] Tokenization error: {e}, using character estimation", file=sys.stderr)
            # Fallback: estimate tokens (rough: 1 token ≈ 4 chars)
            est_tokens = len(sentence) // 4
            sentence_tokens.append((sentence, [], est_tokens))
    
    if not sentence_tokens:
        return []
    
    # Check if entire text fits in one chunk
    total_tokens = sum(count for _, _, count in sentence_tokens)
    if total_tokens <= max_tokens:
        return [Chunk(
            text=text,
            article_title=title,
            href=href,
            chunk_idx=0,
            total_chunks=1,
            content_type=content_type
        )]
    
    # Build chunks from tokenized sentences
    chunks = []
    current_chunk_sentences = []
    current_token_count = 0
    overlap_sentences = []  # Sentences to include in next chunk for overlap
    
    for sentence, tokens, token_count in sentence_tokens:
        # If single sentence exceeds max_tokens, split it by words
        if token_count > max_tokens:
            # Save current chunk if it has content
            if current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    article_title=title,
                    href=href,
                    chunk_idx=len(chunks),
                    total_chunks=0,  # Will update later
                    content_type=content_type
                ))
                current_chunk_sentences = []
                current_token_count = 0
            
            # Split long sentence by words (character-based fallback)
            words = sentence.split()
            word_chunk = []
            word_token_count = 0
            
            for word in words:
                try:
                    word_tokens = tokenizer(word + ' ')
                    word_token_count_inc = len(word_tokens)
                except:
                    word_token_count_inc = len(word) // 4  # Fallback estimate
                
                if word_token_count + word_token_count_inc > max_tokens and word_chunk:
                    chunk_text = ' '.join(word_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        article_title=title,
                        href=href,
                        chunk_idx=len(chunks),
                        total_chunks=0,
                        content_type=content_type
                    ))
                    # Start new chunk with overlap
                    overlap_words = word_chunk[-overlap_tokens//10:] if len(word_chunk) > overlap_tokens//10 else word_chunk
                    word_chunk = overlap_words + [word]
                    try:
                        word_token_count = sum(len(tokenizer(w + ' ')) for w in word_chunk)
                    except:
                        word_token_count = sum(len(w) // 4 for w in word_chunk)
                else:
                    word_chunk.append(word)
                    word_token_count += word_token_count_inc
            
            # Add remaining words
            if word_chunk:
                chunk_text = ' '.join(word_chunk)
                chunks.append(Chunk(
                    text=chunk_text,
                    article_title=title,
                    href=href,
                    chunk_idx=len(chunks),
                    total_chunks=0,
                    content_type=content_type
                ))
            
            continue
        
        # Check if adding this sentence would exceed max_tokens
        if current_token_count + token_count > max_tokens and current_chunk_sentences:
            # Save current chunk
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                article_title=title,
                href=href,
                chunk_idx=len(chunks),
                total_chunks=0,
                content_type=content_type
            ))
            
            # Prepare overlap for next chunk
            if overlap_tokens > 0:
                # Find sentences that fit in overlap_tokens
                # Need to look back at sentence_tokens to get token counts
                overlap_sentences = []
                overlap_token_count = 0
                # Find sentences from current_chunk_sentences in sentence_tokens
                for sent_text, _, sent_tokens in reversed(sentence_tokens):
                    if sent_text in current_chunk_sentences:
                        if overlap_token_count + sent_tokens <= overlap_tokens:
                            overlap_sentences.insert(0, sent_text)
                            overlap_token_count += sent_tokens
                        else:
                            break
                
                current_chunk_sentences = overlap_sentences + [sentence]
                current_token_count = overlap_token_count + token_count
            else:
                current_chunk_sentences = [sentence]
                current_token_count = token_count
        else:
            current_chunk_sentences.append(sentence)
            current_token_count += token_count
    
    # Add final chunk
    if current_chunk_sentences:
        chunk_text = ' '.join(current_chunk_sentences)
        chunks.append(Chunk(
            text=chunk_text,
            article_title=title,
            href=href,
            chunk_idx=len(chunks),
            total_chunks=0,
            content_type=content_type
        ))
    
    # Update total_chunks for all chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total
    
    return chunks

