"""Sentence-level fact verification for RAG system."""

import sys
import re
from typing import List, Optional, Tuple
from dataclasses import dataclass

from kiwix_chat.models import Chunk
from kiwix_chat.rag.embeddings import generate_embedding, initialize_embedding_model


@dataclass
class VerificationResult:
    """Result of verifying a sentence against source chunks."""
    sentence: str
    is_verified: bool
    confidence: float  # 0.0 to 1.0, higher = more confident
    matched_chunk: Optional[Chunk] = None
    matched_text: Optional[str] = None
    claims: List[str] = None  # Extracted factual claims
    
    def __post_init__(self):
        if self.claims is None:
            self.claims = []


class FactVerifier:
    """Verifies factual claims in sentences against source chunks."""
    
    def __init__(self):
        """Initialize fact verifier."""
        self._embedding_model = None
    
    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = initialize_embedding_model()
        return self._embedding_model
    
    def extract_factual_claims(self, sentence: str) -> List[str]:
        """Extract factual claims from a sentence.
        
        Looks for:
        - Numbers (including with units: "29.78 km/s", "149.6 million km")
        - Dates (various formats)
        - Names (capitalized words/phrases)
        - Locations (capitalized words that might be places)
        - Specific factual statements
        
        Args:
            sentence: Sentence to extract claims from
            
        Returns:
            List of extracted factual claims (strings)
        """
        claims = []
        
        # Extract numbers with units (e.g., "29.78 km/s", "149.6 million km")
        number_patterns = [
            r'\d+\.?\d*\s*(?:km/s|km/h|m/s|mph|km|m|cm|mm|kg|g|mg|years?|months?|days?|hours?|minutes?)',
            r'\d+\.?\d*\s*(?:million|billion|trillion)\s*(?:km|m|years?|dollars?|people)',
            r'\d+\.?\d*%',
        ]
        for pattern in number_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            claims.extend(matches)
        
        # Extract dates (with better precision)
        date_patterns = [
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(?:19|20)\d{2}\b',  # Years
        ]
        for pattern in date_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            claims.extend(matches)
        
        # Extract full date phrases (e.g., "September 13, 1996" or "on September 7 1996")
        # This helps catch date errors more precisely
        full_date_pattern = r'\b(?:on|in|during|at)\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        full_dates = re.findall(full_date_pattern, sentence, re.IGNORECASE)
        claims.extend(full_dates)
        
        # Extract capitalized names/phrases (likely proper nouns)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        names = re.findall(name_pattern, sentence)
        common_starters = {'The', 'This', 'That', 'These', 'Those', 'It', 'They', 'We', 'You'}
        names = [n for n in names if not any(n.startswith(c + ' ') for c in common_starters)]
        claims.extend(names[:5])
        
        # Extract vehicle models and brands (generic pattern)
        vehicle_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+\d+\b'
        vehicles = re.findall(vehicle_pattern, sentence)
        claims.extend(vehicles[:3])  # Limit to avoid noise
        
        # Extract specific locations (generic pattern)
        location_patterns = [
            r'\b[A-Z][a-z]+\s+(?:Hotel|Arena|Center|Hospital|Casino|Road|Lane|Street|Avenue)\b'
        ]
        for pattern in location_patterns:
            locations = re.findall(pattern, sentence)
            claims.extend(locations[:3])  # Limit to avoid noise
        
        # Extract event names (boxing matches, etc.)
        event_pattern = r'\b(?:between|vs\.?|versus)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:and|vs\.?|versus)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        events = re.findall(event_pattern, sentence)
        for event in events:
            claims.extend([f"{event[0]} vs {event[1]}"])
        
        # Extract specific factual statements (sentences with numbers or dates)
        if re.search(r'\d+', sentence) or re.search(r'\b(?:19|20)\d{2}\b', sentence):
            # Extract the core claim (subject + verb + object with fact)
            # Simple heuristic: take the part of sentence containing the number/date
            for claim in claims:
                if claim in sentence:
                    # Find surrounding context (10 words before/after)
                    words = sentence.split()
                    try:
                        claim_idx = sentence.lower().find(claim.lower())
                        if claim_idx >= 0:
                            # Extract surrounding context
                            start = max(0, claim_idx - 50)
                            end = min(len(sentence), claim_idx + len(claim) + 50)
                            context = sentence[start:end].strip()
                            if context and context not in claims:
                                claims.append(context)
                    except:
                        pass
        
        # Remove duplicates while preserving order
        seen = set()
        unique_claims = []
        for claim in claims:
            claim_lower = claim.lower().strip()
            if claim_lower and claim_lower not in seen:
                seen.add(claim_lower)
                unique_claims.append(claim.strip())
        
        return unique_claims
    
    def check_claim_against_source(self, claim: str, chunk: Chunk) -> float:
        """Check how well a claim matches a source chunk using semantic similarity.
        
        For dates and numbers, also performs exact matching for higher precision.
        
        Args:
            claim: Factual claim to check
            chunk: Source chunk to check against
            
        Returns:
            Similarity score (0.0 to 1.0), higher = better match
        """
        if not claim or not chunk.text:
            return 0.0
        
        try:
            import numpy as np
            
            # For dates and numbers, check for exact matches first (higher precision)
            claim_lower = claim.lower().strip()
            chunk_lower = chunk.text.lower()
            
            # Check if claim contains a date pattern
            date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            claim_dates = re.findall(date_pattern, claim, re.IGNORECASE)
            
            if claim_dates:
                # For dates, check exact match first
                chunk_dates = re.findall(date_pattern, chunk.text, re.IGNORECASE)
                if chunk_dates:
                    # Normalize dates for comparison (remove commas, normalize spacing)
                    claim_date_norm = re.sub(r'[,\s]+', ' ', claim_dates[0]).strip()
                    chunk_dates_norm = [re.sub(r'[,\s]+', ' ', d).strip() for d in chunk_dates]
                    
                    if claim_date_norm in chunk_dates_norm:
                        # Exact date match - high confidence
                        return 0.95
                    else:
                        # Date mismatch - this is a critical error, return very low score
                        # Dates must match exactly, semantic similarity doesn't help here
                        print(f"[verifier] ⚠ Date mismatch detected: '{claim_date_norm}' not found in source dates {chunk_dates_norm}", file=sys.stderr)
                        return 0.2  # Very low score for date mismatches
            
            # For proper nouns, check exact match (generic - no domain-specific knowledge)
            claim_lower = claim.lower()
            chunk_lower = chunk.text.lower()
            
            # Check if claim is a proper noun (starts with capital, multiple words)
            is_proper_noun = bool(re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', claim))
            
            if is_proper_noun:
                # For proper nouns, require exact match (case-insensitive)
                if claim_lower in chunk_lower:
                    return 0.9  # High confidence for exact match
                else:
                    return 0.3  # Low - proper noun not found
            
            # For other claims, use semantic similarity
            claim_embedding = generate_embedding(claim, is_query=True)
            chunk_embedding = generate_embedding(chunk.text, is_query=False)
            
            # Compute cosine similarity
            claim_vec = np.array(claim_embedding)
            chunk_vec = np.array(chunk_embedding)
            
            claim_norm = np.linalg.norm(claim_vec)
            chunk_norm = np.linalg.norm(chunk_vec)
            
            if claim_norm == 0 or chunk_norm == 0:
                return 0.0
            
            similarity = np.dot(claim_vec, chunk_vec) / (claim_norm * chunk_norm)
            return float(similarity)
        except Exception as e:
            print(f"[verifier] Error checking claim: {e}", file=sys.stderr)
            return 0.0
    
    def verify_sentence(self, sentence: str, source_chunks: List[Chunk], similarity_threshold: float = 0.6) -> VerificationResult:
        """Verify a sentence against source chunks.
        
        Process:
        1. Extract factual claims from sentence
        2. For each claim, find best matching source chunk
        3. Compute overall confidence based on claim matches
        4. Return verification result
        
        Args:
            sentence: Sentence to verify
            source_chunks: List of source chunks to check against
            similarity_threshold: Minimum similarity score to consider a match (0.0 to 1.0)
            
        Returns:
            VerificationResult with verification status and confidence
        """
        if not sentence or not sentence.strip():
            return VerificationResult(
                sentence=sentence,
                is_verified=False,
                confidence=0.0
            )
        
        # Extract factual claims
        claims = self.extract_factual_claims(sentence)
        
        if not claims:
            # No factual claims found - might be narrative/opinion
            # Check if sentence semantically matches any source
            best_match = None
            best_score = 0.0
            best_chunk = None
            
            for chunk in source_chunks:
                score = self.check_claim_against_source(sentence, chunk)
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
                    best_match = chunk.text[:200]  # First 200 chars
            
            # If semantic similarity is high, consider verified
            is_verified = best_score >= similarity_threshold
            return VerificationResult(
                sentence=sentence,
                is_verified=is_verified,
                confidence=best_score,
                matched_chunk=best_chunk,
                matched_text=best_match,
                claims=[]
            )
        
        # Check each claim against sources
        claim_scores = []
        best_chunk = None
        best_match_text = None
        best_overall_score = 0.0
        
        for claim in claims:
            best_claim_score = 0.0
            best_claim_chunk = None
            
            for chunk in source_chunks:
                score = self.check_claim_against_source(claim, chunk)
                if score > best_claim_score:
                    best_claim_score = score
                    best_claim_chunk = chunk
            
            claim_scores.append(best_claim_score)
            
            # Track overall best match
            if best_claim_score > best_overall_score:
                best_overall_score = best_claim_score
                best_chunk = best_claim_chunk
                if best_chunk:
                    # Find the part of chunk text that matches
                    claim_lower = claim.lower()
                    chunk_lower = best_chunk.text.lower()
                    if claim_lower in chunk_lower:
                        idx = chunk_lower.find(claim_lower)
                        start = max(0, idx - 50)
                        end = min(len(best_chunk.text), idx + len(claim) + 50)
                        best_match_text = best_chunk.text[start:end].strip()
        
        # Overall confidence: average of claim scores, weighted by number of claims
        if claim_scores:
            avg_score = sum(claim_scores) / len(claim_scores)
            # Boost confidence if multiple claims match well
            confidence = min(1.0, avg_score * (1.0 + 0.1 * len(claim_scores)))
        else:
            confidence = 0.0
        
        # Verified if average score meets threshold
        is_verified = confidence >= similarity_threshold
        
        return VerificationResult(
            sentence=sentence,
            is_verified=is_verified,
            confidence=confidence,
            matched_chunk=best_chunk,
            matched_text=best_match_text,
            claims=claims
        )

